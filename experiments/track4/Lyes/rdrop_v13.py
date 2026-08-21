"""Resumable, gated CRF-marginal R-Drop v13 experiment campaign."""

import argparse
import copy
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from utils.track4.Lyes.gated_fusion.config import load_gates
from utils.track4.Lyes.gated_fusion.fusion import predict_with_gated_fallback
from experiments.track4.Lyes.campaign.common import write_prediction_artifacts
from experiments.track4.Lyes.campaign.diagnostics import (
    prediction_diagnostics,
    training_word_types,
)
from experiments.track4.Lyes.campaign.export import export_architecture_ensemble
from experiments.track4.Lyes.campaign.folds import make_balanced_folds, write_records_jsonl
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.config import load_config, validate_config
from utils.track4.Lyes.data import SentenceRecord, load_jsonl, load_raw_sentences
from experiments.track4.Lyes.export_ensemble import (
    checkpoint_groups,
    load_v7_config,
)
from evaluation.track4.Lyes.infer import predict_records
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from evaluation.track4.Lyes.paper_metrics import compute_paper_metrics
from training.track4.Lyes.train import train
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


EXPECTED_CONFIG_KEYS = {
    "schema_version",
    "base_config",
    "train_data",
    "released_dev_data",
    "vocab",
    "v2_gates",
    "v7_campaign",
    "output_root",
    "rdrop_distribution",
    "training_overrides",
    "split_count",
    "split_seeds",
    "coefficients",
    "calibration",
    "final_acceptance",
}
PROTECTED_KEYS = (
    "word_accuracy",
    "sentence_accuracy",
    "oov_accuracy",
    "shadda_accuracy",
    "tanween_accuracy",
)


def load_campaign_config(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != EXPECTED_CONFIG_KEYS:
        raise ValueError("invalid R-Drop v13 campaign configuration keys")
    if int(raw["schema_version"]) != 1:
        raise ValueError("unsupported R-Drop v13 campaign schema")
    seeds = raw["split_seeds"]
    if not isinstance(seeds, dict) or set(seeds) != {"a", "b"}:
        raise ValueError("split_seeds must contain exactly a and b")
    coefficients = [float(value) for value in raw["coefficients"]]
    if coefficients != [0.0, 0.1, 0.3, 1.0]:
        raise ValueError("coefficients must be exactly [0, 0.1, 0.3, 1]")
    if int(raw["split_count"]) != 5:
        raise ValueError("R-Drop calibration requires exactly five folds")
    if raw["rdrop_distribution"] != "emission":
        raise ValueError("v13 campaign is locked to emission R-Drop")
    if raw["training_overrides"] != {
        "batch_size": 32,
        "gradient_accumulation_steps": 2,
    }:
        raise ValueError("v13 training overrides are locked to batch 32 x 2")
    return raw


def _coefficient_key(value: float) -> str:
    normalized = "{:.6g}".format(float(value)).replace(".", "p")
    return "lambda_{}".format(normalized)


def _split_records(
    records: Sequence[SentenceRecord], fold_count: int, seed: int
) -> Tuple[List[SentenceRecord], List[SentenceRecord], Mapping[str, Any]]:
    folds = make_balanced_folds(records, fold_count, seed)
    calibration_indices = set(folds[0])
    inner_train = [
        record for index, record in enumerate(records)
        if index not in calibration_indices
    ]
    calibration = [
        record for index, record in enumerate(records)
        if index in calibration_indices
    ]
    train_ids = {record.sent_id for record in inner_train}
    calibration_ids = {record.sent_id for record in calibration}
    if train_ids & calibration_ids:
        raise RuntimeError("calibration split leaks sentence IDs")
    manifest = {
        "seed": seed,
        "fold_count": fold_count,
        "calibration_fold": 0,
        "train_sentences": len(inner_train),
        "calibration_sentences": len(calibration),
        "train_scored_letters": sum(
            char != " " for record in inner_train for char in record.chars
        ),
        "calibration_scored_letters": sum(
            char != " " for record in calibration for char in record.chars
        ),
        "sentence_disjoint": True,
    }
    return inner_train, calibration, manifest


def prepare_splits(config: Mapping[str, Any]) -> Mapping[str, Any]:
    output_root = Path(str(config["output_root"]))
    manifest_path = output_root / "00_splits" / "MANIFEST.json"
    train_records = load_jsonl(Path(str(config["train_data"])))
    result: Dict[str, Any] = {"schema_version": 1, "splits": {}}
    for name in ("a", "b"):
        split_dir = output_root / "00_splits" / "split_{}".format(name)
        train_path = split_dir / "train.jsonl"
        calibration_path = split_dir / "calibration.jsonl"
        inner_train, calibration, split_manifest = _split_records(
            train_records,
            int(config["split_count"]),
            int(config["split_seeds"][name]),
        )
        if train_path.exists() or calibration_path.exists():
            if not train_path.is_file() or not calibration_path.is_file():
                raise RuntimeError("partial calibration split: {}".format(name))
        else:
            write_records_jsonl(train_path, inner_train)
            write_records_jsonl(calibration_path, calibration)
        split_manifest = dict(split_manifest)
        split_manifest.update(
            {
                "train_path": str(train_path),
                "train_sha256": sha256_file(train_path),
                "calibration_path": str(calibration_path),
                "calibration_sha256": sha256_file(calibration_path),
            }
        )
        result["splits"][name] = split_manifest
    write_json(manifest_path, result)
    return result


def _training_config(
    campaign: Mapping[str, Any],
    train_path: Path,
    dev_path: Path,
    output_dir: Path,
    coefficient: float,
    device_name: str,
    epochs: Optional[int] = None,
    final_only: bool = False,
) -> Dict[str, Any]:
    config = load_config(Path(str(campaign["base_config"])))
    config["data"]["train"] = str(train_path)
    config["data"]["dev"] = str(dev_path)
    config["data"]["vocab"] = str(campaign["vocab"])
    config["output_dir"] = str(output_dir)
    config["training"]["device"] = device_name
    config["training"]["rdrop_coefficient"] = float(coefficient)
    config["training"]["rdrop_distribution"] = str(
        campaign["rdrop_distribution"]
    )
    for key, value in campaign["training_overrides"].items():
        config["training"][str(key)] = value
    if epochs is not None:
        config["training"]["epochs"] = int(epochs)
    if final_only:
        config["training"]["selection_mode"] = "last_epoch"
        config["training"]["dev_evaluation_mode"] = "final_only"
    validate_config(config)
    return config


def _run_training(config: Mapping[str, Any]) -> Mapping[str, Any]:
    output_dir = Path(str(config["output_dir"]))
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "best.pt"
    if summary_path.is_file() and checkpoint_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            "partial training artifacts require manual inspection: {}".format(
                output_dir
            )
        )
    return train(config)


def _flatten_metrics(
    paper: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "correct": int(paper["correct_letters"]),
        "total": int(paper["scored_letters"]),
        "micro_f1": float(paper["micro_f1"]),
        "macro_f1": float(paper["macro_f1"]),
        "word_accuracy": float(paper["word_accuracy"]),
        "sentence_accuracy": float(paper["sentence_accuracy"]),
        "oov_accuracy": float(diagnostics["oov_accuracy"]),
        "seen_accuracy": float(diagnostics["seen_accuracy"]),
        "shadda_accuracy": float(paper["shadda"]["accuracy"]),
        "tanween_accuracy": float(paper["tanween"]["accuracy"]),
        "skeleton_mismatch_count": int(paper["skeleton_mismatch_count"]),
        "paper_metrics": dict(paper),
        "diagnostics": dict(diagnostics),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    train_records: Sequence[SentenceRecord],
    evaluation_records: Sequence[SentenceRecord],
    gates_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(gates_path)
    seen_words = training_word_types(train_records)
    neural = predict_records(
        model,
        evaluation_records,
        vocab,
        device,
        batch_size,
        num_workers,
    )
    v2, gate_statistics = predict_with_gated_fallback(
        model,
        evaluation_records,
        vocab,
        prior,
        gates,
        device,
        batch_size,
        num_workers,
    )
    neural_paper = compute_paper_metrics(evaluation_records, neural)
    v2_paper = compute_paper_metrics(evaluation_records, v2)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "neural": _flatten_metrics(
            neural_paper,
            prediction_diagnostics(evaluation_records, neural, seen_words),
        ),
        "v2": _flatten_metrics(
            v2_paper,
            prediction_diagnostics(evaluation_records, v2, seen_words),
        ),
        "v2_gate_statistics": gate_statistics.to_dict(),
    }


def _protected_regressions(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> Mapping[str, bool]:
    return {
        key: float(candidate[key]) < float(control[key])
        for key in PROTECTED_KEYS
    }


def _split_decision(
    coefficient: float,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    minimum_gain: int,
) -> Mapping[str, Any]:
    neural_gain = int(candidate["neural"]["correct"]) - int(
        control["neural"]["correct"]
    )
    v2_gain = int(candidate["v2"]["correct"]) - int(
        control["v2"]["correct"]
    )
    regressions = _protected_regressions(
        candidate["v2"], control["v2"]
    )
    accepted = (
        neural_gain >= minimum_gain
        and v2_gain >= 0
        and not any(regressions.values())
        and int(candidate["neural"]["skeleton_mismatch_count"]) == 0
        and int(candidate["v2"]["skeleton_mismatch_count"]) == 0
    )
    return {
        "coefficient": coefficient,
        "neural_correct_gain": neural_gain,
        "v2_correct_gain": v2_gain,
        "protected_v2_regressions": regressions,
        "accepted": accepted,
    }


def _run_calibration_member(
    campaign: Mapping[str, Any],
    split_name: str,
    coefficient: float,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    root = Path(str(campaign["output_root"]))
    split_dir = root / "00_splits" / "split_{}".format(split_name)
    output_dir = (
        root
        / ("01_split_a" if split_name == "a" else "02_split_b")
        / _coefficient_key(coefficient)
    )
    config = _training_config(
        campaign,
        split_dir / "train.jsonl",
        split_dir / "calibration.jsonl",
        output_dir,
        coefficient,
        device_name,
    )
    summary = _run_training(config)
    evaluation_path = output_dir / "evaluation.json"
    if evaluation_path.is_file():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    else:
        evaluation = evaluate_checkpoint(
            output_dir / "best.pt",
            load_jsonl(split_dir / "train.jsonl"),
            load_jsonl(split_dir / "calibration.jsonl"),
            Path(str(campaign["v2_gates"])),
            select_device(device_name),
            batch_size,
            num_workers,
        )
        write_json(evaluation_path, evaluation)
    return {"summary": dict(summary), "evaluation": dict(evaluation)}


def run_split_a(
    campaign: Mapping[str, Any],
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    prepare_splits(campaign)
    results = {
        str(value): _run_calibration_member(
            campaign, "a", float(value), device_name, batch_size, num_workers
        )
        for value in campaign["coefficients"]
    }
    control = results["0.0"]["evaluation"]
    decisions = [
        _split_decision(
            float(value),
            results[str(value)]["evaluation"],
            control,
            int(campaign["calibration"]["minimum_split_a_correct_gain"]),
        )
        for value in campaign["coefficients"]
        if float(value) > 0.0
    ]
    accepted = [value for value in decisions if value["accepted"]]
    accepted.sort(
        key=lambda value: (
            -int(value["neural_correct_gain"]),
            float(value["coefficient"]),
        )
    )
    selection = {
        "schema_version": 1,
        "results": results,
        "decisions": decisions,
        "accepted": bool(accepted),
        "selected_coefficient": (
            float(accepted[0]["coefficient"]) if accepted else None
        ),
    }
    write_json(
        Path(str(campaign["output_root"])) / "01_split_a" / "SELECTION.json",
        selection,
    )
    return selection


def run_split_b(
    campaign: Mapping[str, Any],
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    split_a = run_split_a(campaign, device_name, batch_size, num_workers)
    coefficient = split_a["selected_coefficient"]
    if coefficient is None:
        result = {"schema_version": 1, "accepted": False, "reason": "split_a_failed"}
        write_json(
            Path(str(campaign["output_root"])) / "02_split_b" / "SELECTION.json",
            result,
        )
        return result
    control = _run_calibration_member(
        campaign, "b", 0.0, device_name, batch_size, num_workers
    )
    candidate = _run_calibration_member(
        campaign, "b", float(coefficient), device_name, batch_size, num_workers
    )
    decision = _split_decision(
        float(coefficient),
        candidate["evaluation"],
        control["evaluation"],
        1,
    )
    split_a_decision = next(
        value for value in split_a["decisions"]
        if float(value["coefficient"]) == float(coefficient)
    )
    gains = [
        int(split_a_decision["neural_correct_gain"]),
        int(decision["neural_correct_gain"]),
    ]
    mean_gain = sum(gains) / 2.0
    accepted = (
        decision["accepted"]
        and all(gain > 0 for gain in gains)
        and mean_gain >= int(
            campaign["calibration"]["minimum_mean_correct_gain"]
        )
    )
    result = {
        "schema_version": 1,
        "coefficient": coefficient,
        "control": control,
        "candidate": candidate,
        "decision": decision,
        "split_neural_correct_gains": gains,
        "mean_neural_correct_gain": mean_gain,
        "accepted": accepted,
    }
    write_json(
        Path(str(campaign["output_root"])) / "02_split_b" / "SELECTION.json",
        result,
    )
    return result


def _round_half_up(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("cannot average an empty epoch sequence")
    mean = Decimal(sum(values)) / Decimal(len(values))
    return int(mean.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _final_run(
    campaign: Mapping[str, Any],
    coefficient: float,
    epochs: int,
    key: str,
    device_name: str,
) -> Mapping[str, Any]:
    output_dir = Path(str(campaign["output_root"])) / "03_final_seed42" / key
    config = _training_config(
        campaign,
        Path(str(campaign["train_data"])),
        Path(str(campaign["released_dev_data"])),
        output_dir,
        coefficient,
        device_name,
        epochs=epochs,
        final_only=True,
    )
    return _run_training(config)


def _export_standalone(
    campaign: Mapping[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    train_records = load_jsonl(Path(str(campaign["train_data"])))
    test_records = load_raw_sentences(
        Path("Data/test_data/raw_sentences_test.txt"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
    )
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(Path(str(campaign["v2_gates"])))
    neural = predict_records(
        model, test_records, vocab, device, batch_size, num_workers
    )
    v2, statistics = predict_with_gated_fallback(
        model,
        test_records,
        vocab,
        prior,
        gates,
        device,
        batch_size,
        num_workers,
    )
    artifacts_dir = output_dir / "artifacts"
    prefix = "DZIRIFORMER_DUALROPE_CRF_EMISSION_RDROP_V13_SEED42"
    neural_artifacts = write_prediction_artifacts(
        artifacts_dir,
        prefix + "_NEURAL",
        test_records,
        neural,
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    v2_artifacts = write_prediction_artifacts(
        artifacts_dir,
        prefix + "_V2",
        test_records,
        v2,
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    result = {
        "system_name": "DziriFormer-DualRoPE-CRF-EmissionRDrop-v13-seed42",
        "artifact_prefix": prefix,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "v2_gate_statistics": statistics.to_dict(),
        "neural_artifacts": dict(neural_artifacts),
        "v2_artifacts": dict(v2_artifacts),
    }
    write_json(artifacts_dir / (prefix + "_MANIFEST.json"), result)
    return result


def run_final(
    campaign: Mapping[str, Any],
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    split_b = run_split_b(campaign, device_name, batch_size, num_workers)
    root = Path(str(campaign["output_root"]))
    final_dir = root / "03_final_seed42"
    if not split_b.get("accepted", False):
        result = {"schema_version": 1, "accepted": False, "reason": "calibration_failed"}
        write_json(final_dir / "SELECTION.json", result)
        return result
    coefficient = float(split_b["coefficient"])
    split_a = json.loads(
        (root / "01_split_a" / "SELECTION.json").read_text(encoding="utf-8")
    )
    epoch_a = int(
        split_a["results"][str(coefficient)]["summary"]["best_epoch"]
    )
    epoch_b = int(split_b["candidate"]["summary"]["best_epoch"])
    locked_epochs = _round_half_up([epoch_a, epoch_b])
    _final_run(
        campaign, 0.0, locked_epochs, "control_lambda_0", device_name
    )
    _final_run(
        campaign,
        coefficient,
        locked_epochs,
        "rdrop_{}".format(_coefficient_key(coefficient)),
        device_name,
    )
    train_records = load_jsonl(Path(str(campaign["train_data"])))
    dev_records = load_jsonl(Path(str(campaign["released_dev_data"])))
    control_checkpoint = final_dir / "control_lambda_0" / "best.pt"
    candidate_dir = final_dir / "rdrop_{}".format(
        _coefficient_key(coefficient)
    )
    candidate_checkpoint = candidate_dir / "best.pt"
    control = evaluate_checkpoint(
        control_checkpoint,
        train_records,
        dev_records,
        Path(str(campaign["v2_gates"])),
        select_device(device_name),
        batch_size,
        num_workers,
    )
    candidate = evaluate_checkpoint(
        candidate_checkpoint,
        train_records,
        dev_records,
        Path(str(campaign["v2_gates"])),
        select_device(device_name),
        batch_size,
        num_workers,
    )
    write_json(final_dir / "control_evaluation.json", control)
    write_json(final_dir / "candidate_evaluation.json", candidate)
    neural_gain = int(candidate["neural"]["correct"]) - int(
        control["neural"]["correct"]
    )
    v2_gain = int(candidate["v2"]["correct"]) - int(
        control["v2"]["correct"]
    )
    regressions = _protected_regressions(candidate["v2"], control["v2"])
    standalone_accepted = (
        neural_gain >= int(
            campaign["final_acceptance"]["minimum_neural_correct_gain"]
        )
        and v2_gain >= int(
            campaign["final_acceptance"]["minimum_v2_correct_gain"]
        )
        and not any(regressions.values())
        and int(candidate["v2"]["skeleton_mismatch_count"]) == 0
    )
    artifacts = _export_standalone(
        campaign,
        candidate_checkpoint,
        candidate_dir,
        device_name,
        batch_size,
        num_workers,
    )
    ensemble = None
    ensemble_accepted = False
    if standalone_accepted:
        v7_config = load_v7_config(Path(str(campaign["v7_campaign"])))
        groups = checkpoint_groups(v7_config, "crf_final")
        groups[0] = [candidate_checkpoint]
        ensemble = export_architecture_ensemble(
            groups,
            root / "04_final_ensemble",
            "DZIRI_FINAL_EMISSION_RDROP_ENSEMBLE_V13",
            "DziriFinal-EmissionRDrop-Ensemble-v13",
            select_device(device_name),
            batch_size,
            num_workers,
        )
        ensemble_accepted = int(ensemble["dev"]["v2"]["correct"]) >= int(
            campaign["final_acceptance"]["minimum_ensemble_v2_correct"]
        )
    result = {
        "schema_version": 1,
        "coefficient": coefficient,
        "locked_epochs": locked_epochs,
        "calibration_best_epochs": [epoch_a, epoch_b],
        "control": control,
        "candidate": candidate,
        "neural_correct_gain": neural_gain,
        "v2_correct_gain": v2_gain,
        "protected_v2_regressions": regressions,
        "standalone_accepted": standalone_accepted,
        "standalone_artifacts": artifacts,
        "ensemble": ensemble,
        "ensemble_accepted": ensemble_accepted,
        "accepted": standalone_accepted and ensemble_accepted,
    }
    write_json(final_dir / "SELECTION.json", result)
    write_results_markdown(campaign, result)
    return result


def write_results_markdown(
    campaign: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    root = Path(str(campaign["output_root"]))
    lines = [
        "# R-Drop v13 Results",
        "",
        "This report is generated from locked campaign artifacts.",
        "",
        "- Accepted: `{}`".format(result.get("accepted", False)),
        "- Coefficient: `{}`".format(result.get("coefficient")),
        "- Locked epochs: `{}`".format(result.get("locked_epochs")),
        "- Neural correct gain: `{}`".format(result.get("neural_correct_gain")),
        "- V2 correct gain: `{}`".format(result.get("v2_correct_gain")),
        "- Standalone accepted: `{}`".format(
            result.get("standalone_accepted", False)
        ),
        "- Final ensemble accepted: `{}`".format(
            result.get("ensemble_accepted", False)
        ),
        "",
        "Do not submit any v13 CSV unless both acceptance flags are true.",
    ]
    path = root / "FINAL_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/track4/Lyes/rdrop_v13/campaign.json"),
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "split-a", "split-b", "final", "all"),
        default="all",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    campaign = load_campaign_config(args.config)
    if args.stage == "prepare":
        result = prepare_splits(campaign)
    elif args.stage == "split-a":
        result = run_split_a(
            campaign, args.device, args.batch_size, args.num_workers
        )
    elif args.stage == "split-b":
        result = run_split_b(
            campaign, args.device, args.batch_size, args.num_workers
        )
    else:
        result = run_final(
            campaign, args.device, args.batch_size, args.num_workers
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
