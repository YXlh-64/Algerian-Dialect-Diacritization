"""Resumable pre-HGL/HGL campaign orchestrator."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from utils.track4.Lyes.gated_fusion.config import load_gates
from experiments.track4.Lyes.campaign.common import (
    load_campaign_config,
    write_prediction_artifacts,
)
from experiments.track4.Lyes.campaign.diagnostics import (
    prediction_diagnostics,
    training_word_types,
)
from experiments.track4.Lyes.campaign.ensemble import (
    predict_probability_members,
)
from experiments.track4.Lyes.campaign.export import (
    export_architecture_ensemble,
    export_checkpoint_ensemble,
)
from experiments.track4.Lyes.campaign.folds import (
    make_balanced_folds,
    write_records_jsonl,
)
from experiments.track4.Lyes.campaign.oof_gate import (
    LogisticGate,
    apply_logistic_gate,
    collect_training_examples,
    fit_logistic_gate,
)
from experiments.track4.Lyes.campaign.report import write_reports
from utils.track4.Lyes.checkpoint import load_checkpoint
from utils.track4.Lyes.config import load_config
from utils.track4.Lyes.data import load_jsonl, load_raw_sentences
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from training.track4.Lyes.train import train
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


BASE_CONFIGS = {
    "base": Path("configs/track4/Lyes/conv_local_transformer.json"),
    "j16": Path("configs/track4/Lyes/dziriformer_j16_gated_v3.json"),
    "gl": Path("configs/track4/Lyes/dziriformer_gl_v3.json"),
    "mixed": Path("configs/track4/Lyes/dziriformer_mixed_v3.json"),
    "hier": Path("configs/track4/Lyes/dziriformer_hier_v4.json"),
}


def _checkpoint_correct(path: Path) -> int:
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    return int(checkpoint["dev_metrics"]["correct"])


def _completed_training(directory: Path) -> bool:
    required = ("best.pt", "last.pt", "summary.json", "metrics.jsonl")
    existing = [directory / name for name in required]
    if all(path.is_file() for path in existing):
        return True
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"partial training directory: {directory}")
    return False


def _train_one(
    config_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
    train_path: Path = Path(
        "Data/train_data/train_Algerian-DIAC.jsonl"
    ),
    dev_path: Path = Path("Data/dev_data/dev_Algerian-DIAC.jsonl"),
    model_override: Mapping[str, Any] = None,
) -> Path:
    if not _completed_training(output_dir):
        config = load_config(config_path)
        config["seed"] = seed
        config["data"]["train"] = str(train_path)
        config["data"]["dev"] = str(dev_path)
        config["training"]["device"] = device
        config["output_dir"] = str(output_dir)
        config["model"].update(dict(model_override or {}))
        train(config)
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    if str(summary["device"]) != device:
        raise RuntimeError(
            f"{output_dir} trained on {summary['device']}, expected {device}"
        )
    return output_dir / "best.pt"


def _write_campaign_state(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    qualified: Mapping[str, Sequence[Path]],
    stage: str,
    extra: Mapping[str, Any] = None,
) -> None:
    write_json(
        output_root / "campaign_manifest.json",
        {
            "schema_version": 1,
            "stage": stage,
            "rows": list(rows),
            "qualified": sorted(qualified),
            **dict(extra or {}),
        },
    )


def _defer_oof(
    output_root: Path,
    reason: str,
) -> Dict[str, Any]:
    oof_root = output_root / "06_oof_gate"
    architecture_names = tuple(BASE_CONFIGS)
    completed = []
    partial = []
    for fold_index in range(5):
        for architecture in architecture_names:
            run_dir = oof_root / f"fold_{fold_index}" / architecture
            if (run_dir / "summary.json").is_file():
                completed.append(str(run_dir))
            elif run_dir.exists() and any(run_dir.iterdir()):
                partial.append(str(run_dir))
    state = {
        "status": "deferred_incomplete",
        "reason": reason,
        "completed_trainings": len(completed),
        "required_trainings": 25,
        "completed_run_directories": completed,
        "interrupted_partial_directories": partial,
        "gate_fitted": False,
        "valid_oof_result": False,
    }
    write_json(oof_root / "DEFERRED.json", state)
    return {
        "system": "OOF Gate",
        "seed": 42,
        "parameters": 9,
        "epoch": 0,
        "device": "",
        "runtime_seconds": 0.0,
        "neural_f1": 0.0,
        "neural_correct": 0,
        "final_f1": 0.0,
        "correct": 0,
        "diagnostics": {},
        "checkpoint_sha256": "",
        "submission_sha256": "",
        "status": f"deferred ({len(completed)}/25)",
        "submission_path": "",
    }


def _record_row(
    system: str,
    seed: int,
    run_dir: Path,
    manifest: Mapping[str, Any],
    status: str,
) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        parameters = int(summary["parameter_count"])
        epoch = int(summary["best_epoch"])
    else:
        summary = {}
        parameters = sum(
            int(load_checkpoint(path, torch.device("cpu"))["model_state_dict"][key].numel())
            for path in []
            for key in []
        )
        epoch = 0
    return {
        "system": system,
        "seed": seed,
        "parameters": parameters,
        "epoch": epoch,
        "device": str(summary.get("device", "")),
        "runtime_seconds": float(summary.get("elapsed_seconds", 0.0)),
        "neural_f1": float(manifest["dev"]["neural"]["micro_f1"]),
        "neural_correct": int(manifest["dev"]["neural"]["correct"]),
        "final_f1": float(manifest["dev"]["v2"]["micro_f1"]),
        "correct": int(manifest["dev"]["v2"]["correct"]),
        "diagnostics": manifest["dev"],
        "checkpoint_sha256": manifest["checkpoints"][0]["sha256"],
        "submission_sha256": manifest[
            "v2_artifacts"
        ]["submission_sha256"],
        "status": status,
        "submission_path": manifest["v2_artifacts"]["submission_path"],
    }


def _run_oof(
    output_root: Path,
    checkpoint_paths: Sequence[Path],
    device: torch.device,
    device_name: str,
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    oof_root = output_root / "06_oof_gate"
    gate_path = oof_root / "logistic_gate.json"
    train_records = load_jsonl(
        Path("Data/train_data/train_Algerian-DIAC.jsonl")
    )
    folds = make_balanced_folds(train_records, 5, 42)
    fold_features = []
    fold_targets = []
    for outer_index, outer_indices in enumerate(folds):
        fold_dir = oof_root / f"fold_{outer_index}"
        example_path = fold_dir / "gate_examples.pt"
        if example_path.is_file():
            examples = torch.load(example_path)
            fold_features.append(examples["features"])
            fold_targets.append(examples["targets"])
            continue
        outer_set = set(outer_indices)
        remaining = [
            record
            for index, record in enumerate(train_records)
            if index not in outer_set
        ]
        inner_folds = make_balanced_folds(remaining, 10, 420 + outer_index)
        inner_dev_indices = set(inner_folds[0])
        inner_dev = [
            record
            for index, record in enumerate(remaining)
            if index in inner_dev_indices
        ]
        inner_train = [
            record
            for index, record in enumerate(remaining)
            if index not in inner_dev_indices
        ]
        outer = [train_records[index] for index in outer_indices]
        data_dir = fold_dir / "data"
        train_path = data_dir / "train.jsonl"
        dev_path = data_dir / "inner_dev.jsonl"
        outer_path = data_dir / "outer.jsonl"
        write_records_jsonl(train_path, inner_train)
        write_records_jsonl(dev_path, inner_dev)
        write_records_jsonl(outer_path, outer)
        fold_checkpoints = []
        for name, config_path in BASE_CONFIGS.items():
            fold_checkpoints.append(
                _train_one(
                    config_path,
                    fold_dir / name,
                    seed=4200 + outer_index,
                    device=device_name,
                    train_path=train_path,
                    dev_path=dev_path,
                )
            )
        probabilities, member_votes, _ = predict_probability_members(
            fold_checkpoints, outer, device, 128, 0
        )
        prior = WordLabelPrior().fit(remaining)
        features, targets = collect_training_examples(
            outer,
            probabilities,
            member_votes,
            prior,
            smoothing=0.01,
        )
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"features": features, "targets": targets}, example_path
        )
        fold_features.append(features)
        fold_targets.append(targets)
    if gate_path.is_file():
        gate = LogisticGate.from_dict(
            json.loads(gate_path.read_text(encoding="utf-8"))
        )
    else:
        gate = fit_logistic_gate(
            torch.cat(fold_features), torch.cat(fold_targets)
        )
        write_json(gate_path, gate.to_dict())

    full_train = train_records
    prior = WordLabelPrior().fit(full_train)
    dev_records = load_jsonl(
        Path("Data/dev_data/dev_Algerian-DIAC.jsonl")
    )
    dev_probabilities, dev_votes, _ = predict_probability_members(
        checkpoint_paths, dev_records, device, 128, 0
    )
    dev_predictions = apply_logistic_gate(
        dev_records, dev_probabilities, dev_votes, prior, 0.01, gate
    )
    diagnostics = prediction_diagnostics(
        dev_records, dev_predictions, training_word_types(full_train)
    )
    test_records = load_raw_sentences(
        Path("Data/test_data/raw_sentences_test.txt"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
    )
    test_probabilities, test_votes, _ = predict_probability_members(
        checkpoint_paths, test_records, device, 128, 0
    )
    test_predictions = apply_logistic_gate(
        test_records, test_probabilities, test_votes, prior, 0.01, gate
    )
    artifacts = write_prediction_artifacts(
        oof_root / "artifacts",
        "DZIRI_ENSEMBLE_OOF_GATE_V3",
        test_records,
        test_predictions,
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    manifest = {
        "system_name": "DziriEnsemble-OOFGate-v3",
        "dev": {"neural": diagnostics, "v2": diagnostics},
        "v2_artifacts": dict(artifacts),
    }
    write_json(oof_root / "manifest.json", manifest)
    return manifest, {
        "system": "OOF Gate",
        "seed": 42,
        "parameters": 9,
        "epoch": 0,
        "neural_f1": diagnostics["micro_f1"],
        "final_f1": diagnostics["micro_f1"],
        "correct": diagnostics["correct"],
        "status": "candidate" if diagnostics["correct"] > 14935 else "rejected",
        "submission_path": artifacts["submission_path"],
    }


def run_campaign(config_path: Path, device_name: str) -> None:
    campaign = load_campaign_config(config_path)
    device = select_device(device_name)
    if device.type != "mps":
        raise RuntimeError("full campaign requires Apple MPS")
    output_root = Path(campaign["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    uniform_paths = [
        Path(path) for path in campaign["uniform_checkpoints"].values()
    ]
    uniform_dir = output_root / "01_uniform_ensemble"
    uniform_manifest = export_checkpoint_ensemble(
        uniform_paths,
        uniform_dir,
        "DZIRI_ENSEMBLE_UNIFORM_V3",
        "DziriEnsemble-Uniform-v3",
        device,
    )
    if int(uniform_manifest["dev"]["v2"]["correct"]) != 14935:
        raise RuntimeError("uniform ensemble reference score changed")
    rows.append(
        {
            "system": "Uniform Ensemble",
            "seed": 42,
            "parameters": 0,
            "epoch": 0,
            "device": "mps",
            "runtime_seconds": 0.0,
            "neural_f1": uniform_manifest["dev"]["neural"]["micro_f1"],
            "neural_correct": uniform_manifest[
                "dev"
            ]["neural"]["correct"],
            "final_f1": uniform_manifest["dev"]["v2"]["micro_f1"],
            "correct": uniform_manifest["dev"]["v2"]["correct"],
            "diagnostics": uniform_manifest["dev"],
            "checkpoint_sha256": [
                checkpoint["sha256"]
                for checkpoint in uniform_manifest["checkpoints"]
            ],
            "submission_sha256": uniform_manifest[
                "v2_artifacts"
            ]["submission_sha256"],
            "status": "candidate",
            "submission_path": uniform_manifest["v2_artifacts"]["submission_path"],
        }
    )
    _write_campaign_state(
        output_root, rows, {}, "uniform_ensemble_complete"
    )

    qualified: Dict[str, List[Path]] = {}
    experiment_specs = (
        ("hier_mixed", "HierMixed", "02_hier_mixed", 14611),
        ("direct16", "Direct16", "03_direct16", 14585),
        ("gl_curriculum", "GL Curriculum", "04_gl_curriculum", 14585),
    )
    for key, label, folder, control in experiment_specs:
        config_file = Path(campaign["experiments"][key])
        checkpoints = []
        seed42_dir = output_root / folder / "seed_42"
        checkpoint = _train_one(
            config_file, seed42_dir, 42, device_name
        )
        checkpoints.append(checkpoint)
        correct = _checkpoint_correct(checkpoint)
        prefix = f"DZIRIFORMER_{key.upper()}_SEED42"
        manifest = export_checkpoint_ensemble(
            [checkpoint], seed42_dir, prefix, f"DziriFormer-{label}", device
        )
        passed = correct > control
        rows.append(
            _record_row(
                label,
                42,
                seed42_dir,
                manifest,
                "qualified" if passed else "rejected",
            )
        )
        if passed:
            for seed in (43, 44):
                run_dir = output_root / folder / f"seed_{seed}"
                seed_checkpoint = _train_one(
                    config_file, run_dir, seed, device_name
                )
                checkpoints.append(seed_checkpoint)
                seed_manifest = export_checkpoint_ensemble(
                    [seed_checkpoint],
                    run_dir,
                    f"DZIRIFORMER_{key.upper()}_SEED{seed}",
                    f"DziriFormer-{label}",
                    device,
                )
                rows.append(
                    _record_row(
                        label, seed, run_dir, seed_manifest, "validation"
                    )
                )
            passed_count = sum(
                _checkpoint_correct(path) > control for path in checkpoints
            )
            mean_f1 = sum(
                float(
                    load_checkpoint(path, torch.device("cpu"))[
                        "dev_metrics"
                    ]["micro_f1"]
                )
                for path in checkpoints
            ) / len(checkpoints)
            control_f1 = control / 15897
            if passed_count >= 2 and mean_f1 > control_f1:
                qualified[key] = checkpoints
        _write_campaign_state(
            output_root,
            rows,
            qualified,
            f"{key}_complete",
        )

    architecture_groups: List[Sequence[Path]] = [uniform_paths]
    architecture_groups.extend(
        qualified[key] for key in sorted(qualified)
    )
    expanded_manifest = export_architecture_ensemble(
        architecture_groups,
        output_root / "05_expanded_ensemble",
        "DZIRI_ENSEMBLE_EXPANDED_V5",
        "DziriEnsemble-Expanded-v5",
        device,
    )
    expanded_row = {
        "system": "Expanded Ensemble",
        "seed": 42,
        "parameters": 0,
        "epoch": 0,
        "device": "mps",
        "runtime_seconds": 0.0,
        "neural_f1": expanded_manifest["dev"]["neural"]["micro_f1"],
        "neural_correct": expanded_manifest[
            "dev"
        ]["neural"]["correct"],
        "final_f1": expanded_manifest["dev"]["v2"]["micro_f1"],
        "correct": expanded_manifest["dev"]["v2"]["correct"],
        "diagnostics": expanded_manifest["dev"],
        "checkpoint_sha256": [
            [
                checkpoint["sha256"]
                for checkpoint in group
            ]
            for group in expanded_manifest["checkpoint_groups"]
        ],
        "submission_sha256": expanded_manifest[
            "v2_artifacts"
        ]["submission_sha256"],
        "status": "candidate",
        "submission_path": expanded_manifest[
            "v2_artifacts"
        ]["submission_path"],
    }
    rows.append(expanded_row)
    _write_campaign_state(
        output_root, rows, qualified, "expanded_ensemble_complete"
    )

    if campaign["execution"]["run_oof_gate"]:
        _, oof_row = _run_oof(
            output_root, uniform_paths, device, device_name
        )
    else:
        oof_row = _defer_oof(
            output_root,
            campaign["execution"]["oof_deferred_reason"],
        )
    rows.append(oof_row)
    _write_campaign_state(
        output_root,
        rows,
        qualified,
        (
            "oof_complete"
            if campaign["execution"]["run_oof_gate"]
            else "oof_deferred"
        ),
    )

    hgl_override: Dict[str, Any] = {}
    if "hier_mixed" in qualified:
        hgl_override["global_attention_every"] = 3
    if "direct16" in qualified:
        hgl_override.update(
            {"head_mode": "direct", "factorized_head": False}
        )
    hgl_config = Path(campaign["experiments"]["hgl"])
    hgl_dir = output_root / "07_hgl" / "seed_42"
    hgl_checkpoint = _train_one(
        hgl_config,
        hgl_dir,
        42,
        device_name,
        model_override=hgl_override,
    )
    hgl_manifest = export_checkpoint_ensemble(
        [hgl_checkpoint],
        hgl_dir,
        "DZIRIFORMER_HGL_V4_SEED42",
        "DziriFormer-HGL-v4",
        device,
    )
    accepted_labels = {
        "hier_mixed": "HierMixed",
        "direct16": "Direct16",
        "gl_curriculum": "GL Curriculum",
    }
    best_pre_hgl = 0.9191042335031767
    for key in qualified:
        label = accepted_labels[key]
        best_pre_hgl = max(
            best_pre_hgl,
            max(
                row["neural_f1"]
                for row in rows
                if row["system"] == label and row["seed"] == 42
            ),
        )
    hgl_passed = (
        hgl_manifest["dev"]["neural"]["micro_f1"] > best_pre_hgl
    )
    rows.append(
        _record_row(
            "HGL", 42, hgl_dir, hgl_manifest,
            "qualified" if hgl_passed else "rejected"
        )
    )
    hgl_checkpoints = [hgl_checkpoint]
    hgl_seed_manifests = [hgl_manifest]
    if hgl_passed:
        for seed in (43, 44):
            run_dir = output_root / "07_hgl" / f"seed_{seed}"
            checkpoint = _train_one(
                hgl_config,
                run_dir,
                seed,
                device_name,
                model_override=hgl_override,
            )
            manifest = export_checkpoint_ensemble(
                [checkpoint],
                run_dir,
                f"DZIRIFORMER_HGL_V4_SEED{seed}",
                "DziriFormer-HGL-v4",
                device,
            )
            hgl_checkpoints.append(checkpoint)
            hgl_seed_manifests.append(manifest)
            rows.append(
                _record_row("HGL", seed, run_dir, manifest, "validation")
            )
    hgl_accepted = False
    hgl_candidate_row = None
    if hgl_passed:
        control_paths = qualified.get("hier_mixed", [])
        if not control_paths:
            raise RuntimeError(
                "HGL passed seed 42 without an accepted hierarchical control"
            )
        hgl_scores = [
            float(
                load_checkpoint(path, torch.device("cpu"))[
                    "dev_metrics"
                ]["micro_f1"]
            )
            for path in hgl_checkpoints
        ]
        control_scores = [
            float(
                load_checkpoint(path, torch.device("cpu"))[
                    "dev_metrics"
                ]["micro_f1"]
            )
            for path in control_paths
        ]
        hgl_accepted = (
            sum(hgl_scores) / len(hgl_scores)
            > sum(control_scores) / len(control_scores)
            and sum(
                score > control
                for score, control in zip(hgl_scores, control_scores)
            )
            >= 2
        )
        hgl_ensemble_manifest = export_checkpoint_ensemble(
            hgl_checkpoints,
            output_root / "07_hgl" / "ensemble",
            "DZIRIFORMER_HGL_V4_ENSEMBLE",
            "DziriFormer-HGL-v4-Ensemble",
            device,
        )
        hgl_candidate_row = {
            "system": "HGL Ensemble",
            "seed": 42,
            "parameters": int(
                json.loads(
                    (
                        output_root
                        / "07_hgl"
                        / "seed_42"
                        / "summary.json"
                    ).read_text(encoding="utf-8")
                )["parameter_count"]
            ),
            "epoch": 0,
            "device": "mps",
            "runtime_seconds": sum(
                float(
                    json.loads(
                        (
                            output_root
                            / "07_hgl"
                            / f"seed_{seed}"
                            / "summary.json"
                        ).read_text(encoding="utf-8")
                    )["elapsed_seconds"]
                )
                for seed in (42, 43, 44)
            ),
            "neural_f1": hgl_ensemble_manifest[
                "dev"
            ]["neural"]["micro_f1"],
            "neural_correct": hgl_ensemble_manifest[
                "dev"
            ]["neural"]["correct"],
            "final_f1": hgl_ensemble_manifest[
                "dev"
            ]["v2"]["micro_f1"],
            "correct": hgl_ensemble_manifest[
                "dev"
            ]["v2"]["correct"],
            "diagnostics": hgl_ensemble_manifest["dev"],
            "checkpoint_sha256": [
                checkpoint["sha256"]
                for checkpoint in hgl_ensemble_manifest["checkpoints"]
            ],
            "submission_sha256": hgl_ensemble_manifest[
                "v2_artifacts"
            ]["submission_sha256"],
            "status": "candidate" if hgl_accepted else "rejected",
            "submission_path": hgl_ensemble_manifest[
                "v2_artifacts"
            ]["submission_path"],
        }
        rows.append(hgl_candidate_row)
    _write_campaign_state(
        output_root,
        rows,
        qualified,
        "hgl_complete",
        {"hgl_accepted": hgl_accepted},
    )

    candidates = [
        row
        for row in rows
        if (
            row["system"] in (
                "Uniform Ensemble",
                "Expanded Ensemble",
            )
            or (
                row["system"] == "OOF Gate"
                and row["status"] == "candidate"
            )
            or (
                row["system"] == "HGL Ensemble"
                and row["status"] == "candidate"
            )
        )
    ]
    final = max(
        candidates,
        key=lambda row: (
            row["correct"],
            -row["parameters"],
        ),
    )
    final_dir = output_root / "08_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    source = Path(final["submission_path"])
    source_official = source.with_name(
        source.name.replace("_SUBMISSION.csv", "_OFFICIAL_CHECK.csv")
    )
    if (
        not source_official.is_file()
        or source.read_bytes() != source_official.read_bytes()
    ):
        raise RuntimeError(
            "selected submission is not equal to its official check"
        )
    final_submission = final_dir / "DZIRI_FINAL_CAMPAIGN_V5_SUBMISSION.csv"
    final_official = (
        final_dir / "DZIRI_FINAL_CAMPAIGN_V5_OFFICIAL_CHECK.csv"
    )
    final_submission.write_bytes(source.read_bytes())
    final_official.write_bytes(source_official.read_bytes())
    if final_submission.read_bytes() != final_official.read_bytes():
        raise RuntimeError("final official submission verification failed")
    final = {
        **final,
        "source_submission_path": str(source),
        "submission_path": str(final_submission),
        "official_check_path": str(final_official),
        "submission_sha256": sha256_file(final_submission),
    }
    write_json(
        output_root / "campaign_manifest.json",
        {
            "schema_version": 1,
            "stage": "complete",
            "rows": rows,
            "final": final,
            "qualified": sorted(qualified),
            "hgl_accepted": hgl_accepted,
            "campaign_config_path": str(config_path),
            "campaign_config_sha256": sha256_file(config_path),
            "final_submission_sha256": sha256_file(final_submission),
            "oof_status": (
                "complete"
                if campaign["execution"]["run_oof_gate"]
                else "deferred_incomplete"
            ),
        },
    )
    write_reports(
        Path("experiments/pre_hgl_v5"), rows, final
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    run_campaign(args.config, args.device)


if __name__ == "__main__":
    main()
