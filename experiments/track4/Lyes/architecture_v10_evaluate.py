"""Evaluate architecture-v10 ablations under one immutable gate contract."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

from utils.track4.Lyes.gated_fusion.config import load_gates
from utils.track4.Lyes.gated_fusion.fusion import predict_with_gated_fallback
from experiments.track4.Lyes.campaign.common import (
    write_prediction_artifacts,
    write_step_manifest,
)
from experiments.track4.Lyes.campaign.diagnostics import (
    prediction_diagnostics,
    training_word_types,
)
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import load_jsonl, load_raw_sentences
from evaluation.track4.Lyes.infer import predict_records
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from evaluation.track4.Lyes.paper_metrics import compute_paper_metrics
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


ROOT_KEYS = {"schema_version", "output_root", "controls", "experiments"}
CONTROL_KEYS = {
    "neural_correct",
    "minimum_neural_gain",
    "neural_oov_correct",
    "neural_word_correct",
    "neural_shadda_accuracy",
    "maximum_shadda_regression",
    "neural_tanween_accuracy",
    "maximum_tanween_regression",
    "production_v2_correct",
}
EXPERIMENT_KEYS = {
    "directory",
    "checkpoint",
    "system_name",
    "artifact_prefix",
    "expected_head_mode",
    "word_position_features",
    "description",
}


def load_evaluation_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or set(config) != ROOT_KEYS:
        raise ValueError("invalid architecture-v10 evaluation root keys")
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported architecture-v10 evaluation schema")
    controls = config["controls"]
    if not isinstance(controls, dict) or set(controls) != CONTROL_KEYS:
        raise ValueError("invalid architecture-v10 control keys")
    experiments = config["experiments"]
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError("architecture-v10 experiments cannot be empty")
    for slug, experiment in experiments.items():
        if re.fullmatch(r"[a-z0-9_]+", slug) is None:
            raise ValueError("invalid architecture-v10 experiment slug")
        if not isinstance(experiment, dict) or set(experiment) != EXPERIMENT_KEYS:
            raise ValueError("invalid architecture-v10 experiment keys")
        if not isinstance(experiment["word_position_features"], bool):
            raise ValueError("word_position_features must be boolean")
        prefix = experiment["artifact_prefix"]
        if re.fullmatch(r"[A-Z0-9_]+", prefix) is None:
            raise ValueError("invalid artifact prefix")
    return config


def _gate_results(
    controls: Mapping[str, Any],
    neural_paper: Mapping[str, Any],
    neural_diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    minimum_correct = (
        int(controls["neural_correct"])
        + int(controls["minimum_neural_gain"])
    )
    results = {
        "overall_correct": {
            "value": int(neural_paper["correct_letters"]),
            "operator": ">=",
            "threshold": minimum_correct,
            "passed": int(neural_paper["correct_letters"]) >= minimum_correct,
        },
        "oov_correct": {
            "value": int(neural_diagnostics["oov_correct"]),
            "operator": ">",
            "threshold": int(controls["neural_oov_correct"]),
            "passed": int(neural_diagnostics["oov_correct"])
            > int(controls["neural_oov_correct"]),
        },
        "word_correct": {
            "value": int(neural_paper["word_correct"]),
            "operator": ">",
            "threshold": int(controls["neural_word_correct"]),
            "passed": int(neural_paper["word_correct"])
            > int(controls["neural_word_correct"]),
        },
        "shadda_accuracy": {
            "value": float(neural_paper["shadda"]["accuracy"]),
            "operator": ">=",
            "threshold": float(controls["neural_shadda_accuracy"])
            - float(controls["maximum_shadda_regression"]),
            "passed": float(neural_paper["shadda"]["accuracy"])
            >= float(controls["neural_shadda_accuracy"])
            - float(controls["maximum_shadda_regression"]),
        },
        "tanween_accuracy": {
            "value": float(neural_paper["tanween"]["accuracy"]),
            "operator": ">=",
            "threshold": float(controls["neural_tanween_accuracy"])
            - float(controls["maximum_tanween_regression"]),
            "passed": float(neural_paper["tanween"]["accuracy"])
            >= float(controls["neural_tanween_accuracy"])
            - float(controls["maximum_tanween_regression"]),
        },
    }
    results["all_passed"] = all(
        bool(value["passed"])
        for key, value in results.items()
        if key != "all_passed"
    )
    return results


def run(
    config_path: Path,
    experiment_slug: str,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    config = load_evaluation_config(config_path)
    if experiment_slug not in config["experiments"]:
        raise ValueError("unknown architecture-v10 experiment")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("invalid evaluation loader settings")
    experiment = config["experiments"][experiment_slug]
    checkpoint_path = Path(str(experiment["checkpoint"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "experiment checkpoint is missing: {}".format(checkpoint_path)
        )
    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    if model.config.resolved_head_mode != experiment["expected_head_mode"]:
        raise ValueError("checkpoint head mode does not match experiment")
    if (
        model.config.word_position_features
        != experiment["word_position_features"]
    ):
        raise ValueError("checkpoint word-position mode does not match experiment")

    train_records = load_jsonl(
        Path("Data/train_data/train_Algerian-DIAC.jsonl")
    )
    dev_records = load_jsonl(
        Path("Data/dev_data/dev_Algerian-DIAC.jsonl")
    )
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(Path("configs/track4/Lyes/gates.json"))
    seen_words = training_word_types(train_records)
    neural_predictions = predict_records(
        model, dev_records, vocab, device, batch_size, num_workers
    )
    v2_predictions, dev_gate_statistics = predict_with_gated_fallback(
        model,
        dev_records,
        vocab,
        prior,
        gates,
        device,
        batch_size,
        num_workers,
    )
    neural_paper = compute_paper_metrics(dev_records, neural_predictions)
    v2_paper = compute_paper_metrics(dev_records, v2_predictions)
    neural_diagnostics = prediction_diagnostics(
        dev_records, neural_predictions, seen_words
    )
    v2_diagnostics = prediction_diagnostics(
        dev_records, v2_predictions, seen_words
    )
    gate_results = _gate_results(
        config["controls"], neural_paper, neural_diagnostics
    )
    architecture_accepted = bool(gate_results["all_passed"])
    competitive = (
        architecture_accepted
        and int(v2_paper["correct_letters"])
        > int(config["controls"]["production_v2_correct"])
    )

    output_dir = Path(str(config["output_root"])) / str(
        experiment["directory"]
    )
    write_json(output_dir / "neural_paper_metrics.json", neural_paper)
    write_json(output_dir / "v2_paper_metrics.json", v2_paper)
    write_json(
        output_dir / "diagnostics.json",
        {
            "neural": neural_diagnostics,
            "v2": v2_diagnostics,
            "v2_gate_statistics": dev_gate_statistics.to_dict(),
        },
    )
    neural_artifacts: Mapping[str, Any] = {}
    v2_artifacts: Mapping[str, Any] = {}
    if competitive:
        test_records = load_raw_sentences(
            Path("Data/test_data/raw_sentences_test.txt"),
            Path("Data/test_data/raw_sentences_test_ids.txt"),
        )
        test_neural = predict_records(
            model, test_records, vocab, device, batch_size, num_workers
        )
        test_v2, _ = predict_with_gated_fallback(
            model,
            test_records,
            vocab,
            prior,
            gates,
            device,
            batch_size,
            num_workers,
        )
        artifact_dir = output_dir / "artifacts"
        prefix = str(experiment["artifact_prefix"])
        neural_artifacts = write_prediction_artifacts(
            artifact_dir,
            prefix + "_NEURAL",
            test_records,
            test_neural,
            Path("Data/test_data/sample_submission.csv"),
            Path("Data/test_data/raw_sentences_test_ids.txt"),
            Path("Data/test_data/raw_sentences_test.txt"),
        )
        v2_artifacts = write_prediction_artifacts(
            artifact_dir,
            prefix + "_V2",
            test_records,
            test_v2,
            Path("Data/test_data/sample_submission.csv"),
            Path("Data/test_data/raw_sentences_test_ids.txt"),
            Path("Data/test_data/raw_sentences_test.txt"),
        )
    selection = {
        "schema_version": 1,
        "experiment": experiment_slug,
        "system_name": experiment["system_name"],
        "description": experiment["description"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
        },
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "device": str(device),
        "neural_correct": neural_paper["correct_letters"],
        "neural_accuracy": neural_paper["accuracy"],
        "v2_correct": v2_paper["correct_letters"],
        "v2_accuracy": v2_paper["accuracy"],
        "gates": gate_results,
        "architecture_accepted": architecture_accepted,
        "production_v2_must_exceed": config["controls"][
            "production_v2_correct"
        ],
        "competitive_submission": competitive,
        "recommended_submission_path": (
            v2_artifacts.get("submission_path") if competitive else None
        ),
        "neural_artifacts": dict(neural_artifacts),
        "v2_artifacts": dict(v2_artifacts),
    }
    write_json(output_dir / "SELECTION.json", selection)
    write_step_manifest(output_dir / "MANIFEST.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/track4/Lyes/architecture_v10/evaluation.json"),
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    run(
        config_path=args.config,
        experiment_slug=args.experiment,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
