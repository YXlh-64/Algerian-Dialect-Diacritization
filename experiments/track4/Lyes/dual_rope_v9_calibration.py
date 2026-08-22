"""Cross-fitted calibration/stacking for the production v7 ensemble."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from utils.track4.Lyes.gated_fusion.config import load_gates
from experiments.track4.Lyes.calibrated_stacking import (
    CalibratedStacker,
    fit_calibrated_stacker,
    stack_record_probabilities,
)
from experiments.track4.Lyes.campaign.common import (
    write_prediction_artifacts,
    write_step_manifest,
)
from experiments.track4.Lyes.campaign.ensemble import (
    apply_lexical_gate,
    average_probability_groups,
    predict_probability_groups,
)
from experiments.track4.Lyes.campaign.folds import make_balanced_folds
from utils.track4.Lyes.data import SentenceRecord, load_jsonl, load_raw_sentences
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from evaluation.track4.Lyes.paper_metrics import compute_paper_metrics
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


ARTIFACT_PREFIX = "DZIRI_FINAL_CALIBRATED_STACK_V9"
SYSTEM_NAME = "DziriFinal-CalibratedStack-v9"
DESCRIPTION = (
    "Five-fold sentence-disjoint temperature calibration and simplex "
    "architecture stacking over the frozen production v7 groups. Four "
    "positive temperatures and four nonnegative weights summing to one are "
    "learned by multiclass NLL on four folds and evaluated on the held fold. "
    "The unchanged V2 lexical fallback is applied after calibrated stacking."
)
EXPECTED_ROOT_KEYS = {
    "schema_version",
    "output_root",
    "production_groups",
    "calibration",
}
EXPECTED_CALIBRATION_KEYS = {
    "fold_count",
    "fold_seed",
    "expected_baseline_correct",
    "minimum_correct_gain",
    "minimum_improved_folds",
    "maximum_regressed_folds",
    "lexical_smoothing",
}


def load_calibration_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or set(config) != EXPECTED_ROOT_KEYS:
        raise ValueError("invalid v9 calibration configuration keys")
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported v9 calibration schema")
    groups = config["production_groups"]
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError("production_groups must contain at least two groups")
    names = []
    for group in groups:
        if not isinstance(group, dict) or set(group) != {
            "name",
            "checkpoints",
        }:
            raise ValueError("invalid production group")
        if not isinstance(group["name"], str) or not group["name"]:
            raise ValueError("production group name must be nonempty")
        checkpoints = group["checkpoints"]
        if (
            not isinstance(checkpoints, list)
            or not checkpoints
            or not all(isinstance(value, str) for value in checkpoints)
        ):
            raise ValueError("production group checkpoints are invalid")
        names.append(group["name"])
    if len(names) != len(set(names)):
        raise ValueError("production group names must be unique")
    calibration = config["calibration"]
    if (
        not isinstance(calibration, dict)
        or set(calibration) != EXPECTED_CALIBRATION_KEYS
    ):
        raise ValueError("invalid v9 calibration gate")
    if int(calibration["fold_count"]) != 5:
        raise ValueError("v9 calibration requires five folds")
    if int(calibration["expected_baseline_correct"]) <= 0:
        raise ValueError("expected baseline correct must be positive")
    if int(calibration["minimum_correct_gain"]) <= 0:
        raise ValueError("minimum gain must be positive")
    if not 1 <= int(calibration["minimum_improved_folds"]) <= 5:
        raise ValueError("minimum improved fold count is invalid")
    if not 0 <= int(calibration["maximum_regressed_folds"]) <= 5:
        raise ValueError("maximum regressed fold count is invalid")
    if float(calibration["lexical_smoothing"]) <= 0.0:
        raise ValueError("lexical smoothing must be positive")
    return config


def checkpoint_groups(config: Mapping[str, Any]) -> List[List[Path]]:
    groups = [
        [Path(value) for value in group["checkpoints"]]
        for group in config["production_groups"]
    ]
    missing = [
        str(path)
        for group in groups
        for path in group
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "v9 calibration checkpoints are missing: {}".format(missing)
        )
    return groups


def flatten_scored_probabilities(
    records: Sequence[SentenceRecord],
    group_probabilities: Sequence[Sequence[torch.Tensor]],
    indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not group_probabilities:
        raise ValueError("at least one probability group is required")
    if any(len(group) != len(records) for group in group_probabilities):
        raise ValueError("group probabilities must align with records")
    probability_rows: List[torch.Tensor] = []
    targets: List[int] = []
    for index in indices:
        record = records[index]
        if record.labels is None:
            raise ValueError("calibration requires gold labels")
        scored_positions = [
            position
            for position, char in enumerate(record.chars)
            if char != " "
        ]
        probability_rows.append(
            torch.stack(
                [
                    group[index].index_select(
                        0, torch.tensor(scored_positions, dtype=torch.long)
                    )
                    for group in group_probabilities
                ],
                dim=0,
            )
        )
        targets.extend(record.labels[position] for position in scored_positions)
    if not probability_rows:
        raise ValueError("calibration selection contains no scored letters")
    return (
        torch.cat(probability_rows, dim=1).to(dtype=torch.float64),
        torch.tensor(targets, dtype=torch.long),
    )


def _subset(values: Sequence[Any], indices: Sequence[int]) -> List[Any]:
    return [values[index] for index in indices]


def robust_gate(
    baseline_correct: int,
    candidate_correct: int,
    fold_deltas: Sequence[int],
    minimum_gain: int,
    minimum_improved_folds: int,
    maximum_regressed_folds: int,
) -> Dict[str, Any]:
    gain = candidate_correct - baseline_correct
    improved = sum(delta > 0 for delta in fold_deltas)
    regressed = sum(delta < 0 for delta in fold_deltas)
    accepted = (
        gain >= minimum_gain
        and improved >= minimum_improved_folds
        and regressed <= maximum_regressed_folds
    )
    return {
        "accepted": accepted,
        "correct_gain": gain,
        "minimum_correct_gain": minimum_gain,
        "improved_folds": improved,
        "minimum_improved_folds": minimum_improved_folds,
        "regressed_folds": regressed,
        "maximum_regressed_folds": maximum_regressed_folds,
        "fold_deltas": list(fold_deltas),
    }


def run(
    config_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    config = load_calibration_config(config_path)
    calibration = config["calibration"]
    groups = checkpoint_groups(config)
    device = select_device(device_name)
    train_records = load_jsonl(
        Path("Data/train_data/train_Algerian-DIAC.jsonl")
    )
    dev_records = load_jsonl(
        Path("Data/dev_data/dev_Algerian-DIAC.jsonl")
    )
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(Path("configs/track4/Lyes/gates.json"))
    group_probabilities, _ = predict_probability_groups(
        groups, dev_records, device, batch_size, num_workers
    )
    baseline_probabilities = average_probability_groups(group_probabilities)
    baseline_predictions = apply_lexical_gate(
        dev_records, baseline_probabilities, prior, gates
    )
    baseline_metrics = compute_paper_metrics(
        dev_records, baseline_predictions
    )
    expected_baseline = int(calibration["expected_baseline_correct"])
    if int(baseline_metrics["correct_letters"]) != expected_baseline:
        raise RuntimeError(
            "v9 baseline drift: {} != {}".format(
                baseline_metrics["correct_letters"], expected_baseline
            )
        )

    folds = make_balanced_folds(
        dev_records,
        int(calibration["fold_count"]),
        int(calibration["fold_seed"]),
    )
    all_indices = set(range(len(dev_records)))
    oof_predictions: List[Any] = [None] * len(dev_records)
    fold_summaries = []
    fold_deltas = []
    for fold_index, held_values in enumerate(folds):
        held_indices = list(held_values)
        train_indices = sorted(all_indices - set(held_indices))
        train_probabilities, train_targets = flatten_scored_probabilities(
            dev_records, group_probabilities, train_indices
        )
        stacker = fit_calibrated_stacker(
            train_probabilities, train_targets
        )
        held_group_probabilities = [
            _subset(group, held_indices) for group in group_probabilities
        ]
        held_probabilities = stack_record_probabilities(
            held_group_probabilities, stacker
        )
        held_records = _subset(dev_records, held_indices)
        held_predictions = apply_lexical_gate(
            held_records, held_probabilities, prior, gates
        )
        held_baseline_predictions = _subset(
            baseline_predictions, held_indices
        )
        held_metrics = compute_paper_metrics(
            held_records, held_predictions
        )
        held_baseline_metrics = compute_paper_metrics(
            held_records, held_baseline_predictions
        )
        delta = int(held_metrics["correct_letters"]) - int(
            held_baseline_metrics["correct_letters"]
        )
        fold_deltas.append(delta)
        for index, prediction in zip(held_indices, held_predictions):
            if oof_predictions[index] is not None:
                raise RuntimeError("duplicate v9 OOF prediction")
            oof_predictions[index] = prediction
        fold_summaries.append(
            {
                "fold": fold_index,
                "train_sentences": len(train_indices),
                "held_sentences": len(held_indices),
                "held_baseline_correct": held_baseline_metrics[
                    "correct_letters"
                ],
                "held_candidate_correct": held_metrics["correct_letters"],
                "correct_delta": delta,
                "stacker": stacker.to_dict(),
            }
        )
    if any(value is None for value in oof_predictions):
        raise RuntimeError("incomplete v9 OOF prediction coverage")
    calibrated_predictions = [
        value for value in oof_predictions if value is not None
    ]
    calibrated_metrics = compute_paper_metrics(
        dev_records, calibrated_predictions
    )
    gate_result = robust_gate(
        int(baseline_metrics["correct_letters"]),
        int(calibrated_metrics["correct_letters"]),
        fold_deltas,
        int(calibration["minimum_correct_gain"]),
        int(calibration["minimum_improved_folds"]),
        int(calibration["maximum_regressed_folds"]),
    )

    full_probabilities, full_targets = flatten_scored_probabilities(
        dev_records, group_probabilities, list(range(len(dev_records)))
    )
    deployment_stacker = fit_calibrated_stacker(
        full_probabilities, full_targets
    )
    output_dir = Path(str(config["output_root"])) / "01_calibrated_stacking"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "baseline_paper_metrics.json", baseline_metrics)
    write_json(
        output_dir / "crossfit_paper_metrics.json", calibrated_metrics
    )
    write_json(
        output_dir / "deployment_stacker.json", deployment_stacker.to_dict()
    )

    artifacts: Mapping[str, Any] = {}
    canonical_path = Path(str(config["output_root"])) / (
        "SUBMIT_THIS_DZIRI_CALIBRATED_STACK_V9.csv"
    )
    if gate_result["accepted"]:
        test_records = load_raw_sentences(
            Path("Data/test_data/raw_sentences_test.txt"),
            Path("Data/test_data/raw_sentences_test_ids.txt"),
        )
        test_group_probabilities, _ = predict_probability_groups(
            groups, test_records, device, batch_size, num_workers
        )
        test_probabilities = stack_record_probabilities(
            test_group_probabilities, deployment_stacker
        )
        test_predictions = apply_lexical_gate(
            test_records, test_probabilities, prior, gates
        )
        artifacts = write_prediction_artifacts(
            output_dir / "artifacts",
            ARTIFACT_PREFIX,
            test_records,
            test_predictions,
            Path("Data/test_data/sample_submission.csv"),
            Path("Data/test_data/raw_sentences_test_ids.txt"),
            Path("Data/test_data/raw_sentences_test.txt"),
        )
        shutil.copyfile(str(artifacts["submission_path"]), canonical_path)
        if sha256_file(canonical_path) != artifacts["submission_sha256"]:
            raise RuntimeError("v9 canonical submission hash mismatch")
    elif canonical_path.exists():
        raise RuntimeError("stale v9 calibrated submission exists")

    selection = {
        "schema_version": 1,
        "system_name": SYSTEM_NAME,
        "description": DESCRIPTION,
        "baseline": {
            "correct": baseline_metrics["correct_letters"],
            "accuracy": baseline_metrics["accuracy"],
        },
        "crossfit_candidate": {
            "correct": calibrated_metrics["correct_letters"],
            "accuracy": calibrated_metrics["accuracy"],
        },
        "robust_gate": gate_result,
        "folds": fold_summaries,
        "deployment_stacker": deployment_stacker.to_dict(),
        "recommended_submission_path": (
            str(canonical_path) if gate_result["accepted"] else None
        ),
        "artifacts": dict(artifacts),
    }
    write_json(output_dir / "SELECTION.json", selection)
    write_step_manifest(
        output_dir / "MANIFEST.json",
        {
            "system_name": SYSTEM_NAME,
            "description": DESCRIPTION,
            "group_names": [
                group["name"] for group in config["production_groups"]
            ],
            "selection": selection,
        },
    )
    print(json.dumps(selection, indent=2, sort_keys=True))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/track4/Lyes/dual_rope_v9/campaign.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    run(
        config_path=args.config,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
