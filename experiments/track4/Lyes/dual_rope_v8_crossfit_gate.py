"""Sentence-disjoint cross-fitted lexical gate for the v8 final ensemble."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from experiments.track4.Lyes.campaign.common import (
    write_prediction_artifacts,
    write_step_manifest,
)
from experiments.track4.Lyes.campaign.diagnostics import (
    prediction_diagnostics,
    training_word_types,
)
from experiments.track4.Lyes.campaign.ensemble import (
    predict_probability_group_members,
)
from experiments.track4.Lyes.campaign.folds import make_balanced_folds
from experiments.track4.Lyes.campaign.oof_gate import (
    LogisticGate,
    apply_logistic_gate,
    collect_training_examples,
    fit_logistic_gate,
)
from utils.track4.Lyes.data import SentenceRecord, load_jsonl, load_raw_sentences
from experiments.track4.Lyes.dual_rope_boundary_crf_v8 import load_v8_config
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


ARTIFACT_PREFIX = "DZIRI_FINAL_BOUNDARY_CRF_CROSSFIT_GATE_V8"
SYSTEM_NAME = "DziriFinal-BoundaryCRF-CrossFitGate-v8"
DESCRIPTION = (
    "The accepted four-group BoundaryCRF-v8 ensemble remains the neural "
    "expert. A standardized eight-feature logistic switch chooses between "
    "its label and the training-only word prior only when they disagree. "
    "Five sentence-disjoint dev folds produce an honest cross-fitted gate "
    "score; each held-out fold is predicted by a gate fitted on the other "
    "four folds. The deployable gate is then refit on all released-dev "
    "disagreements. It uses deterministic full-batch LBFGS and a fixed 0.5 "
    "decision threshold, with no interpolation weight or tuned threshold."
)


def checkpoint_groups(config: Mapping[str, Any]) -> List[List[Path]]:
    raw_groups = [
        [str(config["boundary_crf_checkpoint"])],
        [str(value) for value in config["dual_seed_checkpoints"]],
        [str(value) for value in config["hgl_checkpoints"]],
        [str(value) for value in config["legacy_ensemble_checkpoints"]],
    ]
    groups = [[Path(value) for value in group] for group in raw_groups]
    missing = [
        str(path)
        for group in groups
        for path in group
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "crossfit checkpoint files are missing: {}".format(missing)
        )
    return groups


def _subset(
    values: Sequence[Any], indices: Sequence[int]
) -> List[Any]:
    return [values[index] for index in indices]


def cross_fitted_gate_predictions(
    records: Sequence[SentenceRecord],
    probabilities: Sequence[torch.Tensor],
    group_votes: Sequence[torch.Tensor],
    prior: WordLabelPrior,
    smoothing: float,
    folds: Sequence[Sequence[int]],
) -> Tuple[List[List[int]], List[Mapping[str, Any]]]:
    if len(records) != len(probabilities) or len(records) != len(group_votes):
        raise ValueError("crossfit inputs must have identical record counts")
    flattened = sorted(index for fold in folds for index in fold)
    if flattened != list(range(len(records))):
        raise ValueError("crossfit folds must partition every record exactly")

    oof_predictions: List[List[int] | None] = [None] * len(records)
    fold_summaries: List[Mapping[str, Any]] = []
    all_indices = set(range(len(records)))
    for fold_index, held_indices_value in enumerate(folds):
        held_indices = list(held_indices_value)
        train_indices = sorted(all_indices - set(held_indices))
        train_features, train_targets = collect_training_examples(
            _subset(records, train_indices),
            _subset(probabilities, train_indices),
            _subset(group_votes, train_indices),
            prior,
            smoothing,
        )
        gate = fit_logistic_gate(train_features, train_targets)
        held_predictions = apply_logistic_gate(
            _subset(records, held_indices),
            _subset(probabilities, held_indices),
            _subset(group_votes, held_indices),
            prior,
            smoothing,
            gate,
        )
        for index, prediction in zip(held_indices, held_predictions):
            if oof_predictions[index] is not None:
                raise RuntimeError("crossfit record predicted more than once")
            oof_predictions[index] = prediction
        fold_summaries.append(
            {
                "fold": fold_index,
                "train_sentence_count": len(train_indices),
                "held_sentence_count": len(held_indices),
                "training_disagreement_examples": int(
                    train_features.size(0)
                ),
                "lexical_target_rate": float(train_targets.mean().item()),
                "gate": gate.to_dict(),
            }
        )
    if any(prediction is None for prediction in oof_predictions):
        raise RuntimeError("crossfit prediction coverage is incomplete")
    return [
        prediction
        for prediction in oof_predictions
        if prediction is not None
    ], fold_summaries


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
    config = load_v8_config(config_path)
    gate_config = config["crossfit_gate"]
    fold_count = int(gate_config["fold_count"])
    fold_seed = int(gate_config["fold_seed"])
    smoothing = float(gate_config["lexical_smoothing"])
    groups = checkpoint_groups(config)
    device = select_device(device_name)

    train_records = load_jsonl(
        Path("Data/train_data/train_Algerian-DIAC.jsonl")
    )
    dev_records = load_jsonl(
        Path("Data/dev_data/dev_Algerian-DIAC.jsonl")
    )
    test_records = load_raw_sentences(
        Path("Data/test_data/raw_sentences_test.txt"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
    )
    prior = WordLabelPrior().fit(train_records)
    dev_probabilities, dev_votes, _ = (
        predict_probability_group_members(
            groups, dev_records, device, batch_size, num_workers
        )
    )
    folds = make_balanced_folds(dev_records, fold_count, fold_seed)
    oof_predictions, fold_summaries = cross_fitted_gate_predictions(
        dev_records,
        dev_probabilities,
        dev_votes,
        prior,
        smoothing,
        folds,
    )
    diagnostics = prediction_diagnostics(
        dev_records, oof_predictions, training_word_types(train_records)
    )

    full_features, full_targets = collect_training_examples(
        dev_records,
        dev_probabilities,
        dev_votes,
        prior,
        smoothing,
    )
    deployment_gate = fit_logistic_gate(full_features, full_targets)
    test_probabilities, test_votes, _ = (
        predict_probability_group_members(
            groups, test_records, device, batch_size, num_workers
        )
    )
    test_predictions = apply_logistic_gate(
        test_records,
        test_probabilities,
        test_votes,
        prior,
        smoothing,
        deployment_gate,
    )
    output_dir = Path(str(config["output_root"])) / "03_crossfit_gate"
    artifacts = write_prediction_artifacts(
        output_dir / "artifacts",
        ARTIFACT_PREFIX,
        test_records,
        test_predictions,
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    manifest = {
        "schema_version": 1,
        "system_name": SYSTEM_NAME,
        "artifact_prefix": ARTIFACT_PREFIX,
        "description": DESCRIPTION,
        "crossfit": {
            "fold_count": fold_count,
            "fold_seed": fold_seed,
            "lexical_smoothing": smoothing,
            "decision_threshold": 0.5,
            "folds": [list(fold) for fold in folds],
            "fold_summaries": fold_summaries,
        },
        "deployment_training": {
            "disagreement_examples": int(full_features.size(0)),
            "lexical_target_rate": float(full_targets.mean().item()),
            "gate": deployment_gate.to_dict(),
        },
        "dev": diagnostics,
        "artifacts": dict(artifacts),
    }
    write_step_manifest(
        output_dir / "artifacts" / f"{ARTIFACT_PREFIX}_MANIFEST.json",
        manifest,
    )
    write_json(output_dir / "deployment_gate.json", deployment_gate.to_dict())

    correct = int(diagnostics["correct"])
    threshold = int(
        config["acceptance"]["crossfit_gate_must_exceed_correct"]
    )
    accepted = correct > threshold
    candidate_path = Path(str(artifacts["submission_path"]))
    canonical_path = Path(str(config["output_root"])) / (
        "SUBMIT_THIS_DZIRI_FINAL_CROSSFIT_GATE_V8.csv"
    )
    if accepted:
        shutil.copyfile(candidate_path, canonical_path)
        if sha256_file(candidate_path) != sha256_file(canonical_path):
            raise RuntimeError("crossfit canonical copy hash mismatch")
    elif canonical_path.exists():
        raise RuntimeError(
            "stale crossfit canonical submission exists for a rejected result"
        )
    selection = {
        "schema_version": 1,
        "system_name": SYSTEM_NAME,
        "description": DESCRIPTION,
        "crossfit_dev_micro_f1": diagnostics["micro_f1"],
        "crossfit_dev_correct": correct,
        "acceptance_operator": ">",
        "acceptance_correct": threshold,
        "accepted": accepted,
        "recommended_submission_path": (
            str(canonical_path) if accepted else None
        ),
        "candidate_submission_path": str(candidate_path),
        "submission_sha256": (
            sha256_file(candidate_path) if accepted else None
        ),
    }
    write_json(output_dir / "SELECTION.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/track4/Lyes/dual_rope_v8/campaign.json"),
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
