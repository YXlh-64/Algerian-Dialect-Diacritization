"""Evaluate and export the controlled BoundaryCRF-v8 seed-42 ablation."""

import argparse
import json
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
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


ARTIFACT_PREFIX = "DZIRIFORMER_DUALROPE_BOUNDARY_CRF_V8_SEED42"
SYSTEM_NAME = "DziriFormer-DualRoPE-BoundaryCRF-v8-seed42"
DESCRIPTION = (
    "Controlled Track 4 ablation retaining the complete DualRoPE-v6 encoder "
    "and direct 16-class emissions. Its linear-chain CRF learns separate "
    "16x16 label-transition matrices for within-word transitions and "
    "transitions into the first scored letter after a space. Sentence starts "
    "retain the independent CRF start vector. Spaces remain encoder inputs "
    "but are excluded from loss and decoding. The V2 artifact applies the "
    "unchanged confidence-gated lexical fallback to CRF marginals and "
    "Viterbi labels."
)
EXPECTED_CONFIG_KEYS = {
    "schema_version",
    "output_root",
    "boundary_crf_checkpoint",
    "dual_seed_checkpoints",
    "hgl_checkpoints",
    "legacy_ensemble_checkpoints",
    "acceptance",
    "crossfit_gate",
}
EXPECTED_ACCEPTANCE_KEYS = {
    "boundary_neural_must_exceed_correct",
    "boundary_v2_must_exceed_correct",
    "boundary_ensemble_v2_must_exceed_correct",
    "crossfit_gate_must_exceed_correct",
}
EXPECTED_CROSSFIT_KEYS = {
    "fold_count",
    "fold_seed",
    "lexical_smoothing",
    "decision_threshold",
}


def load_v8_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or set(config) != EXPECTED_CONFIG_KEYS:
        raise ValueError("invalid BoundaryCRF v8 campaign configuration keys")
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported BoundaryCRF v8 campaign schema")
    if not isinstance(config["boundary_crf_checkpoint"], str):
        raise ValueError("boundary_crf_checkpoint must be a path string")
    for key, expected_length in (
        ("dual_seed_checkpoints", 3),
        ("hgl_checkpoints", 3),
        ("legacy_ensemble_checkpoints", 5),
    ):
        values = config[key]
        if (
            not isinstance(values, list)
            or len(values) != expected_length
            or not all(isinstance(value, str) for value in values)
        ):
            raise ValueError(
                "{} must contain exactly {} path strings".format(
                    key, expected_length
                )
            )
    acceptance = config["acceptance"]
    if (
        not isinstance(acceptance, dict)
        or set(acceptance) != EXPECTED_ACCEPTANCE_KEYS
    ):
        raise ValueError("invalid BoundaryCRF v8 acceptance gates")
    if any(int(value) <= 0 for value in acceptance.values()):
        raise ValueError("all BoundaryCRF v8 acceptance gates must be positive")
    crossfit = config["crossfit_gate"]
    if (
        not isinstance(crossfit, dict)
        or set(crossfit) != EXPECTED_CROSSFIT_KEYS
    ):
        raise ValueError("invalid BoundaryCRF v8 crossfit gate config")
    if int(crossfit["fold_count"]) != 5:
        raise ValueError("BoundaryCRF v8 requires five crossfit folds")
    if int(crossfit["fold_seed"]) < 0:
        raise ValueError("crossfit fold seed cannot be negative")
    if float(crossfit["lexical_smoothing"]) <= 0.0:
        raise ValueError("crossfit lexical smoothing must be positive")
    if float(crossfit["decision_threshold"]) != 0.5:
        raise ValueError("crossfit decision threshold must be exactly 0.5")
    return config


def run(
    checkpoint_path: Path,
    campaign_config_path: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "BoundaryCRF checkpoint is missing: {}".format(checkpoint_path)
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    campaign_config = load_v8_config(campaign_config_path)
    configured_checkpoint = Path(
        str(campaign_config["boundary_crf_checkpoint"])
    )
    if checkpoint_path.resolve() != configured_checkpoint.resolve():
        raise ValueError(
            "checkpoint does not match the controlled v8 campaign config"
        )

    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    if model.config.resolved_head_mode != "boundary_crf":
        raise ValueError("checkpoint is not a boundary-conditioned CRF model")

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
    seen_words = training_word_types(train_records)
    lexical_prior = WordLabelPrior().fit(train_records)
    gates = load_gates(Path("configs/track4/Lyes/gates.json"))

    dev_neural = predict_records(
        model, dev_records, vocab, device, batch_size, num_workers
    )
    dev_v2, dev_gate_statistics = predict_with_gated_fallback(
        model,
        dev_records,
        vocab,
        lexical_prior,
        gates,
        device,
        batch_size,
        num_workers,
    )
    diagnostics: Dict[str, Any] = {
        "neural": prediction_diagnostics(
            dev_records, dev_neural, seen_words
        ),
        "v2": prediction_diagnostics(dev_records, dev_v2, seen_words),
        "v2_gate_statistics": dev_gate_statistics.to_dict(),
    }

    test_neural = predict_records(
        model, test_records, vocab, device, batch_size, num_workers
    )
    test_v2, test_gate_statistics = predict_with_gated_fallback(
        model,
        test_records,
        vocab,
        lexical_prior,
        gates,
        device,
        batch_size,
        num_workers,
    )
    artifacts_dir = output_dir / "artifacts"
    neural_artifacts = write_prediction_artifacts(
        artifacts_dir,
        ARTIFACT_PREFIX + "_NEURAL",
        test_records,
        test_neural,
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    v2_artifacts = write_prediction_artifacts(
        artifacts_dir,
        ARTIFACT_PREFIX + "_V2",
        test_records,
        test_v2,
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    manifest = {
        "schema_version": 1,
        "system_name": SYSTEM_NAME,
        "artifact_prefix": ARTIFACT_PREFIX,
        "description": DESCRIPTION,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "dev_metrics": checkpoint.get("dev_metrics"),
        },
        "dev": diagnostics,
        "test_v2_gate_statistics": test_gate_statistics.to_dict(),
        "neural_artifacts": dict(neural_artifacts),
        "v2_artifacts": dict(v2_artifacts),
    }
    write_step_manifest(
        artifacts_dir / f"{ARTIFACT_PREFIX}_MANIFEST.json", manifest
    )

    neural_correct = int(diagnostics["neural"]["correct"])
    v2_correct = int(diagnostics["v2"]["correct"])
    acceptance = campaign_config["acceptance"]
    neural_gate = int(
        acceptance["boundary_neural_must_exceed_correct"]
    )
    v2_gate = int(acceptance["boundary_v2_must_exceed_correct"])
    architecture_accepted = neural_correct > neural_gate
    competitive_submission_accepted = v2_correct > v2_gate
    selection = {
        "schema_version": 1,
        "system_name": SYSTEM_NAME,
        "description": DESCRIPTION,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "neural_correct": neural_correct,
        "neural_micro_f1": diagnostics["neural"]["micro_f1"],
        "neural_must_exceed": neural_gate,
        "architecture_accepted": architecture_accepted,
        "v2_correct": v2_correct,
        "v2_micro_f1": diagnostics["v2"]["micro_f1"],
        "v2_must_exceed": v2_gate,
        "competitive_submission_accepted": (
            competitive_submission_accepted
        ),
        "recommended_submission_path": (
            v2_artifacts["submission_path"]
            if competitive_submission_accepted
            else None
        ),
        "neural_ablation_path": neural_artifacts["submission_path"],
        "do_not_submit_v2_path": (
            None
            if competitive_submission_accepted
            else v2_artifacts["submission_path"]
        ),
    }
    write_json(output_dir / "SELECTION.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/dziriformer_dual_rope_boundary_crf_v8_seed42/best.pt"
        ),
    )
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=Path("configs/track4/Lyes/dual_rope_v8/campaign.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dual_rope_v8/01_boundary_crf_seed42"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    run(
        checkpoint_path=args.checkpoint,
        campaign_config_path=args.campaign_config,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
