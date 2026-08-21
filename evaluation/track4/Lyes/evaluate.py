"""Evaluate and export the controlled DualRoPE-CRF-v7 seed-42 ablation."""

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
from experiments.track4.Lyes.export_ensemble import load_v7_config
from evaluation.track4.Lyes.infer import predict_records
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


ARTIFACT_PREFIX = "DZIRIFORMER_DUALROPE_CRF_V7_SEED42"
SYSTEM_NAME = "DziriFormer-DualRoPE-CRF-v7-seed42"
DESCRIPTION = (
    "Controlled Track 4 ablation retaining the complete 9.89M DualRoPE-v6 "
    "encoder and replacing independent 16-class CE decoding with a "
    "first-order linear-chain CRF over packed scored-letter positions. "
    "Spaces remain encoder inputs but are excluded from the CRF chain. "
    "The V2 artifact applies the unchanged confidence-gated lexical fallback "
    "to CRF marginal probabilities and Viterbi labels."
)


def run(
    checkpoint_path: Path,
    campaign_config_path: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError("CRF checkpoint is missing: {}".format(
            checkpoint_path
        ))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    campaign_config = load_v7_config(campaign_config_path)
    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    if model.config.resolved_head_mode != "crf":
        raise ValueError("checkpoint is not a CRF model")

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
    neural_gate = int(
        campaign_config["acceptance"]["crf_neural_must_exceed_correct"]
    )
    v2_gate = int(
        campaign_config["acceptance"]["crf_v2_must_exceed_correct"]
    )
    architecture_accepted = neural_correct > neural_gate
    competitive_submission_accepted = v2_correct > v2_gate
    selection = {
        "schema_version": 1,
        "system_name": SYSTEM_NAME,
        "description": DESCRIPTION,
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
            "outputs/dziriformer_dual_rope_crf_v7_seed42/best.pt"
        ),
    )
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=Path("configs/track4/Lyes/campaign.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dual_rope_v7/03_crf_seed42"),
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
