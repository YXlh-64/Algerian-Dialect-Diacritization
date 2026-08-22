"""Evaluate and export the standalone DziriFusion-Gated-v2 system."""

import argparse
import re
from pathlib import Path
from typing import Any, Dict

from utils.track4.Lyes.gated_fusion.config import load_gates
from utils.track4.Lyes.gated_fusion.fusion import predict_with_gated_fallback
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import load_jsonl, load_raw_sentences
from evaluation.track4.Lyes.dziri_fusion import evaluate_predictions
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from evaluation.track4.Lyes.submission import write_submission, write_vocalized_predictions
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


def _scored_label_count(records: Any) -> int:
    return sum(
        char != " " for record in records for char in record.chars
    )


def run(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            "{} does not exist; train the Track 4 backbone first".format(
                args.checkpoint
            )
        )

    gates = load_gates(args.gates)
    artifact_prefix = (
        gates.artifact_prefix
        if args.artifact_prefix is None
        else args.artifact_prefix
    )
    if not re.fullmatch(r"[A-Z0-9_]+", artifact_prefix):
        raise ValueError("--artifact-prefix must contain only A-Z, 0-9, _")
    system_name = (
        gates.system_name if args.system_name is None else args.system_name
    )
    if not system_name.strip():
        raise ValueError("--system-name cannot be blank")
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)

    train_records = load_jsonl(args.train_data)
    lexical_prior = WordLabelPrior().fit(train_records)

    dev_evaluation = None
    if not args.skip_dev_evaluation:
        dev_records = load_jsonl(args.dev_data)
        dev_predictions, dev_statistics = predict_with_gated_fallback(
            model=model,
            records=dev_records,
            vocab=vocab,
            lexical_prior=lexical_prior,
            gates=gates,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        dev_metrics = evaluate_predictions(dev_records, dev_predictions)
        dev_evaluation = {
            "micro_f1": dev_metrics["micro_f1"],
            "correct": dev_metrics["correct"],
            "total": dev_metrics["total"],
            "gated_fallback_statistics": dev_statistics.to_dict(),
        }
        print(
            "{} dev Micro-F1: {:.10f} ({}/{})".format(
                system_name,
                dev_metrics["micro_f1"],
                dev_metrics["correct"],
                dev_metrics["total"],
            )
        )

    test_records = load_raw_sentences(args.input, args.ids)
    test_predictions, test_statistics = predict_with_gated_fallback(
        model=model,
        records=test_records,
        vocab=vocab,
        lexical_prior=lexical_prior,
        gates=gates,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocalized_path = args.output_dir / (
        artifact_prefix + "_TEST_VOCALIZED.txt"
    )
    submission_path = args.output_dir / (
        artifact_prefix + "_SUBMISSION.csv"
    )
    manifest_path = args.output_dir / (
        artifact_prefix + "_MANIFEST.json"
    )
    write_vocalized_predictions(
        vocalized_path, test_records, test_predictions
    )
    write_submission(
        submission_path,
        test_records,
        test_predictions,
        sample_submission_path=args.sample_submission,
    )

    checkpoint_dev_metrics: Dict[str, Any] = checkpoint.get(
        "dev_metrics", {}
    )
    manifest = {
        "schema_version": 1,
        "system_name": system_name,
        "artifact_prefix": artifact_prefix,
        "decision_policy": (
            "Use a lexical label only when the neural and lexical labels "
            "disagree, neural max-softmax confidence is below its gate, "
            "and lexical max probability is at or above its gate."
        ),
        "gates": {
            "path": str(args.gates),
            "sha256": sha256_file(args.gates),
            **gates.to_dict(),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "epoch": checkpoint.get("epoch"),
            "dev_micro_f1": checkpoint_dev_metrics.get("micro_f1"),
        },
        "lexical_prior": {
            "training_data_path": str(args.train_data),
            "training_data_sha256": sha256_file(args.train_data),
            "vocabulary_size": lexical_prior.vocabulary_size,
            "word_observations": lexical_prior.word_observations,
        },
        "dev_evaluation": dev_evaluation,
        "test_prediction": {
            "sentences": len(test_records),
            "scored_labels": _scored_label_count(test_records),
            "gated_fallback_statistics": test_statistics.to_dict(),
            "vocalized_path": str(vocalized_path),
            "submission_path": str(submission_path),
            "submission_sha256": sha256_file(submission_path),
        },
    }
    write_json(manifest_path, manifest)

    print(
        "wrote {} test sentences and {} scored labels".format(
            len(test_records),
            manifest["test_prediction"]["scored_labels"],
        )
    )
    print("vocalized output: {}".format(vocalized_path))
    print("submission: {}".format(submission_path))
    print("manifest: {}".format(manifest_path))
    print(
        "submission SHA-256: {}".format(
            manifest["test_prediction"]["submission_sha256"]
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DziriFusion-Gated-v2 with Transformer-primary, "
            "confidence-gated lexical fallback"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/conv_local_seed42/best.pt"),
    )
    parser.add_argument(
        "--gates",
        type=Path,
        default=Path("configs/track4/Lyes/gates.json"),
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("Data/train_data/train_Algerian-DIAC.jsonl"),
    )
    parser.add_argument(
        "--dev-data",
        type=Path,
        default=Path("Data/dev_data/dev_Algerian-DIAC.jsonl"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Data/test_data/raw_sentences_test.txt"),
    )
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("Data/test_data/raw_sentences_test_ids.txt"),
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("Data/test_data/sample_submission.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dziri_fusion_gated_v2_seed42"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-dev-evaluation", action="store_true")
    parser.add_argument("--system-name")
    parser.add_argument("--artifact-prefix")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
