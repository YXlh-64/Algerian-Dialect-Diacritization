"""DziriFusion-v1: neural character model plus training-only lexical prior."""

import argparse
import re
from pathlib import Path
from typing import Dict, Sequence

import torch

from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import SentenceRecord, load_jsonl, load_raw_sentences
from utils.track4.Lyes.lexical_fusion import WordLabelPrior, predict_with_lexical_fusion
from evaluation.track4.Lyes.metrics import MetricAccumulator
from evaluation.track4.Lyes.submission import write_submission, write_vocalized_predictions
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


DEFAULT_SYSTEM_NAME = "DziriFusion-v1"
DEFAULT_ARTIFACT_PREFIX = "DZIRIFUSION_V1"


def evaluate_predictions(
    records: Sequence[SentenceRecord],
    predictions: Sequence[Sequence[int]],
) -> Dict[str, object]:
    if len(records) != len(predictions):
        raise ValueError("evaluation record/prediction count mismatch")
    metrics = MetricAccumulator()
    for record, record_predictions in zip(records, predictions):
        if record.labels is None:
            raise ValueError("dev records must contain labels")
        targets = torch.tensor(
            [
                label if char != " " else -100
                for char, label in zip(record.chars, record.labels)
            ],
            dtype=torch.long,
        )
        metrics.update(torch.tensor(record_predictions), targets)
    return metrics.compute()


def run(args: argparse.Namespace) -> None:
    if args.prior_strength < 0.0:
        raise ValueError("--prior-strength cannot be negative")
    if args.smoothing <= 0.0:
        raise ValueError("--smoothing must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not re.fullmatch(r"[A-Z0-9_]+", args.artifact_prefix):
        raise ValueError(
            "--artifact-prefix must contain only A-Z, 0-9, and underscore"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            "{} does not exist; train DziriFusion-v1 first".format(
                args.checkpoint
            )
        )

    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    train_records = load_jsonl(args.train_data)
    lexical_prior = WordLabelPrior().fit(train_records)

    dev_metrics = None
    dev_fusion_statistics = None
    if not args.skip_dev_evaluation:
        dev_records = load_jsonl(args.dev_data)
        dev_predictions, dev_statistics = predict_with_lexical_fusion(
            model=model,
            records=dev_records,
            vocab=vocab,
            lexical_prior=lexical_prior,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prior_strength=args.prior_strength,
            smoothing=args.smoothing,
        )
        dev_metrics = evaluate_predictions(dev_records, dev_predictions)
        dev_fusion_statistics = dev_statistics.to_dict()
        print(
                "{} dev Micro-F1: {:.10f} ({}/{})".format(
                args.system_name,
                dev_metrics["micro_f1"],
                dev_metrics["correct"],
                dev_metrics["total"],
            )
        )

    test_records = load_raw_sentences(args.input, args.ids)
    test_predictions, test_statistics = predict_with_lexical_fusion(
        model=model,
        records=test_records,
        vocab=vocab,
        lexical_prior=lexical_prior,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prior_strength=args.prior_strength,
        smoothing=args.smoothing,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocalized_path = args.output_dir / (
        args.artifact_prefix + "_TEST_VOCALIZED.txt"
    )
    submission_path = args.output_dir / (
        args.artifact_prefix + "_SUBMISSION.csv"
    )
    manifest_path = args.output_dir / (
        args.artifact_prefix + "_MANIFEST.json"
    )
    write_vocalized_predictions(vocalized_path, test_records, test_predictions)
    write_submission(
        submission_path,
        test_records,
        test_predictions,
        sample_submission_path=args.sample_submission,
    )

    checkpoint_dev_metrics = checkpoint.get("dev_metrics", {})
    manifest = {
        "system_name": args.system_name,
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
            "prior_strength": args.prior_strength,
            "smoothing": args.smoothing,
        },
        "dev_evaluation": (
            None
            if dev_metrics is None
            else {
                "micro_f1": dev_metrics["micro_f1"],
                "correct": dev_metrics["correct"],
                "total": dev_metrics["total"],
                "fusion_statistics": dev_fusion_statistics,
            }
        ),
        "test_prediction": {
            "sentences": len(test_records),
            "scored_labels": sum(
                char != " " for record in test_records for char in record.chars
            ),
            "fusion_statistics": test_statistics.to_dict(),
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
            "Run DziriFusion-v1: CNN/local-Transformer plus a smoothed "
            "training-only lexical prior"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/dziri_fusion_v1_seed42/best.pt"),
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
        default=Path("outputs/dziri_fusion_v1_seed42"),
    )
    parser.add_argument("--prior-strength", type=float, default=3.0)
    parser.add_argument("--smoothing", type=float, default=0.01)
    parser.add_argument("--system-name", default=DEFAULT_SYSTEM_NAME)
    parser.add_argument(
        "--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-dev-evaluation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
