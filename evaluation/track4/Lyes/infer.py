"""Run checkpoint inference and create competition-ready artifacts."""

import argparse
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import (
    BatchCollator,
    CharacterDataset,
    SentenceRecord,
    load_raw_sentences,
    validate_vocabulary_coverage,
)
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer
from evaluation.track4.Lyes.submission import write_submission, write_vocalized_predictions
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


@torch.inference_mode()
def predict_records(
    model: CharDiacritizer,
    records: Sequence[SentenceRecord],
    vocab: Mapping[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> List[List[int]]:
    validate_vocabulary_coverage(records, vocab)
    maximum_length = max(len(record.chars) + 2 for record in records)
    if maximum_length > model.config.max_length:
        raise ValueError(
            "test sequence length {} exceeds model max_length {}".format(
                maximum_length, model.config.max_length
            )
        )
    loader = DataLoader(
        CharacterDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchCollator(vocab),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    predictions: List[List[int]] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device, non_blocking=True
        )
        batch_predictions = model.predict(input_ids, attention_mask).to("cpu")
        for row, record in enumerate(batch["records"]):
            labels = batch_predictions[row, 1 : len(record.chars) + 1].tolist()
            labels = [
                0 if char == " " else int(label)
                for char, label in zip(record.chars, labels)
            ]
            predictions.append(labels)
    return predictions


def run_inference(
    checkpoint_path: Path,
    input_path: Path,
    ids_path: Path,
    vocalized_output_path: Path,
    submission_path: Path,
    sample_submission_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
    manifest_path: Optional[Path] = None,
    system_name: str = "Track4-Neural",
) -> None:
    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    records = load_raw_sentences(input_path, ids_path)
    predictions = predict_records(
        model,
        records,
        vocab,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    write_vocalized_predictions(vocalized_output_path, records, predictions)
    write_submission(
        submission_path,
        records,
        predictions,
        sample_submission_path=sample_submission_path,
    )
    if manifest_path is not None:
        checkpoint_dev_metrics = checkpoint.get("dev_metrics", {})
        write_json(
            manifest_path,
            {
                "system_name": system_name,
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": sha256_file(checkpoint_path),
                    "epoch": checkpoint.get("epoch"),
                    "dev_micro_f1": checkpoint_dev_metrics.get("micro_f1"),
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                },
                "test_prediction": {
                    "sentences": len(records),
                    "scored_labels": sum(
                        char != " "
                        for record in records
                        for char in record.chars
                    ),
                    "vocalized_path": str(vocalized_output_path),
                    "submission_path": str(submission_path),
                    "submission_sha256": sha256_file(submission_path),
                },
            },
        )
    print(
        "wrote {} vocalized sentences and {} scored labels".format(
            len(records),
            sum(char != " " for record in records for char in record.chars),
        )
    )
    print("vocalized output: {}".format(vocalized_output_path))
    print("submission: {}".format(submission_path))
    if manifest_path is not None:
        print("manifest: {}".format(manifest_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--vocalized-output", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--system-name", default="Track4-Neural")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    run_inference(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        ids_path=args.ids,
        vocalized_output_path=args.vocalized_output,
        submission_path=args.submission,
        sample_submission_path=args.sample_submission,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manifest_path=args.manifest,
        system_name=args.system_name,
    )


if __name__ == "__main__":
    main()
