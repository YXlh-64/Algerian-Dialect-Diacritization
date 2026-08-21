"""Evaluate every registered seed-42 paper model with one metric contract."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from utils.track4.Lyes.gated_fusion.config import load_gates
from utils.track4.Lyes.gated_fusion.fusion import predict_with_gated_fallback
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import SentenceRecord, load_jsonl
from evaluation.track4.Lyes.infer import predict_records
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from evaluation.track4.Lyes.paper_metrics import compute_paper_metrics
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


EXPECTED_ROOT_KEYS = {"schema_version", "output_root", "models"}
EXPECTED_MODEL_KEYS = {"slug", "name", "checkpoint"}


def load_paper_registry(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or set(config) != EXPECTED_ROOT_KEYS:
        raise ValueError("invalid paper model registry keys")
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported paper model registry schema")
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("paper model registry cannot be empty")
    slugs = []
    names = []
    for model in models:
        if not isinstance(model, dict) or set(model) != EXPECTED_MODEL_KEYS:
            raise ValueError("invalid paper model registry entry")
        slug = model["slug"]
        name = model["name"]
        checkpoint = model["checkpoint"]
        if (
            not isinstance(slug, str)
            or re.fullmatch(r"[a-z0-9_]+", slug) is None
        ):
            raise ValueError("paper model slug is invalid")
        if not isinstance(name, str) or not name:
            raise ValueError("paper model name is invalid")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("paper model checkpoint is invalid")
        slugs.append(slug)
        names.append(name)
    if len(slugs) != len(set(slugs)) or len(names) != len(set(names)):
        raise ValueError("paper model slugs and names must be unique")
    return config


def write_prediction_jsonl(
    path: Path,
    records: Sequence[SentenceRecord],
    predictions: Sequence[Sequence[int]],
) -> None:
    if len(records) != len(predictions):
        raise ValueError("prediction JSONL inputs must align")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record, labels in zip(records, predictions):
            if len(labels) != len(record.chars):
                raise ValueError("prediction JSONL label length mismatch")
            stream.write(
                json.dumps(
                    {
                        "sent_id": record.sent_id,
                        "labels": [int(label) for label in labels],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _summary_row(
    name: str,
    slug: str,
    variant: str,
    parameter_count: int,
    epoch: Any,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "model": name,
        "slug": slug,
        "variant": variant,
        "parameters": parameter_count,
        "best_epoch": epoch,
        "accuracy": metrics["accuracy"],
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "macro_f1_present_classes": metrics[
            "macro_f1_present_classes"
        ],
        "wer": metrics["wer"],
        "cer": metrics["cer"],
        "word_accuracy": metrics["word_accuracy"],
        "sentence_accuracy": metrics["sentence_accuracy"],
        "shadda_accuracy": metrics["shadda"]["accuracy"],
        "tanween_accuracy": metrics["tanween"]["accuracy"],
        "skeleton_mismatch_count": metrics[
            "skeleton_mismatch_count"
        ],
        "char_bleu": metrics["char_bleu"]["score"],
        "correct_letters": metrics["correct_letters"],
        "scored_letters": metrics["scored_letters"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty report CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    model_count: int,
    integrity_notes: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Paper Dev Metrics",
        "",
        (
            "All systems use the same 607-sentence released dev split and "
            "the metric definitions embedded in each JSON artifact."
        ),
        "",
        "| Model | Variant | Accuracy | Macro F1 | WER | CER | Word Acc. | Sentence Acc. | Shadda Acc. | Tanween Acc. | char-BLEU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {variant} | {accuracy:.6f} | {macro_f1:.6f} | "
            "{wer:.6f} | {cer:.6f} | {word_accuracy:.6f} | "
            "{sentence_accuracy:.6f} | {shadda_accuracy:.6f} | "
            "{tanween_accuracy:.6f} | {char_bleu:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Registered neural models: {}.".format(model_count),
            "Each model has a neural row and an unchanged V2 lexical-fallback row.",
            "Every skeleton mismatch count is expected to be zero.",
            "",
        ]
    )
    if integrity_notes:
        lines.extend(
            [
                "## Recomputed-score integrity notes",
                "",
                (
                    "Fresh decoded predictions are authoritative for every "
                    "paper metric. Historical checkpoint differences of more "
                    "than one letter fail the report."
                ),
                "",
            ]
        )
        for note in integrity_notes:
            lines.append(
                "- {model}: stored {checkpoint_correct}, freshly decoded "
                "{recomputed_correct} ({delta:+d} letter).".format(**note)
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    registry_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    registry = load_paper_registry(registry_path)
    device = select_device(device_name)
    output_root = Path(str(registry["output_root"]))
    train_records = load_jsonl(
        Path("Data/train_data/train_Algerian-DIAC.jsonl")
    )
    dev_records = load_jsonl(
        Path("Data/dev_data/dev_Algerian-DIAC.jsonl")
    )
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(Path("configs/track4/Lyes/gates.json"))
    summary_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    manifest_models = []
    integrity_notes: List[Dict[str, Any]] = []

    for entry in registry["models"]:
        checkpoint_path = Path(str(entry["checkpoint"]))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "paper checkpoint is missing: {}".format(checkpoint_path)
            )
        checkpoint = load_checkpoint(checkpoint_path, device)
        model, vocab = build_model_from_checkpoint(checkpoint, device)
        neural_predictions = predict_records(
            model,
            dev_records,
            vocab,
            device,
            batch_size,
            num_workers,
        )
        v2_predictions, gate_statistics = predict_with_gated_fallback(
            model,
            dev_records,
            vocab,
            prior,
            gates,
            device,
            batch_size,
            num_workers,
        )
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        model_dir = output_root / "models" / str(entry["slug"])
        checkpoint_metrics = checkpoint.get("dev_metrics", {})
        checkpoint_correct = checkpoint_metrics.get("correct")
        recomputed_correct = None
        variants = (
            ("neural", neural_predictions),
            ("v2", v2_predictions),
        )
        for variant, predictions in variants:
            metrics = compute_paper_metrics(dev_records, predictions)
            if variant == "neural":
                recomputed_correct = int(metrics["correct_letters"])
                if (
                    checkpoint_correct is not None
                    and abs(recomputed_correct - int(checkpoint_correct)) > 1
                ):
                    raise RuntimeError(
                        "paper metric drift exceeds one letter for {}: {} != {}".format(
                            entry["name"],
                            recomputed_correct,
                            checkpoint_correct,
                        )
                    )
                if (
                    checkpoint_correct is not None
                    and recomputed_correct != int(checkpoint_correct)
                ):
                    integrity_notes.append(
                        {
                            "model": entry["name"],
                            "checkpoint_correct": int(checkpoint_correct),
                            "recomputed_correct": recomputed_correct,
                            "delta": (
                                recomputed_correct - int(checkpoint_correct)
                            ),
                        }
                    )
            write_json(model_dir / f"{variant}_metrics.json", metrics)
            write_prediction_jsonl(
                model_dir / f"{variant}_predictions.jsonl",
                dev_records,
                predictions,
            )
            summary_rows.append(
                _summary_row(
                    str(entry["name"]),
                    str(entry["slug"]),
                    variant,
                    parameter_count,
                    checkpoint.get("epoch"),
                    metrics,
                )
            )
            for per_class in metrics["per_class"]:
                per_class_rows.append(
                    {
                        "model": entry["name"],
                        "slug": entry["slug"],
                        "variant": variant,
                        **per_class,
                    }
                )
        manifest_models.append(
            {
                "slug": entry["slug"],
                "name": entry["name"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "parameter_count": parameter_count,
                "epoch": checkpoint.get("epoch"),
                "checkpoint_dev_correct": checkpoint_correct,
                "recomputed_dev_correct": recomputed_correct,
                "recomputed_minus_checkpoint": (
                    None
                    if checkpoint_correct is None
                    or recomputed_correct is None
                    else recomputed_correct - int(checkpoint_correct)
                ),
                "v2_gate_statistics": gate_statistics.to_dict(),
            }
        )

    _write_csv(output_root / "ALL_MODELS_SUMMARY.csv", summary_rows)
    _write_csv(output_root / "PER_CLASS_F1.csv", per_class_rows)
    _write_markdown(
        output_root / "PAPER_METRICS.md",
        summary_rows,
        len(registry["models"]),
        integrity_notes,
    )
    manifest = {
        "schema_version": 1,
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "device": str(device),
        "dev_sentences": len(dev_records),
        "models": manifest_models,
        "summary_csv": str(output_root / "ALL_MODELS_SUMMARY.csv"),
        "per_class_csv": str(output_root / "PER_CLASS_F1.csv"),
        "markdown_report": str(output_root / "PAPER_METRICS.md"),
    }
    write_json(output_root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "device": str(device),
                "models": len(registry["models"]),
                "variants": len(summary_rows),
                "summary": manifest["summary_csv"],
                "per_class": manifest["per_class_csv"],
                "report": manifest["markdown_report"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/track4/Lyes/paper_models.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    run(
        registry_path=args.registry,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
