"""Evaluate a saved Track 2 CANINE-S two-head checkpoint on the dev split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import CanineTokenizer, DataCollatorForTokenClassification

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.track2.canine_twohead import CanineTwoHeadForDiacritization
from training.track2.canine_twohead.finetune_canine_twohead import _first_jsonl
from utils.track2.canine_twohead.data_utils import (
    CLASS_NAMES,
    CanineDiacritizationDataset,
    load_jsonl,
    resolve_dataset_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model_dir: str | Path, data_dir: str | Path | None = None,
             batch_size: int = 32, max_length: int = 512, device: str = "cpu") -> dict:
    data_root = resolve_dataset_dir(data_dir)
    model_root = Path(model_dir)
    tokenizer = CanineTokenizer.from_pretrained(str(model_root))
    model = CanineTwoHeadForDiacritization.from_pretrained(str(model_root))
    target_device = torch.device(device)
    model.to(target_device).eval()

    records = load_jsonl(_first_jsonl(data_root, "dev_data"))
    dataset = CanineDiacritizationDataset(records, tokenizer, max_length=max_length)
    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    true_labels, predicted_labels = [], []
    for batch in loader:
        labels = batch["labels"]
        inputs = {key: value.to(target_device) for key, value in batch.items() if key != "labels"}
        predictions = model(**inputs).logits.argmax(dim=-1).cpu()
        mask = labels != -100
        true_labels.extend(labels[mask].tolist())
        predicted_labels.extend(predictions[mask].tolist())

    accuracy = float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels)))
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "micro_f1": float(f1_score(true_labels, predicted_labels, average="micro")),
        "accuracy": accuracy,
        "DER": 1.0 - accuracy,
        "macro_f1": float(f1_score(
            true_labels, predicted_labels, labels=list(range(len(CLASS_NAMES))),
            average="macro", zero_division=0,
        )),
        "n_characters": len(true_labels),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
    }


def _markdown(result: dict) -> str:
    lines = [
        "# CANINE-S two-head — Strategy A",
        "",
        "Official metrics are computed over Arabic character positions only; "
        "space labels are ignored.",
        "",
        f"- Accuracy / micro-F1: **{result['accuracy']:.4f}**",
        f"- DER: **{result['DER']:.4f}**",
        f"- Macro-F1: **{result['macro_f1']:.4f}**",
        f"- Characters evaluated: **{result['n_characters']}**",
        "",
        "## Per-class report",
        "",
        "```text",
    ]
    lines.extend([
        f"{name:22s} precision={values['precision']:.4f} "
        f"recall={values['recall']:.4f} f1={values['f1-score']:.4f} "
        f"support={int(values['support'])}"
        for name, values in result["classification_report"].items()
        if isinstance(values, dict) and "precision" in values
    ])
    lines.extend(["```", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = evaluate(
        args.model_dir, args.data_dir, args.batch_size, args.max_length, args.device
    )
    print(f"Accuracy / micro-F1: {result['accuracy']:.4f}")
    print(f"DER: {result['DER']:.4f}")
    if args.report_path:
        report_path = Path(args.report_path)
    else:
        report_path = Path(args.model_dir) / "evaluation_report.md"
    report_path.write_text(_markdown(result), encoding="utf-8")
    (report_path.parent / "dev_metrics.json").write_text(
        json.dumps({key: value for key, value in result.items()
                    if key not in {"classification_report", "confusion_matrix"}},
                   indent=2),
        encoding="utf-8",
    )
    print(f"wrote report: {report_path}")


if __name__ == "__main__":
    main()
