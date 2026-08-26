from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, CanineForTokenClassification

from utils.track2.canine_s_model.data_utils import CLASS_NAMES, LABEL_TO_MARKS, load_jsonl, resolve_dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved CANINE-S diacritization model.")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing the saved model and tokenizer.")
    parser.add_argument("--data-dir", type=str, default=None, help="Root data directory containing dev_data.")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Sequence length cap.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Execution device.")
    return parser.parse_args()


def _load_model_and_tokenizer(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = CanineForTokenClassification.from_pretrained(str(model_dir))
    return tokenizer, model


def extract_char_predictions(model, tokenizer, sentence: str, device: torch.device, max_seq_len: int = 512):
    encoding = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_seq_len)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**encoding).logits
    predictions = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    return predictions[1:-1]


def evaluate(model_dir: str | Path, data_dir: str | Path | None = None, max_seq_len: int = 512, device: str = "cpu") -> dict:
    device = torch.device(device)
    model_dir = Path(model_dir)
    data_dir = resolve_dataset_dir(data_dir)
    dev_path = data_dir / "dev_data" / "dev_Algerian-DIAC.jsonl"

    tokenizer, model = _load_model_and_tokenizer(model_dir)
    model.to(device)

    records = load_jsonl(dev_path)
    all_true = []
    all_pred = []

    for record in records:
        sentence = record["input"]
        pred = extract_char_predictions(model, tokenizer, sentence, device, max_seq_len=max_seq_len)
        gold = record["labels"][: len(pred)]
        all_true.extend(gold)
        all_pred.extend(pred)

    metrics = {
        "micro_f1": float(f1_score(all_true, all_pred, average="micro")),
        "accuracy": float(accuracy_score(all_true, all_pred)),
    }
    report = classification_report(
        all_true,
        all_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(CLASS_NAMES))))

    return {**metrics, "report": report, "confusion_matrix": cm.tolist()}


def main() -> None:
    args = parse_args()
    result = evaluate(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )
    print(f"Micro F1: {result['micro_f1']:.4f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(result["report"])


if __name__ == "__main__":
    main()
