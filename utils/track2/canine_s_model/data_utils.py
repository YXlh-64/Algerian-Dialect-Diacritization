from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset

CLASS_NAMES = [
    "No Diacritic",
    "Fatha",
    "Fathatan",
    "Damma",
    "Dammatan",
    "Kasra",
    "Kasratan",
    "Sukoon",
    "Shadda",
    "Shadda+Fatha",
    "Shadda+Fathatan",
    "Shadda+Damma",
    "Shadda+Dammatan",
    "Shadda+Kasra",
    "Shadda+Kasratan",
    "Shadda+Sukoon",
]

LABEL_TO_MARKS = {
    0: "",
    1: "\u064E",
    2: "\u064B",
    3: "\u064F",
    4: "\u064C",
    5: "\u0650",
    6: "\u064D",
    7: "\u0652",
    8: "\u0651",
    9: "\u0651\u064E",
    10: "\u0651\u064B",
    11: "\u0651\u064F",
    12: "\u0651\u064C",
    13: "\u0651\u0650",
    14: "\u0651\u064D",
    15: "\u0651\u0652",
}


def diacritize(chars: Iterable[str], labels: Iterable[int]) -> str:
    """Reconstruct the fully diacritized sentence from character IDs and labels."""
    return "".join(ch + LABEL_TO_MARKS.get(int(label), "") for ch, label in zip(chars, labels))


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def resolve_dataset_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the dataset root from CLI input or known repository layouts."""
    if data_dir is not None:
        candidate = Path(data_dir).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset directory not found: {candidate}")

    repo_root = Path(__file__).resolve().parents[3]
    for candidate in [
        repo_root / "data",
        repo_root / "Data",
        repo_root.parent / "data",
        repo_root.parent / "Data",
    ]:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the data directory. Pass --data-dir to the training/evaluation scripts."
    )


def build_dataset(records: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_dict(
        {
            "sent_id": [r["sent_id"] for r in records],
            "tokens": [r["chars"] for r in records],
            "labels": [r["labels"] for r in records],
            "input": [r["input"] for r in records],
            "target": [r["target"] for r in records],
        }
    )


def tokenize_and_align_labels(examples: Dict[str, Any], tokenizer: Any, max_seq_len: int) -> Dict[str, Any]:
    encoded = tokenizer(examples["input"], truncation=True, max_length=max_seq_len)
    labels = []
    for row_labels, row_ids in zip(examples["labels"], encoded["input_ids"]):
        padded = [-100] + row_labels + [-100]
        padded = padded[: len(row_ids)]
        while len(padded) < len(row_ids):
            padded.append(-100)
        labels.append(padded)
    encoded["labels"] = labels
    return encoded
