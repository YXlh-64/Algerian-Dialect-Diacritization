from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


def _has_splits(root: Path) -> bool:
    return (
        (root / "train_data").is_dir()
        and (root / "dev_data").is_dir()
        and any((root / "train_data").glob("*.jsonl"))
        and any((root / "dev_data").glob("*.jsonl"))
    )


def resolve_dataset_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the dataset root from CLI input or known repository layouts."""
    if data_dir is not None:
        candidate = Path(data_dir).expanduser().resolve()
        if _has_splits(candidate):
            return candidate
        raise FileNotFoundError(
            f"Expected train_data/*.jsonl and dev_data/*.jsonl under {candidate}"
        )

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "data",
        repo_root / "Data",
        Path.cwd() / "data",
        Path.cwd() / "Data",
        repo_root.parent / "data",
        repo_root.parent / "Data",
    ]
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        candidates.extend(path.parent for path in kaggle_root.rglob("train_data"))

    for candidate in candidates:
        if _has_splits(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate the dataset. Pass --data-dir pointing to the folder "
        "that contains train_data/ and dev_data/."
    )


def first_jsonl(data_dir: str | Path, split: str) -> Path:
    """Return the first JSONL file in a dataset split."""

    split_dir = Path(data_dir) / split
    matches = sorted(split_dir.glob("*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No JSONL file found in {split_dir}")
    return matches[0]


def build_dataset(records: List[Dict[str, Any]]) -> Dataset:
    chars = [list(record.get("chars", record["input"])) for record in records]
    return Dataset.from_dict(
        {
            "sent_id": [r["sent_id"] for r in records],
            "chars": chars,
            "labels": [r["labels"] for r in records],
            "input": [r["input"] for r in records],
            "target": [r["target"] for r in records],
        }
    )


def tokenize_and_align_labels(examples: Dict[str, Any], tokenizer: Any, max_seq_len: int) -> Dict[str, Any]:
    encoded = tokenizer(examples["input"], truncation=True, max_length=max_seq_len)
    labels = []
    for row_chars, row_labels, row_ids in zip(
        examples["chars"], examples["labels"], encoded["input_ids"]
    ):
        content_length = max(0, len(row_ids) - 2)
        aligned = [-100]
        aligned.extend(
            -100 if char == " " else int(label)
            for char, label in zip(row_chars[:content_length], row_labels[:content_length])
        )
        aligned.append(-100)
        aligned.extend([-100] * (len(row_ids) - len(aligned)))
        labels.append(aligned[: len(row_ids)])
    encoded["labels"] = labels
    return encoded
