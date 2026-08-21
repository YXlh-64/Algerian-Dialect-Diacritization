"""Dataset loading and character/label alignment for CANINE-S."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


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

NUM_LABELS = len(CLASS_NAMES)

LABEL_TO_MARKS = {
    0: "",
    1: "\u064e",
    2: "\u064b",
    3: "\u064f",
    4: "\u064c",
    5: "\u0650",
    6: "\u064d",
    7: "\u0652",
    8: "\u0651",
    9: "\u0651\u064e",
    10: "\u0651\u064b",
    11: "\u0651\u064f",
    12: "\u0651\u064c",
    13: "\u0651\u0650",
    14: "\u0651\u064d",
    15: "\u0651\u0652",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records without changing their character order."""

    records = []
    with Path(path).open(encoding="utf-8") as handle:
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
    """Find the competition ``data`` directory in local or Kaggle layouts."""

    if data_dir is not None:
        root = Path(data_dir).expanduser().resolve()
        if not _has_splits(root):
            raise FileNotFoundError(
                f"Expected train_data/*.jsonl and dev_data/*.jsonl under {root}"
            )
        return root

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "data",
        repo_root / "Data",
        Path.cwd() / "data",
        Path.cwd() / "Data",
    ]
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        candidates.extend(p.parent for p in kaggle_root.rglob("train_data"))

    for candidate in candidates:
        if _has_splits(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate the dataset. Pass --data-dir pointing to the folder "
        "that contains train_data/ and dev_data/."
    )


def record_chars(record: dict[str, Any]) -> list[str]:
    """Return the exact input character sequence used for label alignment."""

    chars = record.get("chars")
    if chars is None:
        chars = list(record["input"])
    return list(chars)


def normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    chars = record_chars(record)
    labels = [int(label) for label in record["labels"]]
    if len(chars) != len(labels):
        raise ValueError(
            f"Record {record.get('sent_id', '<unknown>')} has "
            f"{len(chars)} chars but {len(labels)} labels"
        )
    return {**record, "chars": chars, "labels": labels}


class CanineDiacritizationDataset(Dataset):
    """Tokenize each sentence and align one label with each CANINE character."""

    def __init__(self, records: Iterable[dict[str, Any]], tokenizer, max_length: int = 512):
        self.records = [normalise_record(record) for record in records]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        chars, labels = record["chars"], record["labels"]
        encoding = self.tokenizer(
            "".join(chars), truncation=True, max_length=self.max_length
        )
        input_ids = encoding["input_ids"]
        content_length = max(0, len(input_ids) - 2)  # [CLS] and [SEP]
        aligned = [-100]
        aligned.extend(
            -100 if char == " " else label
            for char, label in zip(chars[:content_length], labels[:content_length])
        )
        aligned.append(-100)
        aligned.extend([-100] * (len(input_ids) - len(aligned)))
        aligned = aligned[: len(input_ids)]

        item = {key: torch.tensor(value, dtype=torch.long) for key, value in encoding.items()}
        item["labels"] = torch.tensor(aligned, dtype=torch.long)
        return item


def find_test_files(data_dir: str | Path) -> tuple[Path, Path, Path | None]:
    """Return raw test text, IDs, and optional sample submission paths."""

    test_dir = Path(data_dir) / "test_data"
    text = test_dir / "raw_sentences_test.txt"
    ids = test_dir / "raw_sentences_test_ids.txt"
    sample = test_dir / "sample_submission.csv"
    if not text.exists():
        matches = sorted(test_dir.glob("raw_sentences_test_*.txt"))
        text = matches[0] if matches else text
    if not ids.exists():
        matches = sorted(test_dir.glob("raw_sentences_test_ids*.txt"))
        ids = matches[0] if matches else ids
    if not text.exists() or not ids.exists():
        raise FileNotFoundError(f"Could not find raw test text and IDs under {test_dir}")
    return text, ids, sample if sample.exists() else None


def diacritize(chars: Iterable[str], labels: Iterable[int]) -> str:
    """Reconstruct a fully vocalized string from base chars and class IDs."""

    return "".join(
        char + LABEL_TO_MARKS.get(int(label), "")
        for char, label in zip(chars, labels)
    )
