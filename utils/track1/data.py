"""Competition-data loading and record helpers shared across Track 1."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NUM_LABELS = 16
BOUNDARY_PADDING_ID = 5
NUM_BOUNDARY_FEATURES = BOUNDARY_PADDING_ID + 1


@dataclass(frozen=True)
class CompetitionData:
    root: Path
    train_records: list[dict[str, Any]]
    dev_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]
    vocabulary: dict[str, int]
    sample_submission_path: Path


def find_data_root(explicit_root: Path | None = None) -> Path:
    """Locate the competition dataset on Kaggle or in the repository."""
    candidates = [
        explicit_root,
        Path("/kaggle/input/competitions/algerian-dialect-vocalization/Data"),
        Path("/kaggle/input/competitions/algerian-dialect-vocalization"),
        Path("/kaggle/input/algerian-dialect-vocalization/Data"),
        Path("/kaggle/input/algerian-dialect-vocalization"),
        Path("../input/algerian-dialect-vocalization/Data"),
        Path("data"),
        Path("../data"),
        Path("data/kaggle/Data"),
        Path("../data/kaggle/Data"),
    ]
    marker = Path("train_data/train_Algerian-DIAC.jsonl")
    for candidate in candidates:
        if candidate is not None and (candidate / marker).exists():
            return candidate.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.glob(f"**/{marker}"))
        if matches:
            return matches[0].parents[1].resolve()

    raise FileNotFoundError(
        "Could not locate train_data/train_Algerian-DIAC.jsonl. "
        "Attach the Algerian Dialect Vocalization competition data."
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_records(
    records: list[dict[str, Any]],
    vocabulary: dict[str, int],
    *,
    require_labels: bool,
    num_labels: int = NUM_LABELS,
) -> None:
    for row_index, record in enumerate(records):
        chars = record["chars"]
        if "".join(chars) != record["input"]:
            raise ValueError(f"input mismatch at row {row_index}")
        if not all(char in vocabulary for char in chars):
            raise ValueError(f"OOV character at row {row_index}")
        if require_labels:
            labels = record["labels"]
            if len(chars) != len(labels):
                raise ValueError(f"length mismatch at row {row_index}")
            if not all(0 <= label < num_labels for label in labels):
                raise ValueError(f"invalid label at row {row_index}")
            if not all(label == 0 for char, label in zip(chars, labels) if char == " "):
                raise ValueError(f"nonzero space label at row {row_index}")


def load_competition_data(explicit_root: Path | None = None) -> CompetitionData:
    root = find_data_root(explicit_root)
    train_records = read_jsonl(root / "train_data/train_Algerian-DIAC.jsonl")
    dev_records = read_jsonl(root / "dev_data/dev_Algerian-DIAC.jsonl")
    vocabulary = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    test_inputs = (
        (root / "test_data/raw_sentences_test.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    test_ids = (
        (root / "test_data/raw_sentences_test_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if len(test_inputs) != len(test_ids):
        raise ValueError("test sentence and ID counts differ")
    test_records = [
        {"sent_id": sent_id, "chars": list(text), "input": text}
        for sent_id, text in zip(test_ids, test_inputs)
    ]

    validate_records(train_records, vocabulary, require_labels=True)
    validate_records(dev_records, vocabulary, require_labels=True)
    validate_records(test_records, vocabulary, require_labels=False)
    return CompetitionData(
        root=root,
        train_records=train_records,
        dev_records=dev_records,
        test_records=test_records,
        vocabulary=vocabulary,
        sample_submission_path=root / "test_data/sample_submission.csv",
    )


def iter_words(
    record: dict[str, Any], include_labels: bool = True
) -> Iterator[tuple[str, tuple[int, ...] | None, int, int]]:
    """Yield each non-space word, optional labels, and its character span."""
    chars = record["chars"]
    labels = record.get("labels")
    start = 0
    for index, char in enumerate(chars + [" "]):
        if char != " ":
            continue
        if index > start:
            word = "".join(chars[start:index])
            word_labels = (
                tuple(labels[start:index])
                if include_labels and labels is not None
                else None
            )
            yield word, word_labels, start, index
        start = index + 1


def letter_label_counts(
    records: list[dict[str, Any]], num_labels: int = NUM_LABELS
) -> np.ndarray:
    counts = np.zeros(num_labels, dtype=np.int64)
    for record in records:
        for char, label in zip(record["chars"], record["labels"]):
            if char != " ":
                counts[label] += 1
    return counts
