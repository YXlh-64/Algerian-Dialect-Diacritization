import json
from pathlib import Path

import pytest

from utils.track4.Lyes.data import (
    BatchCollator,
    load_jsonl,
    load_raw_sentences,
    load_vocab,
    validate_vocabulary_coverage,
)


DATA_ROOT = Path(__file__).resolve().parents[3] / "Data"


def test_released_dataset_contract_and_counts() -> None:
    train = load_jsonl(
        DATA_ROOT / "train_data" / "train_Algerian-DIAC.jsonl"
    )
    dev = load_jsonl(DATA_ROOT / "dev_data" / "dev_Algerian-DIAC.jsonl")
    test = load_raw_sentences(
        DATA_ROOT / "test_data" / "raw_sentences_test.txt",
        DATA_ROOT / "test_data" / "raw_sentences_test_ids.txt",
    )
    vocab = load_vocab(DATA_ROOT / "vocab.json")
    validate_vocabulary_coverage(train + dev + test, vocab)
    assert len(train) == 4864
    assert len(dev) == 607
    assert len(test) == 608
    assert max(len(record.chars) for record in train) == 274


def test_collator_masks_space_bos_eos_and_padding() -> None:
    records = load_jsonl(
        DATA_ROOT / "train_data" / "train_Algerian-DIAC.jsonl"
    )[:2]
    vocab = load_vocab(DATA_ROOT / "vocab.json")
    batch = BatchCollator(vocab)(records)
    for row, record in enumerate(records):
        assert batch["targets"][row, 0].item() == -100
        assert batch["targets"][row, len(record.chars) + 1].item() == -100
        for index, char in enumerate(record.chars, start=1):
            if char == " ":
                assert batch["targets"][row, index].item() == -100
            else:
                assert batch["targets"][row, index].item() == record.labels[index - 1]


def test_invalid_alignment_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "sent_id": "000001",
                "chars": ["ب", " "],
                "labels": [1],
                "input": "ب ",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="length mismatch"):
        load_jsonl(path)
