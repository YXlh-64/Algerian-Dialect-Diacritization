"""Validated dataset loading, encoding, and length-bucket batching."""

import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler

from utils.track4.Lyes.labels import IGNORE_INDEX, NUM_LABELS


REQUIRED_SPECIAL_TOKENS: Tuple[str, ...] = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")


@dataclass(frozen=True)
class SentenceRecord:
    sent_id: str
    chars: Tuple[str, ...]
    labels: Optional[Tuple[int, ...]]
    input_text: str
    target_text: Optional[str] = None


def load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("vocabulary must be a JSON object")
    vocab = {str(char): int(index) for char, index in raw.items()}
    missing = [token for token in REQUIRED_SPECIAL_TOKENS if token not in vocab]
    if missing:
        raise ValueError("vocabulary is missing special tokens: {}".format(missing))
    indices = sorted(vocab.values())
    if indices != list(range(len(indices))):
        raise ValueError("vocabulary indices must be unique and contiguous from zero")
    if vocab["<PAD>"] != 0:
        raise ValueError("<PAD> must have index 0")
    return vocab


def _validate_record(raw: Mapping[str, Any], source: Path, line_number: int) -> SentenceRecord:
    required = {"sent_id", "chars", "labels", "input"}
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(
            "{}:{} missing keys {}".format(source, line_number, ", ".join(missing))
        )

    sent_id = str(raw["sent_id"])
    chars = tuple(str(char) for char in raw["chars"])
    labels = tuple(int(label) for label in raw["labels"])
    input_text = str(raw["input"])
    target = raw.get("target")
    target_text = str(target) if target is not None else None

    if not chars:
        raise ValueError("{}:{} contains an empty sentence".format(source, line_number))
    if any(len(char) != 1 for char in chars):
        raise ValueError(
            "{}:{} chars must contain one code point each".format(source, line_number)
        )
    if len(chars) != len(labels):
        raise ValueError(
            "{}:{} chars/labels length mismatch".format(source, line_number)
        )
    if "".join(chars) != input_text:
        raise ValueError(
            "{}:{} input does not equal joined chars".format(source, line_number)
        )
    if any(label < 0 or label >= NUM_LABELS for label in labels):
        raise ValueError(
            "{}:{} contains a label outside [0, 15]".format(source, line_number)
        )
    if any(char == " " and label != 0 for char, label in zip(chars, labels)):
        raise ValueError(
            "{}:{} contains a non-zero space label".format(source, line_number)
        )

    return SentenceRecord(
        sent_id=sent_id,
        chars=chars,
        labels=labels,
        input_text=input_text,
        target_text=target_text,
    )


def load_jsonl(path: Path) -> List[SentenceRecord]:
    records: List[SentenceRecord] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError("{}:{} is blank".format(path, line_number))
            raw = json.loads(line)
            record = _validate_record(raw, path, line_number)
            if record.sent_id in seen_ids:
                raise ValueError(
                    "{}:{} duplicate sent_id {}".format(
                        path, line_number, record.sent_id
                    )
                )
            seen_ids.add(record.sent_id)
            records.append(record)
    if not records:
        raise ValueError("{} contains no records".format(path))
    return records


def load_raw_sentences(input_path: Path, ids_path: Path) -> List[SentenceRecord]:
    input_lines = input_path.read_text(encoding="utf-8").splitlines()
    sent_ids = ids_path.read_text(encoding="utf-8").splitlines()
    if len(input_lines) != len(sent_ids):
        raise ValueError(
            "test line mismatch: inputs={} ids={}".format(
                len(input_lines), len(sent_ids)
            )
        )

    records: List[SentenceRecord] = []
    seen_ids = set()
    for line_number, (sent_id, text) in enumerate(
        zip(sent_ids, input_lines), start=1
    ):
        normalized = unicodedata.normalize("NFC", text)
        if not sent_id:
            raise ValueError("empty test ID at line {}".format(line_number))
        if sent_id in seen_ids:
            raise ValueError("duplicate test ID {}".format(sent_id))
        if not normalized:
            raise ValueError("empty test sentence at line {}".format(line_number))
        if normalized != text:
            raise ValueError(
                "test sentence {} is not NFC-normalized".format(sent_id)
            )
        seen_ids.add(sent_id)
        records.append(
            SentenceRecord(
                sent_id=sent_id,
                chars=tuple(normalized),
                labels=None,
                input_text=normalized,
            )
        )
    return records


def validate_vocabulary_coverage(
    records: Iterable[SentenceRecord], vocab: Mapping[str, int]
) -> None:
    unknown = sorted(
        {
            char
            for record in records
            for char in record.chars
            if char not in vocab
        }
    )
    if unknown:
        raise ValueError("characters missing from vocabulary: {!r}".format(unknown))


class CharacterDataset(Dataset):
    def __init__(self, records: Sequence[SentenceRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SentenceRecord:
        return self.records[index]


class BatchCollator:
    """Encode variable-length records with BOS/EOS and dynamic padding."""

    def __init__(self, vocab: Mapping[str, int]) -> None:
        self.vocab = dict(vocab)
        self.pad_id = self.vocab["<PAD>"]
        self.unk_id = self.vocab["<UNK>"]
        self.bos_id = self.vocab["<BOS>"]
        self.eos_id = self.vocab["<EOS>"]

    def __call__(self, records: Sequence[SentenceRecord]) -> Dict[str, Any]:
        if not records:
            raise ValueError("cannot collate an empty batch")

        lengths = [len(record.chars) + 2 for record in records]
        max_length = max(lengths)
        batch_size = len(records)
        input_ids = torch.full(
            (batch_size, max_length), self.pad_id, dtype=torch.long
        )
        targets = torch.full(
            (batch_size, max_length), IGNORE_INDEX, dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, max_length), dtype=torch.bool
        )

        for row, record in enumerate(records):
            token_ids = [self.bos_id]
            token_ids.extend(self.vocab.get(char, self.unk_id) for char in record.chars)
            token_ids.append(self.eos_id)
            length = len(token_ids)
            input_ids[row, :length] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[row, :length] = True

            if record.labels is not None:
                for position, (char, label) in enumerate(
                    zip(record.chars, record.labels), start=1
                ):
                    if char != " ":
                        targets[row, position] = label

        return {
            "input_ids": input_ids,
            "targets": targets,
            "attention_mask": attention_mask,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "records": list(records),
        }


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Deterministic sortish sampling that limits padding without fixed ordering."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_size_multiplier: int = 20,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if bucket_size_multiplier <= 0:
            raise ValueError("bucket_size_multiplier must be positive")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = batch_size * bucket_size_multiplier
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)

        batches: List[List[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.lengths[index], reverse=True)
            batches.extend(
                bucket[offset : offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
