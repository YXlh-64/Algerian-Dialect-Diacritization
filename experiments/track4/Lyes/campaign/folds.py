"""Deterministic balanced sentence folds for leakage-free OOF predictions."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from utils.track4.Lyes.data import SentenceRecord


def make_balanced_folds(
    records: Sequence[SentenceRecord], fold_count: int, seed: int
) -> Tuple[Tuple[int, ...], ...]:
    if fold_count < 2 or fold_count > len(records):
        raise ValueError("invalid fold_count")
    keyed = []
    for index, record in enumerate(records):
        scored = sum(char != " " for char in record.chars)
        tie = hashlib.sha256(
            f"{seed}:{record.sent_id}".encode("utf-8")
        ).hexdigest()
        keyed.append((-scored, tie, index, scored))
    keyed.sort()
    folds: List[List[int]] = [[] for _ in range(fold_count)]
    totals = [0] * fold_count
    for _, _, index, scored in keyed:
        destination = min(
            range(fold_count), key=lambda fold: (totals[fold], fold)
        )
        folds[destination].append(index)
        totals[destination] += scored
    flattened = [index for fold in folds for index in fold]
    if sorted(flattened) != list(range(len(records))):
        raise RuntimeError("fold partition is incomplete")
    return tuple(tuple(sorted(fold)) for fold in folds)


def fold_assignment(
    records: Sequence[SentenceRecord],
    folds: Sequence[Sequence[int]],
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for fold_index, indices in enumerate(folds):
        for index in indices:
            sent_id = records[index].sent_id
            if sent_id in result:
                raise ValueError("sentence appears in multiple folds")
            result[sent_id] = fold_index
    if len(result) != len(records):
        raise ValueError("fold assignment does not cover every sentence")
    return result


def write_records_jsonl(
    path: Path, records: Sequence[SentenceRecord]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            if record.labels is None:
                raise ValueError("training fold records require labels")
            value = {
                "sent_id": record.sent_id,
                "chars": list(record.chars),
                "labels": list(record.labels),
                "input": record.input_text,
            }
            if record.target_text is not None:
                value["target"] = record.target_text
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
