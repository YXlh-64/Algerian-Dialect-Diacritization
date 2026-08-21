"""Prediction serialization and exact Kaggle submission validation."""

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from utils.track4.Lyes.data import SentenceRecord
from utils.track4.Lyes.labels import NUM_LABELS, apply_diacritics


def validate_label_sequences(
    records: Sequence[SentenceRecord],
    label_sequences: Sequence[Sequence[int]],
) -> None:
    if len(records) != len(label_sequences):
        raise ValueError(
            "record/prediction count mismatch: {} != {}".format(
                len(records), len(label_sequences)
            )
        )
    for record, labels in zip(records, label_sequences):
        if len(record.chars) != len(labels):
            raise ValueError(
                "prediction length mismatch for {}: {} != {}".format(
                    record.sent_id, len(labels), len(record.chars)
                )
            )
        for char, label in zip(record.chars, labels):
            if not 0 <= int(label) < NUM_LABELS:
                raise ValueError(
                    "{} contains a prediction outside [0, 15]".format(
                        record.sent_id
                    )
                )
            if char == " " and int(label) != 0:
                raise ValueError(
                    "{} contains a non-zero prediction for a space".format(
                        record.sent_id
                    )
                )


def write_vocalized_predictions(
    path: Path,
    records: Sequence[SentenceRecord],
    label_sequences: Sequence[Sequence[int]],
) -> None:
    validate_label_sequences(records, label_sequences)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record, labels in zip(records, label_sequences):
            stream.write(apply_diacritics(record.chars, labels))
            stream.write("\n")


def expected_submission_ids(records: Iterable[SentenceRecord]) -> List[str]:
    return [
        "{}_{}".format(record.sent_id, index)
        for record in records
        for index, char in enumerate(record.chars)
        if char != " "
    ]


def read_template_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Id", "Label"]:
            raise ValueError(
                "{} must have exactly the columns Id,Label".format(path)
            )
        return [row["Id"] for row in reader]


def write_submission(
    path: Path,
    records: Sequence[SentenceRecord],
    label_sequences: Sequence[Sequence[int]],
    sample_submission_path: Optional[Path] = None,
) -> None:
    validate_label_sequences(records, label_sequences)
    ids = expected_submission_ids(records)
    if sample_submission_path is not None:
        template_ids = read_template_ids(sample_submission_path)
        if ids != template_ids:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(ids, template_ids))
                    if pair[0] != pair[1]
                ),
                min(len(ids), len(template_ids)),
            )
            expected = ids[mismatch] if mismatch < len(ids) else "<end>"
            actual = (
                template_ids[mismatch]
                if mismatch < len(template_ids)
                else "<end>"
            )
            raise ValueError(
                "submission template ID mismatch at row {}: expected {} got {}".format(
                    mismatch + 2, expected, actual
                )
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Id", "Label"])
        for record, labels in zip(records, label_sequences):
            for index, (char, label) in enumerate(zip(record.chars, labels)):
                if char != " ":
                    writer.writerow(
                        ["{}_{}".format(record.sent_id, index), int(label)]
                    )


def read_submission(path: Path) -> List[Tuple[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Id", "Label"]:
            raise ValueError("submission columns must be Id,Label")
        return [(row["Id"], int(row["Label"])) for row in reader]
