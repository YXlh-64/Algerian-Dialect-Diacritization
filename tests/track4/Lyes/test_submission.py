from pathlib import Path

from utils.track4.Lyes.data import load_raw_sentences
from evaluation.track4.Lyes.submission import (
    expected_submission_ids,
    read_submission,
    read_template_ids,
    write_submission,
    write_vocalized_predictions,
)


DATA_ROOT = Path(__file__).resolve().parents[3] / "Data"


def test_released_submission_template_matches_test_contract() -> None:
    records = load_raw_sentences(
        DATA_ROOT / "test_data" / "raw_sentences_test.txt",
        DATA_ROOT / "test_data" / "raw_sentences_test_ids.txt",
    )
    expected = expected_submission_ids(records)
    template = read_template_ids(
        DATA_ROOT / "test_data" / "sample_submission.csv"
    )
    assert expected == template
    assert len(expected) == 16438


def test_writers_preserve_alignment(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    ids_path = tmp_path / "ids.txt"
    sample_path = tmp_path / "sample.csv"
    input_path.write_text("ب ب\n", encoding="utf-8")
    ids_path.write_text("000001\n", encoding="utf-8")
    sample_path.write_text(
        "Id,Label\n000001_0,0\n000001_2,0\n", encoding="utf-8"
    )
    records = load_raw_sentences(input_path, ids_path)
    labels = [[1, 0, 15]]
    vocalized = tmp_path / "pred.txt"
    submission = tmp_path / "submission.csv"
    write_vocalized_predictions(vocalized, records, labels)
    write_submission(submission, records, labels, sample_path)
    assert vocalized.read_text(encoding="utf-8") == "بَ بّْ\n"
    assert read_submission(submission) == [
        ("000001_0", 1),
        ("000001_2", 15),
    ]
