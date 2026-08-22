import json
from pathlib import Path

import pytest

from utils.track4.Lyes.data import SentenceRecord
from utils.track4.Lyes.labels import apply_diacritics
from evaluation.track4.Lyes.paper_metrics import (
    compute_paper_metrics,
    corpus_char_bleu,
    levenshtein_distance,
    load_prediction_jsonl,
    strip_diacritics,
)


def _records() -> list[SentenceRecord]:
    return [
        SentenceRecord(
            sent_id="000001",
            chars=tuple("بت تم"),
            labels=(1, 7, 0, 9, 2),
            input_text="بت تم",
        ),
        SentenceRecord(
            sent_id="000002",
            chars=tuple("ب"),
            labels=(3,),
            input_text="ب",
        ),
    ]


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        (tuple("سارة"), tuple("ساره"), 1),
        (("a", "b", "c"), ("a", "x", "c", "d"), 2),
    ],
)
def test_levenshtein_distance_is_exact(
    reference, hypothesis, expected: int
) -> None:
    assert levenshtein_distance(reference, hypothesis) == expected
    assert levenshtein_distance(hypothesis, reference) == expected


def test_perfect_paper_metrics_are_one_and_skeleton_is_preserved() -> None:
    records = _records()
    predictions = [list(record.labels or ()) for record in records]
    metrics = compute_paper_metrics(records, predictions)
    assert metrics["accuracy"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["word_accuracy"] == 1.0
    assert metrics["sentence_accuracy"] == 1.0
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0
    assert metrics["shadda"]["accuracy"] == 1.0
    assert metrics["tanween"]["accuracy"] == 1.0
    assert metrics["skeleton_mismatch_count"] == 0
    assert metrics["char_bleu"]["score"] == pytest.approx(1.0)
    assert metrics["correct_letters"] == 5
    assert metrics["word_total"] == 3


def test_paper_metrics_golden_errors_and_per_class_f1() -> None:
    records = _records()
    predictions = [
        [7, 7, 0, 1, 1],
        [3],
    ]
    metrics = compute_paper_metrics(records, predictions)
    assert metrics["correct_letters"] == 2
    assert metrics["scored_letters"] == 5
    assert metrics["accuracy"] == pytest.approx(0.4)
    assert metrics["word_correct"] == 1
    assert metrics["word_total"] == 3
    assert metrics["word_accuracy"] == pytest.approx(1.0 / 3.0)
    assert metrics["sentence_correct"] == 1
    assert metrics["sentence_accuracy"] == pytest.approx(0.5)
    assert metrics["wer"] == pytest.approx(2.0 / 3.0)
    assert metrics["shadda"]["false_negative"] == 1
    assert metrics["tanween"]["false_negative"] == 1
    assert metrics["confusion_matrix"][1][7] == 1
    assert metrics["confusion_matrix"][9][1] == 1
    assert metrics["per_class"][3]["f1"] == 1.0
    assert metrics["per_class"][1]["precision"] == 0.0


def test_skeleton_mismatch_uses_explicit_predicted_text() -> None:
    records = _records()
    predictions = [list(record.labels or ()) for record in records]
    texts = [
        apply_diacritics(records[0].chars, predictions[0]),
        "تُ",
    ]
    metrics = compute_paper_metrics(records, predictions, texts)
    assert metrics["skeleton_mismatch_count"] == 1
    assert strip_diacritics(texts[0]) == records[0].input_text


def test_char_bleu_is_deterministic_and_bounded() -> None:
    perfect = corpus_char_bleu(["سَلَام"], ["سَلَام"])
    changed = corpus_char_bleu(["سَلَام"], ["سُلَام"])
    assert perfect["score"] == pytest.approx(1.0)
    assert 0.0 < changed["score"] < perfect["score"]
    assert changed == corpus_char_bleu(["سَلَام"], ["سُلَام"])


def test_prediction_jsonl_contract_is_strict(tmp_path: Path) -> None:
    records = _records()
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "sent_id": record.sent_id,
                    "labels": list(record.labels or ()),
                },
                ensure_ascii=False,
            )
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    predictions, texts = load_prediction_jsonl(path, records)
    assert predictions == [list(record.labels or ()) for record in records]
    assert texts is None
