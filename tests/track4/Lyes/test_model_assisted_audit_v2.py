from pathlib import Path

from experiments.track4.Lyes.model_assisted_audit_v2 import (
    _annotation_review_tables,
    _merge_model_rows,
    _review_queue,
    load_audit_config,
)
from utils.track4.Lyes.data import SentenceRecord


ROOT = Path(__file__).resolve().parents[3]


def test_audit_config_is_strict_and_review_only() -> None:
    config = load_audit_config(ROOT / "configs" / "track4" / "Lyes" /  "model_assisted_audit_v2.json")
    assert config["review_top_n"] == 50
    assert [scope["held_out"] for scope in config["scopes"]] == [False, True, True]


def test_model_rows_merge_deterministically() -> None:
    rows = [
        {"scope": "s", "sent_id": "2", "model": "v13", "loss": 3.0},
        {"scope": "s", "sent_id": "1", "model": "v7", "loss": 1.0},
        {"scope": "s", "sent_id": "2", "model": "v7", "loss": 2.0},
        {"scope": "s", "sent_id": "1", "model": "v13", "loss": 1.5},
    ]
    merged = _merge_model_rows(rows, ("scope", "sent_id"))
    assert [row["sent_id"] for row in merged] == ["1", "2"]
    assert merged[0]["v7_loss"] == 1.0
    assert merged[0]["v13_loss"] == 1.5


def test_review_queue_is_union_without_composite_weight() -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "scope": "s",
                "sent_id": str(index),
                "v7_normalized_sentence_nll": float(index),
                "v13_normalized_sentence_nll": float(3 - index),
                "v7_neural_errors": index,
                "v13_neural_errors": 3 - index,
                "neural_disagreements": index % 2,
                "v2_disagreements": (index + 1) % 2,
            }
        )
    queue = _review_queue(rows, 1)
    assert queue
    assert all("review_reasons" in row for row in queue)
    assert all("reason_count" in row for row in queue)


def test_annotation_review_tables_are_deterministic_and_review_only() -> None:
    records = [
        SentenceRecord(str(index), tuple("اب"), (1, 2), "اب")
        for index in range(5)
    ] + [
        SentenceRecord("minority", tuple("اب"), (1, 3), "اب"),
        SentenceRecord("rare", tuple("ت"), (15,), "ت"),
    ] + [
        SentenceRecord("common{}".format(index), tuple("ج"), (0,), "ج")
        for index in range(300)
    ]
    ambiguous, inconsistencies, rare = _annotation_review_tables(records)
    assert len(ambiguous) == 2
    assert len(inconsistencies) == 1
    assert inconsistencies[0]["label_sequence"] == "1 3"
    assert any(row["label"] == 15 for row in rare)
