import csv
import json
from pathlib import Path

import pytest

from experiments.track4.Lyes.data_quality_audit import run_audit, shannon_entropy


def _row(sent_id: str, chars: list[str], labels: list[int]) -> dict:
    marks = (
        "",
        "\u064e",
        "\u064b",
        "\u064f",
        "\u064c",
        "\u0650",
        "\u064d",
        "\u0652",
        "\u0651",
        "\u0651\u064e",
        "\u0651\u064b",
        "\u0651\u064f",
        "\u0651\u064c",
        "\u0651\u0650",
        "\u0651\u064d",
        "\u0651\u0652",
    )
    return {
        "sent_id": sent_id,
        "chars": chars,
        "labels": labels,
        "input": "".join(chars),
        "target": "".join(
            char if char == " " else char + marks[label]
            for char, label in zip(chars, labels)
        ),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_detects_duplicates_conflicts_and_never_rewrites_labels(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    first = _row("000001", ["ب", " ", "ت"], [1, 0, 7])
    duplicate = dict(first, sent_id="000002")
    conflict = _row("000003", ["ب", " ", "ت"], [3, 0, 7])
    unique = _row("000004", ["ك", "ت", "ب"], [1, 1, 1])
    _write_jsonl(train, [first, duplicate, conflict, unique])
    _write_jsonl(dev, [_row("000010", ["ب", " ", "ت"], [1, 0, 7])])

    output = tmp_path / "audit"
    summary = run_audit(train, dev, output)

    assert summary["splits"]["train"]["exact_duplicate_groups"] == 1
    assert summary["splits"]["train"]["exact_duplicate_copies"] == 1
    assert summary["splits"]["train"]["conflicting_skeleton_groups"] == 1
    assert summary["clean_experiment"]["labels_rewritten"] == 0
    assert summary["clean_experiment"]["kept_records"] == 1
    assert summary["clean_experiment"]["excluded_records"] == 3
    with (output / "clean_experiment_manifest.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["decision"] for row in rows] == ["exclude", "exclude", "exclude", "keep"]
    assert summary["dev_comparison"]["full_skeleton_overlap"]["groups"] == 1
    assert summary["clean_experiment"]["dev_used_for_correction"] is False


def test_audit_reports_rare_transitions_and_refuses_overwrite(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    _write_jsonl(train, [_row("000001", ["ب", "ت"], [1, 7])])
    _write_jsonl(dev, [_row("000002", ["ب", "ت"], [1, 7])])
    output = tmp_path / "audit"

    summary = run_audit(
        train,
        dev,
        output,
        rare_label_threshold=1,
        rare_transition_threshold=1,
    )
    assert summary["splits"]["train"]["rare_labels"] == [1, 7]
    assert summary["splits"]["train"]["rare_observed_transition_types"] == 1
    with pytest.raises(FileExistsError):
        run_audit(train, dev, output)


def test_entropy_is_deterministic() -> None:
    assert shannon_entropy([2, 2]) == 1.0
    assert shannon_entropy([4]) == 0.0


def test_shadda_before_vowel_non_nfc_target_is_review_only(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    row = _row("000001", ["ب"], [9])
    _write_jsonl(train, [row])
    _write_jsonl(dev, [_row("000002", ["ت"], [1])])

    summary = run_audit(train, dev, tmp_path / "audit")

    assert summary["splits"]["train"]["hard_issue_count"] == 0
    assert summary["splits"]["train"]["review_issue_count"] == 1
    assert summary["splits"]["train"]["issue_counts"] == {"non_nfc_target": 1}
    assert summary["clean_experiment"]["kept_records"] == 1
