"""Deterministic, read-only audit of the released Algerian diacritization data.

The audit deliberately separates hard schema/encoding defects from ambiguity.
A repeated undiacritized skeleton with multiple label sequences is reported as
an ambiguity, not automatically relabeled.  The proposed clean-data manifest
only removes exact duplicate copies and excludes irreconcilable full-sentence
conflicts; it never invents or majority-votes a target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from utils.track4.Lyes.labels import LABEL_MARKS, LABEL_NAMES, NUM_LABELS, apply_diacritics


TRAIN_ID_RE = re.compile(r"^[0-9]{6}$")
EVAL_ID_RE = re.compile(r"^[0-9]{6}(?:-[0-9]+)?$")
HARD_ISSUES = frozenset(
    {
        "blank_line",
        "invalid_json",
        "not_json_object",
        "missing_keys",
        "invalid_chars_type",
        "invalid_labels_type",
        "invalid_input_type",
        "invalid_target_type",
        "char_not_single_codepoint",
        "chars_labels_length_mismatch",
        "input_chars_mismatch",
        "label_not_integer",
        "label_out_of_range",
        "nonzero_space_label",
        "target_mismatch",
        "non_nfc_input",
        "duplicate_sent_id",
    }
)


@dataclass(frozen=True)
class AuditRecord:
    line_number: int
    sent_id: str
    chars: Tuple[str, ...]
    labels: Tuple[int, ...]
    input_text: str
    target_text: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue(
    split: str,
    line_number: int,
    sent_id: str,
    code: str,
    detail: str,
) -> Dict[str, Any]:
    return {
        "split": split,
        "line_number": line_number,
        "sent_id": sent_id,
        "issue_code": code,
        "severity": "hard" if code in HARD_ISSUES else "review",
        "detail": detail,
    }


def is_arabic_letter(char: str) -> bool:
    return (
        len(char) == 1
        and unicodedata.category(char).startswith("L")
        and "ARABIC" in unicodedata.name(char, "")
    )


def read_and_audit_jsonl(
    path: Path,
    split: str,
) -> Tuple[List[AuditRecord], List[Dict[str, Any]]]:
    records: List[AuditRecord] = []
    issues: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    id_pattern = TRAIN_ID_RE if split == "train" else EVAL_ID_RE

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                issues.append(_issue(split, line_number, "", "blank_line", "empty JSONL row"))
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                issues.append(
                    _issue(split, line_number, "", "invalid_json", str(error))
                )
                continue
            if not isinstance(raw, dict):
                issues.append(
                    _issue(split, line_number, "", "not_json_object", type(raw).__name__)
                )
                continue

            sent_id = str(raw.get("sent_id", ""))
            required = {"sent_id", "chars", "labels", "input", "target"}
            missing = sorted(required.difference(raw))
            if missing:
                issues.append(
                    _issue(
                        split,
                        line_number,
                        sent_id,
                        "missing_keys",
                        ",".join(missing),
                    )
                )
                continue
            if not id_pattern.fullmatch(sent_id):
                issues.append(
                    _issue(split, line_number, sent_id, "unexpected_sent_id", sent_id)
                )
            if sent_id in seen_ids:
                issues.append(
                    _issue(split, line_number, sent_id, "duplicate_sent_id", sent_id)
                )
            seen_ids.add(sent_id)

            raw_chars = raw["chars"]
            raw_labels = raw["labels"]
            if not isinstance(raw_chars, list):
                issues.append(
                    _issue(split, line_number, sent_id, "invalid_chars_type", type(raw_chars).__name__)
                )
                continue
            if not isinstance(raw_labels, list):
                issues.append(
                    _issue(split, line_number, sent_id, "invalid_labels_type", type(raw_labels).__name__)
                )
                continue
            if not isinstance(raw["input"], str):
                issues.append(
                    _issue(split, line_number, sent_id, "invalid_input_type", type(raw["input"]).__name__)
                )
                continue
            if not isinstance(raw["target"], str):
                issues.append(
                    _issue(split, line_number, sent_id, "invalid_target_type", type(raw["target"]).__name__)
                )
                continue

            chars = tuple(str(char) for char in raw_chars)
            converted_labels: List[int] = []
            labels_valid = True
            for index, label in enumerate(raw_labels):
                if isinstance(label, bool) or not isinstance(label, int):
                    issues.append(
                        _issue(
                            split,
                            line_number,
                            sent_id,
                            "label_not_integer",
                            "position={} value={!r}".format(index, label),
                        )
                    )
                    labels_valid = False
                else:
                    converted_labels.append(label)
            if not labels_valid:
                continue
            labels = tuple(converted_labels)
            input_text = raw["input"]
            target_text = raw["target"]

            if any(len(char) != 1 for char in chars):
                issues.append(
                    _issue(
                        split,
                        line_number,
                        sent_id,
                        "char_not_single_codepoint",
                        "chars contains a non-single-codepoint element",
                    )
                )
            if len(chars) != len(labels):
                issues.append(
                    _issue(
                        split,
                        line_number,
                        sent_id,
                        "chars_labels_length_mismatch",
                        "chars={} labels={}".format(len(chars), len(labels)),
                    )
                )
            if "".join(chars) != input_text:
                issues.append(
                    _issue(
                        split,
                        line_number,
                        sent_id,
                        "input_chars_mismatch",
                        "joined chars do not equal input",
                    )
                )
            invalid_labels = sorted({label for label in labels if not 0 <= label < NUM_LABELS})
            if invalid_labels:
                issues.append(
                    _issue(
                        split,
                        line_number,
                        sent_id,
                        "label_out_of_range",
                        repr(invalid_labels),
                    )
                )
            bad_spaces = [
                index
                for index, (char, label) in enumerate(zip(chars, labels))
                if char == " " and label != 0
            ]
            if bad_spaces:
                issues.append(
                    _issue(
                        split,
                        line_number,
                        sent_id,
                        "nonzero_space_label",
                        repr(bad_spaces[:20]),
                    )
                )

            if unicodedata.normalize("NFC", input_text) != input_text:
                issues.append(_issue(split, line_number, sent_id, "non_nfc_input", "input"))
            if unicodedata.normalize("NFC", target_text) != target_text:
                issues.append(_issue(split, line_number, sent_id, "non_nfc_target", "target"))
            if unicodedata.normalize("NFKC", input_text) != input_text:
                issues.append(_issue(split, line_number, sent_id, "nfkc_changes_input", "input"))
            if input_text.startswith(" ") or input_text.endswith(" "):
                issues.append(_issue(split, line_number, sent_id, "edge_space", repr(input_text)))
            if "  " in input_text:
                issues.append(_issue(split, line_number, sent_id, "repeated_space", repr(input_text)))

            for index, char in enumerate(chars):
                if len(char) != 1:
                    continue
                if char == " ":
                    continue
                if char.isspace():
                    issues.append(
                        _issue(
                            split,
                            line_number,
                            sent_id,
                            "non_ascii_space",
                            "position={} U+{:04X}".format(index, ord(char)),
                        )
                    )
                elif unicodedata.combining(char):
                    issues.append(
                        _issue(
                            split,
                            line_number,
                            sent_id,
                            "combining_mark_in_skeleton",
                            "position={} U+{:04X}".format(index, ord(char)),
                        )
                    )
                elif unicodedata.category(char) == "Cf":
                    issues.append(
                        _issue(
                            split,
                            line_number,
                            sent_id,
                            "format_character_in_skeleton",
                            "position={} U+{:04X}".format(index, ord(char)),
                        )
                    )
                elif not is_arabic_letter(char):
                    issues.append(
                        _issue(
                            split,
                            line_number,
                            sent_id,
                            "non_arabic_letter",
                            "position={} char={!r}".format(index, char),
                        )
                    )

            can_render = (
                len(chars) == len(labels)
                and all(len(char) == 1 for char in chars)
                and all(0 <= label < NUM_LABELS for label in labels)
                and not bad_spaces
            )
            if can_render:
                expected_target = apply_diacritics(chars, labels)
                if expected_target != target_text:
                    issues.append(
                        _issue(
                            split,
                            line_number,
                            sent_id,
                            "target_mismatch",
                            "target does not equal chars plus label marks",
                        )
                    )

            records.append(
                AuditRecord(
                    line_number=line_number,
                    sent_id=sent_id,
                    chars=chars,
                    labels=labels,
                    input_text=input_text,
                    target_text=target_text,
                )
            )
    return records, issues


def shannon_entropy(counts: Iterable[int]) -> float:
    values = [count for count in counts if count > 0]
    total = sum(values)
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in values)


def label_signature(labels: Sequence[int]) -> str:
    return " ".join(str(label) for label in labels)


def _variant_target(chars: Sequence[str], labels: Sequence[int]) -> str:
    try:
        return apply_diacritics(chars, labels)
    except ValueError:
        return ""


def duplicate_tables(records: Sequence[AuditRecord]) -> Dict[str, List[Dict[str, Any]]]:
    semantic: Dict[Tuple[Any, ...], List[AuditRecord]] = defaultdict(list)
    skeletons: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in records:
        semantic[
            (record.chars, record.labels, record.input_text, record.target_text)
        ].append(record)
        skeletons[record.input_text].append(record)

    exact_rows: List[Dict[str, Any]] = []
    for group in semantic.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda record: record.line_number)
        exact_rows.append(
            {
                "skeleton": ordered[0].input_text,
                "target": ordered[0].target_text,
                "occurrences": len(ordered),
                "kept_sent_id": ordered[0].sent_id,
                "duplicate_sent_ids": "|".join(record.sent_id for record in ordered[1:]),
                "line_numbers": "|".join(str(record.line_number) for record in ordered),
            }
        )

    skeleton_rows: List[Dict[str, Any]] = []
    variant_rows: List[Dict[str, Any]] = []
    identical_target_rows: List[Dict[str, Any]] = []
    for skeleton, group in skeletons.items():
        if len(group) < 2:
            continue
        variants: Dict[Tuple[int, ...], List[AuditRecord]] = defaultdict(list)
        for record in group:
            variants[record.labels].append(record)
        counts = [len(variant_group) for variant_group in variants.values()]
        skeleton_rows.append(
            {
                "skeleton": skeleton,
                "occurrences": len(group),
                "distinct_targets": len(variants),
                "conflicting": len(variants) > 1,
                "entropy_bits": round(shannon_entropy(counts), 10),
                "sent_ids": "|".join(
                    record.sent_id for record in sorted(group, key=lambda item: item.line_number)
                ),
            }
        )
        for labels, variant_group in variants.items():
            ordered = sorted(variant_group, key=lambda record: record.line_number)
            row = {
                "skeleton": skeleton,
                "target": _variant_target(ordered[0].chars, labels),
                "label_sequence": label_signature(labels),
                "variant_occurrences": len(ordered),
                "group_occurrences": len(group),
                "variant_probability": round(len(ordered) / len(group), 10),
                "group_entropy_bits": round(shannon_entropy(counts), 10),
                "sent_ids": "|".join(record.sent_id for record in ordered),
            }
            if len(variants) > 1:
                variant_rows.append(row)
            if len(ordered) > 1:
                identical_target_rows.append(row)

    ordering = lambda row: (-int(row.get("occurrences", row.get("group_occurrences", 0))), str(row["skeleton"]), str(row.get("target", "")))
    return {
        "exact_duplicate_records": sorted(exact_rows, key=ordering),
        "duplicate_skeletons": sorted(skeleton_rows, key=ordering),
        "conflicting_skeleton_variants": sorted(variant_rows, key=ordering),
        "skeleton_target_duplicates": sorted(identical_target_rows, key=ordering),
    }


def word_ambiguity_table(records: Sequence[AuditRecord]) -> List[Dict[str, Any]]:
    variants: Dict[str, Counter[Tuple[int, ...]]] = defaultdict(Counter)
    for record in records:
        current_chars: List[str] = []
        current_labels: List[int] = []
        for char, label in zip(record.chars, record.labels):
            if char == " ":
                if current_chars:
                    variants["".join(current_chars)][tuple(current_labels)] += 1
                current_chars = []
                current_labels = []
            else:
                current_chars.append(char)
                current_labels.append(label)
        if current_chars:
            variants["".join(current_chars)][tuple(current_labels)] += 1

    rows: List[Dict[str, Any]] = []
    for word, counter in variants.items():
        if len(counter) < 2:
            continue
        total = sum(counter.values())
        for labels, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "word_skeleton": word,
                    "vocalized_word": _variant_target(tuple(word), labels),
                    "label_sequence": label_signature(labels),
                    "variant_occurrences": count,
                    "word_occurrences": total,
                    "distinct_targets": len(counter),
                    "variant_probability": round(count / total, 10),
                    "entropy_bits": round(shannon_entropy(counter.values()), 10),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["word_occurrences"]),
            str(row["word_skeleton"]),
            -int(row["variant_occurrences"]),
            str(row["label_sequence"]),
        ),
    )


def label_counts(records: Sequence[AuditRecord], split: str, rare_threshold: int) -> List[Dict[str, Any]]:
    counter: Counter[int] = Counter()
    for record in records:
        counter.update(
            label for char, label in zip(record.chars, record.labels) if char != " "
        )
    total = sum(counter.values())
    return [
        {
            "split": split,
            "label": label,
            "name": LABEL_NAMES[label],
            "mark": LABEL_MARKS[label],
            "count": counter[label],
            "frequency": round(counter[label] / total, 12) if total else 0.0,
            "rare": 0 < counter[label] <= rare_threshold,
            "absent": counter[label] == 0,
        }
        for label in range(NUM_LABELS)
    ]


def transition_counts(
    records: Sequence[AuditRecord],
    split: str,
    rare_threshold: int,
) -> List[Dict[str, Any]]:
    counters: Dict[str, Counter[Tuple[int, int]]] = {
        "all_scored": Counter(),
        "within_word": Counter(),
        "cross_word": Counter(),
    }
    for record in records:
        scored = [
            (index, label)
            for index, (char, label) in enumerate(zip(record.chars, record.labels))
            if char != " "
        ]
        for (left_index, left_label), (right_index, right_label) in zip(scored, scored[1:]):
            counters["all_scored"][(left_label, right_label)] += 1
            between = record.chars[left_index + 1 : right_index]
            boundary = "cross_word" if " " in between else "within_word"
            counters[boundary][(left_label, right_label)] += 1

    rows: List[Dict[str, Any]] = []
    for boundary in ("all_scored", "within_word", "cross_word"):
        counter = counters[boundary]
        total = sum(counter.values())
        for left in range(NUM_LABELS):
            for right in range(NUM_LABELS):
                count = counter[(left, right)]
                rows.append(
                    {
                        "split": split,
                        "boundary": boundary,
                        "from_label": left,
                        "from_name": LABEL_NAMES[left],
                        "to_label": right,
                        "to_name": LABEL_NAMES[right],
                        "count": count,
                        "frequency": round(count / total, 12) if total else 0.0,
                        "rare": 0 < count <= rare_threshold,
                        "absent": count == 0,
                    }
                )
    return rows


def split_overlap_table(
    train_records: Sequence[AuditRecord],
    dev_records: Sequence[AuditRecord],
) -> List[Dict[str, Any]]:
    train: Dict[str, List[AuditRecord]] = defaultdict(list)
    dev: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in train_records:
        train[record.input_text].append(record)
    for record in dev_records:
        dev[record.input_text].append(record)
    rows: List[Dict[str, Any]] = []
    for skeleton in sorted(set(train).intersection(dev)):
        train_labels = {record.labels for record in train[skeleton]}
        dev_labels = {record.labels for record in dev[skeleton]}
        rows.append(
            {
                "skeleton": skeleton,
                "train_occurrences": len(train[skeleton]),
                "dev_occurrences": len(dev[skeleton]),
                "shared_exact_target": bool(train_labels.intersection(dev_labels)),
                "train_distinct_targets": len(train_labels),
                "dev_distinct_targets": len(dev_labels),
                "train_sent_ids": "|".join(record.sent_id for record in train[skeleton]),
                "dev_sent_ids": "|".join(record.sent_id for record in dev[skeleton]),
            }
        )
    return rows


def clean_experiment_manifest(
    records: Sequence[AuditRecord],
    issues: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    reasons: Dict[int, set[str]] = defaultdict(set)
    by_line = {record.line_number: record for record in records}
    for issue in issues:
        if issue["severity"] == "hard" and issue["split"] == "train":
            reasons[int(issue["line_number"])].add("hard_issue:" + str(issue["issue_code"]))

    semantic: Dict[Tuple[Any, ...], List[AuditRecord]] = defaultdict(list)
    skeletons: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in records:
        semantic[(record.chars, record.labels, record.input_text, record.target_text)].append(record)
        skeletons[record.input_text].append(record)
    for group in semantic.values():
        ordered = sorted(group, key=lambda record: record.line_number)
        for duplicate in ordered[1:]:
            reasons[duplicate.line_number].add("exact_duplicate_after_first")
    for group in skeletons.values():
        if len({record.labels for record in group}) > 1:
            for record in group:
                reasons[record.line_number].add("conflicting_full_skeleton")

    rows: List[Dict[str, Any]] = []
    for line_number in sorted(by_line):
        record = by_line[line_number]
        record_reasons = sorted(reasons.get(line_number, set()))
        rows.append(
            {
                "line_number": line_number,
                "sent_id": record.sent_id,
                "decision": "exclude" if record_reasons else "keep",
                "reasons": "|".join(record_reasons),
                "skeleton": record.input_text,
                "target": record.target_text,
            }
        )
    return rows


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    resolved_fieldnames = list(fieldnames) if fieldnames else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _count_issue_codes(issues: Sequence[Mapping[str, Any]], split: str) -> Dict[str, int]:
    counter = Counter(
        str(issue["issue_code"]) for issue in issues if issue["split"] == split
    )
    return dict(sorted(counter.items()))


def _split_summary(
    records: Sequence[AuditRecord],
    issues: Sequence[Mapping[str, Any]],
    duplicates: Mapping[str, Sequence[Mapping[str, Any]]],
    label_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    rare_label_threshold: int,
    rare_transition_threshold: int,
) -> Dict[str, Any]:
    letters = sum(sum(char != " " for char in record.chars) for record in records)
    spaces = sum(sum(char == " " for char in record.chars) for record in records)
    conflict_skeletons = {
        row["skeleton"] for row in duplicates["conflicting_skeleton_variants"]
    }
    conflict_records = sum(
        int(row["occurrences"])
        for row in duplicates["duplicate_skeletons"]
        if row["conflicting"]
    )
    duplicate_copies = sum(
        int(row["occurrences"]) - 1 for row in duplicates["exact_duplicate_records"]
    )
    return {
        "sentences": len(records),
        "scored_letters": letters,
        "spaces": spaces,
        "hard_issue_count": sum(
            issue["severity"] == "hard" for issue in issues
        ),
        "review_issue_count": sum(
            issue["severity"] == "review" for issue in issues
        ),
        "issue_counts": _count_issue_codes(issues, str(issues[0]["split"]) if issues else ""),
        "exact_duplicate_groups": len(duplicates["exact_duplicate_records"]),
        "exact_duplicate_copies": duplicate_copies,
        "duplicate_skeleton_groups": len(duplicates["duplicate_skeletons"]),
        "conflicting_skeleton_groups": len(conflict_skeletons),
        "records_in_conflicting_skeleton_groups": conflict_records,
        "rare_label_threshold": rare_label_threshold,
        "rare_labels": [
            int(row["label"]) for row in label_rows if row["rare"]
        ],
        "absent_labels": [
            int(row["label"]) for row in label_rows if row["absent"]
        ],
        "rare_transition_threshold": rare_transition_threshold,
        "rare_observed_transition_types": sum(
            row["boundary"] == "all_scored" and bool(row["rare"])
            for row in transition_rows
        ),
    }


def render_report(summary: Mapping[str, Any]) -> str:
    train = summary["splits"]["train"]
    dev = summary["splits"]["dev"]
    clean = summary["clean_experiment"]
    overlap = summary["dev_comparison"]["full_skeleton_overlap"]
    return """# Algerian Dialect Vocalization — Data Quality Audit v1

## Scope and safeguards

This is a deterministic, read-only audit. All proposed exclusions are derived
from the training split alone. The dev split is reported only as an untouched
comparison and is never used to select a correction, target, or majority label.

No source JSONL was modified. Source and artifact SHA-256 hashes are recorded in
`manifest.json`. No per-sentence training-loss artifact exists in the current
outputs, so this audit does **not** claim a high-loss-sample analysis.

## Main findings

| Measure | Train | Dev (comparison only) |
|---|---:|---:|
| Sentences | {train_sentences:,} | {dev_sentences:,} |
| Scored letters | {train_letters:,} | {dev_letters:,} |
| Exact duplicate groups | {train_exact_groups:,} | {dev_exact_groups:,} |
| Redundant exact copies | {train_exact_copies:,} | {dev_exact_copies:,} |
| Repeated skeleton groups | {train_skeleton_groups:,} | {dev_skeleton_groups:,} |
| Conflicting full-skeleton groups | {train_conflicts:,} | {dev_conflicts:,} |
| Records in conflicting groups | {train_conflict_records:,} | {dev_conflict_records:,} |
| Hard contract issues | {train_hard:,} | {dev_hard:,} |
| Review-only Unicode/format findings | {train_review:,} | {dev_review:,} |

Train/dev full-sentence skeleton overlap: **{overlap_groups} groups** involving
**{overlap_dev_records} dev records**. This is disclosed as a split-comparison
statistic only; it does not drive the proposed train filtering.

Rare train labels (observed count <= {rare_label_threshold}): `{rare_labels}`.
Absent train labels: `{absent_labels}`. Rare labels and transitions are retained;
rarity alone is not evidence of annotation error.

## Interpretation

- `train_exact_duplicate_records.csv` identifies redundant copies with identical
  skeleton, labels, input, and target after ignoring the permanent sentence ID.
- `train_conflicting_skeleton_variants.csv` contains identical full undiacritized
  sentences paired with multiple label sequences. Because the complete model
  input is identical, these examples are irreconcilable supervision for a
  deterministic tagger. The audit excludes the entire group rather than using
  an arbitrary majority target.
- `word_vocalization_variants.csv` is descriptive ambiguity evidence, not a
  correction list. A word can legitimately change vocalization with syntax and
  context.
- `annotation_issues.csv` distinguishes hard contract violations from review-only
  findings. The source loader's alignment checks are independently reproduced.
  The widespread `non_nfc_target` review flag is explained by the dataset's
  deliberate Shadda-before-vowel serialization. Recomputed targets match the
  released labels exactly, so this is not treated as corruption or filtered.
- `label_transitions.csv` separates within-word and cross-word transitions, so a
  rare boundary event is not conflated with a rare internal word pattern.

## Conservative clean-data experiment

`clean_experiment_manifest.csv` defines a deterministic proposal:

1. Keep the first source-order occurrence of an exact semantic duplicate and
   exclude later copies.
2. Exclude every member of a conflicting full-sentence skeleton group; do not
   majority-vote or edit its target.
3. Exclude records with hard schema, alignment, label, target-rendering, or NFC
   violations.
4. Retain ambiguous words, rare labels, rare transitions, and all review-only
   cases. No labels are rewritten.

Proposed retained training records: **{kept:,}/{total:,}**. Proposed exclusions:
**{excluded:,}** (`{reason_counts}`).

### Controlled evaluation gate

- Train the unchanged successful v7 or BoundaryCRF configuration with the same
  seed and hyperparameters on the proposed subset; compare against an otherwise
  identical original-data control.
- Keep the cleaning policy only if untouched-dev correct letters improve by at
  least 10, overall Micro-F1 improves, and OOV, Shadda, and Tanween accuracy do
  not regress by more than 0.2 percentage points each.
- If seed 42 passes, later confirm across seeds 43/44 before making a paper-level
  generalization claim. Do not inspect Kaggle scores to revise the filter rules.

## Reproduction

```bash
python -m experiments.track4.Lyes.data_quality_audit \\
  --train Data/train_data/train_Algerian-DIAC.jsonl \\
  --dev Data/dev_data/dev_Algerian-DIAC.jsonl \\
  --output-dir audits/data_quality_v1_reproduction
```
""".format(
        train_sentences=train["sentences"],
        dev_sentences=dev["sentences"],
        train_letters=train["scored_letters"],
        dev_letters=dev["scored_letters"],
        train_exact_groups=train["exact_duplicate_groups"],
        dev_exact_groups=dev["exact_duplicate_groups"],
        train_exact_copies=train["exact_duplicate_copies"],
        dev_exact_copies=dev["exact_duplicate_copies"],
        train_skeleton_groups=train["duplicate_skeleton_groups"],
        dev_skeleton_groups=dev["duplicate_skeleton_groups"],
        train_conflicts=train["conflicting_skeleton_groups"],
        dev_conflicts=dev["conflicting_skeleton_groups"],
        train_conflict_records=train["records_in_conflicting_skeleton_groups"],
        dev_conflict_records=dev["records_in_conflicting_skeleton_groups"],
        train_hard=train["hard_issue_count"],
        dev_hard=dev["hard_issue_count"],
        train_review=train["review_issue_count"],
        dev_review=dev["review_issue_count"],
        overlap_groups=overlap["groups"],
        overlap_dev_records=overlap["dev_records"],
        rare_label_threshold=train["rare_label_threshold"],
        rare_labels=",".join(str(value) for value in train["rare_labels"]) or "none",
        absent_labels=",".join(str(value) for value in train["absent_labels"]) or "none",
        kept=clean["kept_records"],
        total=clean["total_records"],
        excluded=clean["excluded_records"],
        reason_counts=", ".join(
            "{}={}".format(key, value)
            for key, value in clean["exclusion_reason_counts"].items()
        ) or "none",
    )


def run_audit(
    train_path: Path,
    dev_path: Path,
    output_dir: Path,
    rare_label_threshold: int = 100,
    rare_transition_threshold: int = 5,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if rare_label_threshold < 0 or rare_transition_threshold < 0:
        raise ValueError("rare thresholds must be non-negative")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            "refusing to overwrite non-empty audit directory: {}".format(output_dir)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records, train_issues = read_and_audit_jsonl(train_path, "train")
    dev_records, dev_issues = read_and_audit_jsonl(dev_path, "dev")
    train_duplicates = duplicate_tables(train_records)
    dev_duplicates = duplicate_tables(dev_records)
    train_labels = label_counts(train_records, "train", rare_label_threshold)
    dev_labels = label_counts(dev_records, "dev", rare_label_threshold)
    train_transitions = transition_counts(
        train_records, "train", rare_transition_threshold
    )
    dev_transitions = transition_counts(dev_records, "dev", rare_transition_threshold)
    word_rows = word_ambiguity_table(train_records)
    overlap_rows = split_overlap_table(train_records, dev_records)
    clean_rows = clean_experiment_manifest(train_records, train_issues)

    files: Dict[str, Sequence[Mapping[str, Any]]] = {
        "annotation_issues.csv": sorted(
            train_issues + dev_issues,
            key=lambda row: (
                str(row["split"]),
                int(row["line_number"]),
                str(row["issue_code"]),
            ),
        ),
        "train_exact_duplicate_records.csv": train_duplicates["exact_duplicate_records"],
        "train_duplicate_skeletons.csv": train_duplicates["duplicate_skeletons"],
        "train_skeleton_target_duplicates.csv": train_duplicates["skeleton_target_duplicates"],
        "train_conflicting_skeleton_variants.csv": train_duplicates["conflicting_skeleton_variants"],
        "word_vocalization_variants.csv": word_rows,
        "label_counts.csv": train_labels + dev_labels,
        "label_transitions.csv": train_transitions + dev_transitions,
        "train_dev_skeleton_overlap.csv": overlap_rows,
        "clean_experiment_manifest.csv": clean_rows,
    }
    empty_schemas = {
        "train_conflicting_skeleton_variants.csv": (
            "skeleton",
            "target",
            "label_sequence",
            "variant_occurrences",
            "group_occurrences",
            "variant_probability",
            "group_entropy_bits",
            "sent_ids",
        )
    }
    for filename, rows in files.items():
        write_csv(output_dir / filename, rows, empty_schemas.get(filename))

    reason_counts: Counter[str] = Counter()
    for row in clean_rows:
        if row["decision"] == "exclude":
            reason_counts.update(str(row["reasons"]).split("|"))
    overlap_dev_ids = {
        sent_id
        for row in overlap_rows
        for sent_id in str(row["dev_sent_ids"]).split("|")
        if sent_id
    }
    summary: Dict[str, Any] = {
        "audit_version": 1,
        "scope": "train-only correction proposal; dev comparison only",
        "splits": {
            "train": _split_summary(
                train_records,
                train_issues,
                train_duplicates,
                train_labels,
                train_transitions,
                rare_label_threshold,
                rare_transition_threshold,
            ),
            "dev": _split_summary(
                dev_records,
                dev_issues,
                dev_duplicates,
                dev_labels,
                dev_transitions,
                rare_label_threshold,
                rare_transition_threshold,
            ),
        },
        "train_word_ambiguity": {
            "ambiguous_word_types": len({row["word_skeleton"] for row in word_rows}),
            "variant_rows": len(word_rows),
        },
        "dev_comparison": {
            "full_skeleton_overlap": {
                "groups": len(overlap_rows),
                "dev_records": len(overlap_dev_ids),
                "groups_with_shared_exact_target": sum(
                    bool(row["shared_exact_target"]) for row in overlap_rows
                ),
            }
        },
        "high_loss_analysis": {
            "available": False,
            "reason": "No train per-sentence loss artifact was found; epoch metrics contain only aggregate train loss.",
        },
        "clean_experiment": {
            "total_records": len(clean_rows),
            "kept_records": sum(row["decision"] == "keep" for row in clean_rows),
            "excluded_records": sum(row["decision"] == "exclude" for row in clean_rows),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "labels_rewritten": 0,
            "dev_used_for_correction": False,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text(render_report(summary), encoding="utf-8")

    artifact_paths = sorted(
        path for path in output_dir.iterdir() if path.name != "manifest.json"
    )
    manifest = {
        "audit_version": 1,
        "inputs": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "dev": {"path": str(dev_path), "sha256": sha256_file(dev_path)},
        },
        "parameters": {
            "rare_label_threshold": rare_label_threshold,
            "rare_transition_threshold": rare_transition_threshold,
        },
        "artifacts": {
            path.name: sha256_file(path) for path in artifact_paths
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("Data/train_data/train_Algerian-DIAC.jsonl"),
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=Path("Data/dev_data/dev_Algerian-DIAC.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rare-label-threshold", type=int, default=100)
    parser.add_argument("--rare-transition-threshold", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_audit(
        train_path=args.train,
        dev_path=args.dev,
        output_dir=args.output_dir,
        rare_label_threshold=args.rare_label_threshold,
        rare_transition_threshold=args.rare_transition_threshold,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
