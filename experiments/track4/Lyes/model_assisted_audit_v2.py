"""Deterministic model-assisted, review-only annotation audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import torch
from torch.utils.data import DataLoader

from utils.track4.Lyes.gated_fusion.config import load_gates
from utils.track4.Lyes.gated_fusion.fusion import apply_gated_fallback
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import (
    BatchCollator,
    CharacterDataset,
    SentenceRecord,
    load_jsonl,
    validate_vocabulary_coverage,
)
from utils.track4.Lyes.lexical_fusion import WordLabelPrior, iter_words
from utils.track4.Lyes.labels import LABEL_NAMES
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


EXPECTED_CONFIG_KEYS = {
    "schema_version",
    "output_dir",
    "v2_gates",
    "scopes",
    "review_top_n",
}
EXPECTED_SCOPE_KEYS = {
    "name",
    "train",
    "evaluation",
    "v7_checkpoint",
    "v13_checkpoint",
    "held_out",
}


def load_audit_config(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != EXPECTED_CONFIG_KEYS:
        raise ValueError("invalid model-assisted audit config keys")
    if int(raw["schema_version"]) != 1:
        raise ValueError("unsupported model-assisted audit schema")
    scopes = raw["scopes"]
    if not isinstance(scopes, list) or len(scopes) != 3:
        raise ValueError("audit requires full-train and two held-out scopes")
    names = []
    for scope in scopes:
        if not isinstance(scope, dict) or set(scope) != EXPECTED_SCOPE_KEYS:
            raise ValueError("invalid audit scope")
        names.append(str(scope["name"]))
    if len(set(names)) != len(names):
        raise ValueError("audit scope names must be unique")
    if int(raw["review_top_n"]) != 50:
        raise ValueError("audit review_top_n is locked to 50")
    return raw


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _ambiguity_map(
    records: Sequence[SentenceRecord],
) -> Mapping[Tuple[str, int], Set[int]]:
    variants: Dict[Tuple[str, int], Set[int]] = defaultdict(set)
    for record in records:
        if record.labels is None:
            raise ValueError("ambiguity evidence requires labels")
        for start, end, word in iter_words(record.chars):
            for offset, label in enumerate(record.labels[start:end]):
                variants[(word, offset)].add(int(label))
    return variants


@torch.inference_mode()
def _score_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    scope_name: str,
    train_records: Sequence[SentenceRecord],
    evaluation_records: Sequence[SentenceRecord],
    gates_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Mapping[str, List[List[int]]]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    if model.config.resolved_head_mode != "crf" or model.crf is None:
        raise ValueError("model-assisted audit requires standard CRF checkpoints")
    validate_vocabulary_coverage(evaluation_records, vocab)
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(gates_path)
    ambiguity = _ambiguity_map(train_records)
    loader = DataLoader(
        CharacterDataset(evaluation_records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchCollator(vocab),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    model.eval()
    sentence_rows: List[Dict[str, Any]] = []
    word_rows: List[Dict[str, Any]] = []
    neural_predictions: List[List[int]] = []
    v2_predictions: List[List[int]] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        outputs = model(input_ids, attention_mask)
        mask = outputs["crf_mask"].bool()
        sentence_nll = model.crf.log_partition(
            outputs["logits"], mask
        ) - model.crf.gold_score(outputs["logits"], targets, mask)
        log_marginals = model.log_probabilities(outputs).cpu()
        decoded = model.decode_outputs(outputs).cpu()
        for row, record in enumerate(batch["records"]):
            if record.labels is None:
                raise ValueError("audit evaluation records require labels")
            record_slice = slice(1, len(record.chars) + 1)
            record_log_marginals = log_marginals[row, record_slice]
            neural = decoded[row, record_slice].tolist()
            neural = [
                0 if char == " " else int(label)
                for char, label in zip(record.chars, neural)
            ]
            v2, _ = apply_gated_fallback(
                record,
                record_log_marginals,
                prior,
                gates,
                initial_predictions=torch.tensor(neural),
            )
            neural_predictions.append(neural)
            v2_predictions.append(v2)
            scored_letters = sum(char != " " for char in record.chars)
            neural_errors = sum(
                char != " " and int(prediction) != int(gold)
                for char, prediction, gold in zip(
                    record.chars, neural, record.labels
                )
            )
            v2_errors = sum(
                char != " " and int(prediction) != int(gold)
                for char, prediction, gold in zip(
                    record.chars, v2, record.labels
                )
            )
            nll = float(sentence_nll[row].cpu().item())
            sentence_rows.append(
                {
                    "scope": scope_name,
                    "model": model_name,
                    "sent_id": record.sent_id,
                    "input": record.input_text,
                    "scored_letters": scored_letters,
                    "sentence_nll": nll,
                    "normalized_sentence_nll": nll / scored_letters,
                    "neural_errors": neural_errors,
                    "v2_errors": v2_errors,
                    "neural_sentence_exact": int(neural_errors == 0),
                    "v2_sentence_exact": int(v2_errors == 0),
                }
            )
            for start, end, word in iter_words(record.chars):
                gold = torch.tensor(record.labels[start:end], dtype=torch.long)
                word_log_probs = record_log_marginals[start:end]
                offsets = torch.arange(end - start)
                losses = -word_log_probs[offsets, gold]
                neural_word = neural[start:end]
                v2_word = v2[start:end]
                gold_word = record.labels[start:end]
                neural_word_errors = sum(
                    int(prediction) != int(target)
                    for prediction, target in zip(neural_word, gold_word)
                )
                v2_word_errors = sum(
                    int(prediction) != int(target)
                    for prediction, target in zip(v2_word, gold_word)
                )
                ambiguous_positions = sum(
                    len(ambiguity[(word, offset)]) > 1
                    for offset in range(end - start)
                )
                word_rows.append(
                    {
                        "scope": scope_name,
                        "model": model_name,
                        "sent_id": record.sent_id,
                        "start": start,
                        "end": end,
                        "word": word,
                        "length": end - start,
                        "gold_labels": " ".join(map(str, gold_word)),
                        "gold_marginal_nll": float(losses.sum().item()),
                        "mean_gold_marginal_nll": float(losses.mean().item()),
                        "neural_errors": neural_word_errors,
                        "v2_errors": v2_word_errors,
                        "neural_exact": int(neural_word_errors == 0),
                        "v2_exact": int(v2_word_errors == 0),
                        "ambiguous_positions": ambiguous_positions,
                    }
                )
    return sentence_rows, word_rows, {
        "neural": neural_predictions,
        "v2": v2_predictions,
    }


def _merge_model_rows(
    rows: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped[key][str(row["model"])] = row
    merged: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        models = grouped[key]
        if set(models) != {"v7", "v13"}:
            raise RuntimeError("audit model rows are incomplete")
        result = {field: value for field, value in zip(key_fields, key)}
        for model_name in ("v7", "v13"):
            for field, value in models[model_name].items():
                if field not in set(key_fields) | {"model"}:
                    result["{}_{}".format(model_name, field)] = value
        merged.append(result)
    return merged


def _add_disagreements(
    sentence_rows: List[Dict[str, Any]],
    prediction_pairs: Mapping[Tuple[str, str], Mapping[str, List[List[int]]]],
    scopes: Mapping[str, Sequence[SentenceRecord]],
) -> None:
    by_key = {(row["scope"], row["sent_id"]): row for row in sentence_rows}
    for scope_name, records in scopes.items():
        v7 = prediction_pairs[(scope_name, "v7")]
        v13 = prediction_pairs[(scope_name, "v13")]
        for index, record in enumerate(records):
            row = by_key[(scope_name, record.sent_id)]
            row["neural_disagreements"] = sum(
                char != " " and first != second
                for char, first, second in zip(
                    record.chars, v7["neural"][index], v13["neural"][index]
                )
            )
            row["v2_disagreements"] = sum(
                char != " " and first != second
                for char, first, second in zip(
                    record.chars, v7["v2"][index], v13["v2"][index]
                )
            )


def _review_queue(
    rows: Sequence[Mapping[str, Any]], top_n: int
) -> List[Dict[str, Any]]:
    metrics = (
        "v7_normalized_sentence_nll",
        "v13_normalized_sentence_nll",
        "v7_neural_errors",
        "v13_neural_errors",
        "neural_disagreements",
        "v2_disagreements",
    )
    reasons: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    source: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for metric in metrics:
        ranked = sorted(
            rows,
            key=lambda row: (
                -float(row[metric]), str(row["scope"]), str(row["sent_id"])
            ),
        )[:top_n]
        for row in ranked:
            key = (str(row["scope"]), str(row["sent_id"]))
            reasons[key].add(metric)
            source[key] = row
    queue = []
    for key in sorted(reasons, key=lambda item: (-len(reasons[item]), item)):
        row = dict(source[key])
        row["review_reasons"] = ";".join(sorted(reasons[key]))
        row["reason_count"] = len(reasons[key])
        queue.append(row)
    return queue


def _annotation_review_tables(
    records: Sequence[SentenceRecord],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build deterministic ambiguity, minority-variant, and rare-label tables."""

    variants: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[Tuple[str, Tuple[int, ...]], List[str]] = defaultdict(list)
    label_counts: Counter = Counter()
    for record in records:
        if record.labels is None:
            raise ValueError("annotation review requires labels")
        for char, label in zip(record.chars, record.labels):
            if char != " ":
                label_counts[int(label)] += 1
        for start, end, word in iter_words(record.chars):
            sequence = tuple(int(value) for value in record.labels[start:end])
            variants[word][sequence] += 1
            key = (word, sequence)
            if len(examples[key]) < 5:
                examples[key].append(record.sent_id)
    ambiguous: List[Dict[str, Any]] = []
    likely_inconsistencies: List[Dict[str, Any]] = []
    for word in sorted(variants):
        counts = variants[word]
        if len(counts) <= 1:
            continue
        total = sum(counts.values())
        dominant = max(counts.values())
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for sequence, count in ordered:
            suspicious = count <= 2 and dominant >= 5
            row = {
                "word": word,
                "variant_count": len(counts),
                "total_occurrences": total,
                "label_sequence": " ".join(map(str, sequence)),
                "occurrences": count,
                "variant_rate": count / total,
                "dominant_occurrences": dominant,
                "example_sent_ids": ";".join(examples[(word, sequence)]),
                "likely_annotation_inconsistency": int(suspicious),
            }
            ambiguous.append(row)
            if suspicious:
                likely_inconsistencies.append(dict(row))
    total_letters = sum(label_counts.values())
    rare_ids = {
        label
        for label, count in label_counts.items()
        if count / total_letters < 0.005
    }
    rare_occurrences: List[Dict[str, Any]] = []
    for record in records:
        if record.labels is None:
            continue
        for index, (char, label) in enumerate(zip(record.chars, record.labels)):
            label = int(label)
            if char == " " or label not in rare_ids:
                continue
            rare_occurrences.append(
                {
                    "sent_id": record.sent_id,
                    "char_index": index,
                    "char": char,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "class_occurrences": label_counts[label],
                    "class_frequency": label_counts[label] / total_letters,
                    "input": record.input_text,
                }
            )
    return ambiguous, likely_inconsistencies, rare_occurrences


def _augment_existing_audit(
    config: Mapping[str, Any],
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    required = {
        "ambiguous_word_skeletons": output_dir / "AMBIGUOUS_WORD_SKELETONS.csv",
        "likely_annotation_inconsistencies": output_dir / "LIKELY_ANNOTATION_INCONSISTENCIES.csv",
        "rare_class_occurrences": output_dir / "RARE_CLASS_OCCURRENCES.csv",
    }
    if all(path.is_file() for path in required.values()):
        return manifest
    full_scope = next(
        scope for scope in config["scopes"]
        if str(scope["name"]) == "full_train_in_sample"
    )
    records = load_jsonl(Path(full_scope["evaluation"]))
    ambiguous, inconsistencies, rare = _annotation_review_tables(records)
    _write_csv(required["ambiguous_word_skeletons"], ambiguous)
    _write_csv(required["likely_annotation_inconsistencies"], inconsistencies)
    _write_csv(required["rare_class_occurrences"], rare)
    summary_path = output_dir / "SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "ambiguous_variant_rows": len(ambiguous),
            "likely_annotation_inconsistency_rows": len(inconsistencies),
            "rare_class_occurrence_rows": len(rare),
            "rare_class_policy": "released-train frequency below 0.5 percent",
        }
    )
    write_json(summary_path, summary)
    report_path = output_dir / "REPORT.md"
    report = report_path.read_text(encoding="utf-8").rstrip() + "\n\n"
    report += "## Annotation review tables\n\n"
    report += "- Ambiguous skeleton/variant rows: `{}`.\n".format(len(ambiguous))
    report += "- Minority variants flagged for human consistency review: `{}`.\n".format(len(inconsistencies))
    report += "- Rare-class occurrences below 0.5% released-train frequency: `{}`.\n".format(len(rare))
    report += "- Flags are review aids only and never trigger automatic removal or relabeling.\n"
    report_path.write_text(report, encoding="utf-8")
    artifacts = dict(manifest["artifacts"])
    for name, path in required.items():
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
    artifacts["summary"] = {"path": str(summary_path), "sha256": sha256_file(summary_path)}
    artifacts["report"] = {"path": str(report_path), "sha256": sha256_file(report_path)}
    updated = {**dict(manifest), **summary, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", updated)
    return updated


def run_audit(
    config_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    config = load_audit_config(config_path)
    output_dir = Path(config["output_dir"])
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for artifact in manifest.get("artifacts", {}).values():
            path = Path(artifact["path"])
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise RuntimeError("partial or corrupt completed audit")
        return _augment_existing_audit(config, output_dir, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    gates_path = Path(config["v2_gates"])
    all_sentence_rows: List[Dict[str, Any]] = []
    all_word_rows: List[Dict[str, Any]] = []
    prediction_pairs: Dict[Tuple[str, str], Mapping[str, List[List[int]]]] = {}
    scope_records: Dict[str, Sequence[SentenceRecord]] = {}
    evidence = []
    for scope in config["scopes"]:
        scope_name = str(scope["name"])
        train_path = Path(scope["train"])
        evaluation_path = Path(scope["evaluation"])
        train_records = load_jsonl(train_path)
        evaluation_records = load_jsonl(evaluation_path)
        scope_records[scope_name] = evaluation_records
        if bool(scope["held_out"]):
            if {record.sent_id for record in train_records} & {
                record.sent_id for record in evaluation_records
            }:
                raise RuntimeError("audit held-out scope contains leakage")
        for model_name, field in (("v7", "v7_checkpoint"), ("v13", "v13_checkpoint")):
            checkpoint_path = Path(scope[field])
            sentence_rows, word_rows, predictions = _score_checkpoint(
                checkpoint_path,
                model_name,
                scope_name,
                train_records,
                evaluation_records,
                gates_path,
                device,
                batch_size,
                num_workers,
            )
            all_sentence_rows.extend(sentence_rows)
            all_word_rows.extend(word_rows)
            prediction_pairs[(scope_name, model_name)] = predictions
            evidence.append(
                {
                    "scope": scope_name,
                    "model": model_name,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "train_sha256": sha256_file(train_path),
                    "evaluation_sha256": sha256_file(evaluation_path),
                }
            )
    sentence_rows = _merge_model_rows(
        all_sentence_rows, ("scope", "sent_id")
    )
    _add_disagreements(sentence_rows, prediction_pairs, scope_records)
    word_rows = _merge_model_rows(
        all_word_rows, ("scope", "sent_id", "start", "end", "word")
    )
    review = _review_queue(sentence_rows, int(config["review_top_n"]))
    csv_paths = {
        "sentence_scores": output_dir / "sentence_scores.csv",
        "word_scores": output_dir / "word_scores.csv",
        "review_queue": output_dir / "REVIEW_QUEUE.csv",
    }
    _write_csv(csv_paths["sentence_scores"], sentence_rows)
    _write_csv(csv_paths["word_scores"], word_rows)
    _write_csv(csv_paths["review_queue"], review)
    summary = {
        "schema_version": 1,
        "device": str(device),
        "sentence_rows": len(sentence_rows),
        "word_rows": len(word_rows),
        "review_queue_rows": len(review),
        "held_out_sentence_rows": sum(
            row["scope"].endswith("_heldout") for row in sentence_rows
        ),
        "evidence": evidence,
        "policy": "review_only_no_dataset_mutation",
    }
    write_json(output_dir / "SUMMARY.json", summary)
    report = "\n".join(
        [
            "# Model-Assisted Data Audit v2",
            "",
            "This audit is review-only. It never edits, excludes, relabels, or majority-votes a released record.",
            "",
            "- Device: `{}`".format(device),
            "- Sentence rows: `{}`".format(len(sentence_rows)),
            "- Word rows: `{}`".format(len(word_rows)),
            "- Review queue rows: `{}`".format(len(review)),
            "- Queue rule: union of the top 50 rows for each declared metric; no weighted score.",
            "",
            "Held-out split A/B results are primary evidence. Full-train scores are descriptive because both full checkpoints saw those labels during training.",
        ]
    ) + "\n"
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    artifacts = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in {
            **csv_paths,
            "summary": output_dir / "SUMMARY.json",
            "report": report_path,
        }.items()
    }
    manifest = {**summary, "artifacts": artifacts}
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/track4/Lyes/model_assisted_audit_v2.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid audit loader settings")
    run_audit(
        args.config,
        args.device,
        args.batch_size,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
