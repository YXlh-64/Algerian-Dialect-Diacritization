"""Resumable FilteredWordLattice-v14 campaign and oracle gate."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import DataLoader

from utils.track4.Lyes.gated_fusion.config import load_gates
from utils.track4.Lyes.gated_fusion.fusion import apply_gated_fallback
from experiments.track4.Lyes.campaign.common import write_prediction_artifacts, write_step_manifest
from experiments.track4.Lyes.campaign.diagnostics import prediction_diagnostics, training_word_types
from experiments.track4.Lyes.campaign.ensemble import (
    average_probability_groups,
    predict_probability_groups,
    probabilities_to_predictions,
)
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import (
    BatchCollator,
    CharacterDataset,
    SentenceRecord,
    load_jsonl,
    load_raw_sentences,
    validate_vocabulary_coverage,
)
from utils.track4.Lyes.lexical_fusion import iter_words
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from models.track4.Lyes.dual_stream_crf_head import LinearChainCRF
from evaluation.track4.Lyes.paper_metrics import compute_paper_metrics
from experiments.track4.Lyes.export_ensemble import (
    checkpoint_groups,
    load_v7_config,
)
from utils.track4.Lyes.utils import select_device, sha256_file, write_json
from models.track4.Lyes.word_lattice import WordLattice, build_word_lattice, oracle_predictions
from experiments.track4.Lyes.word_lattice_runtime import (
    evaluate_baseline_bundle,
    evaluate_lattice_scorer,
    load_or_build_cached_bundle,
    load_scorer_checkpoint,
    train_lattice_scorer,
)


EXPECTED_CONFIG_KEYS = {
    "schema_version",
    "output_root",
    "v7_config",
    "v7_full_checkpoint",
    "v7_campaign",
    "v2_gates",
    "released_train",
    "released_dev",
    "splits",
    "candidate_counts",
    "oracle_gate",
    "scorer",
    "training",
    "calibration_gate",
    "final_gate",
}


def load_campaign_config(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != EXPECTED_CONFIG_KEYS:
        raise ValueError("invalid FilteredWordLattice-v14 config keys")
    if int(raw["schema_version"]) != 1:
        raise ValueError("unsupported FilteredWordLattice-v14 schema")
    if set(raw["splits"]) != {"a", "b"}:
        raise ValueError("v14 config requires split a and split b")
    required_split_keys = {"train", "calibration", "checkpoint"}
    for name, split in raw["splits"].items():
        if not isinstance(split, dict) or set(split) != required_split_keys:
            raise ValueError("invalid split {} config".format(name))
    if raw["candidate_counts"] != [4, 8]:
        raise ValueError("v14 candidate counts are locked to [4, 8]")
    oracle_gate = raw["oracle_gate"]
    if set(oracle_gate) != {
        "minimum_recoverable_letters",
        "minimum_recoverable_exact_words",
    }:
        raise ValueError("invalid v14 oracle gate")
    if int(oracle_gate["minimum_recoverable_letters"]) != 20:
        raise ValueError("v14 recoverable-letter gate is locked to 20")
    if int(oracle_gate["minimum_recoverable_exact_words"]) != 10:
        raise ValueError("v14 recoverable-word gate is locked to 10")
    return raw


def _training_word_evidence(
    records: Sequence[SentenceRecord],
) -> Tuple[Set[str], Set[str]]:
    variants: Dict[str, Set[Tuple[int, ...]]] = defaultdict(set)
    for record in records:
        if record.labels is None:
            raise ValueError("training records require labels")
        for start, end, word in iter_words(record.chars):
            variants[word].add(tuple(record.labels[start:end]))
    return set(variants), {
        word for word, labels in variants.items() if len(labels) > 1
    }


@torch.inference_mode()
def generate_lattices(
    checkpoint_path: Path,
    records: Sequence[SentenceRecord],
    k: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> List[WordLattice]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    if model.config.resolved_head_mode != "crf" or model.crf is None:
        raise ValueError("v14 oracle requires a standard CRF checkpoint")
    embedded_data = checkpoint.get("experiment_config", {}).get("data", {})
    if not isinstance(embedded_data, dict):
        raise ValueError("checkpoint experiment data config is missing")
    validate_vocabulary_coverage(records, vocab)
    loader = DataLoader(
        CharacterDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchCollator(vocab),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    cpu_crf = LinearChainCRF(16)
    with torch.no_grad():
        cpu_crf.transitions.copy_(model.crf.transitions.detach().cpu())
    model.eval()
    lattices: List[WordLattice] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        decoded = model.decode_outputs(outputs).cpu()
        emissions = outputs["logits"].detach().cpu()
        for row, record in enumerate(batch["records"]):
            record_slice = slice(1, len(record.chars) + 1)
            baseline = decoded[row, record_slice].tolist()
            baseline = [
                0 if char == " " else int(label)
                for char, label in zip(record.chars, baseline)
            ]
            lattices.append(
                build_word_lattice(
                    record=record,
                    emissions=emissions[row, record_slice],
                    baseline_labels=baseline,
                    crf=cpu_crf,
                    k=k,
                )
            )
    return lattices


def _word_exact_count(
    records: Sequence[SentenceRecord], predictions: Sequence[Sequence[int]]
) -> int:
    correct = 0
    for record, labels in zip(records, predictions):
        if record.labels is None:
            raise ValueError("oracle evaluation requires labels")
        for start, end, _ in iter_words(record.chars):
            correct += tuple(labels[start:end]) == tuple(record.labels[start:end])
    return correct


def _sentence_exact_count(
    records: Sequence[SentenceRecord], predictions: Sequence[Sequence[int]]
) -> int:
    return sum(
        tuple(labels) == tuple(record.labels or ())
        for record, labels in zip(records, predictions)
    )


def _letter_correct_count(
    records: Sequence[SentenceRecord], predictions: Sequence[Sequence[int]]
) -> int:
    return sum(
        int(label) == int(gold)
        for record, labels in zip(records, predictions)
        for char, label, gold in zip(record.chars, labels, record.labels or ())
        if char != " "
    )


def evaluate_oracle(
    train_records: Sequence[SentenceRecord],
    records: Sequence[SentenceRecord],
    lattices: Sequence[WordLattice],
    k: int,
) -> Mapping[str, Any]:
    if len(records) != len(lattices):
        raise ValueError("oracle records and lattices must align")
    seen_words, ambiguous_words = _training_word_evidence(train_records)
    baseline_predictions: List[List[int]] = []
    oracle_outputs: List[List[int]] = []
    coverage = 0
    wrong_word_coverage = 0
    word_total = 0
    baseline_wrong_words = 0
    breakdown: Dict[str, Counter] = defaultdict(Counter)

    for record, lattice in zip(records, lattices):
        if record.sent_id != lattice.sent_id or record.labels is None:
            raise ValueError("oracle lattice alignment failure")
        baseline = list(lattice.baseline_labels)
        oracle, covered, recovered = oracle_predictions(
            lattice, record.labels
        )
        baseline_predictions.append(baseline)
        oracle_outputs.append(oracle)
        coverage += covered
        wrong_word_coverage += recovered
        for (start, end), group, (_, _, word) in zip(
            lattice.spans, lattice.candidates, iter_words(record.chars)
        ):
            gold = tuple(record.labels[start:end])
            baseline_word = tuple(baseline[start:end])
            is_covered = any(candidate.labels == gold for candidate in group)
            is_wrong = baseline_word != gold
            word_total += 1
            baseline_wrong_words += is_wrong
            keys = (
                "length_{}".format(end - start),
                "seen" if word in seen_words else "oov",
                "ambiguous" if word in ambiguous_words else "unambiguous",
            )
            for key in keys:
                breakdown[key]["words"] += 1
                breakdown[key]["gold_covered"] += is_covered
                breakdown[key]["baseline_wrong"] += is_wrong
                breakdown[key]["recoverable_wrong"] += is_wrong and is_covered

    baseline_letters = _letter_correct_count(records, baseline_predictions)
    oracle_letters = _letter_correct_count(records, oracle_outputs)
    baseline_words = _word_exact_count(records, baseline_predictions)
    oracle_words = _word_exact_count(records, oracle_outputs)
    baseline_sentences = _sentence_exact_count(records, baseline_predictions)
    oracle_sentences = _sentence_exact_count(records, oracle_outputs)
    return {
        "k": k,
        "sentences": len(records),
        "words": word_total,
        "baseline_wrong_words": baseline_wrong_words,
        "gold_word_sequence_coverage": coverage,
        "gold_word_sequence_coverage_rate": coverage / word_total,
        "wrong_word_gold_coverage": wrong_word_coverage,
        "wrong_word_gold_coverage_rate": (
            wrong_word_coverage / baseline_wrong_words
            if baseline_wrong_words
            else 0.0
        ),
        "baseline_correct_letters": baseline_letters,
        "oracle_correct_letters": oracle_letters,
        "recoverable_correct_letters": oracle_letters - baseline_letters,
        "baseline_exact_words": baseline_words,
        "oracle_exact_words": oracle_words,
        "recoverable_exact_words": oracle_words - baseline_words,
        "baseline_exact_sentences": baseline_sentences,
        "oracle_exact_sentences": oracle_sentences,
        "recoverable_exact_sentences": oracle_sentences - baseline_sentences,
        "breakdown": {
            key: dict(sorted(values.items()))
            for key, values in sorted(breakdown.items())
        },
    }


def run_oracle(
    config_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    config = load_campaign_config(config_path)
    output_dir = Path(config["output_root"]) / "00_oracle"
    selection_path = output_dir / "ORACLE_SELECTION.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("schema_version") != 1:
            raise RuntimeError("invalid completed oracle selection")
        return selection
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    results: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    evidence: Dict[str, Mapping[str, str]] = {}
    for split_name in ("a", "b"):
        split = config["splits"][split_name]
        train_path = Path(split["train"])
        calibration_path = Path(split["calibration"])
        checkpoint_path = Path(split["checkpoint"])
        for required in (train_path, calibration_path, checkpoint_path):
            if not required.is_file():
                raise FileNotFoundError(str(required))
        checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
        embedded_data = checkpoint.get("experiment_config", {}).get("data", {})
        if embedded_data.get("train") != str(train_path):
            raise RuntimeError("{} checkpoint train split mismatch".format(split_name))
        if embedded_data.get("dev") != str(calibration_path):
            raise RuntimeError("{} checkpoint calibration split mismatch".format(split_name))
        train_records = load_jsonl(train_path)
        calibration_records = load_jsonl(calibration_path)
        train_ids = {record.sent_id for record in train_records}
        calibration_ids = {record.sent_id for record in calibration_records}
        if train_ids & calibration_ids:
            raise RuntimeError("{} train/calibration leakage".format(split_name))
        results[split_name] = {}
        evidence[split_name] = {
            "train_sha256": sha256_file(train_path),
            "calibration_sha256": sha256_file(calibration_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        for k in config["candidate_counts"]:
            lattices = generate_lattices(
                checkpoint_path,
                calibration_records,
                int(k),
                device,
                batch_size,
                num_workers,
            )
            result = evaluate_oracle(
                train_records, calibration_records, lattices, int(k)
            )
            results[split_name][str(k)] = result
            write_json(
                output_dir / "split_{}_k{}.json".format(split_name, k),
                result,
            )

    gates = config["oracle_gate"]
    decisions: Dict[str, Mapping[str, Any]] = {}
    passing: List[int] = []
    for k in config["candidate_counts"]:
        split_passes = {}
        for split_name in ("a", "b"):
            result = results[split_name][str(k)]
            split_passes[split_name] = (
                int(result["recoverable_correct_letters"])
                >= int(gates["minimum_recoverable_letters"])
                and int(result["recoverable_exact_words"])
                >= int(gates["minimum_recoverable_exact_words"])
            )
        accepted = all(split_passes.values())
        decisions[str(k)] = {
            "split_passes": split_passes,
            "accepted": accepted,
        }
        if accepted:
            passing.append(int(k))
    selection = {
        "schema_version": 1,
        "device": str(device),
        "gate": dict(gates),
        "evidence": evidence,
        "results": results,
        "decisions": decisions,
        "passing_candidate_counts": passing,
        "accepted": bool(passing),
        "next_stage": "v14_and_v15" if passing else "stop_word_modeling",
    }
    write_json(selection_path, selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True))
    return selection


PROTECTED_KEYS = (
    "word_accuracy",
    "sentence_accuracy",
    "oov_accuracy",
    "shadda_accuracy",
    "tanween_accuracy",
)


def _protected_regressions(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> Mapping[str, bool]:
    return {
        key: float(candidate[key]) < float(control[key])
        for key in PROTECTED_KEYS
    }


def _calibration_result(
    summary: Mapping[str, Any],
    control: Mapping[str, Any],
) -> Mapping[str, Any]:
    candidate = summary["best_metrics"]
    return {
        "summary": dict(summary),
        "control": dict(control),
        "neural_correct_gain": (
            int(candidate["neural"]["correct"])
            - int(control["neural"]["correct"])
        ),
        "neural_exact_word_gain": (
            int(candidate["neural"]["word_correct"])
            - int(control["neural"]["word_correct"])
        ),
        "v2_correct_gain": (
            int(candidate["v2"]["correct"])
            - int(control["v2"]["correct"])
        ),
        "protected_v2_regressions": _protected_regressions(
            candidate["v2"], control["v2"]
        ),
    }


def run_calibration(
    config_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    config = load_campaign_config(config_path)
    output_root = Path(config["output_root"])
    selection_path = output_root / "02_calibration_b" / "CALIBRATION_SELECTION.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("schema_version") != 1:
            raise RuntimeError("invalid completed v14 calibration")
        return selection
    oracle = run_oracle(config_path, device_name, batch_size, num_workers)
    if not oracle["accepted"]:
        selection = {
            "schema_version": 1,
            "accepted": False,
            "reason": "oracle_gate_failed",
            "oracle_selection": str(
                output_root / "00_oracle" / "ORACLE_SELECTION.json"
            ),
        }
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(selection_path, selection)
        return selection
    device = select_device(device_name)
    gates = load_gates(Path(config["v2_gates"]))
    split_a = config["splits"]["a"]
    train_a_path = Path(split_a["train"])
    dev_a_path = Path(split_a["calibration"])
    checkpoint_a = Path(split_a["checkpoint"])
    train_a = load_jsonl(train_a_path)
    dev_a = load_jsonl(dev_a_path)
    results_a: Dict[str, Mapping[str, Any]] = {}
    for k in config["candidate_counts"]:
        cache_dir = output_root / "cache" / "split_a" / "k{}".format(k)
        train_bundle = load_or_build_cached_bundle(
            cache_dir / "train.pt",
            checkpoint_a,
            train_a_path,
            train_a,
            int(k),
            device,
            batch_size,
            num_workers,
        )
        dev_bundle = load_or_build_cached_bundle(
            cache_dir / "calibration.pt",
            checkpoint_a,
            dev_a_path,
            dev_a,
            int(k),
            device,
            batch_size,
            num_workers,
        )
        control = evaluate_baseline_bundle(
            dev_bundle, train_a, dev_a, gates
        ).metrics
        summary = train_lattice_scorer(
            train_a,
            train_bundle,
            dev_a,
            dev_bundle,
            gates,
            config["scorer"],
            config["training"],
            output_root / "01_calibration_a" / "k{}".format(k),
            device,
        )
        results_a[str(k)] = _calibration_result(summary, control)
    selected_k = max(
        (int(k) for k in config["candidate_counts"]),
        key=lambda value: (
            int(results_a[str(value)]["summary"]["best_metrics"]["neural"]["correct"]),
            int(results_a[str(value)]["summary"]["best_metrics"]["neural"]["word_correct"]),
            -value,
        ),
    )
    selected_a = results_a[str(selected_k)]
    if (
        int(selected_a["neural_correct_gain"]) <= 0
        or int(selected_a["neural_exact_word_gain"]) <= 0
    ):
        selection = {
            "schema_version": 1,
            "accepted": False,
            "reason": "split_a_nonpositive_gain",
            "selected_k": selected_k,
            "split_a": results_a,
        }
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(selection_path, selection)
        return selection

    split_b = config["splits"]["b"]
    train_b_path = Path(split_b["train"])
    dev_b_path = Path(split_b["calibration"])
    checkpoint_b = Path(split_b["checkpoint"])
    train_b = load_jsonl(train_b_path)
    dev_b = load_jsonl(dev_b_path)
    cache_dir_b = output_root / "cache" / "split_b" / "k{}".format(selected_k)
    train_bundle_b = load_or_build_cached_bundle(
        cache_dir_b / "train.pt",
        checkpoint_b,
        train_b_path,
        train_b,
        selected_k,
        device,
        batch_size,
        num_workers,
    )
    dev_bundle_b = load_or_build_cached_bundle(
        cache_dir_b / "calibration.pt",
        checkpoint_b,
        dev_b_path,
        dev_b,
        selected_k,
        device,
        batch_size,
        num_workers,
    )
    control_b = evaluate_baseline_bundle(
        dev_bundle_b, train_b, dev_b, gates
    ).metrics
    summary_b = train_lattice_scorer(
        train_b,
        train_bundle_b,
        dev_b,
        dev_bundle_b,
        gates,
        config["scorer"],
        config["training"],
        output_root / "02_calibration_b" / "k{}".format(selected_k),
        device,
    )
    result_b = _calibration_result(summary_b, control_b)
    mean_letters = (
        int(selected_a["neural_correct_gain"])
        + int(result_b["neural_correct_gain"])
    ) / 2.0
    mean_words = (
        int(selected_a["neural_exact_word_gain"])
        + int(result_b["neural_exact_word_gain"])
    ) / 2.0
    gate = config["calibration_gate"]
    accepted = (
        int(selected_a["neural_correct_gain"]) > 0
        and int(result_b["neural_correct_gain"]) > 0
        and int(selected_a["neural_exact_word_gain"]) > 0
        and int(result_b["neural_exact_word_gain"]) > 0
        and mean_letters >= int(gate["minimum_mean_correct_letters"])
        and mean_words >= int(gate["minimum_mean_exact_words"])
        and int(selected_a["summary"]["best_metrics"]["neural"]["skeleton_mismatch_count"]) == 0
        and int(result_b["summary"]["best_metrics"]["neural"]["skeleton_mismatch_count"]) == 0
    )
    locked_epochs = int(
        math.floor(
            (
                int(selected_a["summary"]["best_epoch"])
                + int(result_b["summary"]["best_epoch"])
            )
            / 2.0
            + 0.5
        )
    )
    selection = {
        "schema_version": 1,
        "device": str(device),
        "selected_k": selected_k,
        "split_a": results_a,
        "split_b": result_b,
        "mean_neural_correct_gain": mean_letters,
        "mean_neural_exact_word_gain": mean_words,
        "locked_epochs": locked_epochs,
        "accepted": accepted,
        "next_stage": "final_seed42" if accepted else "stop_v14",
    }
    write_json(selection_path, selection)
    return selection


def _evaluate_probability_system(
    train_records: Sequence[SentenceRecord],
    records: Sequence[SentenceRecord],
    probabilities: Sequence[torch.Tensor],
    gates: Any,
) -> Mapping[str, Any]:
    prior = WordLabelPrior().fit(train_records)
    seen_words = training_word_types(train_records)
    neural = probabilities_to_predictions(records, probabilities)
    v2 = []
    for record, distribution, initial in zip(records, probabilities, neural):
        labels, _ = apply_gated_fallback(
            record,
            distribution.clamp_min(1.0e-12).log(),
            prior,
            gates,
            initial_predictions=torch.tensor(initial),
        )
        v2.append(labels)
    result: Dict[str, Any] = {
        "neural_predictions": neural,
        "v2_predictions": v2,
    }
    if any(record.labels is None for record in records):
        result["evaluation_skipped"] = "records_have_no_gold_labels"
        return result
    for name, predictions in (("neural", neural), ("v2", v2)):
        paper = compute_paper_metrics(records, predictions)
        diagnostics = prediction_diagnostics(records, predictions, seen_words)
        result[name] = {
            "correct": int(paper["correct_letters"]),
            "micro_f1": float(paper["micro_f1"]),
            "word_correct": int(paper["word_correct"]),
            "word_accuracy": float(paper["word_accuracy"]),
            "sentence_accuracy": float(paper["sentence_accuracy"]),
            "oov_accuracy": float(diagnostics["oov_accuracy"]),
            "shadda_accuracy": float(paper["shadda"]["accuracy"]),
            "tanween_accuracy": float(paper["tanween"]["accuracy"]),
            "skeleton_mismatch_count": int(paper["skeleton_mismatch_count"]),
            "paper_metrics": paper,
            "diagnostics": diagnostics,
        }
    return result


def run_final(
    config_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    config = load_campaign_config(config_path)
    output_root = Path(config["output_root"])
    selection_path = output_root / "03_final_seed42" / "SELECTION.json"
    if selection_path.is_file():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    calibration = run_calibration(
        config_path, device_name, batch_size, num_workers
    )
    if not calibration["accepted"]:
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection = {
            "schema_version": 1,
            "accepted": False,
            "reason": "calibration_gate_failed",
        }
        write_json(selection_path, selection)
        return selection
    device = select_device(device_name)
    gates = load_gates(Path(config["v2_gates"]))
    k = int(calibration["selected_k"])
    epochs = int(calibration["locked_epochs"])
    train_path = Path(config["released_train"])
    dev_path = Path(config["released_dev"])
    checkpoint_path = Path(config["v7_full_checkpoint"])
    train_records = load_jsonl(train_path)
    dev_records = load_jsonl(dev_path)
    cache_dir = output_root / "cache" / "final" / "k{}".format(k)
    train_bundle = load_or_build_cached_bundle(
        cache_dir / "train.pt",
        checkpoint_path,
        train_path,
        train_records,
        k,
        device,
        batch_size,
        num_workers,
    )
    dev_bundle = load_or_build_cached_bundle(
        cache_dir / "dev.pt",
        checkpoint_path,
        dev_path,
        dev_records,
        k,
        device,
        batch_size,
        num_workers,
    )
    control = evaluate_baseline_bundle(
        dev_bundle, train_records, dev_records, gates
    )
    run_dir = output_root / "03_final_seed42" / "model"
    summary = train_lattice_scorer(
        train_records,
        train_bundle,
        dev_records,
        dev_bundle,
        gates,
        config["scorer"],
        config["training"],
        run_dir,
        device,
        fixed_epochs=epochs,
    )
    scorer, scorer_checkpoint = load_scorer_checkpoint(
        Path(summary["best_checkpoint"]), device
    )
    candidate = evaluate_lattice_scorer(
        scorer, dev_bundle, train_records, dev_records, gates, device
    )
    neural_gain = (
        int(candidate.metrics["neural"]["correct"])
        - int(control.metrics["neural"]["correct"])
    )
    word_gain = (
        int(candidate.metrics["neural"]["word_correct"])
        - int(control.metrics["neural"]["word_correct"])
    )
    v2_gain = (
        int(candidate.metrics["v2"]["correct"])
        - int(control.metrics["v2"]["correct"])
    )
    neural_regressions = _protected_regressions(
        candidate.metrics["neural"], control.metrics["neural"]
    )
    v2_regressions = _protected_regressions(
        candidate.metrics["v2"], control.metrics["v2"]
    )
    final_gate = config["final_gate"]
    standalone_accepted = (
        neural_gain >= int(final_gate["minimum_correct_letters"])
        and word_gain >= int(final_gate["minimum_exact_words"])
        and v2_gain >= 0
        and not any(neural_regressions.values())
        and not any(v2_regressions.values())
        and int(candidate.metrics["neural"]["skeleton_mismatch_count"]) == 0
        and int(candidate.metrics["v2"]["skeleton_mismatch_count"]) == 0
    )

    test_records = load_raw_sentences(
        Path("Data/test_data/raw_sentences_test.txt"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
    )
    test_bundle = load_or_build_cached_bundle(
        cache_dir / "test.pt",
        checkpoint_path,
        Path("Data/test_data/raw_sentences_test.txt"),
        test_records,
        k,
        device,
        batch_size,
        num_workers,
    )
    test_candidate = evaluate_lattice_scorer(
        scorer, test_bundle, train_records, test_records, gates, device
    )
    artifacts_dir = run_dir / "artifacts"
    prefix = "DZIRIFORMER_FILTERED_WORD_LATTICE_V14_SEED42"
    neural_artifacts = write_prediction_artifacts(
        artifacts_dir,
        prefix + "_NEURAL",
        test_records,
        [list(values) for values in test_candidate.neural_predictions],
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    v2_artifacts = write_prediction_artifacts(
        artifacts_dir,
        prefix + "_V2",
        test_records,
        [list(values) for values in test_candidate.v2_predictions],
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    standalone_manifest = {
        "system_name": "DziriFormer-FilteredWordLattice-v14-seed42",
        "artifact_prefix": prefix,
        "k": k,
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": sha256_file(checkpoint_path),
        "scorer_checkpoint": str(summary["best_checkpoint"]),
        "scorer_checkpoint_sha256": sha256_file(Path(summary["best_checkpoint"])),
        "dev": candidate.metrics,
        "neural_artifacts": dict(neural_artifacts),
        "v2_artifacts": dict(v2_artifacts),
        "accepted": standalone_accepted,
    }
    write_step_manifest(artifacts_dir / (prefix + "_MANIFEST.json"), standalone_manifest)

    ensemble_manifest: Optional[Mapping[str, Any]] = None
    ensemble_accepted = False
    if standalone_accepted:
        v7_config = load_v7_config(Path(config["v7_campaign"]))
        groups = checkpoint_groups(v7_config, "crf_final")[1:]
        dev_groups, _ = predict_probability_groups(
            groups, dev_records, device, batch_size, num_workers
        )
        dev_probabilities = average_probability_groups(
            [list(candidate.probabilities)] + dev_groups
        )
        ensemble_dev = _evaluate_probability_system(
            train_records, dev_records, dev_probabilities, gates
        )
        test_groups, _ = predict_probability_groups(
            groups, test_records, device, batch_size, num_workers
        )
        test_probabilities = average_probability_groups(
            [list(test_candidate.probabilities)] + test_groups
        )
        ensemble_test = _evaluate_probability_system(
            train_records, test_records, test_probabilities, gates
        )
        ensemble_dir = output_root / "04_final_ensemble" / "artifacts"
        ensemble_prefix = "DZIRI_FINAL_FILTERED_WORD_LATTICE_ENSEMBLE_V14"
        ensemble_neural = write_prediction_artifacts(
            ensemble_dir,
            ensemble_prefix + "_NEURAL",
            test_records,
            ensemble_test["neural_predictions"],
            Path("Data/test_data/sample_submission.csv"),
            Path("Data/test_data/raw_sentences_test_ids.txt"),
            Path("Data/test_data/raw_sentences_test.txt"),
        )
        ensemble_v2 = write_prediction_artifacts(
            ensemble_dir,
            ensemble_prefix + "_V2",
            test_records,
            ensemble_test["v2_predictions"],
            Path("Data/test_data/sample_submission.csv"),
            Path("Data/test_data/raw_sentences_test_ids.txt"),
            Path("Data/test_data/raw_sentences_test.txt"),
        )
        ensemble_accepted = (
            int(ensemble_dev["v2"]["correct"])
            >= int(final_gate["minimum_ensemble_v2_correct"])
        )
        ensemble_manifest = {
            "system_name": "DziriFinal-FilteredWordLattice-Ensemble-v14",
            "artifact_prefix": ensemble_prefix,
            "aggregation": "equal_architecture_probability_mean",
            "dev": {
                "neural": ensemble_dev["neural"],
                "v2": ensemble_dev["v2"],
            },
            "neural_artifacts": dict(ensemble_neural),
            "v2_artifacts": dict(ensemble_v2),
            "accepted": ensemble_accepted,
        }
        write_step_manifest(
            ensemble_dir / (ensemble_prefix + "_MANIFEST.json"),
            ensemble_manifest,
        )
    selection = {
        "schema_version": 1,
        "device": str(device),
        "k": k,
        "locked_epochs": epochs,
        "control": control.metrics,
        "candidate": candidate.metrics,
        "neural_correct_gain": neural_gain,
        "neural_exact_word_gain": word_gain,
        "v2_correct_gain": v2_gain,
        "protected_neural_regressions": neural_regressions,
        "protected_v2_regressions": v2_regressions,
        "standalone_accepted": standalone_accepted,
        "ensemble": ensemble_manifest,
        "ensemble_accepted": ensemble_accepted,
        "accepted": standalone_accepted and ensemble_accepted,
        "standalone_manifest": standalone_manifest,
    }
    write_json(selection_path, selection)
    write_json(
        output_root / "campaign_manifest.json",
        {
            "schema_version": 1,
            "oracle": str(output_root / "00_oracle" / "ORACLE_SELECTION.json"),
            "calibration": str(output_root / "02_calibration_b" / "CALIBRATION_SELECTION.json"),
            "final": str(selection_path),
            "accepted": selection["accepted"],
        },
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/track4/Lyes/filtered_word_lattice_v14/campaign.json"),
    )
    parser.add_argument("--stage", choices=("oracle", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid loader settings")
    selection = run_oracle(
        args.config,
        args.device,
        args.batch_size,
        args.num_workers,
    )
    if args.stage == "all" and selection["accepted"]:
        final = run_final(
            args.config,
            args.device,
            args.batch_size,
            args.num_workers,
        )
        print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
