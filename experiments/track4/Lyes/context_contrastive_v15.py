"""Resumable ContextContrastive-v15 calibration, training, and export."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from utils.track4.Lyes.gated_fusion.config import GatedFusionConfig, load_gates
from utils.track4.Lyes.gated_fusion.fusion import GatedFusionStatistics, apply_gated_fallback
from experiments.track4.Lyes.campaign.common import write_prediction_artifacts, write_step_manifest
from experiments.track4.Lyes.campaign.diagnostics import prediction_diagnostics, training_word_types
from experiments.track4.Lyes.campaign.ensemble import (
    average_probability_groups,
    predict_probability_groups,
    probabilities_to_predictions,
)
from models.track4.Lyes.context_contrastive import (
    AmbiguityIndex,
    ContextContrastiveModel,
    initialize_from_v7,
    load_context_contrastive_checkpoint,
    save_context_contrastive_checkpoint,
)
from utils.track4.Lyes.data import (
    BatchCollator,
    CharacterDataset,
    LengthBucketBatchSampler,
    SentenceRecord,
    load_jsonl,
    load_raw_sentences,
    validate_vocabulary_coverage,
)
from experiments.track4.Lyes.export_ensemble import (
    checkpoint_groups,
    load_v7_config,
)
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from evaluation.track4.Lyes.paper_metrics import compute_paper_metrics
from utils.track4.Lyes.utils import append_jsonl, select_device, sha256_file, write_json


EXPECTED_KEYS = {
    "schema_version",
    "output_root",
    "oracle_selection",
    "released_train",
    "released_dev",
    "v7_full_checkpoint",
    "v7_campaign",
    "v2_gates",
    "splits",
    "auxiliary_coefficients",
    "model",
    "training",
    "final_gate",
}
PROTECTED_KEYS = (
    "word_accuracy",
    "sentence_accuracy",
    "oov_accuracy",
    "shadda_accuracy",
    "tanween_accuracy",
)


def load_campaign_config(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != EXPECTED_KEYS:
        raise ValueError("invalid ContextContrastive-v15 config keys")
    if int(raw["schema_version"]) != 1:
        raise ValueError("unsupported ContextContrastive-v15 schema")
    if raw["auxiliary_coefficients"] != [0.1, 0.3, 1.0]:
        raise ValueError("v15 coefficients are locked to [0.1, 0.3, 1.0]")
    if set(raw["splits"]) != {"a", "b"}:
        raise ValueError("v15 requires split a and split b")
    split_keys = {"train", "calibration", "checkpoint"}
    if any(set(split) != split_keys for split in raw["splits"].values()):
        raise ValueError("invalid v15 split configuration")
    if raw["training"] != {
        "epochs": 15,
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "one_cycle_pct_start": 0.3,
        "one_cycle_div_factor": 25.0,
        "one_cycle_final_div_factor": 10000.0,
        "gradient_clip_norm": 1.0,
        "seed": 42,
        "num_workers": 0,
    }:
        raise ValueError("v15 training configuration differs from the approved plan")
    return raw


def _make_loader(
    records: Sequence[SentenceRecord],
    vocab: Mapping[str, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> Tuple[DataLoader, LengthBucketBatchSampler]:
    sampler = LengthBucketBatchSampler(
        [len(record.chars) + 2 for record in records],
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    if not shuffle:
        loader = DataLoader(
            CharacterDataset(records),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=BatchCollator(vocab),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        return loader, sampler
    loader = DataLoader(
        CharacterDataset(records),
        batch_sampler=sampler,
        collate_fn=BatchCollator(vocab),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def _flatten_metrics(
    paper: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "correct": int(paper["correct_letters"]),
        "total": int(paper["scored_letters"]),
        "micro_f1": float(paper["micro_f1"]),
        "macro_f1": float(paper["macro_f1"]),
        "word_correct": int(paper["word_correct"]),
        "word_accuracy": float(paper["word_accuracy"]),
        "sentence_correct": int(paper["sentence_correct"]),
        "sentence_accuracy": float(paper["sentence_accuracy"]),
        "oov_accuracy": float(diagnostics["oov_accuracy"]),
        "seen_accuracy": float(diagnostics["seen_accuracy"]),
        "shadda_accuracy": float(paper["shadda"]["accuracy"]),
        "tanween_accuracy": float(paper["tanween"]["accuracy"]),
        "skeleton_mismatch_count": int(paper["skeleton_mismatch_count"]),
        "paper_metrics": dict(paper),
        "diagnostics": dict(diagnostics),
    }


@torch.inference_mode()
def predict_model(
    model: ContextContrastiveModel,
    records: Sequence[SentenceRecord],
    vocab: Mapping[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[List[int]], List[torch.Tensor]]:
    validate_vocabulary_coverage(records, vocab)
    loader, _ = _make_loader(records, vocab, batch_size, False, 0, num_workers)
    model.eval()
    predictions: List[List[int]] = []
    probabilities: List[torch.Tensor] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        decoded = model.decode_outputs(outputs).cpu()
        marginal = model.log_probabilities(outputs).exp().cpu()
        for row, record in enumerate(batch["records"]):
            values = decoded[row, 1 : len(record.chars) + 1].tolist()
            predictions.append([
                0 if char == " " else int(label)
                for char, label in zip(record.chars, values)
            ])
            probabilities.append(marginal[row, 1 : len(record.chars) + 1].contiguous())
    return predictions, probabilities


def evaluate_model(
    model: ContextContrastiveModel,
    train_records: Sequence[SentenceRecord],
    records: Sequence[SentenceRecord],
    vocab: Mapping[str, int],
    gates: GatedFusionConfig,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    neural, probabilities = predict_model(
        model, records, vocab, device, batch_size, num_workers
    )
    prior = WordLabelPrior().fit(train_records)
    seen_words = training_word_types(train_records)
    v2: List[List[int]] = []
    statistics = GatedFusionStatistics()
    for record, distribution, initial in zip(records, probabilities, neural):
        labels, row_statistics = apply_gated_fallback(
            record,
            distribution.clamp_min(1.0e-12).log(),
            prior,
            gates,
            initial_predictions=torch.tensor(initial),
        )
        v2.append(labels)
        statistics.update(row_statistics)
    result: Dict[str, Any] = {
        "neural_predictions": neural,
        "v2_predictions": v2,
        "probabilities": probabilities,
        "v2_gate_statistics": statistics.to_dict(),
    }
    if any(record.labels is None for record in records):
        result["evaluation_skipped"] = "records_have_no_gold_labels"
        return result
    for name, values in (("neural", neural), ("v2", v2)):
        paper = compute_paper_metrics(records, values)
        diagnostics = prediction_diagnostics(records, values, seen_words)
        result[name] = _flatten_metrics(paper, diagnostics)
    return result


def _save_checkpoint(
    path: Path,
    model: ContextContrastiveModel,
    vocab: Mapping[str, int],
    gate_hidden_dim: int,
    epoch: int,
    coefficient: float,
    metrics: Mapping[str, Any],
) -> None:
    save_context_contrastive_checkpoint(
        path, model, vocab, gate_hidden_dim, epoch, coefficient, metrics
    )


def train_model(
    base_checkpoint: Path,
    train_records: Sequence[SentenceRecord],
    evaluation_records: Sequence[SentenceRecord],
    gates: GatedFusionConfig,
    coefficient: float,
    gate_hidden_dim: int,
    training: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    fixed_epochs: Optional[int] = None,
) -> Mapping[str, Any]:
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "best.pt"
    if summary_path.is_file() and checkpoint_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise RuntimeError("v15 completed checkpoint hash mismatch")
        return summary
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("partial v15 run requires manual inspection: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    model, vocab, source = initialize_from_v7(base_checkpoint, device, gate_hidden_dim)
    validate_vocabulary_coverage(train_records, vocab)
    validate_vocabulary_coverage(evaluation_records, vocab)
    epochs = int(fixed_epochs or training["epochs"])
    train_loader, train_sampler = _make_loader(
        train_records,
        vocab,
        int(training["batch_size"]),
        True,
        seed,
        int(training["num_workers"]),
    )
    ambiguity = AmbiguityIndex.fit(train_records)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=float(training["learning_rate"]),
        total_steps=updates_per_epoch * epochs,
        pct_start=float(training["one_cycle_pct_start"]),
        div_factor=float(training["one_cycle_div_factor"]),
        final_div_factor=float(training["one_cycle_final_div_factor"]),
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_key = (-1, -1)
    best_epoch = 0
    best_metrics: Optional[Mapping[str, Any]] = None
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_nll = total_auxiliary = 0.0
        batch_count = accumulated = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            ambiguity_targets = ambiguity.targets(
                batch["records"], input_ids.size(1), device
            )
            outputs = model(input_ids, attention_mask)
            loss, nll, auxiliary = model.compute_loss(
                outputs, targets, ambiguity_targets, coefficient
            )
            (loss / accumulation).backward()
            accumulated += 1
            total_loss += float(loss.detach().item())
            total_nll += float(nll.detach().item())
            total_auxiliary += float(auxiliary.detach().item())
            batch_count += 1
            last = batch_index == len(train_loader)
            if accumulated == accumulation or last:
                if last and accumulated < accumulation:
                    correction = accumulation / accumulated
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
        metrics = evaluate_model(
            model,
            train_records,
            evaluation_records,
            vocab,
            gates,
            device,
            int(training["batch_size"]),
            int(training["num_workers"]),
        )
        epoch_record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started,
            "train": {
                "loss": total_loss / batch_count,
                "crf_nll": total_nll / batch_count,
                "gate_bce": total_auxiliary / batch_count,
            },
            "dev": {"neural": metrics["neural"], "v2": metrics["v2"]},
        }
        append_jsonl(metrics_path, epoch_record)
        key = (int(metrics["neural"]["correct"]), int(metrics["neural"]["word_correct"]))
        select = fixed_epochs is None and key > best_key
        if fixed_epochs is not None:
            select = epoch == epochs
        if select:
            best_key = key
            best_epoch = epoch
            best_metrics = {"neural": metrics["neural"], "v2": metrics["v2"]}
            _save_checkpoint(
                checkpoint_path,
                model,
                vocab,
                gate_hidden_dim,
                epoch,
                coefficient,
                best_metrics,
            )
        print(
            "v15 lambda={} epoch={:02d} neural={}/{} v2={}/{} gate_bce={:.6f}".format(
                coefficient,
                epoch,
                metrics["neural"]["correct"],
                metrics["neural"]["total"],
                metrics["v2"]["correct"],
                metrics["v2"]["total"],
                total_auxiliary / batch_count,
            ),
            flush=True,
        )
    if best_metrics is None:
        raise RuntimeError("v15 training did not produce a checkpoint")
    summary = {
        "schema_version": 1,
        "system_name": "DziriFormer-ContextContrastive-v15",
        "device": str(device),
        "source_checkpoint": str(base_checkpoint),
        "source_checkpoint_sha256": sha256_file(base_checkpoint),
        "source_epoch": int(source.get("epoch", 0)),
        "auxiliary_coefficient": float(coefficient),
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    write_json(summary_path, summary)
    return summary


def _evaluate_control(
    checkpoint_path: Path,
    train_records: Sequence[SentenceRecord],
    records: Sequence[SentenceRecord],
    gates: GatedFusionConfig,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    model, vocab, _ = initialize_from_v7(checkpoint_path, device)
    return evaluate_model(
        model, train_records, records, vocab, gates, device, batch_size, num_workers
    )


def _result(summary: Mapping[str, Any], control: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = summary["best_metrics"]
    return {
        "summary": dict(summary),
        "control": {"neural": control["neural"], "v2": control["v2"]},
        "neural_correct_gain": int(metrics["neural"]["correct"]) - int(control["neural"]["correct"]),
        "neural_word_gain": int(metrics["neural"]["word_correct"]) - int(control["neural"]["word_correct"]),
        "v2_correct_gain": int(metrics["v2"]["correct"]) - int(control["v2"]["correct"]),
    }


def run_calibration(config_path: Path, device_name: str) -> Mapping[str, Any]:
    config = load_campaign_config(config_path)
    output_root = Path(config["output_root"])
    selection_path = output_root / "02_calibration_b" / "CALIBRATION_SELECTION.json"
    if selection_path.is_file():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    oracle = json.loads(Path(config["oracle_selection"]).read_text(encoding="utf-8"))
    if not oracle.get("accepted"):
        raise RuntimeError("v15 is gated by the accepted word-lattice oracle")
    device = select_device(device_name)
    gates = load_gates(Path(config["v2_gates"]))
    training = config["training"]
    split_a = config["splits"]["a"]
    train_a = load_jsonl(Path(split_a["train"]))
    dev_a = load_jsonl(Path(split_a["calibration"]))
    control_a = _evaluate_control(
        Path(split_a["checkpoint"]), train_a, dev_a, gates, device,
        int(training["batch_size"]), int(training["num_workers"])
    )
    results_a: Dict[str, Mapping[str, Any]] = {}
    for coefficient in config["auxiliary_coefficients"]:
        summary = train_model(
            Path(split_a["checkpoint"]), train_a, dev_a, gates, float(coefficient),
            int(config["model"]["gate_hidden_dim"]), training,
            output_root / "01_calibration_a" / "lambda_{}".format(str(coefficient).replace(".", "p")),
            device,
        )
        results_a[str(coefficient)] = _result(summary, control_a)
    winner = max(
        (float(value) for value in config["auxiliary_coefficients"]),
        key=lambda value: (
            int(results_a[str(value)]["summary"]["best_metrics"]["neural"]["correct"]),
            int(results_a[str(value)]["summary"]["best_metrics"]["neural"]["word_correct"]),
            -value,
        ),
    )
    split_b = config["splits"]["b"]
    train_b = load_jsonl(Path(split_b["train"]))
    dev_b = load_jsonl(Path(split_b["calibration"]))
    control_b = _evaluate_control(
        Path(split_b["checkpoint"]), train_b, dev_b, gates, device,
        int(training["batch_size"]), int(training["num_workers"])
    )
    summary_b = train_model(
        Path(split_b["checkpoint"]), train_b, dev_b, gates, winner,
        int(config["model"]["gate_hidden_dim"]), training,
        output_root / "02_calibration_b" / "lambda_{}".format(str(winner).replace(".", "p")),
        device,
    )
    result_b = _result(summary_b, control_b)
    result_a = results_a[str(winner)]
    accepted = True
    locked_epochs = int(math.floor((int(result_a["summary"]["best_epoch"]) + int(result_b["summary"]["best_epoch"])) / 2.0 + 0.5))
    selection = {
        "schema_version": 1,
        "device": str(device),
        "split_a": results_a,
        "winner": winner,
        "split_b": result_b,
        "calibration_evidence_only": True,
        "locked_epochs": locked_epochs,
        "accepted": accepted,
        "next_stage": "final_seed42" if accepted else "stop_v15",
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(selection_path, selection)
    return selection


def _protected_regressions(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> Mapping[str, bool]:
    return {key: float(candidate[key]) < float(control[key]) for key in PROTECTED_KEYS}


def _reconcile_final_selection(
    selection: Mapping[str, Any], gate: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Upgrade a completed selection to the current strict acceptance schema."""
    if "candidate" not in selection or "control" not in selection:
        return dict(selection)
    result: Dict[str, Any] = dict(selection)
    candidate = result["candidate"]
    control = result["control"]
    neural_gain = int(candidate["neural"]["correct"]) - int(control["neural"]["correct"])
    v2_gain = int(candidate["v2"]["correct"]) - int(control["v2"]["correct"])
    neural_regressions = _protected_regressions(candidate["neural"], control["neural"])
    v2_regressions = _protected_regressions(candidate["v2"], control["v2"])
    standalone_accepted = (
        neural_gain >= int(gate["minimum_neural_correct_gain"])
        and v2_gain >= int(gate["minimum_v2_correct_gain"])
        and not any(neural_regressions.values())
        and not any(v2_regressions.values())
        and int(candidate["neural"]["skeleton_mismatch_count"]) == 0
        and int(candidate["v2"]["skeleton_mismatch_count"]) == 0
    )
    result["neural_correct_gain"] = neural_gain
    result["v2_correct_gain"] = v2_gain
    result["protected_neural_regressions"] = neural_regressions
    result["protected_v2_regressions"] = v2_regressions
    result["standalone_accepted"] = standalone_accepted
    standalone_manifest = dict(result.get("standalone_manifest") or {})
    if standalone_manifest:
        standalone_manifest["accepted"] = standalone_accepted
        result["standalone_manifest"] = standalone_manifest
    ensemble_accepted = bool(result.get("ensemble_accepted", False))
    result["accepted"] = standalone_accepted and ensemble_accepted
    return result


def _evaluate_probability_system(
    train_records: Sequence[SentenceRecord],
    records: Sequence[SentenceRecord],
    probabilities: Sequence[torch.Tensor],
    gates: GatedFusionConfig,
) -> Mapping[str, Any]:
    neural = probabilities_to_predictions(records, probabilities)
    prior = WordLabelPrior().fit(train_records)
    v2: List[List[int]] = []
    for record, probability, initial in zip(records, probabilities, neural):
        labels, _ = apply_gated_fallback(
            record, probability.clamp_min(1.0e-12).log(), prior, gates,
            initial_predictions=torch.tensor(initial)
        )
        v2.append(labels)
    result: Dict[str, Any] = {"neural_predictions": neural, "v2_predictions": v2}
    if any(record.labels is None for record in records):
        return result
    seen_words = training_word_types(train_records)
    for name, values in (("neural", neural), ("v2", v2)):
        paper = compute_paper_metrics(records, values)
        result[name] = _flatten_metrics(paper, prediction_diagnostics(records, values, seen_words))
    return result


def run_final(config_path: Path, device_name: str) -> Mapping[str, Any]:
    config = load_campaign_config(config_path)
    output_root = Path(config["output_root"])
    selection_path = output_root / "03_final_seed42" / "SELECTION.json"
    if selection_path.is_file():
        stored = json.loads(selection_path.read_text(encoding="utf-8"))
        reconciled = _reconcile_final_selection(stored, config["final_gate"])
        if reconciled != stored:
            write_json(selection_path, reconciled)
            write_json(output_root / "campaign_manifest.json", reconciled)
            standalone_manifest = reconciled.get("standalone_manifest")
            if standalone_manifest:
                prefix = str(standalone_manifest["artifact_prefix"])
                manifest_path = (
                    output_root / "03_final_seed42" / "model" / "artifacts"
                    / (prefix + "_MANIFEST.json")
                )
                write_step_manifest(manifest_path, standalone_manifest)
        return reconciled
    calibration = run_calibration(config_path, device_name)
    if not calibration["accepted"]:
        selection = {"schema_version": 1, "accepted": False, "reason": "calibration_gate_failed"}
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(selection_path, selection)
        return selection
    device = select_device(device_name)
    gates = load_gates(Path(config["v2_gates"]))
    training = config["training"]
    train_records = load_jsonl(Path(config["released_train"]))
    dev_records = load_jsonl(Path(config["released_dev"]))
    run_dir = output_root / "03_final_seed42" / "model"
    summary = train_model(
        Path(config["v7_full_checkpoint"]), train_records, dev_records, gates,
        float(calibration["winner"]), int(config["model"]["gate_hidden_dim"]),
        training, run_dir, device, fixed_epochs=int(calibration["locked_epochs"])
    )
    model, vocab, _ = load_context_contrastive_checkpoint(Path(summary["checkpoint"]), device)
    candidate = evaluate_model(
        model, train_records, dev_records, vocab, gates, device,
        int(training["batch_size"]), int(training["num_workers"])
    )
    control = _evaluate_control(
        Path(config["v7_full_checkpoint"]), train_records, dev_records, gates, device,
        int(training["batch_size"]), int(training["num_workers"])
    )
    neural_gain = int(candidate["neural"]["correct"]) - int(control["neural"]["correct"])
    v2_gain = int(candidate["v2"]["correct"]) - int(control["v2"]["correct"])
    neural_regressions = _protected_regressions(
        candidate["neural"], control["neural"]
    )
    v2_regressions = _protected_regressions(
        candidate["v2"], control["v2"]
    )
    gate = config["final_gate"]
    standalone_accepted = (
        neural_gain >= int(gate["minimum_neural_correct_gain"])
        and v2_gain >= int(gate["minimum_v2_correct_gain"])
        and not any(neural_regressions.values())
        and not any(v2_regressions.values())
        and int(candidate["neural"]["skeleton_mismatch_count"]) == 0
        and int(candidate["v2"]["skeleton_mismatch_count"]) == 0
    )
    test_records = load_raw_sentences(
        Path("Data/test_data/raw_sentences_test.txt"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
    )
    test_candidate = evaluate_model(
        model, train_records, test_records, vocab, gates, device,
        int(training["batch_size"]), int(training["num_workers"])
    )
    artifacts_dir = run_dir / "artifacts"
    prefix = "DZIRIFORMER_CONTEXT_CONTRASTIVE_V15_SEED42"
    neural_artifacts = write_prediction_artifacts(
        artifacts_dir, prefix + "_NEURAL", test_records,
        test_candidate["neural_predictions"], Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"), Path("Data/test_data/raw_sentences_test.txt")
    )
    v2_artifacts = write_prediction_artifacts(
        artifacts_dir, prefix + "_V2", test_records,
        test_candidate["v2_predictions"], Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"), Path("Data/test_data/raw_sentences_test.txt")
    )
    standalone_manifest = {
        "system_name": "DziriFormer-ContextContrastive-v15-seed42",
        "artifact_prefix": prefix,
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "dev": {"neural": candidate["neural"], "v2": candidate["v2"]},
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
            groups, dev_records, device, int(training["batch_size"]), int(training["num_workers"])
        )
        ensemble_dev = _evaluate_probability_system(
            train_records, dev_records,
            average_probability_groups([candidate["probabilities"]] + dev_groups), gates
        )
        test_groups, _ = predict_probability_groups(
            groups, test_records, device, int(training["batch_size"]), int(training["num_workers"])
        )
        ensemble_test = _evaluate_probability_system(
            train_records, test_records,
            average_probability_groups([test_candidate["probabilities"]] + test_groups), gates
        )
        ensemble_dir = output_root / "04_final_ensemble" / "artifacts"
        ensemble_prefix = "DZIRI_FINAL_CONTEXT_CONTRASTIVE_ENSEMBLE_V15"
        ensemble_neural = write_prediction_artifacts(
            ensemble_dir, ensemble_prefix + "_NEURAL", test_records,
            ensemble_test["neural_predictions"], Path("Data/test_data/sample_submission.csv"),
            Path("Data/test_data/raw_sentences_test_ids.txt"), Path("Data/test_data/raw_sentences_test.txt")
        )
        ensemble_v2 = write_prediction_artifacts(
            ensemble_dir, ensemble_prefix + "_V2", test_records,
            ensemble_test["v2_predictions"], Path("Data/test_data/sample_submission.csv"),
            Path("Data/test_data/raw_sentences_test_ids.txt"), Path("Data/test_data/raw_sentences_test.txt")
        )
        ensemble_accepted = int(ensemble_dev["v2"]["correct"]) >= int(gate["minimum_ensemble_v2_correct"])
        ensemble_manifest = {
            "system_name": "DziriFinal-ContextContrastive-Ensemble-v15",
            "artifact_prefix": ensemble_prefix,
            "aggregation": "equal_architecture_probability_mean",
            "dev": {"neural": ensemble_dev["neural"], "v2": ensemble_dev["v2"]},
            "neural_artifacts": dict(ensemble_neural),
            "v2_artifacts": dict(ensemble_v2),
            "accepted": ensemble_accepted,
        }
        write_step_manifest(ensemble_dir / (ensemble_prefix + "_MANIFEST.json"), ensemble_manifest)
    selection = {
        "schema_version": 1,
        "device": str(device),
        "auxiliary_coefficient": calibration["winner"],
        "locked_epochs": calibration["locked_epochs"],
        "control": {"neural": control["neural"], "v2": control["v2"]},
        "candidate": {"neural": candidate["neural"], "v2": candidate["v2"]},
        "neural_correct_gain": neural_gain,
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
    write_json(output_root / "campaign_manifest.json", selection)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/track4/Lyes/context_contrastive_v15/campaign.json")
    )
    parser.add_argument("--stage", choices=("calibration", "all"), default="all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    selection = run_calibration(args.config, args.device)
    if args.stage == "all" and selection["accepted"]:
        selection = run_final(args.config, args.device)
    print(json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
