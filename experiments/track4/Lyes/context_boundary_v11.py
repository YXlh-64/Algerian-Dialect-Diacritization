"""Evaluate and export the context-conditioned low-rank CRF-v11 model."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from utils.track4.Lyes.gated_fusion.config import load_gates
from utils.track4.Lyes.gated_fusion.fusion import predict_with_gated_fallback
from experiments.track4.Lyes.campaign.common import (
    write_prediction_artifacts,
    write_step_manifest,
)
from experiments.track4.Lyes.campaign.diagnostics import (
    prediction_diagnostics,
    training_word_types,
)
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import (
    BatchCollator,
    CharacterDataset,
    SentenceRecord,
    load_jsonl,
    load_raw_sentences,
)
from evaluation.track4.Lyes.infer import predict_records
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer
from evaluation.track4.Lyes.paper_metrics import compute_paper_metrics
from utils.track4.Lyes.utils import select_device, sha256_file, write_json


CONFIG_KEYS = {
    "schema_version",
    "checkpoint",
    "output_dir",
    "system_name",
    "artifact_prefix",
    "expected_head_mode",
    "controls",
}
CONTROL_KEYS = {
    "neural_correct",
    "minimum_neural_gain",
    "neural_oov_correct",
    "neural_word_correct",
    "neural_shadda_accuracy",
    "maximum_shadda_regression",
    "neural_tanween_accuracy",
    "maximum_tanween_regression",
    "production_v2_correct",
}
DESCRIPTION = (
    "Controlled Track 4 extension of DualRoPE-CRF-v7. The encoder, direct "
    "16-class emissions, data, optimizer, and update budget are unchanged. "
    "At every scored transition, a rank-2 residual is scaled by a learned "
    "sigmoid gate computed from the contextual DualRoPE state and an explicit "
    "word-boundary bit: T_i = T_shared + g_i U V^T. U is randomly initialized, "
    "V is zero initialized, and the gate starts at 0.5, making the initial "
    "transition matrix exactly equal to the ordinary CRF while preserving "
    "gradient flow. V2 uses the unchanged training-only confidence-gated "
    "lexical fallback."
)


def load_evaluation_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("invalid context-boundary v11 evaluation keys")
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported context-boundary v11 schema")
    for key in ("checkpoint", "output_dir", "system_name"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a non-empty string")
    if config["expected_head_mode"] != "context_low_rank_boundary_crf":
        raise ValueError("unexpected context-boundary head mode")
    if re.fullmatch(r"[A-Z0-9_]+", str(config["artifact_prefix"])) is None:
        raise ValueError("invalid context-boundary artifact prefix")
    controls = config["controls"]
    if not isinstance(controls, dict) or set(controls) != CONTROL_KEYS:
        raise ValueError("invalid context-boundary control keys")
    integer_controls = (
        "neural_correct",
        "minimum_neural_gain",
        "neural_oov_correct",
        "neural_word_correct",
        "production_v2_correct",
    )
    if any(int(controls[key]) <= 0 for key in integer_controls):
        raise ValueError("count-based controls must be positive")
    for key in (
        "neural_shadda_accuracy",
        "maximum_shadda_regression",
        "neural_tanween_accuracy",
        "maximum_tanween_regression",
    ):
        if not 0.0 <= float(controls[key]) <= 1.0:
            raise ValueError("accuracy controls must be in [0, 1]")
    return config


def _gate_summary(values: torch.Tensor) -> Dict[str, Any]:
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "minimum": None,
            "maximum": None,
        }
    values = values.detach().to(dtype=torch.float64, device="cpu")
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
    }


@torch.no_grad()
def transition_gate_diagnostics(
    model: CharDiacritizer,
    records: Sequence[SentenceRecord],
    vocab: Mapping[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("invalid transition-gate loader settings")
    loader = DataLoader(
        CharacterDataset(tuple(records)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchCollator(vocab),
        num_workers=num_workers,
    )
    all_values = []
    boundary_values = []
    within_values = []
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        if "crf_transition_gate" not in outputs:
            raise ValueError("model did not produce contextual CRF gates")
        gate = outputs["crf_transition_gate"]
        mask = outputs["crf_mask"].bool()
        boundary = outputs["crf_boundary_indicator"].bool()
        all_values.append(gate[mask].cpu())
        boundary_values.append(gate[boundary].cpu())
        within_values.append(gate[mask & ~boundary].cpu())
    all_tensor = torch.cat(all_values) if all_values else torch.empty(0)
    boundary_tensor = (
        torch.cat(boundary_values) if boundary_values else torch.empty(0)
    )
    within_tensor = (
        torch.cat(within_values) if within_values else torch.empty(0)
    )
    return {
        "all_scored_positions": _gate_summary(all_tensor),
        "word_initial_positions": _gate_summary(boundary_tensor),
        "within_word_positions": _gate_summary(within_tensor),
        "boundary_minus_within_mean": (
            None
            if boundary_tensor.numel() == 0 or within_tensor.numel() == 0
            else float(
                boundary_tensor.to(torch.float64).mean().item()
                - within_tensor.to(torch.float64).mean().item()
            )
        ),
    }


def _acceptance(
    controls: Mapping[str, Any],
    neural_paper: Mapping[str, Any],
    neural_diagnostics: Mapping[str, Any],
    v2_paper: Mapping[str, Any],
) -> Dict[str, Any]:
    minimum_correct = int(controls["neural_correct"]) + int(
        controls["minimum_neural_gain"]
    )
    gates: Dict[str, Any] = {
        "overall_correct": {
            "value": int(neural_paper["correct_letters"]),
            "operator": ">=",
            "threshold": minimum_correct,
        },
        "oov_correct": {
            "value": int(neural_diagnostics["oov_correct"]),
            "operator": ">",
            "threshold": int(controls["neural_oov_correct"]),
        },
        "word_correct": {
            "value": int(neural_paper["word_correct"]),
            "operator": ">",
            "threshold": int(controls["neural_word_correct"]),
        },
        "shadda_accuracy": {
            "value": float(neural_paper["shadda"]["accuracy"]),
            "operator": ">=",
            "threshold": float(controls["neural_shadda_accuracy"])
            - float(controls["maximum_shadda_regression"]),
        },
        "tanween_accuracy": {
            "value": float(neural_paper["tanween"]["accuracy"]),
            "operator": ">=",
            "threshold": float(controls["neural_tanween_accuracy"])
            - float(controls["maximum_tanween_regression"]),
        },
        "production_v2_correct": {
            "value": int(v2_paper["correct_letters"]),
            "operator": ">",
            "threshold": int(controls["production_v2_correct"]),
        },
    }
    for result in gates.values():
        if result["operator"] == ">=":
            result["passed"] = result["value"] >= result["threshold"]
        else:
            result["passed"] = result["value"] > result["threshold"]
    architecture_keys = (
        "overall_correct",
        "oov_correct",
        "word_correct",
        "shadda_accuracy",
        "tanween_accuracy",
    )
    return {
        "gates": gates,
        "architecture_accepted": all(
            bool(gates[key]["passed"]) for key in architecture_keys
        ),
        "competitive_submission": all(
            bool(gates[key]["passed"]) for key in architecture_keys
        )
        and bool(gates["production_v2_correct"]["passed"]),
    }


def run(
    config_path: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, Any]:
    config = load_evaluation_config(config_path)
    checkpoint_path = Path(str(config["checkpoint"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"context-boundary checkpoint is missing: {checkpoint_path}"
        )
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("invalid evaluation loader settings")
    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, vocab = build_model_from_checkpoint(checkpoint, device)
    if model.config.resolved_head_mode != config["expected_head_mode"]:
        raise ValueError("checkpoint head mode does not match experiment")

    train_records = load_jsonl(
        Path("Data/train_data/train_Algerian-DIAC.jsonl")
    )
    dev_records = load_jsonl(Path("Data/dev_data/dev_Algerian-DIAC.jsonl"))
    test_records = load_raw_sentences(
        Path("Data/test_data/raw_sentences_test.txt"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
    )
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(Path("configs/track4/Lyes/gates.json"))
    seen_words = training_word_types(train_records)

    dev_neural = predict_records(
        model, dev_records, vocab, device, batch_size, num_workers
    )
    dev_v2, dev_v2_gate_statistics = predict_with_gated_fallback(
        model,
        dev_records,
        vocab,
        prior,
        gates,
        device,
        batch_size,
        num_workers,
    )
    neural_paper = compute_paper_metrics(dev_records, dev_neural)
    v2_paper = compute_paper_metrics(dev_records, dev_v2)
    neural_diagnostics = prediction_diagnostics(
        dev_records, dev_neural, seen_words
    )
    v2_diagnostics = prediction_diagnostics(dev_records, dev_v2, seen_words)
    context_gate_diagnostics = transition_gate_diagnostics(
        model, dev_records, vocab, device, batch_size, num_workers
    )
    decision = _acceptance(
        config["controls"], neural_paper, neural_diagnostics, v2_paper
    )

    test_neural = predict_records(
        model, test_records, vocab, device, batch_size, num_workers
    )
    test_v2, test_v2_gate_statistics = predict_with_gated_fallback(
        model,
        test_records,
        vocab,
        prior,
        gates,
        device,
        batch_size,
        num_workers,
    )
    output_dir = Path(str(config["output_dir"]))
    artifact_dir = output_dir / "artifacts"
    prefix = str(config["artifact_prefix"])
    neural_artifacts = write_prediction_artifacts(
        artifact_dir,
        prefix + "_NEURAL",
        test_records,
        test_neural,
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )
    v2_artifacts = write_prediction_artifacts(
        artifact_dir,
        prefix + "_V2",
        test_records,
        test_v2,
        Path("Data/test_data/sample_submission.csv"),
        Path("Data/test_data/raw_sentences_test_ids.txt"),
        Path("Data/test_data/raw_sentences_test.txt"),
    )

    write_json(output_dir / "neural_paper_metrics.json", neural_paper)
    write_json(output_dir / "v2_paper_metrics.json", v2_paper)
    diagnostics = {
        "neural": neural_diagnostics,
        "v2": v2_diagnostics,
        "context_transition_gate": context_gate_diagnostics,
        "dev_v2_gate_statistics": dev_v2_gate_statistics.to_dict(),
        "test_v2_gate_statistics": test_v2_gate_statistics.to_dict(),
    }
    write_json(output_dir / "diagnostics.json", diagnostics)

    recommended_path = (
        v2_artifacts["submission_path"]
        if decision["competitive_submission"]
        else None
    )
    selection = {
        "schema_version": 1,
        "system_name": config["system_name"],
        "artifact_prefix": prefix,
        "description": DESCRIPTION,
        "device": str(device),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "dev_metrics": checkpoint.get("dev_metrics"),
        },
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "neural_correct": int(neural_paper["correct_letters"]),
        "neural_accuracy": float(neural_paper["accuracy"]),
        "v2_correct": int(v2_paper["correct_letters"]),
        "v2_accuracy": float(v2_paper["accuracy"]),
        **decision,
        "recommended_submission_path": recommended_path,
        "neural_artifacts": dict(neural_artifacts),
        "v2_artifacts": dict(v2_artifacts),
    }
    write_json(output_dir / "SELECTION.json", selection)
    write_step_manifest(output_dir / "MANIFEST.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/track4/Lyes/context_boundary_v11/evaluation.json"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    run(
        config_path=args.config,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
