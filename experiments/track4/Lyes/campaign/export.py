"""Evaluate checkpoints/ensembles and export distinctive verified artifacts."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from utils.track4.Lyes.gated_fusion.config import load_gates
from experiments.track4.Lyes.campaign.common import (
    write_prediction_artifacts,
    write_step_manifest,
)
from experiments.track4.Lyes.campaign.diagnostics import (
    prediction_diagnostics,
    training_word_types,
)
from experiments.track4.Lyes.campaign.ensemble import (
    apply_lexical_gate,
    predict_probability_group_ensemble,
    predict_probability_ensemble,
    probabilities_to_predictions,
)
from utils.track4.Lyes.data import load_jsonl, load_raw_sentences
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from utils.track4.Lyes.utils import sha256_file, write_json


def _load_completed_export(
    output_dir: Path,
    artifact_prefix: str,
) -> Mapping[str, Any]:
    manifest_path = (
        output_dir
        / "artifacts"
        / f"{artifact_prefix}_MANIFEST.json"
    )
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_prefix") != artifact_prefix:
        raise RuntimeError(f"artifact prefix mismatch: {manifest_path}")
    for key in ("neural_artifacts", "v2_artifacts"):
        artifacts = manifest.get(key, {})
        for path_key in (
            "vocalized_path",
            "submission_path",
            "official_check_path",
        ):
            path = Path(str(artifacts.get(path_key, "")))
            if not path.is_file():
                raise RuntimeError(
                    f"partial/corrupt export missing {path_key}: "
                    f"{manifest_path}"
                )
        submission = Path(str(artifacts["submission_path"]))
        if sha256_file(submission) != artifacts.get("submission_sha256"):
            raise RuntimeError(
                f"submission hash mismatch: {submission}"
            )
    return manifest


def export_checkpoint_ensemble(
    checkpoint_paths: Sequence[Path],
    output_dir: Path,
    artifact_prefix: str,
    system_name: str,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 0,
    train_path: Path = Path(
        "Data/train_data/train_Algerian-DIAC.jsonl"
    ),
    dev_path: Path = Path("Data/dev_data/dev_Algerian-DIAC.jsonl"),
    input_path: Path = Path("Data/test_data/raw_sentences_test.txt"),
    ids_path: Path = Path("Data/test_data/raw_sentences_test_ids.txt"),
    sample_submission: Path = Path(
        "Data/test_data/sample_submission.csv"
    ),
    gates_path: Path = Path("configs/track4/Lyes/gates.json"),
) -> Mapping[str, Any]:
    completed = _load_completed_export(output_dir, artifact_prefix)
    if completed:
        return completed
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = load_jsonl(train_path)
    dev_records = load_jsonl(dev_path)
    test_records = load_raw_sentences(input_path, ids_path)
    seen_words = training_word_types(train_records)
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(gates_path)

    dev_probabilities, _ = predict_probability_ensemble(
        checkpoint_paths, dev_records, device, batch_size, num_workers
    )
    dev_neural = probabilities_to_predictions(
        dev_records, dev_probabilities
    )
    dev_v2 = apply_lexical_gate(
        dev_records, dev_probabilities, prior, gates
    )
    diagnostics: Dict[str, Any] = {
        "neural": prediction_diagnostics(
            dev_records, dev_neural, seen_words
        ),
        "v2": prediction_diagnostics(dev_records, dev_v2, seen_words),
    }
    write_json(output_dir / "diagnostics.json", diagnostics)

    test_probabilities, _ = predict_probability_ensemble(
        checkpoint_paths, test_records, device, batch_size, num_workers
    )
    test_neural = probabilities_to_predictions(
        test_records, test_probabilities
    )
    test_v2 = apply_lexical_gate(
        test_records, test_probabilities, prior, gates
    )
    artifacts_dir = output_dir / "artifacts"
    neural_artifacts = write_prediction_artifacts(
        artifacts_dir,
        f"{artifact_prefix}_NEURAL",
        test_records,
        test_neural,
        sample_submission,
        ids_path,
        input_path,
    )
    v2_artifacts = write_prediction_artifacts(
        artifacts_dir,
        f"{artifact_prefix}_V2",
        test_records,
        test_v2,
        sample_submission,
        ids_path,
        input_path,
    )
    manifest: Dict[str, Any] = {
        "system_name": system_name,
        "artifact_prefix": artifact_prefix,
        "checkpoints": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in checkpoint_paths
        ],
        "aggregation": "equal_arithmetic_probability_mean",
        "dev": diagnostics,
        "neural_artifacts": dict(neural_artifacts),
        "v2_artifacts": dict(v2_artifacts),
    }
    write_step_manifest(
        artifacts_dir / f"{artifact_prefix}_MANIFEST.json", manifest
    )
    return manifest


def export_architecture_ensemble(
    checkpoint_groups: Sequence[Sequence[Path]],
    output_dir: Path,
    artifact_prefix: str,
    system_name: str,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 0,
    train_path: Path = Path(
        "Data/train_data/train_Algerian-DIAC.jsonl"
    ),
    dev_path: Path = Path("Data/dev_data/dev_Algerian-DIAC.jsonl"),
    input_path: Path = Path("Data/test_data/raw_sentences_test.txt"),
    ids_path: Path = Path("Data/test_data/raw_sentences_test_ids.txt"),
    sample_submission: Path = Path(
        "Data/test_data/sample_submission.csv"
    ),
    gates_path: Path = Path("configs/track4/Lyes/gates.json"),
) -> Mapping[str, Any]:
    """Export an equal architecture ensemble with seed means as experts."""
    completed = _load_completed_export(output_dir, artifact_prefix)
    if completed:
        return completed
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = load_jsonl(train_path)
    dev_records = load_jsonl(dev_path)
    test_records = load_raw_sentences(input_path, ids_path)
    seen_words = training_word_types(train_records)
    prior = WordLabelPrior().fit(train_records)
    gates = load_gates(gates_path)

    dev_probabilities, _ = predict_probability_group_ensemble(
        checkpoint_groups,
        dev_records,
        device,
        batch_size,
        num_workers,
    )
    dev_neural = probabilities_to_predictions(
        dev_records, dev_probabilities
    )
    dev_v2 = apply_lexical_gate(
        dev_records, dev_probabilities, prior, gates
    )
    diagnostics: Dict[str, Any] = {
        "neural": prediction_diagnostics(
            dev_records, dev_neural, seen_words
        ),
        "v2": prediction_diagnostics(dev_records, dev_v2, seen_words),
    }
    write_json(output_dir / "diagnostics.json", diagnostics)

    test_probabilities, _ = predict_probability_group_ensemble(
        checkpoint_groups,
        test_records,
        device,
        batch_size,
        num_workers,
    )
    test_neural = probabilities_to_predictions(
        test_records, test_probabilities
    )
    test_v2 = apply_lexical_gate(
        test_records, test_probabilities, prior, gates
    )
    artifacts_dir = output_dir / "artifacts"
    neural_artifacts = write_prediction_artifacts(
        artifacts_dir,
        f"{artifact_prefix}_NEURAL",
        test_records,
        test_neural,
        sample_submission,
        ids_path,
        input_path,
    )
    v2_artifacts = write_prediction_artifacts(
        artifacts_dir,
        f"{artifact_prefix}_V2",
        test_records,
        test_v2,
        sample_submission,
        ids_path,
        input_path,
    )
    manifest: Dict[str, Any] = {
        "system_name": system_name,
        "artifact_prefix": artifact_prefix,
        "checkpoint_groups": [
            [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in group
            ]
            for group in checkpoint_groups
        ],
        "aggregation": (
            "equal_seed_mean_within_architecture_then_"
            "equal_architecture_probability_mean"
        ),
        "dev": diagnostics,
        "neural_artifacts": dict(neural_artifacts),
        "v2_artifacts": dict(v2_artifacts),
    }
    write_step_manifest(
        artifacts_dir / f"{artifact_prefix}_MANIFEST.json", manifest
    )
    return manifest
