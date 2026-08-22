"""Strict campaign configuration and artifact helpers."""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from utils.track4.Lyes.data import SentenceRecord
from evaluation.track4.Lyes.submission import write_submission, write_vocalized_predictions
from utils.track4.Lyes.utils import sha256_file, write_json


ARTIFACT_PATTERN = re.compile(r"[A-Z0-9_]+")


def load_campaign_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "schema_version",
        "output_root",
        "seeds",
        "uniform_checkpoints",
        "experiments",
        "gates",
        "oof_folds",
        "execution",
    }
    if set(config) != required:
        raise ValueError("invalid campaign configuration keys")
    if int(config["schema_version"]) != 1:
        raise ValueError("unsupported campaign schema")
    if list(config["seeds"]) != [42, 43, 44]:
        raise ValueError("campaign seeds must be [42, 43, 44]")
    if int(config["oof_folds"]) != 5:
        raise ValueError("campaign requires five OOF folds")
    execution = config["execution"]
    if set(execution) != {
        "run_oof_gate",
        "oof_deferred_reason",
    }:
        raise ValueError("invalid campaign execution configuration")
    if not isinstance(execution["run_oof_gate"], bool):
        raise ValueError("run_oof_gate must be boolean")
    if not isinstance(execution["oof_deferred_reason"], str):
        raise ValueError("oof_deferred_reason must be a string")
    return config


def validate_artifact_prefix(prefix: str) -> str:
    if not ARTIFACT_PATTERN.fullmatch(prefix):
        raise ValueError("artifact prefix must contain only A-Z, 0-9, _")
    return prefix


def write_prediction_artifacts(
    output_dir: Path,
    prefix: str,
    records: Sequence[SentenceRecord],
    predictions: Sequence[Sequence[int]],
    sample_submission: Path,
    ids_path: Path,
    input_path: Path,
) -> Mapping[str, str]:
    validate_artifact_prefix(prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocalized = output_dir / f"{prefix}_TEST_VOCALIZED.txt"
    submission = output_dir / f"{prefix}_SUBMISSION.csv"
    official = output_dir / f"{prefix}_OFFICIAL_CHECK.csv"
    write_vocalized_predictions(vocalized, records, predictions)
    write_submission(
        submission,
        records,
        predictions,
        sample_submission_path=sample_submission,
    )
    subprocess.run(
        [
            "python",
            "Data/test_data/make_submission.py",
            "--ids",
            str(ids_path),
            "--input",
            str(input_path),
            "--pred",
            str(vocalized),
            "--out",
            str(official),
        ],
        check=True,
    )
    if submission.read_bytes() != official.read_bytes():
        raise RuntimeError("official submission verification failed")
    return {
        "vocalized_path": str(vocalized),
        "submission_path": str(submission),
        "official_check_path": str(official),
        "submission_sha256": sha256_file(submission),
    }


def write_step_manifest(path: Path, values: Mapping[str, Any]) -> None:
    write_json(path, {"schema_version": 1, **dict(values)})
