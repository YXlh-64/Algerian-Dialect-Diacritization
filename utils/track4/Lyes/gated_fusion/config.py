"""Strict loading of the authoritative V2 confidence gates."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


EXPECTED_KEYS = {
    "schema_version",
    "system_name",
    "artifact_prefix",
    "confidence_measure",
    "neural_confidence_threshold",
    "lexical_confidence_threshold",
    "lexical_smoothing",
}


@dataclass(frozen=True)
class GatedFusionConfig:
    schema_version: int
    system_name: str
    artifact_prefix: str
    confidence_measure: str
    neural_confidence_threshold: float
    lexical_confidence_threshold: float
    lexical_smoothing: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_gates(path: Path) -> GatedFusionConfig:
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("V2 gates must be a JSON object")
    actual_keys = set(raw)
    if actual_keys != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - actual_keys)
        unknown = sorted(actual_keys - EXPECTED_KEYS)
        raise ValueError(
            "invalid V2 gate keys; missing={} unknown={}".format(
                missing, unknown
            )
        )

    config = GatedFusionConfig(
        schema_version=int(raw["schema_version"]),
        system_name=str(raw["system_name"]),
        artifact_prefix=str(raw["artifact_prefix"]),
        confidence_measure=str(raw["confidence_measure"]),
        neural_confidence_threshold=float(
            raw["neural_confidence_threshold"]
        ),
        lexical_confidence_threshold=float(
            raw["lexical_confidence_threshold"]
        ),
        lexical_smoothing=float(raw["lexical_smoothing"]),
    )
    if config.schema_version != 1:
        raise ValueError("unsupported V2 gate schema")
    if not config.system_name:
        raise ValueError("system_name cannot be empty")
    if not re.fullmatch(r"[A-Z0-9_]+", config.artifact_prefix):
        raise ValueError(
            "artifact_prefix must contain only A-Z, 0-9, and underscore"
        )
    if config.confidence_measure != "max_softmax_probability":
        raise ValueError("unsupported confidence_measure")
    if not 0.0 < config.neural_confidence_threshold < 1.0:
        raise ValueError("neural confidence threshold must be in (0, 1)")
    if not 0.0 < config.lexical_confidence_threshold <= 1.0:
        raise ValueError("lexical confidence threshold must be in (0, 1]")
    if config.lexical_smoothing <= 0.0:
        raise ValueError("lexical_smoothing must be positive")
    return config
