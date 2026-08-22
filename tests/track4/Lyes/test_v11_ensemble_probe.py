import json
from pathlib import Path

import pytest

from experiments.track4.Lyes.v11_ensemble_probe import load_probe_config


CONFIG_PATH = Path("configs/track4/Lyes/context_boundary_v11/ensemble_probe.json")


def test_v11_ensemble_probe_contract_is_fixed_and_equal_grouped() -> None:
    config = load_probe_config(CONFIG_PATH)

    assert config["baseline_v2_correct"] == 14_977
    assert config["minimum_meaningful_gain"] == 10
    assert config["artifact_prefix"] == (
        "DZIRI_FINAL_V7_PLUS_CONTEXT_V11_EQUAL5_PROBE"
    )
    assert config["v11_checkpoint"].endswith(
        "context_boundary_v11/01_seed42/best.pt"
    )


def test_v11_ensemble_probe_rejects_uncontrolled_artifact_prefix(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["artifact_prefix"] = "uncontrolled-weight"
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_prefix"):
        load_probe_config(config_path)
