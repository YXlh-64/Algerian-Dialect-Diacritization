import copy
import json
from pathlib import Path

import pytest

from experiments.track4.Lyes.export_ensemble import (
    checkpoint_groups,
    load_v7_config,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "track4" / "Lyes" / "campaign.json"


def test_v7_config_and_seed42_groups_are_strict() -> None:
    config = load_v7_config(CONFIG_PATH)
    groups = checkpoint_groups(config, "seed42")
    assert [len(group) for group in groups] == [1, 3, 5]
    assert groups[0][0].name == "best.pt"


def test_v7_multiseed_fails_until_all_seed_checkpoints_exist() -> None:
    config = load_v7_config(CONFIG_PATH)
    missing = [
        seed
        for seed, path in config["dual_seed_checkpoints"].items()
        if not (ROOT / path).is_file()
    ]
    if missing:
        with pytest.raises(FileNotFoundError, match="DualRoPE"):
            checkpoint_groups(config, "multiseed")
    else:
        assert [
            len(group)
            for group in checkpoint_groups(config, "multiseed")
        ] == [
            3,
            3,
            5,
        ]
        assert [
            len(group)
            for group in checkpoint_groups(config, "dual_multiseed")
        ] == [3]
        assert [
            len(group)
            for group in checkpoint_groups(config, "crf_final")
        ] == [1, 3, 3, 5]


def test_v7_config_rejects_unknown_acceptance_gate(
    tmp_path: Path,
) -> None:
    config = load_v7_config(CONFIG_PATH)
    invalid = copy.deepcopy(config)
    invalid["acceptance"]["unknown"] = 1
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="acceptance"):
        load_v7_config(invalid_path)
