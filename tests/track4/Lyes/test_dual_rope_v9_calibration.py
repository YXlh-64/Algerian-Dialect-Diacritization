from pathlib import Path

import torch

from utils.track4.Lyes.data import SentenceRecord
from experiments.track4.Lyes.dual_rope_v9_calibration import (
    flatten_scored_probabilities,
    load_calibration_config,
    robust_gate,
)


def test_v9_calibration_config_is_strict() -> None:
    config = load_calibration_config(
        Path("configs/track4/Lyes/dual_rope_v9/campaign.json")
    )
    assert len(config["production_groups"]) == 4
    assert config["calibration"]["minimum_correct_gain"] == 10
    assert config["calibration"]["minimum_improved_folds"] == 4
    assert config["calibration"]["maximum_regressed_folds"] == 0


def test_flatten_scored_probabilities_excludes_spaces() -> None:
    records = [
        SentenceRecord(
            sent_id="000001",
            chars=tuple("ب ت"),
            labels=(1, 0, 7),
            input_text="ب ت",
        )
    ]
    groups = [
        [torch.full((3, 16), 1.0 / 16)],
        [torch.full((3, 16), 1.0 / 16)],
    ]
    probabilities, targets = flatten_scored_probabilities(
        records, groups, [0]
    )
    assert probabilities.shape == (2, 2, 16)
    assert targets.tolist() == [1, 7]


def test_robust_gate_requires_gain_fold_consistency_and_no_regression() -> None:
    accepted = robust_gate(100, 111, [3, 2, 4, 2, 0], 10, 4, 0)
    assert accepted["accepted"] is True
    insufficient_gain = robust_gate(100, 109, [3, 2, 2, 2, 0], 10, 4, 0)
    assert insufficient_gain["accepted"] is False
    regression = robust_gate(100, 111, [4, 4, 4, 4, -5], 10, 4, 0)
    assert regression["accepted"] is False
