from pathlib import Path

from experiments.track4.Lyes.architecture_v10_snapshot import load_snapshot_config


def test_snapshot_campaign_config_is_controlled() -> None:
    config = load_snapshot_config(
        Path("configs/track4/Lyes/architecture_v10/campaign.json")
    )
    snapshot = config["snapshot"]
    assert len(snapshot["checkpoints"]) == 2
    assert snapshot["control_v2_correct"] == 14962
    assert snapshot["minimum_correct_gain"] == 10
    assert snapshot["production_correct"] == 14977
