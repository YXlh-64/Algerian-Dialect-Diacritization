import json
from pathlib import Path

from utils.track4.Lyes.data import SentenceRecord
from experiments.track4.Lyes.rdrop_v13 import (
    _coefficient_key,
    _round_half_up,
    _split_records,
    load_campaign_config,
)


def _records(count: int):
    return [
        SentenceRecord(
            sent_id="s{:03d}".format(index),
            chars=tuple("ب" * (index % 7 + 1)),
            labels=tuple([index % 16] * (index % 7 + 1)),
            input_text="ب" * (index % 7 + 1),
        )
        for index in range(count)
    ]


def test_calibration_split_is_deterministic_disjoint_and_complete() -> None:
    records = _records(31)
    first_train, first_calibration, first_manifest = _split_records(
        records, 5, 1301
    )
    second_train, second_calibration, second_manifest = _split_records(
        records, 5, 1301
    )
    assert first_train == second_train
    assert first_calibration == second_calibration
    assert first_manifest == second_manifest
    train_ids = {record.sent_id for record in first_train}
    calibration_ids = {record.sent_id for record in first_calibration}
    assert not train_ids & calibration_ids
    assert train_ids | calibration_ids == {
        record.sent_id for record in records
    }


def test_epoch_lock_uses_decimal_half_up() -> None:
    assert _round_half_up([20, 21]) == 21
    assert _round_half_up([19, 20]) == 20
    assert _round_half_up([21, 21]) == 21


def test_coefficient_artifact_names_are_stable() -> None:
    assert _coefficient_key(0.0) == "lambda_0"
    assert _coefficient_key(0.1) == "lambda_0p1"
    assert _coefficient_key(1.0) == "lambda_1"


def test_campaign_config_rejects_coefficient_search_drift(
    tmp_path: Path,
) -> None:
    source = Path("configs/track4/Lyes/rdrop_v13/campaign.json")
    valid = load_campaign_config(source)
    assert valid["coefficients"] == [0.0, 0.1, 0.3, 1.0]
    invalid = dict(valid)
    invalid["coefficients"] = [0.0, 0.2]
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    try:
        load_campaign_config(path)
    except ValueError as error:
        assert "coefficients" in str(error)
    else:
        raise AssertionError("drifted coefficient grid was accepted")
