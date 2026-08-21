import copy

from utils.track4.Lyes.config import DEFAULT_CONFIG, validate_config


def test_last_epoch_selection_mode_is_valid_for_final_refit() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["training"]["selection_mode"] = "last_epoch"
    validate_config(config)


def test_unknown_selection_mode_is_rejected() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["training"]["selection_mode"] = "leaky_best"
    try:
        validate_config(config)
    except ValueError as error:
        assert "selection_mode" in str(error)
    else:
        raise AssertionError("unknown selection mode must fail")
