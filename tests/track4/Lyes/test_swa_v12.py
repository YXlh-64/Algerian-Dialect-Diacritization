import random
from pathlib import Path

import pytest
import torch

from experiments.track4.Lyes.swa_v12 import (
    _restore_rng_state,
    average_model_state_dicts,
    load_swa_config,
    update_prefix_average,
)


def test_swa_campaign_is_controlled() -> None:
    config = load_swa_config(Path("configs/track4/Lyes/swa_v12/campaign.json"))
    assert set(config["systems"]) == {"crf_v7", "boundary_crf_v8"}
    for system in config["systems"].values():
        assert system["tail_epochs"] == 8
        assert system["maximum_tail_epochs"] == 12
        assert system["baseline"]["minimum_correct_gain"] == 10


def test_checkpoint_weight_average_is_exact_and_deterministic() -> None:
    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor(2, dtype=torch.int64),
    }
    second = {
        "weight": torch.tensor([3.0, 7.0]),
        "counter": torch.tensor(2, dtype=torch.int64),
    }
    expected = torch.tensor([2.0, 5.0])
    one = average_model_state_dicts([first, second])
    two = average_model_state_dicts([first, second])
    assert torch.equal(one["weight"], expected)
    assert torch.equal(one["weight"], two["weight"])
    assert one["counter"].item() == 2


def test_checkpoint_weight_average_rejects_different_nonfloat_state() -> None:
    first = {"counter": torch.tensor(1, dtype=torch.int64)}
    second = {"counter": torch.tensor(2, dtype=torch.int64)}
    with pytest.raises(ValueError, match="non-floating state differs"):
        average_model_state_dicts([first, second])


def test_prefix_average_matches_direct_three_member_mean() -> None:
    first = {"weight": torch.tensor([1.0, 4.0])}
    second = {"weight": torch.tensor([2.0, 5.0])}
    third = {"weight": torch.tensor([6.0, 9.0])}
    first_two = average_model_state_dicts([first, second])
    prefix = update_prefix_average(first_two, third, member_count=3)
    direct = average_model_state_dicts([first, second, third])
    assert torch.equal(prefix["weight"], direct["weight"])


def test_restore_rng_state_requires_and_restores_cpu_byte_tensor() -> None:
    saved_python = random.getstate()
    saved_torch = torch.get_rng_state().clone()
    random.seed(991)
    torch.manual_seed(991)

    _restore_rng_state(
        {
            "python": saved_python,
            "torch_cpu": saved_torch.to(dtype=torch.int16),
        },
        torch.device("cpu"),
    )

    assert random.getstate() == saved_python
    assert torch.equal(torch.get_rng_state(), saved_torch)


def test_restore_rng_state_rejects_non_tensor_state() -> None:
    with pytest.raises(TypeError, match="CPU RNG state"):
        _restore_rng_state(
            {"python": random.getstate(), "torch_cpu": [1, 2, 3]},
            torch.device("cpu"),
        )
