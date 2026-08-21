from pathlib import Path

import torch

from utils.track4.Lyes.config import load_config
from utils.track4.Lyes.labels import IGNORE_INDEX
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer, LinearChainCRF, ModelConfig


CONFIG_PATH = Path(
    "configs/track4/Lyes/architecture_v10/low_rank_boundary_crf_v10.json"
)


def _model() -> CharDiacritizer:
    resolved = load_config(CONFIG_PATH)
    config = ModelConfig.from_mapping(
        resolved["model"],
        vocab_size=43,
        pad_id=0,
        space_id=4,
        bos_id=2,
        eos_id=3,
    )
    return CharDiacritizer(config)


def test_low_rank_boundary_is_rank_two_residual_with_64_parameters() -> None:
    model = _model()
    assert model.config.resolved_head_mode == "low_rank_boundary_crf"
    assert model.crf is not None
    assert model.crf.boundary_transitions is None
    assert model.crf.boundary_left is not None
    assert model.crf.boundary_right is not None
    assert tuple(model.crf.boundary_left.shape) == (16, 2)
    assert tuple(model.crf.boundary_right.shape) == (2, 16)
    assert sum(parameter.numel() for parameter in model.parameters()) == 9_890_160
    assert torch.equal(
        model.crf._boundary_transition_matrix(), model.crf.transitions
    )


def test_low_rank_boundary_crf_forward_backward_and_boundary_mask() -> None:
    model = _model()
    input_ids = torch.tensor([[2, 10, 11, 4, 12, 13, 3]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    targets = torch.tensor(
        [[IGNORE_INDEX, 1, 9, IGNORE_INDEX, 7, 0, IGNORE_INDEX]]
    )
    outputs = model(input_ids, attention_mask)
    assert outputs["crf_boundary_mask"].tolist() == [
        [False, False, False, False, True, False, False]
    ]
    loss = model.compute_loss(outputs, targets, shadda_loss_weight=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.crf.boundary_left.grad is not None
    assert model.crf.boundary_right.grad is not None


def test_crf_rejects_ambiguous_full_and_low_rank_boundary_modes() -> None:
    try:
        LinearChainCRF(16, boundary_conditioned=True, boundary_rank=2)
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("ambiguous boundary modes must fail")
