from pathlib import Path

import torch

from utils.track4.Lyes.config import load_config
from utils.track4.Lyes.labels import IGNORE_INDEX
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer, ModelConfig


CONFIG_PATH = Path(
    "configs/track4/Lyes/architecture_v10/factorized_emission_crf_v10.json"
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


def test_factorized_emission_crf_has_only_two_emission_heads() -> None:
    model = _model()
    assert model.config.resolved_head_mode == "factorized_crf"
    assert model.base_head is not None
    assert model.shadda_head is not None
    assert model.label_head is None
    assert model.crf is not None
    assert model.crf.boundary_transitions is None
    assert sum(parameter.numel() for parameter in model.parameters()) == 9_888_554


def test_factorized_emissions_are_normalized_and_train_with_crf_nll() -> None:
    model = _model()
    input_ids = torch.tensor([[2, 10, 11, 4, 12, 13, 3]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    targets = torch.tensor(
        [[IGNORE_INDEX, 1, 9, IGNORE_INDEX, 7, 0, IGNORE_INDEX]]
    )
    outputs = model(input_ids, attention_mask)
    probabilities = outputs["logits"].exp().sum(dim=-1)
    assert torch.allclose(probabilities, torch.ones_like(probabilities))
    loss = model.compute_loss(outputs, targets, shadda_loss_weight=99.0)
    loss.backward()
    decoded = model.decode_outputs(outputs)
    assert torch.isfinite(loss)
    assert decoded.shape == input_ids.shape
    assert model.base_head.weight.grad is not None
    assert model.shadda_head.weight.grad is not None
    assert model.crf.transitions.grad is not None
