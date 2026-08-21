import copy
import itertools
import math

import pytest
import torch

from utils.track4.Lyes.config import DEFAULT_CONFIG, validate_config
from utils.track4.Lyes.labels import IGNORE_INDEX
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer, LinearChainCRF, ModelConfig
from training.track4.Lyes.rdrop import (
    symmetric_crf_marginal_kl,
    symmetric_emission_kl,
    symmetric_log_probability_kl,
)


def _small_crf_model(dropout: float) -> CharDiacritizer:
    return CharDiacritizer(
        ModelConfig(
            vocab_size=8,
            pad_id=0,
            space_id=4,
            bos_id=2,
            eos_id=3,
            architecture="plain_transformer",
            d_model=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            dropout=dropout,
            max_length=16,
            attention_window=4,
            factorized_head=False,
            head_mode="crf",
        )
    )


def test_symmetric_kl_matches_exact_golden_value() -> None:
    probabilities_a = torch.tensor([[[0.75, 0.25]]], dtype=torch.float64)
    probabilities_b = torch.tensor([[[0.50, 0.50]]], dtype=torch.float64)
    actual = symmetric_log_probability_kl(
        probabilities_a.log(),
        probabilities_b.log(),
        torch.tensor([[True]]),
    )
    expected = 0.5 * (
        0.75 * math.log(0.75 / 0.50)
        + 0.25 * math.log(0.25 / 0.50)
        + 0.50 * math.log(0.50 / 0.75)
        + 0.50 * math.log(0.50 / 0.25)
    )
    assert float(actual) == pytest.approx(expected, abs=1e-12)


def test_symmetric_kl_is_symmetric_zero_for_identical_and_masks_spaces() -> None:
    first = torch.tensor(
        [[[0.7, 0.3], [0.2, 0.8]]], dtype=torch.float64
    ).log()
    second = torch.tensor(
        [[[0.6, 0.4], [0.9, 0.1]]], dtype=torch.float64
    ).log()
    mask = torch.tensor([[True, False]])
    forward = symmetric_log_probability_kl(first, second, mask)
    reverse = symmetric_log_probability_kl(second, first, mask)
    changed_masked_value = second.clone()
    changed_masked_value[:, 1] = torch.tensor([0.01, 0.99]).log()
    changed = symmetric_log_probability_kl(first, changed_masked_value, mask)
    assert torch.equal(forward, reverse)
    assert torch.equal(forward, changed)
    assert float(symmetric_log_probability_kl(first, first, mask)) == 0.0


def test_crf_marginals_are_normalized_and_differentiable() -> None:
    crf = LinearChainCRF(num_labels=3)
    emissions = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor(
        [[True, False, True, True], [False, True, True, False]]
    )
    log_marginals = crf.log_marginals(emissions, mask)
    assert torch.allclose(
        log_marginals[mask].logsumexp(dim=-1),
        torch.zeros(int(mask.sum())),
        atol=1e-6,
    )
    loss = log_marginals[mask].square().sum()
    loss.backward()
    assert emissions.grad is not None
    assert torch.isfinite(emissions.grad).all()
    assert crf.transitions.grad is not None
    assert torch.isfinite(crf.transitions.grad).all()


@pytest.mark.parametrize(
    "crf,boundary_mask,transition_gate",
    [
        (LinearChainCRF(3), None, None),
        (
            LinearChainCRF(3, boundary_conditioned=True),
            torch.tensor([[False, False, True, False]]),
            None,
        ),
        (
            LinearChainCRF(3, boundary_rank=2),
            torch.tensor([[False, False, True, False]]),
            None,
        ),
        (
            LinearChainCRF(3, boundary_rank=2, context_conditioned=True),
            None,
            torch.tensor([[0.0, 0.2, 0.8, 0.0]]),
        ),
    ],
)
def test_vectorized_crf_marginals_match_brute_force(
    crf: LinearChainCRF,
    boundary_mask: torch.Tensor,
    transition_gate: torch.Tensor,
) -> None:
    torch.manual_seed(29)
    with torch.no_grad():
        for parameter in crf.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)
    emissions = torch.randn(1, 4, 3, dtype=torch.float32)
    mask = torch.tensor([[False, True, True, False]])
    actual = crf.log_marginals(
        emissions, mask, boundary_mask, transition_gate
    )

    positions = mask[0].nonzero(as_tuple=False).squeeze(1).tolist()
    sequences = list(itertools.product(range(3), repeat=len(positions)))
    scores = []
    for sequence in sequences:
        score = crf.start_transitions[sequence[0]]
        score = score + emissions[0, positions[0], sequence[0]]
        for packed_index in range(1, len(positions)):
            position = positions[packed_index]
            boundary = (
                None
                if boundary_mask is None
                else boundary_mask[:, position]
            )
            gate = (
                None
                if transition_gate is None
                else transition_gate[:, position]
            )
            transition = crf._transition_matrix(boundary, gate)[0]
            score = score + transition[
                sequence[packed_index - 1], sequence[packed_index]
            ]
            score = score + emissions[
                0, position, sequence[packed_index]
            ]
        score = score + crf.end_transitions[sequence[-1]]
        scores.append(score)
    all_scores = torch.stack(scores)
    normalizer = torch.logsumexp(all_scores, dim=0)
    expected = torch.full_like(actual, -math.log(3.0))
    for packed_index, position in enumerate(positions):
        for label in range(3):
            label_scores = torch.stack(
                [
                    score
                    for score, sequence in zip(scores, sequences)
                    if sequence[packed_index] == label
                ]
            )
            expected[0, position, label] = (
                torch.logsumexp(label_scores, dim=0) - normalizer
            )
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_crf_rdrop_cpu_forward_backward_reaches_model_and_crf() -> None:
    torch.manual_seed(7)
    model = _small_crf_model(dropout=0.2)
    model.train()
    input_ids = torch.tensor(
        [[2, 5, 6, 4, 7, 3], [2, 6, 4, 5, 7, 3]], dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    targets = torch.tensor(
        [
            [IGNORE_INDEX, 1, 7, IGNORE_INDEX, 0, IGNORE_INDEX],
            [IGNORE_INDEX, 3, IGNORE_INDEX, 5, 1, IGNORE_INDEX],
        ],
        dtype=torch.long,
    )
    first = model(input_ids, attention_mask)
    second = model(input_ids, attention_mask)
    nll = 0.5 * (
        model.compute_loss(first, targets, 1.0)
        + model.compute_loss(second, targets, 1.0)
    )
    consistency = symmetric_crf_marginal_kl(model, first, second)
    loss = nll + 0.3 * consistency
    loss.backward()
    assert float(consistency.detach()) >= 0.0
    assert model.label_head is not None
    assert model.label_head.weight.grad is not None
    assert torch.isfinite(model.label_head.weight.grad).all()
    assert model.crf is not None
    assert model.crf.transitions.grad is not None
    assert torch.isfinite(model.crf.transitions.grad).all()


def test_crf_rdrop_is_zero_when_dropout_is_disabled() -> None:
    torch.manual_seed(11)
    model = _small_crf_model(dropout=0.0)
    model.train()
    input_ids = torch.tensor([[2, 5, 4, 6, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    first = model(input_ids, attention_mask)
    second = model(input_ids, attention_mask)
    consistency = symmetric_crf_marginal_kl(model, first, second)
    assert float(consistency.detach()) == pytest.approx(0.0, abs=1e-8)


def test_emission_rdrop_cpu_forward_backward_is_finite() -> None:
    torch.manual_seed(13)
    model = _small_crf_model(dropout=0.2)
    model.train()
    input_ids = torch.tensor([[2, 5, 4, 6, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    first = model(input_ids, attention_mask)
    second = model(input_ids, attention_mask)
    consistency = symmetric_emission_kl(first, second)
    consistency.backward()
    assert float(consistency.detach()) >= 0.0
    assert model.label_head is not None
    assert model.label_head.weight.grad is not None
    assert torch.isfinite(model.label_head.weight.grad).all()


def test_rdrop_configuration_is_fail_closed() -> None:
    default = copy.deepcopy(DEFAULT_CONFIG)
    validate_config(default)
    assert default["training"]["rdrop_coefficient"] == 0.0
    assert default["training"]["rdrop_distribution"] == "emission"
    assert default["training"]["dev_evaluation_mode"] == "each_epoch"

    invalid_head = copy.deepcopy(DEFAULT_CONFIG)
    invalid_head["training"]["rdrop_coefficient"] = 0.1
    with pytest.raises(ValueError, match="head_mode=crf"):
        validate_config(invalid_head)

    invalid_selection = copy.deepcopy(DEFAULT_CONFIG)
    invalid_selection["training"]["dev_evaluation_mode"] = "final_only"
    with pytest.raises(ValueError, match="last_epoch"):
        validate_config(invalid_selection)

    valid = copy.deepcopy(DEFAULT_CONFIG)
    valid["model"]["factorized_head"] = False
    valid["model"]["head_mode"] = "crf"
    valid["training"]["rdrop_coefficient"] = 0.3
    valid["training"]["selection_mode"] = "last_epoch"
    valid["training"]["dev_evaluation_mode"] = "final_only"
    validate_config(valid)
