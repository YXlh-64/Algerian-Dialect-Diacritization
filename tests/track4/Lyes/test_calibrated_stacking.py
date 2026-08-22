import torch

from experiments.track4.Lyes.calibrated_stacking import (
    CalibratedStacker,
    fit_calibrated_stacker,
    stack_record_probabilities,
)


def test_calibrated_stacker_constraints_and_determinism() -> None:
    probabilities = torch.tensor(
        [
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.20, 0.80],
                [0.10, 0.90],
            ],
            [
                [0.55, 0.45],
                [0.45, 0.55],
                [0.55, 0.45],
                [0.45, 0.55],
            ],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 0, 1, 1])
    first = fit_calibrated_stacker(probabilities, targets)
    second = fit_calibrated_stacker(probabilities, targets)
    assert torch.equal(first.log_temperatures, second.log_temperatures)
    assert torch.equal(first.weight_logits, second.weight_logits)
    assert torch.all(first.temperatures > 0.0)
    assert torch.all(first.weights >= 0.0)
    assert torch.allclose(
        first.weights.sum(),
        torch.tensor(1.0, dtype=first.weights.dtype),
    )
    combined = first.combine(probabilities)
    assert torch.equal(combined.argmax(dim=-1), targets)
    assert torch.allclose(
        combined.sum(dim=-1),
        torch.ones(combined.size(0), dtype=combined.dtype),
    )


def test_equal_unity_stacker_reproduces_arithmetic_mean() -> None:
    probabilities = torch.tensor(
        [
            [[0.8, 0.2], [0.4, 0.6]],
            [[0.2, 0.8], [0.6, 0.4]],
        ]
    )
    stacker = CalibratedStacker(
        log_temperatures=torch.zeros(2),
        weight_logits=torch.zeros(2),
    )
    assert torch.allclose(stacker.combine(probabilities), probabilities.mean(0))


def test_record_stacking_preserves_record_shapes() -> None:
    groups = [
        [torch.tensor([[0.8, 0.2]]), torch.tensor([[0.3, 0.7]])],
        [torch.tensor([[0.2, 0.8]]), torch.tensor([[0.9, 0.1]])],
    ]
    stacker = CalibratedStacker(torch.zeros(2), torch.zeros(2))
    result = stack_record_probabilities(groups, stacker)
    assert len(result) == 2
    assert torch.allclose(result[0], torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(result[1], torch.tensor([[0.6, 0.4]]))
