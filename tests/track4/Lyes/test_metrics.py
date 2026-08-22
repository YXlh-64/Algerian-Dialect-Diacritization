import pytest
import torch

from evaluation.track4.Lyes.metrics import MetricAccumulator


def test_metric_ignores_masked_positions_and_equals_micro_f1() -> None:
    predictions = torch.tensor([[1, 15, 7, 0, 5]])
    targets = torch.tensor([[1, -100, 0, -100, 5]])
    metrics = MetricAccumulator()
    metrics.update(predictions, targets, loss=0.75)
    result = metrics.compute()
    assert result["correct"] == 2
    assert result["total"] == 3
    assert result["accuracy"] == pytest.approx(2.0 / 3.0)
    assert result["micro_f1"] == pytest.approx(2.0 / 3.0)
    assert result["loss"] == pytest.approx(0.75)
