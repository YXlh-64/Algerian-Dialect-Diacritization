"""Deterministic temperature calibration and simplex architecture stacking."""

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CalibratedStacker:
    log_temperatures: torch.Tensor
    weight_logits: torch.Tensor

    @property
    def temperatures(self) -> torch.Tensor:
        return self.log_temperatures.exp()

    @property
    def weights(self) -> torch.Tensor:
        return self.weight_logits.softmax(dim=0)

    def combine(self, group_probabilities: torch.Tensor) -> torch.Tensor:
        if group_probabilities.ndim != 3:
            raise ValueError(
                "group probabilities must have shape [groups, items, labels]"
            )
        if group_probabilities.size(0) != self.log_temperatures.numel():
            raise ValueError("calibrator group count mismatch")
        log_probabilities = group_probabilities.clamp_min(1.0e-12).log()
        calibrated = F.softmax(
            log_probabilities / self.temperatures[:, None, None],
            dim=-1,
        )
        return (
            calibrated * self.weights[:, None, None]
        ).sum(dim=0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperatures": self.temperatures.tolist(),
            "weights": self.weights.tolist(),
            "log_temperatures": self.log_temperatures.tolist(),
            "weight_logits": self.weight_logits.tolist(),
            "objective": "multiclass_negative_log_likelihood",
            "weight_constraint": "softmax_nonnegative_sum_one",
        }


def fit_calibrated_stacker(
    group_probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> CalibratedStacker:
    if group_probabilities.ndim != 3:
        raise ValueError(
            "group probabilities must have shape [groups, items, labels]"
        )
    if targets.ndim != 1 or targets.numel() != group_probabilities.size(1):
        raise ValueError("stacking targets must match item count")
    if group_probabilities.size(0) < 2:
        raise ValueError("stacking requires at least two architecture groups")
    if group_probabilities.size(2) < 2:
        raise ValueError("stacking requires at least two labels")
    if not torch.isfinite(group_probabilities).all():
        raise ValueError("group probabilities must be finite")
    if group_probabilities.lt(0).any():
        raise ValueError("group probabilities cannot be negative")
    if targets.lt(0).any() or targets.ge(group_probabilities.size(2)).any():
        raise ValueError("stacking targets are outside label range")

    values = group_probabilities.to(dtype=torch.float64)
    labels = targets.to(dtype=torch.long)
    group_count = values.size(0)
    log_temperatures = torch.zeros(
        group_count, dtype=torch.float64, requires_grad=True
    )
    weight_logits = torch.zeros(
        group_count, dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [log_temperatures, weight_logits],
        max_iter=150,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperatures = log_temperatures.clamp(-3.0, 3.0).exp()
        weights = weight_logits.softmax(dim=0)
        calibrated = F.softmax(
            values.clamp_min(1.0e-12).log()
            / temperatures[:, None, None],
            dim=-1,
        )
        combined = (calibrated * weights[:, None, None]).sum(dim=0)
        loss = F.nll_loss(combined.clamp_min(1.0e-12).log(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return CalibratedStacker(
        log_temperatures.detach().clamp(-3.0, 3.0),
        weight_logits.detach(),
    )


def stack_record_probabilities(
    group_probabilities: Sequence[Sequence[torch.Tensor]],
    stacker: CalibratedStacker,
) -> list[torch.Tensor]:
    if not group_probabilities:
        raise ValueError("at least one group is required")
    record_count = len(group_probabilities[0])
    if any(len(group) != record_count for group in group_probabilities):
        raise ValueError("stacking groups must have aligned records")
    return [
        stacker.combine(
            torch.stack(
                [group[index] for group in group_probabilities], dim=0
            ).to(dtype=torch.float64)
        ).to(dtype=torch.float32)
        for index in range(record_count)
    ]
