"""Official letter-only Micro-F1/accuracy accumulation."""

from typing import Any, Dict, List

import torch

from utils.track4.Lyes.labels import IGNORE_INDEX, LABEL_NAMES, NUM_LABELS


class MetricAccumulator:
    def __init__(self) -> None:
        self.confusion = torch.zeros(
            (NUM_LABELS, NUM_LABELS), dtype=torch.long
        )
        self.loss_sum = 0.0
        self.loss_weight = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float = 0.0,
    ) -> None:
        predictions = predictions.detach().reshape(-1).to("cpu")
        targets = targets.detach().reshape(-1).to("cpu")
        mask = targets.ne(IGNORE_INDEX)
        valid_predictions = predictions[mask]
        valid_targets = targets[mask]
        if valid_targets.numel() == 0:
            return
        if torch.any(valid_predictions.lt(0)) or torch.any(
            valid_predictions.ge(NUM_LABELS)
        ):
            raise ValueError("predictions contain labels outside [0, 15]")
        encoded = valid_targets * NUM_LABELS + valid_predictions
        counts = torch.bincount(
            encoded, minlength=NUM_LABELS * NUM_LABELS
        ).reshape(NUM_LABELS, NUM_LABELS)
        self.confusion += counts
        count = int(valid_targets.numel())
        self.loss_sum += float(loss) * count
        self.loss_weight += count

    def compute(self) -> Dict[str, Any]:
        total = int(self.confusion.sum().item())
        correct = int(torch.diag(self.confusion).sum().item())
        accuracy = correct / total if total else 0.0
        per_label: List[Dict[str, Any]] = []
        for label, name in enumerate(LABEL_NAMES):
            support = int(self.confusion[label].sum().item())
            true_positive = int(self.confusion[label, label].item())
            recall = true_positive / support if support else None
            per_label.append(
                {
                    "label": label,
                    "name": name,
                    "support": support,
                    "recall": recall,
                }
            )
        return {
            "loss": self.loss_sum / self.loss_weight if self.loss_weight else 0.0,
            "accuracy": accuracy,
            "micro_f1": accuracy,
            "correct": correct,
            "total": total,
            "per_label": per_label,
            "confusion_matrix": self.confusion.tolist(),
        }
