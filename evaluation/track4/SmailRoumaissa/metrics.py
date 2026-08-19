from typing import List


class MicroF1Accumulator:
    """Micro-F1 over a single-label multi-class problem reduces to plain
    accuracy (pooled TP == pooled correct predictions, and every character
    contributes exactly one TP+FN and one TP+FP)."""
    
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, preds: List[int], golds: List[int]):
        for p, g in zip(preds, golds):
            self.total += 1
            if p == g:
                self.correct += 1

    @property
    def score(self) -> float:
        return self.correct / max(self.total, 1)
