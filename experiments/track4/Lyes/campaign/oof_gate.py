"""Explainable OOF-trained neural-versus-lexical logistic gate."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from utils.track4.Lyes.data import SentenceRecord
from utils.track4.Lyes.lexical_fusion import WordLabelPrior, iter_words


FEATURE_NAMES = (
    "neural_confidence",
    "neural_margin",
    "normalized_entropy",
    "model_disagreement",
    "lexical_confidence",
    "log_word_frequency",
    "normalized_position",
    "clipped_word_length",
)


@dataclass(frozen=True)
class LogisticGate:
    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor

    def probability(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (features - self.mean) / self.scale
        return torch.sigmoid(standardized @ self.weight + self.bias)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weight": self.weight.tolist(),
            "bias": float(self.bias.item()),
            "threshold": 0.5,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "LogisticGate":
        if tuple(values["feature_names"]) != FEATURE_NAMES:
            raise ValueError("OOF gate feature schema mismatch")
        return cls(
            mean=torch.tensor(values["mean"], dtype=torch.float64),
            scale=torch.tensor(values["scale"], dtype=torch.float64),
            weight=torch.tensor(values["weight"], dtype=torch.float64),
            bias=torch.tensor(float(values["bias"]), dtype=torch.float64),
        )


def _letter_features(
    probability: torch.Tensor,
    model_predictions: torch.Tensor,
    lexical_probability: torch.Tensor,
    frequency: int,
    position: int,
    word_length: int,
) -> torch.Tensor:
    top = probability.topk(k=2)
    entropy = -(
        probability.clamp_min(1.0e-12)
        * probability.clamp_min(1.0e-12).log()
    ).sum() / math.log(probability.numel())
    votes = torch.bincount(model_predictions, minlength=probability.numel())
    disagreement = 1.0 - float(votes.max().item()) / model_predictions.numel()
    return torch.tensor(
        [
            float(top.values[0]),
            float(top.values[0] - top.values[1]),
            float(entropy),
            disagreement,
            float(lexical_probability.max()),
            math.log1p(frequency),
            position / max(1, word_length - 1),
            min(word_length, 20) / 20.0,
        ],
        dtype=torch.float64,
    )


def collect_training_examples(
    records: Sequence[SentenceRecord],
    ensemble_probabilities: Sequence[torch.Tensor],
    member_predictions: Sequence[torch.Tensor],
    prior: WordLabelPrior,
    smoothing: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features: List[torch.Tensor] = []
    targets: List[float] = []
    for record, probabilities, member_votes in zip(
        records, ensemble_probabilities, member_predictions
    ):
        if record.labels is None:
            raise ValueError("OOF gate training requires labels")
        for start, end, word in iter_words(record.chars):
            lexical_log = prior.log_probabilities(word, smoothing)
            if lexical_log is None:
                continue
            lexical_probability = lexical_log.exp()
            frequency = prior.observation_count(word)
            word_length = end - start
            for position in range(word_length):
                absolute = start + position
                neural = int(probabilities[absolute].argmax())
                lexical = int(lexical_probability[position].argmax())
                if neural == lexical:
                    continue
                target = int(record.labels[absolute])
                neural_correct = neural == target
                lexical_correct = lexical == target
                if neural_correct == lexical_correct:
                    continue
                features.append(
                    _letter_features(
                        probabilities[absolute],
                        member_votes[:, absolute],
                        lexical_probability[position],
                        frequency,
                        position,
                        word_length,
                    )
                )
                targets.append(float(lexical_correct))
    if not features or len(set(targets)) != 2:
        raise ValueError("OOF gate needs both disagreement outcomes")
    return torch.stack(features), torch.tensor(targets, dtype=torch.float64)


def fit_logistic_gate(
    features: torch.Tensor, targets: torch.Tensor
) -> LogisticGate:
    if features.ndim != 2 or features.size(1) != len(FEATURE_NAMES):
        raise ValueError("invalid OOF feature matrix")
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1.0e-8)
    standardized = (features - mean) / scale
    weight = torch.zeros(
        standardized.size(1), dtype=torch.float64, requires_grad=True
    )
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        max_iter=100,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = standardized @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return LogisticGate(
        mean.detach(),
        scale.detach(),
        weight.detach(),
        bias.detach(),
    )


def apply_logistic_gate(
    records: Sequence[SentenceRecord],
    ensemble_probabilities: Sequence[torch.Tensor],
    member_predictions: Sequence[torch.Tensor],
    prior: WordLabelPrior,
    smoothing: float,
    gate: LogisticGate,
) -> List[List[int]]:
    results: List[List[int]] = []
    for record, probabilities, member_votes in zip(
        records, ensemble_probabilities, member_predictions
    ):
        labels = probabilities.argmax(dim=-1).clone()
        for start, end, word in iter_words(record.chars):
            lexical_log = prior.log_probabilities(word, smoothing)
            if lexical_log is None:
                continue
            lexical_probability = lexical_log.exp()
            frequency = prior.observation_count(word)
            word_length = end - start
            for position in range(word_length):
                absolute = start + position
                neural = int(labels[absolute])
                lexical = int(lexical_probability[position].argmax())
                if neural == lexical:
                    continue
                features = _letter_features(
                    probabilities[absolute],
                    member_votes[:, absolute],
                    lexical_probability[position],
                    frequency,
                    position,
                    word_length,
                )
                if float(gate.probability(features)) >= 0.5:
                    labels[absolute] = lexical
        results.append(
            [
                0 if char == " " else int(label)
                for char, label in zip(record.chars, labels.tolist())
            ]
        )
    return results
