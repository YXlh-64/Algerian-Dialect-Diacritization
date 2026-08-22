from pathlib import Path

import pytest
import torch

from utils.track4.Lyes.gated_fusion.config import GatedFusionConfig, load_gates
from utils.track4.Lyes.gated_fusion.fusion import apply_gated_fallback
from utils.track4.Lyes.data import SentenceRecord
from utils.track4.Lyes.lexical_fusion import WordLabelPrior


ROOT = Path(__file__).resolve().parents[3]


def _record(
    sent_id: str, text: str, labels=None
) -> SentenceRecord:
    return SentenceRecord(
        sent_id=sent_id,
        chars=tuple(text),
        labels=None if labels is None else tuple(labels),
        input_text=text,
    )


def _gates(
    neural_threshold: float = 0.99,
    lexical_threshold: float = 0.60,
) -> GatedFusionConfig:
    return GatedFusionConfig(
        schema_version=1,
        system_name="test",
        artifact_prefix="TEST",
        confidence_measure="max_softmax_probability",
        neural_confidence_threshold=neural_threshold,
        lexical_confidence_threshold=lexical_threshold,
        lexical_smoothing=0.01,
    )


def _log_probabilities(
    preferred_labels, preferred_logit: float
) -> torch.Tensor:
    logits = torch.zeros((len(preferred_labels), 16))
    for index, label in enumerate(preferred_labels):
        logits[index, label] = preferred_logit
    return torch.log_softmax(logits, dim=-1)


def test_authoritative_gate_file_has_expected_policy() -> None:
    gates = load_gates(ROOT / "configs" / "track4" / "Lyes" / "gates.json")
    assert gates.system_name == "DziriFusion-Gated-v2"
    assert gates.artifact_prefix == "DZIRIFUSION_GATED_V2"
    assert gates.neural_confidence_threshold == pytest.approx(0.99)
    assert gates.lexical_confidence_threshold == pytest.approx(0.60)
    assert gates.lexical_smoothing == pytest.approx(0.01)


def test_low_neural_confidence_uses_strong_lexical_fallback() -> None:
    prior = WordLabelPrior().fit([_record("train", "ب", [1])])
    predictions, statistics = apply_gated_fallback(
        _record("test", "ب"),
        _log_probabilities([0], preferred_logit=2.0),
        prior,
        _gates(),
    )
    assert predictions == [1]
    assert statistics.fallback_changes == 1
    assert statistics.neural_lexical_disagreements == 1


def test_high_neural_confidence_retains_transformer_prediction() -> None:
    prior = WordLabelPrior().fit([_record("train", "ب", [1])])
    predictions, statistics = apply_gated_fallback(
        _record("test", "ب"),
        _log_probabilities([0], preferred_logit=10.0),
        prior,
        _gates(),
    )
    assert predictions == [0]
    assert statistics.fallback_changes == 0
    assert statistics.retained_due_to_neural_confidence == 1


def test_weak_lexical_evidence_does_not_override_transformer() -> None:
    prior = WordLabelPrior().fit(
        [
            _record("train-1", "ب", [1]),
            _record("train-2", "ب", [2]),
        ]
    )
    predictions, statistics = apply_gated_fallback(
        _record("test", "ب"),
        _log_probabilities([0], preferred_logit=2.0),
        prior,
        _gates(),
    )
    assert predictions == [0]
    assert statistics.fallback_changes == 0
    assert statistics.retained_due_to_weak_lexical_evidence == 1


def test_unseen_words_remain_neural_and_spaces_are_zero() -> None:
    prior = WordLabelPrior().fit([_record("train", "ب", [1])])
    predictions, statistics = apply_gated_fallback(
        _record("test", "ج "),
        _log_probabilities([3, 4], preferred_logit=2.0),
        prior,
        _gates(),
    )
    assert predictions == [3, 0]
    assert statistics.known_words == 0
    assert statistics.fallback_changes == 0
