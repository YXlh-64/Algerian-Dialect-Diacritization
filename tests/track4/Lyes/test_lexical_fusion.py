import torch

from utils.track4.Lyes.data import SentenceRecord
from utils.track4.Lyes.lexical_fusion import WordLabelPrior, iter_words


def _record(
    sent_id: str, text: str, labels=None
) -> SentenceRecord:
    return SentenceRecord(
        sent_id=sent_id,
        chars=tuple(text),
        labels=None if labels is None else tuple(labels),
        input_text=text,
    )


def test_iter_words_preserves_character_offsets() -> None:
    assert list(iter_words(tuple("با بي"))) == [
        (0, 2, "با"),
        (3, 5, "بي"),
    ]


def test_smoothed_word_prior_is_normalized() -> None:
    prior = WordLabelPrior().fit(
        [
            _record("1", "بب", [1, 7]),
            _record("2", "بب", [1, 1]),
        ]
    )
    probabilities = prior.log_probabilities(
        "بب", smoothing=0.01
    ).exp()
    assert probabilities.shape == (2, 16)
    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones(2), atol=1e-12
    )
    assert prior.vocabulary_size == 1
    assert prior.word_observations == 2


def test_fusion_changes_seen_word_and_preserves_unseen_word() -> None:
    prior = WordLabelPrior().fit([_record("train", "بب", [1, 7])])
    record = _record("test", "بب ج")
    logits = torch.full((4, 16), -8.0)
    logits[0, 0] = 0.0
    logits[1, 0] = 0.0
    logits[2, 5] = 0.0
    logits[3, 3] = 0.0
    log_probabilities = torch.log_softmax(logits, dim=-1)

    predictions, statistics = prior.fuse_record(
        record,
        log_probabilities,
        prior_strength=3.0,
        smoothing=0.01,
    )

    assert predictions == [1, 7, 0, 3]
    assert statistics.total_words == 2
    assert statistics.known_words == 1
    assert statistics.total_letters == 3
    assert statistics.known_word_letters == 2
    assert statistics.labels_changed_by_prior == 2


def test_zero_prior_strength_keeps_neural_letter_predictions() -> None:
    prior = WordLabelPrior().fit([_record("train", "ب", [1])])
    record = _record("test", "ب")
    log_probabilities = torch.log_softmax(
        torch.tensor([[4.0] + [0.0] * 15]), dim=-1
    )
    predictions, statistics = prior.fuse_record(
        record,
        log_probabilities,
        prior_strength=0.0,
        smoothing=0.01,
    )
    assert predictions == [0]
    assert statistics.labels_changed_by_prior == 0
