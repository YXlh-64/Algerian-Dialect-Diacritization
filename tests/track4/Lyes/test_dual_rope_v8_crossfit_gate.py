from pathlib import Path

import torch

from experiments.track4.Lyes.campaign.folds import make_balanced_folds
from utils.track4.Lyes.data import SentenceRecord
from experiments.track4.Lyes.dual_rope_v8_crossfit_gate import (
    cross_fitted_gate_predictions,
)
from utils.track4.Lyes.lexical_fusion import WordLabelPrior


def _synthetic_records() -> list[SentenceRecord]:
    records = []
    for index in range(20):
        word = "بب" if index % 2 == 0 else "تت"
        lexical_label = 1 if word == "بب" else 3
        label = lexical_label if index % 4 < 2 else 7
        records.append(
            SentenceRecord(
                sent_id=f"{index:06d}",
                chars=tuple(word),
                labels=(label, label),
                input_text=word,
            )
        )
    return records


def test_crossfit_predictions_are_deterministic_and_cover_each_sentence() -> None:
    records = _synthetic_records()
    prior_records = [
        SentenceRecord(
            sent_id=f"prior-{index:06d}",
            chars=record.chars,
            labels=(
                (1 if record.input_text == "بب" else 3),
                (1 if record.input_text == "بب" else 3),
            ),
            input_text=record.input_text,
        )
        for index, record in enumerate(records)
    ]
    prior = WordLabelPrior().fit(prior_records)
    probabilities = []
    votes = []
    for index, record in enumerate(records):
        distribution = torch.full((len(record.chars), 16), 0.001)
        lexical_label = 1 if record.input_text == "بب" else 3
        neural_label = 7
        distribution[:, neural_label] = 0.985
        distribution /= distribution.sum(dim=-1, keepdim=True)
        probabilities.append(distribution)
        votes.append(
            torch.tensor(
                [
                    [neural_label, neural_label],
                    [neural_label, neural_label],
                    [lexical_label, lexical_label],
                    [neural_label, neural_label],
                ]
            )
        )
    folds = make_balanced_folds(records, 5, 842)
    first, first_summaries = cross_fitted_gate_predictions(
        records, probabilities, votes, prior, 0.01, folds
    )
    second, second_summaries = cross_fitted_gate_predictions(
        records, probabilities, votes, prior, 0.01, folds
    )
    assert first == second
    assert first_summaries == second_summaries
    assert len(first) == len(records)
    assert all(len(prediction) == 2 for prediction in first)


def test_crossfit_rejects_incomplete_fold_partition() -> None:
    records = _synthetic_records()
    prior = WordLabelPrior().fit(records)
    probabilities = [torch.full((2, 16), 1.0 / 16)] * len(records)
    votes = [torch.zeros((4, 2), dtype=torch.long)] * len(records)
    try:
        cross_fitted_gate_predictions(
            records,
            probabilities,
            votes,
            prior,
            0.01,
            ((0, 1),),
        )
    except ValueError as error:
        assert "partition" in str(error)
    else:
        raise AssertionError("incomplete folds must fail closed")
