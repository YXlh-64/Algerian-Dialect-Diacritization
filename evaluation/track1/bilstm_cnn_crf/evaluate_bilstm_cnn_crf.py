"""Metric, structured-ensemble, and submission utilities for Track 1."""

from __future__ import annotations

import itertools
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.track1.data import NUM_LABELS, iter_words, letter_label_counts

LABEL_TO_MARKS = {
    0: "",
    1: "\u064e",
    2: "\u064b",
    3: "\u064f",
    4: "\u064c",
    5: "\u0650",
    6: "\u064d",
    7: "\u0652",
    8: "\u0651",
    9: "\u0651\u064e",
    10: "\u0651\u064b",
    11: "\u0651\u064f",
    12: "\u0651\u064c",
    13: "\u0651\u0650",
    14: "\u0651\u064d",
    15: "\u0651\u0652",
}


def per_class_f1(
    targets: Iterable[int], predictions: Iterable[int], num_labels: int = NUM_LABELS
) -> np.ndarray:
    targets = np.asarray(list(targets), dtype=np.int64)
    predictions = np.asarray(list(predictions), dtype=np.int64)
    scores = np.zeros(num_labels, dtype=np.float64)
    for label in range(num_labels):
        tp = np.sum((targets == label) & (predictions == label))
        fp = np.sum((targets != label) & (predictions == label))
        fn = np.sum((targets == label) & (predictions != label))
        denominator = 2 * tp + fp + fn
        scores[label] = 0.0 if denominator == 0 else 2 * tp / denominator
    return scores


def metric_summary(
    targets: Iterable[int], predictions: Iterable[int]
) -> dict[str, Any]:
    targets = np.asarray(list(targets), dtype=np.int64)
    predictions = np.asarray(list(predictions), dtype=np.int64)
    scores = per_class_f1(targets, predictions)
    support = np.bincount(targets, minlength=NUM_LABELS)
    return {
        "macro_f1_16": float(scores.mean()),
        "macro_f1_supported": float(scores[support > 0].mean()),
        "accuracy": float(np.mean(targets == predictions)),
        "per_class_f1": scores,
        "support": support,
    }


def score_record_predictions(
    records: list[dict[str, Any]], predictions: list[np.ndarray]
) -> dict[str, Any]:
    targets_flat, predictions_flat = [], []
    for record, prediction in zip(records, predictions):
        assert len(prediction) == len(record["chars"])
        for char, target, predicted in zip(
            record["chars"], record["labels"], prediction
        ):
            if char != " ":
                targets_flat.append(target)
                predictions_flat.append(int(predicted))
    return metric_summary(targets_flat, predictions_flat)


def build_word_log_priors(
    records: list[dict[str, Any]], smoothing: float = 0.10
) -> dict[str, np.ndarray]:
    word_counts: dict[str, np.ndarray] = {}
    for record in records:
        for word, labels, _, _ in iter_words(record):
            if word not in word_counts:
                word_counts[word] = np.zeros((len(word), NUM_LABELS), dtype=np.float64)
            for position, label in enumerate(labels):
                word_counts[word][position, label] += 1.0
    priors = {}
    for word, counts in word_counts.items():
        probabilities = (counts + smoothing) / (
            counts.sum(axis=1, keepdims=True) + smoothing * NUM_LABELS
        )
        log_probabilities = np.log(probabilities)
        priors[word] = log_probabilities - log_probabilities.max(axis=1, keepdims=True)
    return priors


def word_prior_matrix(
    record: dict[str, Any], priors: dict[str, np.ndarray]
) -> np.ndarray:
    matrix = np.zeros((len(record["chars"]), NUM_LABELS), dtype=np.float64)
    for word, _, start, end in iter_words(record, include_labels=False):
        if word in priors:
            matrix[start:end] = priors[word]
    return matrix


def build_sentence_memory(
    records: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    candidates: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        candidates[record["input"]][tuple(record["labels"])] += 1
    return {
        text: np.asarray(counter.most_common(1)[0][0], dtype=np.int64)
        for text, counter in candidates.items()
    }


def class_log_prior(records: list[dict[str, Any]]) -> np.ndarray:
    counts = letter_label_counts(records).astype(np.float64) + 1.0
    prior = np.log(counts / counts.sum())
    return prior - prior.mean()


def simplex_grid(model_count: int, denominator: int = 4) -> list[np.ndarray]:
    if model_count == 1:
        return [np.ones(1)]
    return [
        np.asarray(allocation, dtype=np.float64) / denominator
        for allocation in itertools.product(range(denominator + 1), repeat=model_count)
        if sum(allocation) == denominator
    ]


def blend_transition(
    weights: np.ndarray,
    transitions: list[dict[str, np.ndarray] | None],
) -> dict[str, np.ndarray] | None:
    usable = [
        (weight, transition)
        for weight, transition in zip(weights, transitions)
        if transition is not None and weight > 0
    ]
    if not usable:
        return None
    normalizer = sum(weight for weight, _ in usable)
    return {
        key: sum(weight * transition[key] for weight, transition in usable) / normalizer
        for key in ("start", "end", "transitions")
    }


def viterbi_numpy(
    emissions: np.ndarray,
    transition: dict[str, np.ndarray],
    strength: float,
) -> np.ndarray:
    if strength <= 0:
        return emissions.argmax(axis=1).astype(np.int64)
    start = strength * transition["start"]
    end = strength * transition["end"]
    transitions = strength * transition["transitions"]
    score = start + emissions[0]
    backpointers = []
    for position in range(1, len(emissions)):
        candidates = score[:, None] + transitions
        best_previous = candidates.argmax(axis=0)
        score = candidates[best_previous, np.arange(NUM_LABELS)] + emissions[position]
        backpointers.append(best_previous)
    tag = int(np.argmax(score + end))
    path = [tag]
    for backpointer in reversed(backpointers):
        tag = int(backpointer[tag])
        path.append(tag)
    return np.asarray(list(reversed(path)), dtype=np.int64)


def decode_ensemble(
    model_outputs: list[list[dict[str, Any]]],
    records: list[dict[str, Any]],
    transitions: list[dict[str, np.ndarray] | None],
    weights: np.ndarray,
    lexical_priors: dict[str, np.ndarray],
    lexical_strength: float,
    frequency_prior: np.ndarray,
    frequency_strength: float,
    transition_strength: float,
    sentence_memory: dict[str, np.ndarray] | None = None,
    exact_sentence_memory: bool = False,
) -> list[np.ndarray]:
    blended_transition = blend_transition(weights, transitions)
    predictions = []
    for record_index, record in enumerate(records):
        scores = sum(
            weight * outputs[record_index]["log_probs"]
            for weight, outputs in zip(weights, model_outputs)
        )
        if lexical_strength > 0:
            scores = scores + lexical_strength * word_prior_matrix(
                record, lexical_priors
            )
        if frequency_strength > 0:
            scores = scores - frequency_strength * frequency_prior[None, :]
        for position, char in enumerate(record["chars"]):
            if char == " ":
                scores[position, :] = -1e4
                scores[position, 0] = 0.0
        if blended_transition is None:
            prediction = scores.argmax(axis=1).astype(np.int64)
        else:
            prediction = viterbi_numpy(scores, blended_transition, transition_strength)
        if (
            exact_sentence_memory
            and sentence_memory is not None
            and record["input"] in sentence_memory
        ):
            memorized = sentence_memory[record["input"]]
            if len(memorized) == len(prediction):
                prediction = memorized.copy()
        predictions.append(prediction)
    return predictions


def tune_ensemble(
    model_outputs: list[list[dict[str, Any]]],
    transitions: list[dict[str, np.ndarray] | None],
    records: list[dict[str, Any]],
    prior_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[np.ndarray], pd.DataFrame]:
    lexical_priors = build_word_log_priors(prior_records)
    frequency_prior = class_log_prior(prior_records)
    sentence_memory = build_sentence_memory(prior_records)
    candidates = []
    for weights in simplex_grid(len(model_outputs), denominator=4):
        for lexical_strength in (0.0, 0.20, 0.45, 0.75):
            for frequency_strength in (0.0, 0.05, 0.10, 0.20):
                predictions = decode_ensemble(
                    model_outputs,
                    records,
                    transitions,
                    weights,
                    lexical_priors,
                    lexical_strength,
                    frequency_prior,
                    frequency_strength,
                    transition_strength=0.0,
                    sentence_memory=sentence_memory,
                )
                metrics = score_record_predictions(records, predictions)
                candidates.append(
                    {
                        "weights": weights,
                        "lexical_strength": lexical_strength,
                        "frequency_strength": frequency_strength,
                        "transition_strength": 0.0,
                        "macro_f1_16": metrics["macro_f1_16"],
                        "accuracy": metrics["accuracy"],
                    }
                )
    stage_one = sorted(
        candidates,
        key=lambda row: (row["macro_f1_16"], row["accuracy"]),
        reverse=True,
    )[:5]
    structured = []
    for candidate in stage_one:
        for transition_strength in (0.0, 0.25, 0.50, 0.75, 1.0):
            predictions = decode_ensemble(
                model_outputs,
                records,
                transitions,
                candidate["weights"],
                lexical_priors,
                candidate["lexical_strength"],
                frequency_prior,
                candidate["frequency_strength"],
                transition_strength,
                sentence_memory,
            )
            metrics = score_record_predictions(records, predictions)
            structured.append(
                {
                    **candidate,
                    "transition_strength": transition_strength,
                    "macro_f1_16": metrics["macro_f1_16"],
                    "accuracy": metrics["accuracy"],
                    "predictions": predictions,
                    "metrics": metrics,
                }
            )
    best = max(structured, key=lambda row: (row["macro_f1_16"], row["accuracy"]))
    config = {
        "weights": best["weights"],
        "lexical_strength": best["lexical_strength"],
        "frequency_strength": best["frequency_strength"],
        "transition_strength": best["transition_strength"],
    }
    leaderboard = pd.DataFrame(
        [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"predictions", "metrics", "weights"}
                },
                "weights": np.round(row["weights"], 3).tolist(),
            }
            for row in structured
        ]
    ).sort_values(["macro_f1_16", "accuracy"], ascending=False)
    return config, best["predictions"], leaderboard


def write_submission(
    records: list[dict[str, Any]],
    predictions: list[np.ndarray],
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    for record, prediction in zip(records, predictions):
        for char_index, (char, label) in enumerate(zip(record["chars"], prediction)):
            if char != " ":
                rows.append(
                    {
                        "Id": f"{record['sent_id']}_{char_index}",
                        "Label": int(label),
                    }
                )
    submission = pd.DataFrame(rows, columns=["Id", "Label"])
    submission.to_csv(output_path, index=False)
    return submission


def vocalize(chars: list[str], labels: Iterable[int]) -> str:
    return "".join(
        char + LABEL_TO_MARKS[int(label)] for char, label in zip(chars, labels)
    )
