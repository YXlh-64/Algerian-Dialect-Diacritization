"""Cross-track error metrics for aligned character-label predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Return Levenshtein distance using memory linear in the shorter input."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def aligned_word_error_metrics(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[Sequence[int]],
    *,
    space_char: str = " ",
) -> dict[str, Any]:
    """Compute shared DER/WER metrics for aligned label sequences.

    Spaces are structural and excluded. Starred metrics exclude each
    multi-character word's final position; one-character words do not
    contribute to starred denominators. This is the repository-wide metric
    definition used by both Track 1 and Track 3.
    """
    if len(records) != len(predictions):
        raise ValueError("record and prediction counts differ")

    total_chars = char_errors = 0
    total_chars_star = char_errors_star = 0
    total_words = word_errors = 0
    total_words_star = word_errors_star = 0
    sentence_count = exact_sentence_count = 0
    class_totals: dict[int, int] = defaultdict(int)
    class_errors: dict[int, int] = defaultdict(int)
    confusion_pairs: Counter[tuple[int, int]] = Counter()

    for record, prediction in zip(records, predictions):
        chars = record["chars"]
        labels = record.get("labels")
        if labels is None:
            raise ValueError("word-error metrics require labeled records")
        if len(chars) != len(labels) or len(chars) != len(prediction):
            raise ValueError("character, label, and prediction lengths differ")

        sentence_count += 1
        sentence_is_exact = True
        words: list[list[tuple[int, int]]] = []
        current_word: list[tuple[int, int]] = []
        for index, char in enumerate(chars):
            if char == space_char:
                if current_word:
                    words.append(current_word)
                current_word = []
                continue

            target = int(labels[index])
            predicted = int(prediction[index])
            current_word.append((predicted, target))
            class_totals[target] += 1
            if predicted != target:
                class_errors[target] += 1
                confusion_pairs[(target, predicted)] += 1
                sentence_is_exact = False
        if current_word:
            words.append(current_word)
        exact_sentence_count += int(sentence_is_exact)

        for word in words:
            errors = [predicted != target for predicted, target in word]
            total_chars += len(errors)
            char_errors += sum(errors)
            total_words += 1
            word_errors += int(any(errors))

            if len(errors) > 1:
                starred_errors = errors[:-1]
                total_chars_star += len(starred_errors)
                char_errors_star += sum(starred_errors)
                total_words_star += 1
                word_errors_star += int(any(starred_errors))

    per_class_der = {
        label: class_errors[label] / total
        for label, total in class_totals.items()
        if total > 0
    }
    return {
        "DER": char_errors / max(total_chars, 1),
        "DER_star": char_errors_star / max(total_chars_star, 1),
        "WER": word_errors / max(total_words, 1),
        "WER_star": word_errors_star / max(total_words_star, 1),
        "sentence_exact_match": exact_sentence_count / max(sentence_count, 1),
        "per_class_der": per_class_der,
        "top_confusions": confusion_pairs.most_common(15),
        "n_chars": total_chars,
        "n_words": total_words,
        "n_sentences": sentence_count,
    }


def word_level_metrics_from_predict_fn(
    predict_fn: Callable[[Sequence[str]], Sequence[int]],
    records: Sequence[dict[str, Any]],
    *,
    space_char: str = " ",
) -> dict[str, Any]:
    """Compatibility adapter for Track-3 model prediction functions."""
    predictions = [predict_fn(record["chars"]) for record in records]
    return aligned_word_error_metrics(records, predictions, space_char=space_char)
