"""Deterministic paper metrics for Algerian Arabic vocalization systems."""

import argparse
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from utils.track4.Lyes.data import SentenceRecord, load_jsonl
from utils.track4.Lyes.labels import LABEL_MARKS, LABEL_NAMES, NUM_LABELS, apply_diacritics
from utils.track4.Lyes.utils import write_json


TANWEEN_BASE_LABELS = frozenset((2, 4, 6))
DIACRITIC_MARKS = frozenset(
    mark for label_mark in LABEL_MARKS for mark in label_mark
)


def levenshtein_distance(
    reference: Sequence[Hashable], hypothesis: Sequence[Hashable]
) -> int:
    """Exact Myers bit-vector Levenshtein distance with linear scan time."""
    if len(reference) > len(hypothesis):
        reference, hypothesis = hypothesis, reference
    pattern_length = len(reference)
    if pattern_length == 0:
        return len(hypothesis)

    equality_masks: Dict[Hashable, int] = {}
    for index, token in enumerate(reference):
        equality_masks[token] = equality_masks.get(token, 0) | (1 << index)

    active_mask = (1 << pattern_length) - 1
    highest_bit = 1 << (pattern_length - 1)
    positive_vertical = active_mask
    negative_vertical = 0
    distance = pattern_length
    for token in hypothesis:
        equality = equality_masks.get(token, 0)
        vertical_or_equality = equality | negative_vertical
        horizontal = (
            ((equality & positive_vertical) + positive_vertical)
            ^ positive_vertical
        ) | equality
        positive_horizontal = negative_vertical | ~(
            horizontal | positive_vertical
        )
        negative_horizontal = positive_vertical & horizontal
        if positive_horizontal & highest_bit:
            distance += 1
        elif negative_horizontal & highest_bit:
            distance -= 1
        positive_horizontal = ((positive_horizontal << 1) | 1) & active_mask
        negative_horizontal = (negative_horizontal << 1) & active_mask
        positive_vertical = (
            negative_horizontal
            | ~(vertical_or_equality | positive_horizontal)
        ) & active_mask
        negative_vertical = positive_horizontal & vertical_or_equality
    return distance


def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return "".join(char for char in normalized if char not in DIACRITIC_MARKS)


def _binary_metrics(
    targets: Sequence[bool], predictions: Sequence[bool]
) -> Dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("binary metric inputs must have identical lengths")
    true_positive = sum(
        target and predicted
        for target, predicted in zip(targets, predictions)
    )
    true_negative = sum(
        not target and not predicted
        for target, predicted in zip(targets, predictions)
    )
    false_positive = sum(
        not target and predicted
        for target, predicted in zip(targets, predictions)
    )
    false_negative = sum(
        target and not predicted
        for target, predicted in zip(targets, predictions)
    )
    total = len(targets)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = (
        true_positive / recall_denominator if recall_denominator else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "accuracy": (
            (true_positive + true_negative) / total if total else 0.0
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support_positive": recall_denominator,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "total": total,
    }


def _character_tokens(text: str) -> Tuple[str, ...]:
    return tuple(
        char
        for char in unicodedata.normalize("NFC", text)
        if not char.isspace()
    )


def corpus_char_bleu(
    references: Sequence[str],
    hypotheses: Sequence[str],
    maximum_order: int = 4,
) -> Dict[str, Any]:
    """Corpus character BLEU with add-one smoothing and effective order."""
    if len(references) != len(hypotheses) or not references:
        raise ValueError("char-BLEU requires aligned nonempty corpora")
    if maximum_order <= 0:
        raise ValueError("maximum_order must be positive")
    clipped = [0] * maximum_order
    possible = [0] * maximum_order
    reference_length = 0
    hypothesis_length = 0
    for reference_text, hypothesis_text in zip(references, hypotheses):
        reference = _character_tokens(reference_text)
        hypothesis = _character_tokens(hypothesis_text)
        reference_length += len(reference)
        hypothesis_length += len(hypothesis)
        for order in range(1, maximum_order + 1):
            if len(hypothesis) < order:
                continue
            hypothesis_ngrams = Counter(
                tuple(hypothesis[index : index + order])
                for index in range(len(hypothesis) - order + 1)
            )
            reference_ngrams = Counter(
                tuple(reference[index : index + order])
                for index in range(max(0, len(reference) - order + 1))
            )
            possible[order - 1] += sum(hypothesis_ngrams.values())
            clipped[order - 1] += sum(
                min(count, reference_ngrams.get(ngram, 0))
                for ngram, count in hypothesis_ngrams.items()
            )
    effective_orders = [
        index for index, count in enumerate(possible) if count > 0
    ]
    precisions = [
        (
            (clipped[index] + 1.0) / (possible[index] + 1.0)
            if index > 0
            else clipped[index] / possible[index]
        )
        for index in effective_orders
    ]
    if (
        not precisions
        or precisions[0] <= 0.0
        or hypothesis_length == 0
    ):
        score = 0.0
        brevity_penalty = 0.0 if hypothesis_length == 0 else 1.0
    else:
        log_precision = sum(math.log(value) for value in precisions) / len(
            precisions
        )
        brevity_penalty = (
            1.0
            if hypothesis_length >= reference_length
            else math.exp(1.0 - reference_length / hypothesis_length)
        )
        score = brevity_penalty * math.exp(log_precision)
    return {
        "score": score,
        "maximum_order": maximum_order,
        "effective_order": len(effective_orders),
        "precisions": precisions,
        "clipped_matches": clipped,
        "possible_matches": possible,
        "reference_characters": reference_length,
        "hypothesis_characters": hypothesis_length,
        "brevity_penalty": brevity_penalty,
        "tokenization": "NFC Unicode codepoints excluding whitespace",
        "smoothing": "add-one for orders 2-4; none for unigram",
    }


def compute_paper_metrics(
    records: Sequence[SentenceRecord],
    predictions: Sequence[Sequence[int]],
    predicted_texts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if not records or len(records) != len(predictions):
        raise ValueError("paper metrics require aligned nonempty predictions")
    if predicted_texts is not None and len(predicted_texts) != len(records):
        raise ValueError("predicted_texts must align with records")

    confusion = [[0] * NUM_LABELS for _ in range(NUM_LABELS)]
    reference_texts: List[str] = []
    hypothesis_texts: List[str] = []
    shadda_targets: List[bool] = []
    shadda_predictions: List[bool] = []
    tanween_targets: List[bool] = []
    tanween_predictions: List[bool] = []
    word_correct = 0
    word_total = 0
    sentence_correct = 0
    character_edits = 0
    reference_character_total = 0
    word_edits = 0
    reference_word_total = 0
    skeleton_mismatches = 0

    for record_index, (record, labels_value) in enumerate(
        zip(records, predictions)
    ):
        if record.labels is None:
            raise ValueError("paper metrics require reference labels")
        labels = tuple(int(label) for label in labels_value)
        if len(labels) != len(record.chars):
            raise ValueError(
                "prediction length mismatch for {}".format(record.sent_id)
            )
        if any(label < 0 or label >= NUM_LABELS for label in labels):
            raise ValueError(
                "prediction outside label range for {}".format(record.sent_id)
            )
        if any(
            char == " " and label != 0
            for char, label in zip(record.chars, labels)
        ):
            raise ValueError(
                "space has nonzero prediction for {}".format(record.sent_id)
            )

        reference_text = apply_diacritics(record.chars, record.labels)
        hypothesis_text = (
            apply_diacritics(record.chars, labels)
            if predicted_texts is None
            else unicodedata.normalize("NFC", predicted_texts[record_index])
        )
        reference_texts.append(reference_text)
        hypothesis_texts.append(hypothesis_text)
        normalized_reference_skeleton = " ".join(
            record.input_text.split()
        )
        normalized_hypothesis_skeleton = " ".join(
            strip_diacritics(hypothesis_text).split()
        )
        skeleton_mismatches += int(
            normalized_hypothesis_skeleton != normalized_reference_skeleton
        )

        reference_characters = tuple(unicodedata.normalize("NFC", reference_text))
        hypothesis_characters = tuple(
            unicodedata.normalize("NFC", hypothesis_text)
        )
        character_edits += levenshtein_distance(
            reference_characters, hypothesis_characters
        )
        reference_character_total += len(reference_characters)
        reference_words = tuple(reference_text.split())
        hypothesis_words = tuple(hypothesis_text.split())
        word_edits += levenshtein_distance(reference_words, hypothesis_words)
        reference_word_total += len(reference_words)

        sentence_matches = True
        word_start = 0
        for position in range(len(record.chars) + 1):
            at_boundary = (
                position == len(record.chars)
                or record.chars[position] == " "
            )
            if at_boundary and position > word_start:
                word_total += 1
                word_correct += int(
                    labels[word_start:position]
                    == record.labels[word_start:position]
                )
            if at_boundary:
                word_start = position + 1

        for char, target, predicted in zip(
            record.chars, record.labels, labels
        ):
            if char == " ":
                continue
            confusion[target][predicted] += 1
            sentence_matches &= target == predicted
            shadda_targets.append(target >= 8)
            shadda_predictions.append(predicted >= 8)
            tanween_targets.append(target % 8 in TANWEEN_BASE_LABELS)
            tanween_predictions.append(
                predicted % 8 in TANWEEN_BASE_LABELS
            )
        sentence_correct += int(sentence_matches)

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[label][label] for label in range(NUM_LABELS))
    per_class: List[Dict[str, Any]] = []
    for label, name in enumerate(LABEL_NAMES):
        true_positive = confusion[label][label]
        support = sum(confusion[label])
        predicted_count = sum(row[label] for row in confusion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append(
            {
                "label": label,
                "name": name,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    present_f1 = [
        value["f1"] for value in per_class if value["support"] > 0
    ]
    char_bleu = corpus_char_bleu(reference_texts, hypothesis_texts)
    return {
        "schema_version": 1,
        "definitions": {
            "accuracy": "exact 16-class accuracy over non-space letters",
            "macro_f1": "unweighted mean F1 over all fixed 16 classes; zero for undefined classes",
            "wer": "corpus Levenshtein word edits over fully vocalized whitespace tokens divided by reference words",
            "cer": "corpus Levenshtein edits over NFC Unicode codepoints of fully vocalized sentences, including spaces and combining marks, divided by reference codepoints",
            "word_accuracy": "fraction of whitespace-delimited words whose complete aligned label sequence is exact",
            "sentence_accuracy": "fraction of sentences whose complete non-space label sequence is exact",
            "shadda_accuracy": "binary presence/absence accuracy for labels 8-15 over all scored letters",
            "tanween_accuracy": "binary presence/absence accuracy for base labels 2, 4, 6 including their Shadda combinations over all scored letters",
            "skeleton_mismatch_count": "sentences whose predicted text stripped of competition diacritics differs from the normalized input skeleton",
            "char_bleu": "corpus BLEU-4 over NFC non-whitespace Unicode codepoints",
        },
        "sentences": len(records),
        "words": reference_word_total,
        "scored_letters": total,
        "correct_letters": correct,
        "accuracy": correct / total if total else 0.0,
        "micro_f1": correct / total if total else 0.0,
        "macro_f1": sum(value["f1"] for value in per_class) / NUM_LABELS,
        "macro_f1_present_classes": (
            sum(present_f1) / len(present_f1) if present_f1 else 0.0
        ),
        "per_class": per_class,
        "word_accuracy": word_correct / word_total if word_total else 0.0,
        "word_correct": word_correct,
        "word_total": word_total,
        "sentence_accuracy": sentence_correct / len(records),
        "sentence_correct": sentence_correct,
        "sentence_total": len(records),
        "wer": word_edits / reference_word_total if reference_word_total else 0.0,
        "word_edits": word_edits,
        "reference_word_total": reference_word_total,
        "cer": (
            character_edits / reference_character_total
            if reference_character_total
            else 0.0
        ),
        "character_edits": character_edits,
        "reference_character_total": reference_character_total,
        "shadda": _binary_metrics(shadda_targets, shadda_predictions),
        "tanween": _binary_metrics(tanween_targets, tanween_predictions),
        "confusion_matrix": confusion,
        "skeleton_mismatch_count": skeleton_mismatches,
        "char_bleu": char_bleu,
    }


def load_prediction_jsonl(
    path: Path, records: Sequence[SentenceRecord]
) -> Tuple[List[List[int]], Optional[List[str]]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(records):
        raise ValueError("prediction JSONL row count does not match dev")
    predictions: List[List[int]] = []
    texts: List[str] = []
    has_text = []
    for record, row in zip(records, rows):
        if row.get("sent_id") != record.sent_id:
            raise ValueError("prediction JSONL sentence order mismatch")
        labels = row.get("labels")
        if not isinstance(labels, list):
            raise ValueError("prediction JSONL labels must be a list")
        predictions.append([int(label) for label in labels])
        text = row.get("text")
        has_text.append(text is not None)
        texts.append("" if text is None else str(text))
    if any(has_text) and not all(has_text):
        raise ValueError("prediction JSONL must provide text for every row or none")
    return predictions, texts if all(has_text) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev",
        type=Path,
        default=Path("Data/dev_data/dev_Algerian-DIAC.jsonl"),
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_jsonl(args.dev)
    predictions, texts = load_prediction_jsonl(args.predictions, records)
    metrics = compute_paper_metrics(records, predictions, texts)
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
