"""Comparable neural/fusion diagnostics for every campaign prediction set."""

from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence, Set

from utils.track4.Lyes.data import SentenceRecord
from evaluation.track4.Lyes.dziri_fusion import evaluate_predictions
from utils.track4.Lyes.lexical_fusion import iter_words


def training_word_types(records: Sequence[SentenceRecord]) -> Set[str]:
    return {
        word
        for record in records
        for _, _, word in iter_words(record.chars)
    }


def prediction_diagnostics(
    records: Sequence[SentenceRecord],
    predictions: Sequence[Sequence[int]],
    seen_words: Set[str],
) -> Dict[str, Any]:
    metrics = evaluate_predictions(records, predictions)
    counts = defaultdict(int)
    for record, labels in zip(records, predictions):
        if record.labels is None:
            raise ValueError("diagnostics require gold labels")
        for start, end, word in iter_words(record.chars):
            group = "seen" if word in seen_words else "oov"
            for index in range(start, end):
                target = int(record.labels[index])
                predicted = int(labels[index])
                counts[f"{group}_total"] += 1
                counts[f"{group}_correct"] += int(target == predicted)
                shadda = target >= 8
                counts["shadda_total"] += int(shadda)
                counts["shadda_correct"] += int(
                    shadda and target == predicted
                )
                counts["fatha_to_sukoon"] += int(
                    target == 1 and predicted == 7
                )
                counts["sukoon_to_fatha"] += int(
                    target == 7 and predicted == 1
                )
    result: Dict[str, Any] = {
        "micro_f1": metrics["micro_f1"],
        "correct": metrics["correct"],
        "total": metrics["total"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    for group in ("seen", "oov", "shadda"):
        total = counts[f"{group}_total"]
        correct = counts[f"{group}_correct"]
        result[f"{group}_accuracy"] = correct / total if total else 0.0
        result[f"{group}_correct"] = correct
        result[f"{group}_total"] = total
    result["fatha_to_sukoon"] = counts["fatha_to_sukoon"]
    result["sukoon_to_fatha"] = counts["sukoon_to_fatha"]
    return result
