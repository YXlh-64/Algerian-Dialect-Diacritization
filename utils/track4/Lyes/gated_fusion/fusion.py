"""Token-level confidence-gated lexical fallback for V2."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from utils.track4.Lyes.gated_fusion.config import GatedFusionConfig
from utils.track4.Lyes.data import (
    BatchCollator,
    CharacterDataset,
    SentenceRecord,
    validate_vocabulary_coverage,
)
from utils.track4.Lyes.labels import NUM_LABELS
from utils.track4.Lyes.lexical_fusion import WordLabelPrior, iter_words
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer


@dataclass
class GatedFusionStatistics:
    total_words: int = 0
    known_words: int = 0
    total_letters: int = 0
    known_word_letters: int = 0
    neural_lexical_disagreements: int = 0
    fallback_changes: int = 0
    retained_due_to_neural_confidence: int = 0
    retained_due_to_weak_lexical_evidence: int = 0

    def update(self, other: "GatedFusionStatistics") -> None:
        self.total_words += other.total_words
        self.known_words += other.known_words
        self.total_letters += other.total_letters
        self.known_word_letters += other.known_word_letters
        self.neural_lexical_disagreements += (
            other.neural_lexical_disagreements
        )
        self.fallback_changes += other.fallback_changes
        self.retained_due_to_neural_confidence += (
            other.retained_due_to_neural_confidence
        )
        self.retained_due_to_weak_lexical_evidence += (
            other.retained_due_to_weak_lexical_evidence
        )

    def to_dict(self) -> Dict[str, float]:
        result = asdict(self)
        result["known_word_rate"] = (
            self.known_words / self.total_words if self.total_words else 0.0
        )
        result["known_letter_rate"] = (
            self.known_word_letters / self.total_letters
            if self.total_letters
            else 0.0
        )
        result["fallback_change_rate"] = (
            self.fallback_changes / self.total_letters
            if self.total_letters
            else 0.0
        )
        return result


def apply_gated_fallback(
    record: SentenceRecord,
    neural_log_probabilities: torch.Tensor,
    lexical_prior: WordLabelPrior,
    gates: GatedFusionConfig,
    initial_predictions: Optional[torch.Tensor] = None,
) -> Tuple[List[int], GatedFusionStatistics]:
    expected_shape = (len(record.chars), NUM_LABELS)
    if tuple(neural_log_probabilities.shape) != expected_shape:
        raise ValueError(
            "log-probability shape mismatch for {}: {} != {}".format(
                record.sent_id,
                tuple(neural_log_probabilities.shape),
                expected_shape,
            )
        )
    if not torch.isfinite(neural_log_probabilities).all():
        raise ValueError("neural log probabilities must be finite")

    neural_probabilities = neural_log_probabilities.exp()
    if initial_predictions is None:
        neural_confidence, neural_predictions = neural_probabilities.max(
            dim=-1
        )
    else:
        if initial_predictions.shape != neural_log_probabilities.shape[:1]:
            raise ValueError(
                "initial_predictions must match the record length"
            )
        neural_predictions = initial_predictions.to(dtype=torch.long)
        if neural_predictions.lt(0).any() or neural_predictions.ge(
            NUM_LABELS
        ).any():
            raise ValueError("initial_predictions are outside label range")
        neural_confidence = neural_probabilities.gather(
            -1, neural_predictions.unsqueeze(-1)
        ).squeeze(-1)
    final_predictions = neural_predictions.clone()
    statistics = GatedFusionStatistics()

    for start, end, word in iter_words(record.chars):
        statistics.total_words += 1
        statistics.total_letters += end - start
        lexical_log_probabilities = lexical_prior.log_probabilities(
            word,
            smoothing=gates.lexical_smoothing,
            dtype=neural_log_probabilities.dtype,
        )
        if lexical_log_probabilities is None:
            continue

        statistics.known_words += 1
        statistics.known_word_letters += end - start
        lexical_probabilities = lexical_log_probabilities.exp()
        lexical_confidence, lexical_predictions = lexical_probabilities.max(
            dim=-1
        )

        for position in range(end - start):
            absolute_position = start + position
            neural_label = int(neural_predictions[absolute_position])
            lexical_label = int(lexical_predictions[position])
            if neural_label == lexical_label:
                continue

            statistics.neural_lexical_disagreements += 1
            if (
                float(neural_confidence[absolute_position])
                >= gates.neural_confidence_threshold
            ):
                statistics.retained_due_to_neural_confidence += 1
                continue
            if (
                float(lexical_confidence[position])
                < gates.lexical_confidence_threshold
            ):
                statistics.retained_due_to_weak_lexical_evidence += 1
                continue

            final_predictions[absolute_position] = lexical_label
            statistics.fallback_changes += 1

    for index, char in enumerate(record.chars):
        if char == " ":
            final_predictions[index] = 0
    return [int(label) for label in final_predictions.tolist()], statistics


@torch.inference_mode()
def predict_with_gated_fallback(
    model: CharDiacritizer,
    records: Sequence[SentenceRecord],
    vocab: Mapping[str, int],
    lexical_prior: WordLabelPrior,
    gates: GatedFusionConfig,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[List[int]], GatedFusionStatistics]:
    if not records:
        raise ValueError("cannot predict an empty record collection")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    validate_vocabulary_coverage(records, vocab)
    maximum_length = max(len(record.chars) + 2 for record in records)
    if maximum_length > model.config.max_length:
        raise ValueError(
            "sequence length {} exceeds model max_length {}".format(
                maximum_length, model.config.max_length
            )
        )

    loader = DataLoader(
        CharacterDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchCollator(vocab),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    predictions: List[List[int]] = []
    aggregate_statistics = GatedFusionStatistics()

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device, non_blocking=True
        )
        outputs = model(input_ids, attention_mask)
        log_probabilities = model.log_probabilities(outputs).to("cpu")
        decoded = model.decode_outputs(outputs).to("cpu")

        for row, record in enumerate(batch["records"]):
            record_log_probabilities = log_probabilities[
                row, 1 : len(record.chars) + 1
            ]
            initial_predictions = decoded[
                row, 1 : len(record.chars) + 1
            ]
            record_predictions, statistics = apply_gated_fallback(
                record,
                record_log_probabilities,
                lexical_prior,
                gates,
                initial_predictions=initial_predictions,
            )
            predictions.append(record_predictions)
            aggregate_statistics.update(statistics)
    return predictions, aggregate_statistics
