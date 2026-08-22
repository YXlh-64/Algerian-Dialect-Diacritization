"""Training-only word priors for deterministic neural/lexical fusion."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.track4.Lyes.data import (
    BatchCollator,
    CharacterDataset,
    SentenceRecord,
    validate_vocabulary_coverage,
)
from utils.track4.Lyes.labels import NUM_LABELS
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer


def iter_words(chars: Sequence[str]) -> Iterator[Tuple[int, int, str]]:
    start = 0
    for end in range(len(chars) + 1):
        if end == len(chars) or chars[end] == " ":
            if end > start:
                yield start, end, "".join(chars[start:end])
            start = end + 1


@dataclass
class FusionStatistics:
    total_words: int = 0
    known_words: int = 0
    total_letters: int = 0
    known_word_letters: int = 0
    labels_changed_by_prior: int = 0

    def update(self, other: "FusionStatistics") -> None:
        self.total_words += other.total_words
        self.known_words += other.known_words
        self.total_letters += other.total_letters
        self.known_word_letters += other.known_word_letters
        self.labels_changed_by_prior += other.labels_changed_by_prior

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
        result["changed_label_rate"] = (
            self.labels_changed_by_prior / self.total_letters
            if self.total_letters
            else 0.0
        )
        return result


class WordLabelPrior:
    """Per-position label counts for words observed in training only."""

    def __init__(self) -> None:
        self._counts: Dict[str, torch.Tensor] = {}
        self.word_observations = 0

    @property
    def vocabulary_size(self) -> int:
        return len(self._counts)

    def fit(self, records: Sequence[SentenceRecord]) -> "WordLabelPrior":
        counts: Dict[str, torch.Tensor] = {}
        observations = 0
        for record in records:
            if record.labels is None:
                raise ValueError("lexical-prior training records need labels")
            for start, end, word in iter_words(record.chars):
                labels = record.labels[start:end]
                word_counts = counts.get(word)
                if word_counts is None:
                    word_counts = torch.zeros(
                        (len(word), NUM_LABELS), dtype=torch.float64
                    )
                    counts[word] = word_counts
                elif word_counts.size(0) != len(word):
                    raise RuntimeError("inconsistent lexical-prior word length")
                for position, label in enumerate(labels):
                    word_counts[position, label] += 1.0
                observations += 1
        if not counts:
            raise ValueError("cannot fit lexical prior on an empty dataset")
        self._counts = counts
        self.word_observations = observations
        return self

    def contains(self, word: str) -> bool:
        return word in self._counts

    def observation_count(self, word: str) -> int:
        counts = self._counts.get(word)
        if counts is None or counts.size(0) == 0:
            return 0
        return int(round(float(counts[0].sum().item())))

    def log_probabilities(
        self,
        word: str,
        smoothing: float,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        if smoothing <= 0.0:
            raise ValueError("smoothing must be positive")
        counts = self._counts.get(word)
        if counts is None:
            return None
        smoothed = counts + smoothing
        probabilities = smoothed / smoothed.sum(dim=-1, keepdim=True)
        return probabilities.log().to(dtype=dtype)

    def fuse_record(
        self,
        record: SentenceRecord,
        neural_log_probabilities: torch.Tensor,
        prior_strength: float,
        smoothing: float,
    ) -> Tuple[List[int], FusionStatistics]:
        if prior_strength < 0.0:
            raise ValueError("prior_strength cannot be negative")
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

        neural_predictions = neural_log_probabilities.argmax(dim=-1)
        fused_predictions = neural_predictions.clone()
        statistics = FusionStatistics()

        for start, end, word in iter_words(record.chars):
            statistics.total_words += 1
            statistics.total_letters += end - start
            lexical_log_probabilities = self.log_probabilities(
                word,
                smoothing=smoothing,
                dtype=neural_log_probabilities.dtype,
            )
            if lexical_log_probabilities is None:
                continue
            statistics.known_words += 1
            statistics.known_word_letters += end - start
            fused_scores = (
                neural_log_probabilities[start:end]
                + prior_strength * lexical_log_probabilities
            )
            fused_predictions[start:end] = fused_scores.argmax(dim=-1)

        for index, char in enumerate(record.chars):
            if char == " ":
                fused_predictions[index] = 0
            elif fused_predictions[index] != neural_predictions[index]:
                statistics.labels_changed_by_prior += 1
        return [int(label) for label in fused_predictions.tolist()], statistics


@torch.inference_mode()
def predict_with_lexical_fusion(
    model: CharDiacritizer,
    records: Sequence[SentenceRecord],
    vocab: Dict[str, int],
    lexical_prior: WordLabelPrior,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prior_strength: float,
    smoothing: float,
) -> Tuple[List[List[int]], FusionStatistics]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if not records:
        raise ValueError("cannot predict an empty record collection")
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
    all_predictions: List[List[int]] = []
    aggregate_statistics = FusionStatistics()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device, non_blocking=True
        )
        outputs = model(input_ids, attention_mask)
        log_probabilities = F.log_softmax(outputs["logits"], dim=-1).to("cpu")
        for row, record in enumerate(batch["records"]):
            record_log_probabilities = log_probabilities[
                row, 1 : len(record.chars) + 1
            ]
            predictions, statistics = lexical_prior.fuse_record(
                record,
                record_log_probabilities,
                prior_strength=prior_strength,
                smoothing=smoothing,
            )
            all_predictions.append(predictions)
            aggregate_statistics.update(statistics)
    return all_predictions, aggregate_statistics
