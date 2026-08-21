"""Equal-weight probability ensembles with optional lexical arbitration."""

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from utils.track4.Lyes.gated_fusion.config import GatedFusionConfig
from utils.track4.Lyes.gated_fusion.fusion import apply_gated_fallback
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint, load_checkpoint
from utils.track4.Lyes.data import (
    BatchCollator,
    CharacterDataset,
    SentenceRecord,
    validate_vocabulary_coverage,
)
from utils.track4.Lyes.lexical_fusion import WordLabelPrior


def average_probability_groups(
    groups: Sequence[Sequence[torch.Tensor]],
) -> List[torch.Tensor]:
    """Average aligned probability tensors with equal group weight."""
    if not groups:
        raise ValueError("at least one probability group is required")
    record_count = len(groups[0])
    if any(len(group) != record_count for group in groups):
        raise ValueError("probability groups must have equal record counts")
    return [
        torch.stack([group[row] for group in groups], dim=0).mean(dim=0)
        for row in range(record_count)
    ]


@torch.inference_mode()
def predict_probability_members(
    checkpoint_paths: Sequence[Path],
    records: Sequence[SentenceRecord],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Mapping[str, int]]:
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")
    models = []
    shared_vocab: Dict[str, int] = {}
    for index, path in enumerate(checkpoint_paths):
        checkpoint = load_checkpoint(path, device)
        model, vocab = build_model_from_checkpoint(checkpoint, device)
        if index == 0:
            shared_vocab = vocab
        elif vocab != shared_vocab:
            raise ValueError("ensemble checkpoint vocabularies differ")
        models.append(model)
    validate_vocabulary_coverage(records, shared_vocab)
    loader = DataLoader(
        CharacterDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BatchCollator(shared_vocab),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    probabilities: List[torch.Tensor] = []
    member_predictions: List[torch.Tensor] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        model_outputs = [
            model(input_ids, attention_mask) for model in models
        ]
        model_probabilities = [
            model.probabilities(outputs)
            for model, outputs in zip(models, model_outputs)
        ]
        mean_probabilities = torch.stack(model_probabilities).mean(dim=0)
        for row, record in enumerate(batch["records"]):
            record_slice = slice(1, len(record.chars) + 1)
            probabilities.append(
                mean_probabilities[row, record_slice].to("cpu")
            )
            member_predictions.append(
                torch.stack(
                    [
                        values[row, record_slice].argmax(dim=-1).to("cpu")
                        for values in model_probabilities
                    ]
                )
            )
    return probabilities, member_predictions, shared_vocab


def predict_probability_ensemble(
    checkpoint_paths: Sequence[Path],
    records: Sequence[SentenceRecord],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[torch.Tensor], Mapping[str, int]]:
    probabilities, _, vocab = predict_probability_members(
        checkpoint_paths,
        records,
        device,
        batch_size,
        num_workers,
    )
    return probabilities, vocab


def predict_probability_group_ensemble(
    checkpoint_groups: Sequence[Sequence[Path]],
    records: Sequence[SentenceRecord],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[torch.Tensor], Mapping[str, int]]:
    """Average seeds within each architecture, then architectures equally."""
    group_probabilities, shared_vocab = predict_probability_groups(
        checkpoint_groups, records, device, batch_size, num_workers
    )
    averaged = average_probability_groups(group_probabilities)
    return averaged, shared_vocab


def predict_probability_groups(
    checkpoint_groups: Sequence[Sequence[Path]],
    records: Sequence[SentenceRecord],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[List[torch.Tensor]], Mapping[str, int]]:
    """Return one aligned probability collection per architecture group."""
    if not checkpoint_groups:
        raise ValueError("at least one checkpoint group is required")
    group_probabilities: List[List[torch.Tensor]] = []
    shared_vocab: Mapping[str, int] = {}
    for index, group in enumerate(checkpoint_groups):
        probabilities, vocab = predict_probability_ensemble(
            group, records, device, batch_size, num_workers
        )
        if index == 0:
            shared_vocab = vocab
        elif vocab != shared_vocab:
            raise ValueError("architecture checkpoint vocabularies differ")
        group_probabilities.append(probabilities)
    return group_probabilities, shared_vocab


def predict_probability_group_members(
    checkpoint_groups: Sequence[Sequence[Path]],
    records: Sequence[SentenceRecord],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Mapping[str, int]]:
    """Return equal-group probabilities and one hard vote per group."""
    group_probabilities, shared_vocab = predict_probability_groups(
        checkpoint_groups, records, device, batch_size, num_workers
    )
    averaged = average_probability_groups(group_probabilities)
    group_votes = [
        torch.stack(
            [group[row].argmax(dim=-1) for group in group_probabilities]
        )
        for row in range(len(records))
    ]
    return averaged, group_votes, shared_vocab


def probabilities_to_predictions(
    records: Sequence[SentenceRecord],
    probabilities: Sequence[torch.Tensor],
) -> List[List[int]]:
    predictions: List[List[int]] = []
    for record, distribution in zip(records, probabilities):
        labels = distribution.argmax(dim=-1).tolist()
        predictions.append(
            [
                0 if char == " " else int(label)
                for char, label in zip(record.chars, labels)
            ]
        )
    return predictions


def apply_lexical_gate(
    records: Sequence[SentenceRecord],
    probabilities: Sequence[torch.Tensor],
    prior: WordLabelPrior,
    gates: GatedFusionConfig,
) -> List[List[int]]:
    predictions: List[List[int]] = []
    for record, distribution in zip(records, probabilities):
        labels, _ = apply_gated_fallback(
            record,
            distribution.clamp_min(1.0e-12).log(),
            prior,
            gates,
        )
        predictions.append(labels)
    return predictions
