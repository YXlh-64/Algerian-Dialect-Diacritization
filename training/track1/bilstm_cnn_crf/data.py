"""Batching, sampling, and reproducibility utilities for Track 1."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.track1.data import BOUNDARY_PADDING_ID, NUM_LABELS, letter_label_counts


@dataclass(frozen=True)
class DataSettings:
    vocabulary: dict[str, int]
    pad_id: int
    unk_id: int
    num_labels: int = NUM_LABELS
    sampler_max_weight: float = 5.0
    num_workers: int = 2
    dual_gpu_active: bool = False


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def boundary_features(chars: list[str]) -> list[int]:
    """Return 0=space, 1=start, 2=middle, 3=end, 4=single-letter word."""
    features = []
    for index, char in enumerate(chars):
        if char == " ":
            features.append(0)
            continue
        at_start = index == 0 or chars[index - 1] == " "
        at_end = index == len(chars) - 1 or chars[index + 1] == " "
        if at_start and at_end:
            features.append(4)
        elif at_start:
            features.append(1)
        elif at_end:
            features.append(3)
        else:
            features.append(2)
    return features


class DiacritizationDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], settings: DataSettings):
        self.items = []
        for index, record in enumerate(records):
            chars = record["chars"]
            item = {
                "index": index,
                "sent_id": record["sent_id"],
                "chars": chars,
                "tokens": torch.tensor(
                    [settings.vocabulary.get(char, settings.unk_id) for char in chars],
                    dtype=torch.long,
                ),
                "boundaries": torch.tensor(boundary_features(chars), dtype=torch.long),
                "spaces": torch.tensor(
                    [char == " " for char in chars], dtype=torch.bool
                ),
            }
            if "labels" in record:
                item["labels"] = torch.tensor(record["labels"], dtype=torch.long)
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def collate_batch(items: list[dict[str, Any]], *, pad_id: int) -> dict[str, Any]:
    lengths = torch.tensor([len(item["tokens"]) for item in items], dtype=torch.long)
    max_length = int(lengths.max())
    batch_size = len(items)
    tokens = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
    boundaries = torch.full(
        (batch_size, max_length), BOUNDARY_PADDING_ID, dtype=torch.long
    )
    spaces = torch.zeros((batch_size, max_length), dtype=torch.bool)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    has_labels = "labels" in items[0]
    labels = (
        torch.zeros((batch_size, max_length), dtype=torch.long) if has_labels else None
    )

    for row, item in enumerate(items):
        length = len(item["tokens"])
        tokens[row, :length] = item["tokens"]
        boundaries[row, :length] = item["boundaries"]
        spaces[row, :length] = item["spaces"]
        mask[row, :length] = True
        if labels is not None:
            labels[row, :length] = item["labels"]

    return {
        "indices": [item["index"] for item in items],
        "sent_ids": [item["sent_id"] for item in items],
        "chars": [item["chars"] for item in items],
        "tokens": tokens,
        "boundaries": boundaries,
        "spaces": spaces,
        "mask": mask,
        "lengths": lengths,
        "labels": labels,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("tokens", "boundaries", "spaces", "mask", "lengths", "labels"):
        if moved.get(key) is not None:
            moved[key] = moved[key].to(device, non_blocking=True)
    return moved


def sentence_sampling_weights(
    records: list[dict[str, Any]], settings: DataSettings
) -> torch.DoubleTensor:
    counts = letter_label_counts(records, settings.num_labels)
    reference = float(np.median(counts[counts > 0]))
    class_weights = np.ones(settings.num_labels, dtype=np.float64)
    for label, count in enumerate(counts):
        if count > 0:
            class_weights[label] = min(
                settings.sampler_max_weight, math.sqrt(reference / count)
            )
    class_weights = np.maximum(class_weights, 1.0)
    weights = []
    for record in records:
        present = {
            label
            for char, label in zip(record["chars"], record["labels"])
            if char != " "
        }
        weights.append(max([class_weights[label] for label in present] or [1.0]))
    return torch.as_tensor(weights, dtype=torch.double)


def make_loader(
    records: list[dict[str, Any]],
    settings: DataSettings,
    *,
    batch_size: int,
    training: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    dataset = DiacritizationDataset(records, settings)
    sampler = None
    if training:
        generator = torch.Generator()
        generator.manual_seed(seed)
        weights = sentence_sampling_weights(records, settings)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=generator
        )
    loader_workers = 0 if settings.dual_gpu_active else settings.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_batch, pad_id=settings.pad_id),
        persistent_workers=loader_workers > 0,
    )


def effective_number_weights(
    records: list[dict[str, Any]],
    *,
    num_labels: int,
    beta: float,
    cap: float,
) -> torch.Tensor:
    counts = letter_label_counts(records, num_labels).astype(np.float64)
    weights = np.zeros(num_labels, dtype=np.float64)
    present = counts > 0
    weights[present] = (1.0 - beta) / (1.0 - np.power(beta, counts[present]))
    weights[present] /= weights[present].mean()
    weights[present] = np.clip(weights[present], 0.25, cap)
    return torch.tensor(weights, dtype=torch.float32)
