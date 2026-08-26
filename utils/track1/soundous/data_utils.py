"""JSONL loading + length-bucketed Dataset/DataLoader for the diacritization tagging task.

`chars` is carried through collate_batch (as a plain python list, not tensorized) because
evaluation/track1/soundousndous/metrics.py needs word-boundary (space) positions to compute WER/WordAcc.
"""
import json
import math
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class DiacritizationDataset(Dataset):
    def __init__(self, rows, char2idx, label2idx, unk_idx: int, no_diac_idx: int = 0):
        self.rows = rows
        self.char2idx = char2idx
        self.label2idx = label2idx
        self.unk_idx = unk_idx
        self.no_diac_idx = no_diac_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        chars = row["chars"]
        labels = row["labels"]
        if labels and isinstance(labels[0], str):
            labels = [self.label2idx.get(l, self.no_diac_idx) for l in labels]
        char_ids = [self.char2idx.get(c, self.unk_idx) for c in chars]
        return {
            "sent_id": row.get("sent_id", str(i)),
            "chars": chars,
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "length": len(char_ids),
        }


def make_collate_fn(pad_idx: int, no_diac_idx: int = 0):
    def collate_batch(batch):
        batch = sorted(batch, key=lambda x: x["length"], reverse=True)
        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
        max_len = lengths.max().item()
        char_ids = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
        labels = torch.full((len(batch), max_len), no_diac_idx, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, b in enumerate(batch):
            L = b["length"]
            char_ids[i, :L] = b["char_ids"]
            labels[i, :L] = b["labels"]
            mask[i, :L] = True
        return {
            "char_ids": char_ids, "labels": labels, "mask": mask, "lengths": lengths,
            "sent_ids": [b["sent_id"] for b in batch], "chars": [b["chars"] for b in batch],
        }
    return collate_batch


class BucketBatchSampler(torch.utils.data.Sampler):
    """Groups similar-length examples into the same batch (less pad waste), shuffles batch
    ORDER each epoch (not within-batch composition, which would defeat bucketing)."""

    def __init__(self, lengths, batch_size, shuffle=True, bucket_mult=50):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = batch_size * bucket_mult

    def __iter__(self):
        idx = np.arange(len(self.lengths))
        if self.shuffle:
            np.random.shuffle(idx)
        batches = []
        for i in range(0, len(idx), self.bucket_size):
            bucket = idx[i:i + self.bucket_size]
            bucket = bucket[np.argsort(self.lengths[bucket])]
            for j in range(0, len(bucket), self.batch_size):
                batches.append(bucket[j:j + self.batch_size].tolist())
        if self.shuffle:
            np.random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)


def make_loader(rows, char2idx, label2idx, pad_idx, unk_idx, no_diac_idx=0,
                 batch_size=64, shuffle=True) -> DataLoader:
    ds = DiacritizationDataset(rows, char2idx, label2idx, unk_idx, no_diac_idx)
    lengths = np.array([len(r["chars"]) for r in rows])
    sampler = BucketBatchSampler(lengths, batch_size, shuffle=shuffle)
    collate_fn = make_collate_fn(pad_idx, no_diac_idx)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn)


class CurriculumBatchSampler(BucketBatchSampler):
    """Shortest-first for the first `curriculum_epochs` epochs (curriculum learning), then
    falls back to normal bucketed+shuffled behavior. Call set_epoch(epoch) at the top of each
    training epoch before iterating the DataLoader."""

    def __init__(self, lengths, batch_size, curriculum_epochs=5, **kwargs):
        super().__init__(lengths, batch_size, **kwargs)
        self.curriculum_epochs = curriculum_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        if self.current_epoch < self.curriculum_epochs:
            idx = np.argsort(self.lengths)
            return iter([idx[i:i + self.batch_size].tolist() for i in range(0, len(idx), self.batch_size)])
        return super().__iter__()


def make_curriculum_loader(rows, char2idx, label2idx, pad_idx, unk_idx, no_diac_idx=0,
                            batch_size=64, curriculum_epochs=5):
    ds = DiacritizationDataset(rows, char2idx, label2idx, unk_idx, no_diac_idx)
    lengths = np.array([len(r["chars"]) for r in rows])
    sampler = CurriculumBatchSampler(lengths, batch_size, curriculum_epochs=curriculum_epochs, shuffle=True)
    collate_fn = make_collate_fn(pad_idx, no_diac_idx)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn)
    return loader, sampler
