import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from utils.track4.SmailRoumaissa.constants import SPACE


class Vocab:
    def __init__(self, char2id: Dict[str, int]):
        self.char2id = dict(char2id)
        self.id2char = {i: c for c, i in self.char2id.items()}

    @classmethod
    def load(cls, path: str) -> "Vocab":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw)

    def encode(self, chars: List[str]) -> List[int]:
        unk = self.char2id["<UNK>"]
        return [self.char2id.get(c, unk) for c in chars]

    @property
    def pad_id(self) -> int:
        return self.char2id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.char2id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.char2id["<EOS>"]

    def __len__(self) -> int:
        return len(self.char2id)


class DiacritizationDataset(Dataset):
    def __init__(self, path: str, vocab: Vocab, max_chars: int = 500,
                 has_labels: bool = True, char_dropout_prob: float = 0.0):
        self.examples = []
        self.vocab = vocab
        self.has_labels = has_labels
        self.char_dropout_prob = char_dropout_prob
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                chars = rec["chars"]
                if len(chars) > max_chars - 2:
                    continue
                labels = rec.get("labels") if has_labels else None
                self.examples.append((rec["sent_id"], chars, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        sent_id, chars, labels = self.examples[idx]
        ids = [self.vocab.bos_id] + self.vocab.encode(chars) + [self.vocab.eos_id]

        if self.char_dropout_prob > 0.0:
            unk_id = self.vocab.char2id["<UNK>"]
            for i, c in enumerate(chars, start=1):
                if c != SPACE and random.random() < self.char_dropout_prob:
                    ids[i] = unk_id

        is_letter = [False] + [c != SPACE for c in chars] + [False]
        item = {
            "sent_id": sent_id,
            "chars": chars,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "is_letter": torch.tensor(is_letter, dtype=torch.bool),
        }
        if labels is not None:
            lab = [-100] + [(-100 if c == SPACE else l) for c, l in zip(chars, labels)] + [-100]
            item["labels"] = torch.tensor(lab, dtype=torch.long)
        return item


def collate(batch: List[dict], pad_id: int) -> dict:
    maxlen = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, maxlen), dtype=torch.bool)
    is_letter = torch.zeros((B, maxlen), dtype=torch.bool)
    has_labels = "labels" in batch[0]
    labels = torch.full((B, maxlen), -100, dtype=torch.long) if has_labels else None

    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = b["input_ids"]
        attn_mask[i, :L] = True
        is_letter[i, :L] = b["is_letter"]
        if has_labels:
            labels[i, :L] = b["labels"]

    out = {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "is_letter": is_letter,
        "sent_ids": [b["sent_id"] for b in batch],
        "chars": [b["chars"] for b in batch],
    }
    if has_labels:
        out["labels"] = labels
    return out
