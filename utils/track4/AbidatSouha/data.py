"""Dataset loading, encoding and batching (notebook §2, plus build_char_prior from §7).

One change from the notebook: `DiacData` took a `Cfg` and joined hard-coded
subpaths onto `cfg.data_dir`. Here it takes a `DataPaths` from
`configs.track4.AbidatSouha.paths`, so the same code runs locally and on Kaggle.
"""

import collections
import json
from pathlib import Path

import numpy as np
import torch

from configs.track4.AbidatSouha.paths import DataPaths
from utils.track4.AbidatSouha.constants import NUM_CLASSES
from utils.track4.AbidatSouha.features import featurize, word_ids


def load_jsonl(p):
    with Path(p).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


class DiacData:
    def __init__(self, paths: DataPaths):
        with Path(paths.vocab).open(encoding="utf-8") as stream:
            self.vocab = json.load(stream)
        self.unk   = self.vocab["<UNK>"]
        self.train = load_jsonl(paths.train)
        self.dev   = load_jsonl(paths.dev)
        # train lexicon -> evaluation buckets
        self.lex = collections.defaultdict(collections.Counter)
        for r in self.train:
            for wi, wo in zip(r["input"].split(), r["target"].split()):
                self.lex[wi][wo] += 1
        self.ambiguous = {k for k, v in self.lex.items() if len(v) > 1}

    def encode(self, recs):
        out = []
        for r in recs:
            chars = r["chars"]
            out.append(dict(ids=[self.vocab.get(c, self.unk) for c in chars],
                            labels=r["labels"], feats=featurize(chars),
                            wid=word_ids(chars), chars=chars))
        return out


def collate(batch, char_dropout=0.0, unk=1, train=False):
    B = len(batch); T = max(len(b["ids"]) for b in batch)
    ids   = torch.zeros(B, T, dtype=torch.long)
    lab   = torch.full((B, T), -100, dtype=torch.long)
    mask  = torch.zeros(B, T, dtype=torch.bool)
    feats = torch.zeros(6, B, T, dtype=torch.long)
    wid   = torch.full((B, T), -2, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n]  = torch.tensor(b["ids"])
        lab[i, :n]  = torch.tensor(b["labels"])
        mask[i, :n] = True
        wid[i, :n]  = torch.tensor(b["wid"])
        for k in range(6):
            feats[k, i, :n] = torch.tensor(b["feats"][k])
    if train and char_dropout > 0:
        drop = (torch.rand(B, T) < char_dropout) & mask
        ids = torch.where(drop, torch.full_like(ids, unk), ids)
    return ids, feats, lab, mask, wid


def build_char_prior(data, train_enc, smoothing=1.0):
    V = len(data.vocab)
    cnt = np.full((V, NUM_CLASSES), smoothing)
    for r in train_enc:
        for i, l in zip(r["ids"], r["labels"]):
            cnt[i, l] += 1
    return torch.tensor(np.log(cnt / cnt.sum(1, keepdims=True)), dtype=torch.float32)
