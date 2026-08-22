"""The full tagger (notebook §7).

**Factorized output head.** The label scheme satisfies `label = 8*shadda + base`
exactly (base 0..7 = none / fatha / fathatan / damma / dammatan / kasra /
kasratan / sukoon). Predicting the two factors separately lets *shadda+kasra* --
which has only 1,108 training examples on its own -- borrow all 10,521 kasra
examples for its vowel decision and all 5,315 shadda examples for its gemination
decision.

**Why the interaction table is needed.** A purely additive head assumes shadda
and base are conditionally independent. They are not: label 8 (shadda with no
vowel) occurs **18** times, whereas independence predicts **1,632**. A
context-independent `2x8` table (16 parameters) restores full 16-way
expressivity while keeping the statistical sharing.

**Per-character logit prior.** Initialised from training log-frequencies (see
`utils.track4.AbidatSouha.data.build_char_prior`), so the model learns the *residual*
from the unigram distribution instead of re-deriving `alif -> bare` from
scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.track4.AbidatSouha.model_config import ModelConfig
from models.track4.AbidatSouha.cnn import ConvFrontEnd
from models.track4.AbidatSouha.crf import CRF
from models.track4.AbidatSouha.layers import RMSNorm, SinPos
from models.track4.AbidatSouha.transformer import EncoderLayer, T5RelBias


class DiacModel(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int = None,
                 char_prior: torch.Tensor = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        K = cfg.num_classes
        # the loaded vocab.json is the source of truth; cfg.vocab_size is the default
        vocab_size = cfg.vocab_size if vocab_size is None else vocab_size

        self.emb = nn.Embedding(vocab_size, d, padding_idx=cfg.pad_id)
        if cfg.use_features:
            self.fembs = nn.ModuleList([nn.Embedding(s, d) for s in cfg.feat_sizes])
        self.sin = SinPos(d) if cfg.rel_pos == "sinusoidal" else None
        self.rel = (T5RelBias(cfg.n_heads, cfg.rel_buckets, cfg.rel_max_dist)
                    if cfg.rel_pos == "t5" else None)
        self.in_norm = RMSNorm(d); self.in_drop = nn.Dropout(cfg.dropout)
        self.conv = ConvFrontEnd(d, cfg.conv_kernels, cfg.dropout) if cfg.use_conv else None
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(d)

        if cfg.factorized_head:
            self.h_shadda = nn.Linear(d, 2)
            self.h_base   = nn.Linear(d, 8)
            self.inter_tab = nn.Parameter(torch.zeros(2, 8)) if cfg.interaction else None
        else:
            self.head = nn.Linear(d, K)

        if cfg.char_prior:
            p = char_prior if char_prior is not None else torch.zeros(vocab_size, K)
            self.prior = nn.Parameter(p.clone())
        else:
            self.prior = None

        self.aux = nn.Linear(d, 2) if cfg.aux_diac_head else None
        self.crf = CRF(K, cfg.split_crf) if cfg.use_crf else None

    def encode(self, ids, feats, mask, wid):
        x = self.emb(ids)
        if self.cfg.use_features:
            for k, e in enumerate(self.fembs):
                x = x + e(feats[k])
        if self.sin is not None:
            x = self.sin(x)
        x = self.in_drop(self.in_norm(x)) * mask.unsqueeze(-1)
        if self.conv is not None:
            x = self.conv(x, mask)
        rel = self.rel(ids.size(1), ids.device) if self.rel is not None else None
        for l in self.layers:
            x = l(x, mask, rel, wid)
        return self.out_norm(x)

    def emissions(self, ids, feats, mask, wid):
        h = self.encode(ids, feats, mask, wid)
        if self.cfg.factorized_head:
            ls = F.log_softmax(self.h_shadda(h), -1)      # (B,T,2)
            lb = F.log_softmax(self.h_base(h),   -1)      # (B,T,8)
            em = ls.unsqueeze(-1) + lb.unsqueeze(-2)      # (B,T,2,8)
            if self.inter_tab is not None:
                em = em + self.inter_tab
            em = em.reshape(*h.shape[:2], self.cfg.num_classes)  # label = 8*shadda + base
        else:
            em = self.head(h)
        if self.prior is not None:
            em = em + self.prior[ids]
        return em, h
