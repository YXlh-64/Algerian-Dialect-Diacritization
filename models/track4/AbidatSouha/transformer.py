"""Transformer encoder written from scratch (notebook §3).

`nn.TransformerEncoderLayer` and `nn.MultiheadAttention` are deliberately not
used -- the track requires the encoder to be built by hand.
"""

import math

import torch
import torch.nn as nn

from configs.track4.AbidatSouha.model_config import ModelConfig
from models.track4.AbidatSouha.layers import RMSNorm, SwiGLU


class T5RelBias(nn.Module):
    "Bidirectional bucketed relative position bias, added to the attention logits."

    def __init__(self, n_heads, n_buckets=32, max_dist=64):
        super().__init__()
        self.n_buckets, self.max_dist = n_buckets, max_dist
        self.emb = nn.Embedding(n_buckets, n_heads)
        nn.init.normal_(self.emb.weight, std=0.02)

    def _bucket(self, rel):
        num = self.n_buckets // 2
        ret = (rel > 0).long() * num
        n = rel.abs()
        max_exact = num // 2
        large = max_exact + (torch.log(n.float().clamp(min=1) / max_exact)
                             / math.log(self.max_dist / max_exact)
                             * (num - max_exact)).long()
        return ret + torch.where(n < max_exact, n, large.clamp(max=num - 1))

    def forward(self, T, device):
        pos = torch.arange(T, device=device)
        b = self._bucket(pos[None, :] - pos[:, None])
        return self.emb(b).permute(2, 0, 1).unsqueeze(0)          # (1,H,T,T)


class MHSA(nn.Module):
    "Multi-head self-attention, implemented by hand."

    def __init__(self, d, h, p, same_word_bias=True):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(p)
        self.sw = nn.Parameter(torch.zeros(h)) if same_word_bias else None

    def forward(self, x, mask, rel=None, wid=None):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)          # (B,H,T,T)
        if rel is not None:
            att = att + rel
        if self.sw is not None and wid is not None:
            same = (wid.unsqueeze(2) == wid.unsqueeze(1)) & (wid.unsqueeze(2) >= 0)
            att = att + same.unsqueeze(1).float() * self.sw.view(1, -1, 1, 1)
        att = att.masked_fill(~mask[:, None, None, :], float("-inf"))
        att = self.drop(att.softmax(-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o(y)


class EncoderLayer(nn.Module):
    "Pre-norm block: x + Attn(RMSNorm(x)) ; x + FFN(RMSNorm(x))"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.att = MHSA(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.same_word_bias)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask, rel, wid):
        x = x + self.drop(self.att(self.n1(x), mask, rel, wid))
        x = x + self.ffn(self.n2(x))
        return x
