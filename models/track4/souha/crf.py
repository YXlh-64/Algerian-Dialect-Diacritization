"""Hand-written linear-chain CRF (notebook §5).

Two transition matrices, selected per position: **intra-word** and
**inter-word**. Motivation from the data -- word-final positions emit
essentially only `{sukoon 52%, bare 46%, fatha 1.4%}`, so transitions *across* a
word boundary are close to independent, and forcing them through the same matrix
as word-internal transitions is a modelling error.

`tests/track4/souha/test_crf.py` checks the forward algorithm and the Viterbi
decode against brute-force enumeration over all K**T label sequences.
"""

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_classes, split=True):
        super().__init__()
        K = num_classes
        self.num_classes, self.split = K, split
        self.intra = nn.Parameter(torch.randn(K, K) * 0.01)
        self.inter = nn.Parameter(torch.randn(K, K) * 0.01) if split else None
        self.start = nn.Parameter(torch.randn(K) * 0.01)
        self.end = nn.Parameter(torch.randn(K) * 0.01)

    def _trans(self, is_intra):                    # (B,) bool -> (B,K,K)
        if not self.split:
            return self.intra.unsqueeze(0).expand(is_intra.size(0), -1, -1)
        return torch.where(is_intra[:, None, None], self.intra, self.inter)

    def _logZ(self, em, mask, is_intra):
        B, T, K = em.shape
        a = self.start.unsqueeze(0) + em[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(a.unsqueeze(2) + self._trans(is_intra[:, t])
                                  + em[:, t].unsqueeze(1), dim=1)
            a = torch.where(mask[:, t].unsqueeze(1), nxt, a)
        return torch.logsumexp(a + self.end.unsqueeze(0), dim=1)

    def _score(self, em, tags, mask, is_intra):
        B, T, K = em.shape
        s = self.start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        ar = torch.arange(B, device=em.device)
        for t in range(1, T):
            tr = self._trans(is_intra[:, t])
            step = tr[ar, tags[:, t - 1], tags[:, t]] \
                 + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            s = s + step * mask[:, t].float()
        last = mask.sum(1) - 1
        return s + self.end[tags[ar, last]]

    def nll(self, em, tags, mask, is_intra):
        tags = tags.clamp(min=0)
        return (self._logZ(em, mask, is_intra)
                - self._score(em, tags, mask, is_intra)).mean()

    @torch.no_grad()
    def decode(self, em, mask, is_intra):
        B, T, K = em.shape
        a = self.start.unsqueeze(0) + em[:, 0]
        ptr = []
        for t in range(1, T):
            sc = a.unsqueeze(2) + self._trans(is_intra[:, t]) + em[:, t].unsqueeze(1)
            best, idx = sc.max(1)
            ptr.append(idx)
            a = torch.where(mask[:, t].unsqueeze(1), best, a)
        a = a + self.end.unsqueeze(0)
        lens = mask.sum(1)
        out = torch.zeros(B, T, dtype=torch.long, device=em.device)
        last = a.argmax(1)
        for b in range(B):
            L = int(lens[b]); y = int(last[b]); out[b, L - 1] = y
            for t in range(L - 1, 0, -1):
                y = int(ptr[t - 1][b, y]); out[b, t - 1] = y
        return out


def is_intra_mask(wid):
    "(B,T) bool: does the transition t-1 -> t stay inside one word?"
    prev = torch.cat([wid[:, :1], wid[:, :-1]], dim=1)
    return (wid == prev) & (wid >= 0)
