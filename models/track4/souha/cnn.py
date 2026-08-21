"""CNN front-end (notebook §4).

Depthwise-separable convolutions with GLU gating, kernel widths 3/5/7, applied
*before* the encoder. Arabic vocalisation is driven by templatic morphology --
root-and-pattern skeletons, the `al-` article, the `-sh` negation clitic, verbal
prefixes. Those are character n-grams: fixed-width, translation-invariant
patterns that a convolution learns from far fewer examples than self-attention
needs to learn them from scratch.
"""

import torch
import torch.nn as nn

from models.track4.souha.layers import RMSNorm


class ConvFrontEnd(nn.Module):
    def __init__(self, d, kernels, p):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                norm=RMSNorm(d),
                dw=nn.Conv1d(d, 2 * d, k, padding=k // 2, groups=d),  # depthwise
                pw=nn.Conv1d(d, d, 1),                                # pointwise
            )) for k in kernels])
        self.drop = nn.Dropout(p)

    def forward(self, x, mask):
        for b in self.blocks:
            h = b["norm"](x).transpose(1, 2)
            h = b["dw"](h)
            a, g = h.chunk(2, dim=1)
            h = a * torch.sigmoid(g)                                  # GLU
            h = b["pw"](h).transpose(1, 2)
            x = x + self.drop(h) * mask.unsqueeze(-1)
        return x
