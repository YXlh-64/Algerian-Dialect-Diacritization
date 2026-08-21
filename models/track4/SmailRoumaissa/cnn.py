import torch
import torch.nn as nn


class DepthwiseConv1d(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class MultiKernelCNNFrontend(nn.Module):
    """Local character n-gram features (letter shape / adjacency cues) fed
    in before the Transformer blocks, e.g. detecting sun/moon-letter
    assimilation contexts or common trigrams.

    Fusion across the parallel kernel branches is now GATED: the concatenated
    branch outputs are projected to 2*dim and split into (value, gate), so the
    model can learn -- per position, per channel -- how much of each kernel's
    local-context signal to let through, instead of blending all branches
    with a fixed linear combination.
    """
    def __init__(self, dim: int, kernels=(3, 5, 7), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([DepthwiseConv1d(dim, k) for k in kernels])
        self.proj = nn.Linear(dim * len(kernels), dim * 2)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(~pad_mask.unsqueeze(-1), 0.0)
        branch_out = torch.cat([b(x) for b in self.branches], dim=-1)
        value, gate = self.proj(branch_out).chunk(2, dim=-1)
        h = value * torch.sigmoid(gate)
        h = h.masked_fill(~pad_mask.unsqueeze(-1), 0.0)
        return self.norm(x + self.drop(h))
