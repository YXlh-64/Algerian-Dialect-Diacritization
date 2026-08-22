import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.track4.SmailRoumaissa.cnn import MultiKernelCNNFrontend


class RelativePositionBias(nn.Module):
    """Learned scalar bias per (head, bucketed relative distance), added
    directly to attention logits before softmax. Bidirectional buckets:
    exact for small |distance|, log-spaced for larger ones, so it extends
    gracefully to sequence lengths beyond anything seen in training --
    important here since diacritic dependencies (case endings, gemination,
    assimilation) are fundamentally about *distance*, not absolute position.
    Computed once per forward pass in `Backbone` and shared by every block,
    which is the standard T5 setup and keeps the extra parameter count tiny
    (num_buckets * n_heads).
    """
    def __init__(self, n_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.embedding = nn.Embedding(num_buckets, n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int, max_distance: int) -> torch.Tensor:
        num_buckets_half = num_buckets // 2
        ret = (relative_position > 0).long() * num_buckets_half
        n = relative_position.abs()

        max_exact = num_buckets_half // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6) / math.log(max_distance / max_exact)
            * (num_buckets_half - max_exact)
        ).long()
        val_if_large = torch.clamp(val_if_large, max=num_buckets_half - 1)

        return ret + torch.where(is_small, n, val_if_large)

    def forward(self, qlen: int, klen: int, device) -> torch.Tensor:
        q_pos = torch.arange(qlen, device=device)[:, None]
        k_pos = torch.arange(klen, device=device)[None, :]
        rel_pos = k_pos - q_pos
        bucket = self._relative_position_bucket(rel_pos, self.num_buckets, self.max_distance)
        values = self.embedding(bucket)
        return values.permute(2, 0, 1).unsqueeze(0)


class RelativeMultiHeadAttention(nn.Module):
    """Self-attention with an additive relative position bias baked into the
    logits (in place of `nn.MultiheadAttention` + absolute position
    embeddings). Hand-rolled since `nn.MultiheadAttention` has no hook for
    injecting a custom bias term."""
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor, rel_bias: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores + rel_bias

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))

        probs = self.attn_drop(F.softmax(scores, dim=-1))
        out = (probs @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = RelativeMultiHeadAttention(dim, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor, rel_bias: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, key_padding_mask, rel_bias)
        x = x + self.ff(self.ln2(x))
        return x


class Backbone(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, dim: int = 256, n_layers: int = 6,
                 n_heads: int = 8, ff_dim: int = 1024, kernels=(3, 5, 7),
                 max_len: int = 512, dropout: float = 0.15,
                 rel_pos_buckets: int = 32, rel_pos_max_distance: int = 128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.emb_norm = nn.LayerNorm(dim)
        self.emb_drop = nn.Dropout(dropout)
        self.cnn = MultiKernelCNNFrontend(dim, kernels, dropout)
        self.rel_pos_bias = RelativePositionBias(n_heads, rel_pos_buckets, rel_pos_max_distance)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        x = self.emb_drop(self.emb_norm(x))
        x = self.cnn(x, attn_mask)
        key_padding_mask = ~attn_mask
        T = x.size(1)
        rel_bias = self.rel_pos_bias(T, T, x.device)
        for blk in self.blocks:
            x = blk(x, key_padding_mask, rel_bias)
        return self.final_norm(x)
