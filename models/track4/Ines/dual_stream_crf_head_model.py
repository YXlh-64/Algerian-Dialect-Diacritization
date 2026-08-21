"""Ines's dual-stream RoPE Transformer with a letter-only CRF head."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    '''Precomputed rotary position embedding table, applied to Q/K per attention call.'''

    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires an even head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)   # (seq, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (seq, dim)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x):
        T = x.size(2)
        return self.cos[:, :, :T, :], self.sin[:, :, :T, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def build_window_mask(seq_len, window, device):
    '''True = blocked (cannot attend). |i - j| > window is disallowed.'''
    idx = torch.arange(seq_len, device=device)
    dist = (idx[None, :] - idx[:, None]).abs()
    return dist > window


# ---------------------------------------------------------------------------
# Attention / transformer blocks
# ---------------------------------------------------------------------------
class MultiHeadAttentionRoPE(nn.Module):
    '''Self-attention with RoPE, an optional local window mask, and key-padding mask.'''

    def __init__(self, dim, n_heads, dropout=0.1, max_seq_len=512):
        super().__init__()
        assert dim % n_heads == 0
        self.h = n_heads
        self.d = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.d, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, window=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.d).transpose(1, 2)

        cos, sin = self.rope(q)
        q, k = apply_rope(q, k, cos, sin)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)

        if window is not None:
            block_mask = build_window_mask(T, window, x.device)
            attn_scores = attn_scores.masked_fill(block_mask[None, None, :, :], float("-inf"))
        if key_padding_mask is not None:
            kp = key_padding_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(kp, float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # guard fully-masked (padded) query rows
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    '''Pre-norm block: MHA(+RoPE, +window) -> residual -> FFN(GELU) -> residual.'''

    def __init__(self, dim, n_heads, dropout=0.1, max_seq_len=512, ff_mult=4):
        super().__init__()
        self.attn = MultiHeadAttentionRoPE(dim, n_heads, dropout, max_seq_len)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, window=None):
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask, window))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    '''Q = local stream, K = V = global stream, via nn.MultiheadAttention.'''

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, local_feats, global_feats, key_padding_mask=None):
        q = self.norm_q(local_feats)
        kv = self.norm_kv(global_feats)
        out, _ = self.mha(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return out


class AdaptiveFusionGate(nn.Module):
    '''gate = sigmoid(Linear([local ; cross])); out = gate*local + (1-gate)*cross.'''

    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim * 2, dim)

    def forward(self, local_feats, cross_feats):
        gate = torch.sigmoid(self.gate_proj(torch.cat([local_feats, cross_feats], dim=-1)))
        return gate * local_feats + (1 - gate) * cross_feats


# ---------------------------------------------------------------------------
# CRF head
# ---------------------------------------------------------------------------
class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.01)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.01)
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.01)

    def forward(self, emissions, tags, mask):
        '''Negative log-likelihood. emissions:(B,T,K) tags:(B,T) mask:(B,T) bool prefix-style.'''
        nonempty = mask.any(dim=1)
        if not nonempty.any():
            return emissions.sum() * 0.0
        gold = self._score(emissions[nonempty], tags[nonempty], mask[nonempty])
        logZ = self._partition(emissions[nonempty], mask[nonempty])
        return (logZ - gold).mean()

    def _score(self, emissions, tags, mask):
        mask = mask.float()
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        T = emissions.size(1)
        for t in range(1, T):
            emit = emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (trans + emit) * mask[:, t]
        seq_lens = mask.sum(1).long()
        last_tag_idx = (seq_lens - 1).clamp(min=0)
        last_tags = tags.gather(1, last_tag_idx.unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def _partition(self, emissions, mask):
        mask = mask.float()
        T = emissions.size(1)
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            broadcast = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            new_alpha = torch.logsumexp(broadcast, dim=1)
            m = mask[:, t].unsqueeze(1)
            alpha = new_alpha * m + alpha * (1 - m)
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask):
        '''Viterbi. Returns a python list of tag-id lists, one per batch item.'''
        B, T, K = emissions.shape
        mask_b = mask.bool()
        history = []
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            best_score, best_idx = broadcast.max(dim=1)
            m = mask_b[:, t].unsqueeze(1)
            score = torch.where(m, best_score, score)
            history.append(best_idx)
        score = score + self.end_transitions.unsqueeze(0)
        seq_lens = mask_b.sum(1).long()

        best_paths = []
        for b in range(B):
            L = seq_lens[b].item()
            if L == 0:
                best_paths.append([])
                continue
            best_last = score[b].argmax().item()
            path = [best_last]
            for t in range(L - 2, -1, -1):
                best_last = history[t][b, best_last].item()
                path.append(best_last)
            path.reverse()
            best_paths.append(path)
        return best_paths


def gather_letters(emissions, tags, letter_mask):
    '''Compact each sequence to just its letter positions (front-packed, zero-padded).
    Returns compact_emissions, compact_tags, compact_mask, orig_idx (for scattering back), lengths.'''
    B, T, K = emissions.shape
    lengths = letter_mask.sum(1)
    max_len = max(lengths.max().item(), 1)
    device = emissions.device

    compact_emissions = torch.zeros(B, max_len, K, device=device, dtype=emissions.dtype)
    compact_tags = torch.zeros(B, max_len, device=device, dtype=tags.dtype)
    compact_mask = torch.zeros(B, max_len, device=device, dtype=torch.bool)
    orig_idx = torch.zeros(B, max_len, device=device, dtype=torch.long)

    for b in range(B):
        idx = letter_mask[b].nonzero(as_tuple=True)[0]
        L = idx.numel()
        if L == 0:
            continue
        compact_emissions[b, :L] = emissions[b, idx]
        compact_tags[b, :L] = tags[b, idx]
        compact_mask[b, :L] = True
        orig_idx[b, :L] = idx
    return compact_emissions, compact_tags, compact_mask, orig_idx, lengths


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class Track4DualStreamCRF(nn.Module):
    '''Dual-stream (local-window + global) cross-attention transformer with a
    CRF head decoded over letter positions only (spaces/pads excluded).
    Was `DSCAT` in the original notebook -- renamed to match the repo's
    Track<N><HeadName> naming convention (see Track3Diacritizer /
    Track3BiLSTMCRF in models/track3/).'''

    def __init__(self, vocab_size, num_labels, dim=256, n_heads=8,
                 local_layers=6, global_layers=4, final_layers=2,
                 local_window=16, dropout=0.15, max_seq_len=512,
                 unscored_label_id=0):
        super().__init__()
        if not 0 <= unscored_label_id < num_labels:
            raise ValueError("unscored_label_id must be a valid label id")
        self.unscored_label_id = unscored_label_id
        self.embed = nn.Embedding(vocab_size, dim)

        self.local_blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout, max_seq_len) for _ in range(local_layers)
        ])
        self.global_blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout, max_seq_len) for _ in range(global_layers)
        ])
        self.local_window = local_window

        self.cross_attn = CrossAttention(dim, n_heads, dropout)
        self.fusion = AdaptiveFusionGate(dim)

        self.final_blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout, max_seq_len) for _ in range(final_layers)
        ])
        self.out_proj = nn.Linear(dim, num_labels)
        self.crf = CRF(num_labels)

    def encode(self, char_ids, pad_mask):
        x = self.embed(char_ids)

        local = x
        for blk in self.local_blocks:
            local = blk(local, key_padding_mask=pad_mask, window=self.local_window)

        glob = x
        for blk in self.global_blocks:
            glob = blk(glob, key_padding_mask=pad_mask, window=None)

        cross_feats = self.cross_attn(local, glob, key_padding_mask=pad_mask)
        fused = self.fusion(local, cross_feats)

        out = fused
        for blk in self.final_blocks:
            out = blk(out, key_padding_mask=pad_mask, window=None)

        return self.out_proj(out)

    def loss(self, char_ids, tags, pad_mask, is_space):
        '''pad_mask: True=PAD. is_space: True=space character.'''
        logits = self.encode(char_ids, pad_mask)
        letter_mask = (~pad_mask) & (~is_space)
        ce, ct, cm, _, _ = gather_letters(logits, tags, letter_mask)
        return self.crf(ce, ct.long(), cm)

    @torch.no_grad()
    def predict(self, char_ids, pad_mask, is_space):
        '''Full-length prediction with spaces/pads set to the unscored label.'''
        logits = self.encode(char_ids, pad_mask)
        letter_mask = (~pad_mask) & (~is_space)
        B, T = char_ids.shape
        dummy_tags = torch.zeros_like(char_ids)
        ce, _, cm, orig_idx, lengths = gather_letters(logits, dummy_tags, letter_mask)
        paths = self.crf.decode(ce, cm)
        full_pred = torch.full(
            (B, T),
            self.unscored_label_id,
            dtype=torch.long,
            device=char_ids.device,
        )
        for b in range(B):
            L = lengths[b].item()
            if L == 0:
                continue
            idx = orig_idx[b, :L]
            full_pred[b, idx] = torch.tensor(paths[b], dtype=torch.long, device=char_ids.device)
        return full_pred

    def forward(self, char_ids, pad_mask, is_space, labels=None):
        '''Thin dispatcher so this model can be called the same way as the
        track3 heads: labels given -> loss, labels=None -> decode.'''
        if labels is not None:
            return self.loss(char_ids, labels, pad_mask, is_space)
        return self.predict(char_ids, pad_mask, is_space)


@torch.no_grad()
def majority_vote_decode(models, char_ids, pad_mask, is_space):
    '''Ensembles multiple Track4DualStreamCRF checkpoints (e.g. different
    seeds/folds) by majority-voting each model's own full-length prediction
    per position. Mirrors models/track3/bilstm_crf_head's
    majority_vote_decode, adapted to this model's (char_ids, pad_mask,
    is_space) call signature instead of track3's token/char-alignment one.'''
    all_preds = torch.stack([m.predict(char_ids, pad_mask, is_space) for m in models], dim=0)  # (M,B,T)
    voted, _ = torch.mode(all_preds, dim=0)  # ties -> lowest tag id, deterministic
    return voted
