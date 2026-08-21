# The Transformer encoder

Written by hand. `nn.TransformerEncoderLayer` and `nn.MultiheadAttention` are
deliberately not used — the track requires the encoder to be built from
primitives. Only `nn.Linear`, `nn.Embedding` and `nn.Dropout` are borrowed, and
those are plain tensor operations.

Implementation:
[`models/track4/souha/transformer.py`](../../../models/track4/souha/transformer.py)
and [`models/track4/souha/layers.py`](../../../models/track4/souha/layers.py).

## Configuration

| Parameter | Value | Note |
|---|---:|---|
| `d_model` | 192 | |
| `n_layers` | 4 | |
| `n_heads` | 4 | head dimension 192/4 = 48 |
| `d_ff` | 512 | 2.67 × `d_model` |
| `dropout` | 0.25 | attention weights, FFN output, residual branches |
| `rel_buckets` | 32 | 16 for each direction |
| `rel_max_dist` | 64 | beyond this, all distances share a bucket |

1,774,096 parameters across the four layers — 92.9% of the model. Per layer:
attention 148,228, feed-forward 294,912, norms 384.

## Layer structure — pre-norm residual

```
        x ──────────────────────────────────┐
        │                                   │
     RMSNorm                                │
        │                                   │
      MHSA(mask, rel, wid)                  │
        │                                   │
     Dropout                                │
        │                                   │
        └────────────► (+) ◄────────────────┘
                        │
        ┌───────────────┴───────────────────┐
        │                                   │
     RMSNorm                                │
        │                                   │
      SwiGLU  (has its own dropout)         │
        │                                   │
        └────────────► (+) ◄────────────────┘
                        │
                     output
```

**Why pre-norm.** Normalising *inside* the residual branch rather than after it
leaves an unnormalised identity path from input to output. Gradients reach early
layers without passing through a normalisation at every step, which is what makes
deep stacks trainable without delicate initialisation. Post-norm would work at
four layers, but pre-norm is strictly less fragile and costs nothing.

## RMSNorm

```python
return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

Root-mean-square normalisation: rescale by the RMS of the feature vector, then
apply a learned per-channel gain. 192 parameters per instance.

**Why not LayerNorm.** LayerNorm subtracts the mean and adds a learned bias.
RMSNorm drops both. The mean-centring step turns out to contribute little — the
useful part of the operation is the rescaling — and removing it saves a
reduction pass and 192 bias parameters per norm. With 10 norm instances in the
model the parameter saving is trivial; the simplification is the point.

## SwiGLU feed-forward

```python
return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))
```

Three projections, all bias-free: `w1: 192→512`, `w2: 192→512`, `w3: 512→192`.
The SiLU-activated branch is gated elementwise by the ungated branch.

**Why gated.** A standard `Linear→ReLU→Linear` FFN applies the same
transformation everywhere and relies on ReLU to zero out unwanted dimensions.
The gate `w2(x)` is a second, learned, input-dependent signal controlling how
much of each hidden dimension passes through. Gated variants consistently
outperform ReLU FFNs at equal hidden width.

**What it costs.** Three matrices instead of two: 294,912 parameters against
196,608 for a vanilla FFN at `d_ff=512`. That 1.5× is why `d_ff` is 512 rather
than the conventional 4×`d_model` = 768 — the width was reduced to keep the
model at a size appropriate for 133k training positions.

## Attention

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)     # (B,4,T,T)
att = att + rel                                          # relative position
att = att + same.unsqueeze(1) * self.sw.view(1,-1,1,1)   # same-word bias
att = att.masked_fill(~mask[:, None, None, :], -inf)     # padding
att = self.drop(att.softmax(-1))
y   = (att @ v)                                          # (B,4,T,48)
```

Standard scaled dot-product attention with `q`, `k`, `v`, `o` projections
(192→192, with bias), plus **three additive terms on the logits** before the
softmax. Adding biases to logits rather than to values keeps the operation a
proper attention distribution — every term shifts the *preference* for a
key, and the softmax renormalises.

### 1. T5 relative position bias

A learned scalar per (bucket, head), added to every query–key logit according to
the signed distance `j - i`.

```
n_buckets = 32, max_dist = 64  →  16 buckets per direction

  sign of (j - i)   ─────────────►  buckets 0–15 (key at or before query)
                                    buckets 16–31 (key after query)

  |j - i| = 0..7    ─────────────►  one bucket each, exact
  |j - i| = 8..63   ─────────────►  logarithmically binned into 8 buckets
  |j - i| ≥ 64      ─────────────►  saturates in the last bucket
```

Concretely, distance 16 lands in bucket 10, distance 32 in bucket 13, distance
64 and beyond in bucket 15. The table is `nn.Embedding(32, 4)` — 128 parameters
— computed once per forward pass and shared by all four layers.

**Why relative rather than absolute.** Diacritization is translation-invariant:
the rule that `ال` + sun letter takes a shadda holds identically at character 3
and at character 200. Absolute sinusoidal encodings force the model to learn
that invariance; relative biases have it by construction.

**Why logarithmic bucketing.** Morphological dependencies are local — a few
characters — while syntactic context is diffuse. Exact resolution for distances
0–7 covers within-word structure precisely; coarse log-spaced buckets beyond
that give a cheap "somewhere to the left, fairly far" signal without spending
parameters on distinctions that carry no information.

**Why it generalises.** Training sentences reach 274 characters, test only 143.
An absolute encoding must have seen a position to have learned it; a relative
bias handles any length because the bucket function is defined for all distances.

The plain baseline `T1` uses `SinPos` — classic absolute sinusoidal encoding —
instead, via `rel_pos="sinusoidal"`.

### 2. Same-word bias

```python
same = (wid.unsqueeze(2) == wid.unsqueeze(1)) & (wid.unsqueeze(2) >= 0)
```

A boolean `(B,T,T)` matrix marking query–key pairs that belong to the same word,
with spaces (`wid = -1`) and padding (`wid = -2`) excluded. One learned scalar
per head — **4 parameters for the whole model** — is added to those logits.

**Why.** The word is the natural unit of this task. Which vocalization a word
takes is mostly determined inside the word, with sentence context only
disambiguating between candidate readings. A per-head scalar lets the model
allocate that division itself: heads that learn a positive value specialise in
within-word morphology, heads that stay near zero or go negative attend across
the sentence. Initialised to zero, so the model starts unbiased and the split is
learned, not imposed.

Four parameters is arguably the best value-per-parameter in the architecture,
though see [results.md](results.md) — the ablation that would prove it has not
been run.

### 3. Padding mask

`mask[:, None, None, :]` masks **keys** only: no query may attend to a padded
position. Query rows at padded positions still produce output, which is
harmless — those positions carry label `-100` and are dropped by the loss, and
are sliced away at inference. Every row of `mask` has at least one `True`, so
the softmax can never see an all-`-inf` row.

## What is deliberately absent

- **No causal mask.** Tagging is bidirectional; the diacritic on a character
  depends on what follows it as much as what precedes.
- **No `[CLS]`/`[SEP]`, no BOS/EOS.** The output must be length-preserving, one
  label per input character. Adding boundary tokens would require stripping them
  back out and offsetting every index in the submission.
- **No pretrained weights.** Every parameter is randomly initialised and trained
  on `train_Algerian-DIAC.jsonl` alone.
