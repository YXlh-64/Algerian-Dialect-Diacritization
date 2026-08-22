# Architecture

End-to-end view of the model. Each component has its own document; this one
shows how they compose, what shapes flow between them, and what each costs.

Notation: `B` = batch, `T` = padded sequence length, `D` = `d_model` = 192,
`K` = `num_classes` = 16.

## Component chart

```
                      INPUT  (one sentence, characters)
                      "الشمس القمر"
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
   char ids (B,T)                        6 morphological streams (6,B,T)
   vocab.json, 43 symbols                pos_in_word, dist_start, dist_end,
   <PAD>=0  <UNK>=1                      wlen, mater, sun          → features.md
        │                                           │
   nn.Embedding(43, 192)                 6 × nn.Embedding(size, 192)
   padding_idx=0                                    │
        │                                           │
        └──────────────────► (+) ◄──────────────────┘
                              │
                       (B,T,192)
                              │
                    RMSNorm → Dropout(0.25) → × mask
                              │
        ┌─────────────────────┴─────────────────────┐
        │           CNN FRONT-END  (3 blocks)       │   → cnn-frontend.md
        │  k=3 ─► k=5 ─► k=7                        │
        │  each: RMSNorm → depthwise Conv1d(192→384)│
        │        → GLU gate → pointwise Conv1d      │
        │        → residual add, masked             │
        └─────────────────────┬─────────────────────┘
                       (B,T,192)
                              │
        ┌─────────────────────┴─────────────────────┐
        │      TRANSFORMER ENCODER  (4 layers)      │   → transformer.md
        │                                           │
        │  per layer, pre-norm:                     │
        │    x + MHSA(RMSNorm(x))                   │
        │    x + SwiGLU(RMSNorm(x))                 │
        │                                           │
        │  attention logits (B,4,T,T) receive:      │
        │    + T5 relative position bias (1,4,T,T)  │
        │    + same-word bias, one scalar per head  │
        │    + padding mask (−inf)                  │
        └─────────────────────┬─────────────────────┘
                              │
                          RMSNorm
                       h = (B,T,192)
                              │
              ┌───────────────┴────────────────┐
              │                                │
    ┌─────────┴──────────┐          ┌──────────┴──────────┐
    │   OUTPUT HEAD      │          │  AUXILIARY HEAD     │  → output-head.md
    │  → output-head.md  │          │  Linear(192→2)      │
    │                    │          │  "is this position  │
    │ Linear(192→2)      │          │   diacritic-bearing?"│
    │   → log_softmax    │          │  train-time only,   │
    │ Linear(192→8)      │          │  weight 0.3         │
    │   → log_softmax    │          └─────────────────────┘
    │        │           │
    │  outer sum (B,T,2,8)
    │      + interaction table (2,8)
    │      reshape → (B,T,16)
    │      + char prior[ids] (B,T,16)
    └─────────┬──────────┘
              │
       emissions (B,T,16)
              │
    ┌─────────┴──────────────────────────────────┐
    │  LINEAR-CHAIN CRF          → crf.md        │
    │  start (16) · end (16)                     │
    │  intra-word transitions (16,16)            │
    │  inter-word transitions (16,16)            │
    │  selected per position by is_intra_mask    │
    │                                            │
    │  train    → negative log-likelihood        │
    │  inference→ Viterbi decode                 │
    └─────────┬──────────────────────────────────┘
              │
       labels (B,T), one per character
              │
    render() → "الشَّمْس القَمَر"
```

## Forward path in code

`DiacModel.encode` then `DiacModel.emissions` in
[`models/track4/AbidatSouha/tagger.py`](../../../models/track4/AbidatSouha/tagger.py):

| Step | Operation | Output shape |
|---|---|---|
| 1 | `emb(ids)` | `(B,T,192)` |
| 2 | `+ Σ fembs[k](feats[k])` for k=0..5 | `(B,T,192)` |
| 3 | `SinPos` — **only** when `rel_pos="sinusoidal"` (the T1 baseline) | `(B,T,192)` |
| 4 | `in_drop(in_norm(x)) * mask` | `(B,T,192)` |
| 5 | `ConvFrontEnd` — 3 residual blocks | `(B,T,192)` |
| 6 | `T5RelBias(T)` computed once, shared by all layers | `(1,4,T,T)` |
| 7 | 4 × `EncoderLayer(x, mask, rel, wid)` | `(B,T,192)` |
| 8 | `out_norm` → `h` | `(B,T,192)` |
| 9 | factorized head + interaction + prior | `(B,T,16)` |
| 10 | CRF `nll` (train) or `decode` (inference) | scalar / `(B,T)` |

The auxiliary head branches off `h` at step 8 and exists only during training.

## Parameter budget

1,910,460 parameters total for the full model.

| Component | Parameters | Share |
|---|---:|---:|
| Encoder layers (4×) | 1,774,096 | 92.9% |
| CNN front-end | 118,656 | 6.2% |
| Character embedding | 8,256 | 0.4% |
| Feature embeddings (6×) | 5,376 | 0.3% |
| Base head `Linear(192→8)` | 1,544 | 0.1% |
| Character prior `(43,16)` | 688 | <0.1% |
| CRF transitions | 544 | <0.1% |
| Shadda head `Linear(192→2)` | 386 | <0.1% |
| Auxiliary head | 386 | <0.1% |
| Input / output RMSNorm | 384 | <0.1% |
| T5 relative bias table | 128 | <0.1% |
| Interaction table `(2,8)` | 16 | <0.1% |
| **Total** | **1,910,460** | |

Two observations that shape the rest of the design:

**The encoder dominates.** 92.9% of parameters sit in four attention+FFN blocks.
Everything else — every linguistic feature, the entire CRF, the whole output
head — is under 1% combined. The cheap components are effectively free, so the
question for each is only "does it help?", never "is it worth the size?".

**The CRF costs 544 parameters** and is the single most expensive component at
*runtime*, because its forward and Viterbi recursions are sequential loops over
sequence length. Parameter count and compute cost are unrelated here.

For comparison, the plain baseline `T1` (`PLAIN_BASELINE` in
`configs/track4/AbidatSouha/model_config.py`) has 1,785,808 parameters — 93.5% of the
full model. The entire architecture beyond a vanilla Transformer accounts for
6.5% of the weights.

## Ablation switches

Every component is individually switchable through `ModelConfig`, so the
architecture can be reduced to a plain char-level Transformer without touching
model code:

| Switch | Default | Off means |
|---|---|---|
| `rel_pos` | `"t5"` | `"sinusoidal"` for absolute PE, or `"none"` |
| `same_word_bias` | `True` | no per-head intra-word attention bias |
| `use_features` | `True` | character embedding only, no morphology |
| `use_conv` | `True` | no CNN front-end |
| `use_crf` | `True` | per-position softmax, cross-entropy loss |
| `split_crf` | `True` | one shared transition matrix |
| `factorized_head` | `True` | single `Linear(192→16)` |
| `interaction` | `True` | additive head, shadda ⟂ base |
| `char_prior` | `True` | no unigram prior |
| `aux_diac_head` | `True` | no auxiliary loss |

`PLAIN_BASELINE` sets all nine to off; that configuration is the required
track-4 plain char-level Transformer.
