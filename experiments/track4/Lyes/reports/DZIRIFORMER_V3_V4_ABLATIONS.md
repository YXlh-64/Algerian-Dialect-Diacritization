# DziriFormer v3/v4 controlled ablations

## Decision

Keep `DziriFusion-Gated-v2` as the current Kaggle submission. None of the new
seed-42 checkpoints improves its `0.9360885702` dev Micro-F1 after the same
lexical fallback is applied.

`DziriFormer-Hier-v4` is the strongest new neural model. It improves neural
dev Micro-F1 from `0.9174687048` to `0.9191042335` and OOV-letter accuracy
from `0.8403281792` to `0.8466393184`. This supports retaining hierarchical
word context in the future HGL architecture.

## Controlled experiment contract

Every run uses:

- the released 4,864-sentence train split;
- the released 607-sentence dev split;
- seed 42;
- the supplied 43-character vocabulary;
- AdamW, warm-up, cosine decay, early stopping, and the same batch size;
- no pretrained model, pretrained embedding, external corpus, tokenizer, or
  morphological analyzer;
- official 16-class letter-level Micro-F1 for checkpoint selection.

The current `DziriFusion-Gated-v2` fallback was then applied unchanged. Its
thresholds were not retuned for any new checkpoint.

## Results

| Model | Parameters | Epoch | Neural correct | Neural F1 | OOV F1 | V2 correct | V2 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original local | 5,085,962 | 23 | 14,585 | 0.9174687048 | 0.8403281792 | 14,881 | **0.9360885702** |
| J16-Gated-v3 | 5,090,331 | 26 | 14,585 | 0.9174687048 | 0.8444304197 | 14,870 | 0.9353966157 |
| GL-v3 | 5,090,314 | 47 | 14,568 | 0.9163993206 | 0.8359103818 | 14,845 | 0.9338239919 |
| Mixed-v3 | 5,085,962 | 53 | 14,603 | 0.9186009939 | 0.8428526349 | 14,872 | 0.9355224256 |
| Hier-v4 | 6,420,491 | 18 | **14,611** | **0.9191042335** | **0.8466393184** | **14,879** | **0.9359627603** |

## 1. DziriFormer-J16-Gated-v3

### Architecture

The shared character encoder feeds three experts:

1. an eight-class base-diacritic head;
2. a binary Shadda head;
3. a direct 16-class head.

The factorized heads produce a normalized 16-class distribution. A learned
per-character scalar gate mixes that distribution with the direct 16-class
distribution in probability space. The gate is initialized neutrally at
`0.5`, and the complete mixture is trained with one official 16-class
cross-entropy.

There are no static auxiliary-loss weights. The legacy
`shadda_loss_weight` configuration value is ignored when
`head_mode="gated_joint"`.

On dev, the learned joint-expert gate averaged:

| Population | Mean joint-expert weight |
|---|---:|
| All scored letters | 0.75854 |
| Correct predictions | 0.76228 |
| Errors | 0.71696 |
| Shadda labels | 0.69943 |
| Non-Shadda labels | 0.76105 |

The model gained 13 correct OOV letters but lost 13 seen-word letters, giving
the same overall neural score as the original model.

### Train

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_j16_gated_v3.json
```

### V2 fallback artifact

```text
outputs/dziriformer_j16_gated_v3_seed42/fusion_v2/
└── DZIRIFUSION_GATED_V2_SUBMISSION.csv
```

Kaggle description:

> Track 4 from-scratch 5.09M character CNN and shifted-window Transformer with a learned per-character probability gate between a direct 16-class expert and factorized base/Shadda expert. One joint cross-entropy, no static auxiliary-head weights. Existing V2 lexical fallback applied unchanged. Seed 42, best epoch 26.

Decision: do not submit; its fused dev score is 11 letters below current V2.

## 2. DziriFormer-GL-v3

### Architecture

This is the requested direct 2SDiac-style experiment. A 17-entry embedding
represents blank plus the 16 official diacritic labels. Each training example
samples a masking level uniformly from `{0, 0.1, ..., 1.0}`. Gold hints are
revealed according to that level. Dev and test use all-blank hints.

The exact schedule finished 17 neural letters below baseline. On this small
corpus, the likely issue is the gap between partially revealed training inputs
and completely blank inference inputs.

### Train

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_gl_v3.json
```

### V2 fallback artifact

```text
outputs/dziriformer_gl_v3_seed42/fusion_v2/
└── DZIRIFUSION_GATED_V2_SUBMISSION.csv
```

Kaggle description:

> Track 4 from-scratch 5.09M guided-label character Transformer. Training uses 2SDiac-style random masking over gold diacritic hints; validation and test use blank hints only. Existing V2 lexical fallback applied unchanged. Seed 42, best epoch 47.

Decision: do not submit; both neural and fused dev scores regress.

## 3. DziriFormer-Mixed-v3

### Architecture

The six character Transformer blocks use:

```text
windowed, shifted-windowed, full,
shifted-windowed, windowed, full
```

This is a controlled test of mixing windowed and universal/full attention.
It adds no parameters and improves the baseline by 18 neural letters.

### Train

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_mixed_v3.json
```

### V2 fallback artifact

```text
outputs/dziriformer_mixed_v3_seed42/fusion_v2/
└── DZIRIFUSION_GATED_V2_SUBMISSION.csv
```

Kaggle description:

> Track 4 from-scratch 5.09M character CNN Transformer mixing shifted 64-character local attention with full-sequence attention in blocks 3 and 6. Existing V2 lexical fallback applied unchanged. Seed 42, best epoch 53.

Decision: useful architecture signal, but do not submit; fused dev remains nine
letters below current V2.

## 4. DziriFormer-Hier-v4

### Architecture

The model adds:

- learned forward and reverse within-word position embeddings;
- gated character-to-word pooling;
- two full word-level Transformer blocks;
- learned word-context-to-character feature fusion.

Word boundaries are derived deterministically from spaces. Word vectors are
composed from character states, not looked up, so OOV words remain supported.

The hierarchy gains 20 OOV letters and six seen-word letters relative to the
original neural model. The gain survives neural evaluation but is mostly
absorbed by the existing lexical fallback.

### Train

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_hier_v4.json
```

### V2 fallback artifact

```text
outputs/dziriformer_hier_v4_seed42/fusion_v2/
└── DZIRIFUSION_GATED_V2_SUBMISSION.csv
```

Kaggle description:

> Track 4 from-scratch 6.42M hierarchical word-character Transformer. Character-composed word vectors are globally contextualized by two word-level Transformer blocks and gated back into each character; no external embeddings or analyzer. Existing V2 lexical fallback applied unchanged. Seed 42, best epoch 18.

Decision: retain for HGL design, but do not replace current submission; fused
dev is two letters lower.

## HGL go/no-go discussion

Do not combine every mechanism blindly. The ablations support:

- **keep hierarchy**, because it gives the largest neural and OOV gain;
- **keep periodic full attention as an option**, because it gives a small
  positive result without extra parameters;
- **do not carry the current J16 gate by default**, because its net gain is
  zero;
- **do not carry the exact GL masking distribution unchanged**, because it
  regresses.

A defensible HGL experiment should therefore begin as
`Hier + mixed attention`, then test a revised guided curriculum separately:

1. train initially with the exact GL masking distribution;
2. progressively increase the probability of fully blank hints;
3. finish with a blank-only fine-tuning phase;
4. select checkpoints only on blank-hint dev Micro-F1.

This schedule must be implemented and ablated before calling the result
`DziriFormer-HGL-v4`.
