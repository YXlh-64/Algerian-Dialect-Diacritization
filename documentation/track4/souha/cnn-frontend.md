# CNN front-end

Three residual depthwise-separable convolution blocks with GLU gating, applied
**before** the transformer encoder.

Implementation: [`models/track4/souha/cnn.py`](../../../models/track4/souha/cnn.py).
Enabled by `ModelConfig.use_conv`; kernel widths from `ModelConfig.conv_kernels`,
default `(3, 5, 7)`.

## Why convolution before attention

Arabic vocalisation is driven by **templatic morphology**: root-and-pattern
skeletons, the `al-` article, the `-ش` negation clitic, verbal prefixes like
`ن-` and `ي-`. These are character n-grams — fixed-width, translation-invariant
patterns that mean the same thing wherever they appear in a word.

Self-attention can represent such patterns, but it has to *learn* that position
is what matters, using content-based matching plus a position signal, from
133,032 labelled positions. A convolution has translation invariance and locality
built into its structure, so it learns the same patterns from far fewer examples.

The division of labour is deliberate:

- **Convolution** handles local, fixed-width morphology — prefixes, suffixes,
  consonant clusters, the `ال` sequence.
- **Attention** handles the long-range, content-dependent decisions — agreement
  across a word, disambiguating a form from sentence context.

Placing the convolution first means the encoder receives representations in
which local morphology has already been resolved.

## Block structure

For each kernel width `k` in (3, 5, 7), in sequence:

```
x ──► RMSNorm ──► transpose to (B,192,T)
                       │
                       ▼
        depthwise Conv1d(192 → 384, kernel=k, groups=192, padding=k//2)
                       │
                  chunk into a, g   each (B,192,T)
                       │
                 a * sigmoid(g)          ← GLU gating
                       │
        pointwise Conv1d(192 → 192, kernel=1)
                       │
              transpose back to (B,T,192)
                       │
                   Dropout(0.25)
                       │
                   × mask               ← padded positions zeroed
                       │
x ──────────────► (+) ─────────────────► output
```

Three design points:

**Depthwise separable.** `groups=192` means each channel is convolved
independently, then a 1×1 pointwise convolution mixes channels. A dense
`Conv1d(192, 192, k=7)` would cost 192×192×7 ≈ 258k parameters; the separable
form costs 192×2×7 + 192×192 ≈ 39k. The three blocks together are 118,656
parameters, 6.2% of the model.

**GLU gating.** The depthwise convolution outputs `2×192` channels, split into a
value `a` and a gate `g`, combined as `a * sigmoid(g)`. The gate lets the block
suppress its own output where the n-gram pattern is not present, rather than
adding noise to every position.

**Masked residual.** The residual branch is multiplied by the padding mask
before being added, so padded positions cannot leak content into real positions
through the convolution window. Without this, a kernel of width 7 near the end
of a short sentence would mix padding into the last three real characters.

**Increasing widths.** 3 → 5 → 7 applied in sequence gives an effective
receptive field of 13 characters, which comfortably covers a word (mean word
length in this data is well under that) plus its immediate context — enough for
`ال` + stem, or a prefix + root, without reaching across the sentence.

## Cost

118,656 parameters. Unlike the CRF, this component is cheap at runtime too:
convolutions parallelise across the sequence, so it adds little wall-clock time
compared to the encoder it feeds.
