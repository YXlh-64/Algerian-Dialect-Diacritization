# DziriFormer-DualRoPE-CE-v6

## Status

`DziriFormer-DualRoPE-CE-v6` is implemented as a new, self-contained Track 4
experiment. The architecture, configuration, CPU forward/backward tests,
checkpoint round-trip test, and real-dev-data CPU smoke forward are complete.
No full seed-42 training result is claimed until the MPS command below
finishes and writes `summary.json`.

Verified locally before handoff:

| Check | Result |
|---|---:|
| Full repository tests | 54 passed |
| Dedicated v6 tests | 7 passed |
| CPU smoke epochs | 1 |
| CPU smoke optimizer updates | 608 |
| CPU smoke dev Micro-F1 | `0.6964207083` |
| CPU smoke runtime | 4.10 seconds |

The small-model smoke score is only a pipeline check. It is not an experiment
result and its checkpoint must not be submitted.

Authoritative configuration:

```text
configs/track4/Lyes/dziriformer_dual_rope_ce_v6.json
```

Expected output:

```text
outputs/dziriformer_dual_rope_ce_v6_seed42/
```

## Research question

The previous Track 4 systems used one character stream: a CNN followed by
local/shifted-local attention, with optional periodic full attention or a
word-level hierarchy. This experiment tests a different hypothesis:

> Can independent local and sentence-global character representations,
> aligned with RoPE and combined by a learned gate, improve direct
> 16-class vocalization?

This is deliberately the **CE variant**. It isolates the parallel RoPE
encoder from a CRF and from lexical post-processing. A CRF can be tested later
as a separate ablation only if this encoder is useful.

## Complete architecture

```text
Input character IDs, including spaces; maximum length 512
                         │
                         ▼
Shared token embedding: vocabulary 43 × 256
No learned absolute position embedding
                         │
              LayerNorm + dropout 0.15
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
      LOCAL STREAM              GLOBAL STREAM
  6 pre-LN RoPE blocks       4 pre-LN RoPE blocks
  window = 16 characters     full bidirectional attention
  8 heads × 32 dimensions    8 heads × 32 dimensions
  FFN 256 → 1024 → 256       FFN 256 → 1024 → 256
  GELU, dropout, residual     GELU, dropout, residual
  fixed non-shifted windows   complete sentence context
            │                         │
            └────────────┬────────────┘
                         ▼
              RoPE cross-attention
                 Q = local states
                 K,V = global states
                         │
          residual + dropout + LayerNorm
                         │
                         ▼
           Feature-wise adaptive fusion
      g = sigmoid(Wg [local ; cross] + bg)
      fused = g ⊙ local + (1-g) ⊙ cross
      gate shape = batch × length × 256
      Wg and bg initialize to zero, so g = 0.5
                         │
                    LayerNorm
                         │
                         ▼
       2 full-attention pre-LN RoPE blocks
       8 heads, FFN 1024, dropout 0.15
                         │
                    LayerNorm
                         │
                         ▼
             Linear 256 → 16 logits
                         │
                         ▼
      Unweighted official-label cross-entropy
                         │
                         ▼
       One label 0–15 per scored Arabic letter
```

The model contains exactly:

| Component | Parameters |
|---|---:|
| Shared token embedding | 11,008 |
| Input LayerNorm | 512 |
| Dual streams, cross-attention, gate, refinement | 9,873,664 |
| Final LayerNorm | 512 |
| Direct 16-class head | 4,112 |
| **Total** | **9,889,808** |

There are 12 Transformer blocks in total: 6 local, 4 global, and 2
refinement blocks. The cross-attention module is additional.

## RoPE contract

For every attention head, pairs of query/key channels are rotated by their
absolute character position:

```text
theta(i, k) = i / 10000^(2k / head_dim)
RoPE(x, i)  = x ⊙ cos(theta) + rotate_pairs(x) ⊙ sin(theta)
```

RoPE is applied to:

- Q and K in all six local blocks;
- Q and K in all four global blocks;
- local Q and global K in cross-attention;
- Q and K in both refinement blocks.

Values are not rotated. Padded positions are masked. Window packing retains
the original absolute character indices, so local windows do not reset the
RoPE position to zero.

The six local blocks use the fixed 16-character windows described by the
inspiring architecture. They are not shifted. Cross-window communication is
provided by the separate global stream, cross-attention, and the two final
full-attention blocks.

## Fusion behavior

The gate is a learned vector, not a hand-chosen scalar or fixed interpolation
weight:

```text
g[b, i, :] ∈ (0, 1)^256
```

Each character and each hidden feature can choose a different balance between
local and global evidence. Zero initialization makes the first forward pass an
exact `0.5 / 0.5` mixture before learning. `fusion_gate` is returned by the
model for diagnostics but is not an auxiliary training target.

## Output and loss

This model uses one direct 16-class linear head and one loss:

```text
loss = CrossEntropy(label_logits, official_label, ignore_index=-100)
```

There is:

- no factorized base/Shadda head;
- no manually weighted auxiliary loss;
- no CRF or Viterbi decoding;
- no CNN frontend;
- no learned absolute position table;
- no guided-label input;
- no word hierarchy;
- no lexical prior during neural training or neural inference.

Spaces remain ordinary encoder inputs, preserving word boundaries. Spaces,
BOS, EOS, and padding receive `IGNORE_INDEX=-100`, so they do not contribute
to the loss or official Micro-F1.

## Training configuration

| Setting | Value |
|---|---:|
| Seed | 42 |
| Epoch limit | 25 |
| Early-stopping patience | 10 |
| Batch size | 64 |
| Optimizer | AdamW |
| Maximum OneCycle learning rate | `3e-4` |
| Initial learning rate | `3e-4 / 25` |
| OneCycle increasing phase | 30% of optimizer updates |
| Final division factor | 10,000 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Model selection | highest released-dev Micro-F1 |
| Device | MPS |
| MPS arithmetic | FP32 |

The shared trainer enables AMP only on CUDA. Therefore `amp: true` does not
silently enable unsupported CUDA autocast on MPS; the MPS run is FP32 and the
summary must report `"device": "mps"`.

## Exact commands

### 1. Verify the implementation

```bash
source .venv/bin/activate
python -m pytest tests/test_dual_rope_ce_v6.py
```

Optional one-epoch small-model CPU pipeline run:

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_dual_rope_ce_v6_smoke.json
```

### 2. Train seed 42 on the Mac GPU

```bash
source .venv/bin/activate
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_dual_rope_ce_v6.json
```

If macOS blocks DataLoader shared memory, use:

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_dual_rope_ce_v6.json \
  --num-workers 0
```

Do not use the smoke checkpoint for Kaggle.

### 3. Create the neural Kaggle submission

```bash
python -m evaluation.track4.Lyes.infer \
  --checkpoint outputs/dziriformer_dual_rope_ce_v6_seed42/best.pt \
  --input Data/test_data/raw_sentences_test.txt \
  --ids Data/test_data/raw_sentences_test_ids.txt \
  --vocalized-output outputs/dziriformer_dual_rope_ce_v6_seed42/DZIRIFORMER_DUALROPE_CE_V6_TEST_VOCALIZED.txt \
  --submission outputs/dziriformer_dual_rope_ce_v6_seed42/DZIRIFORMER_DUALROPE_CE_V6_NEURAL_SUBMISSION.csv \
  --sample-submission Data/test_data/sample_submission.csv \
  --manifest outputs/dziriformer_dual_rope_ce_v6_seed42/DZIRIFORMER_DUALROPE_CE_V6_NEURAL_MANIFEST.json \
  --system-name DziriFormer-DualRoPE-CE-v6 \
  --device mps
```

Independently reproduce the CSV with the official converter:

```bash
python Data/test_data/make_submission.py \
  --ids Data/test_data/raw_sentences_test_ids.txt \
  --input Data/test_data/raw_sentences_test.txt \
  --pred outputs/dziriformer_dual_rope_ce_v6_seed42/DZIRIFORMER_DUALROPE_CE_V6_TEST_VOCALIZED.txt \
  --out outputs/dziriformer_dual_rope_ce_v6_seed42/DZIRIFORMER_DUALROPE_CE_V6_OFFICIAL_CHECK.csv

cmp \
  outputs/dziriformer_dual_rope_ce_v6_seed42/DZIRIFORMER_DUALROPE_CE_V6_NEURAL_SUBMISSION.csv \
  outputs/dziriformer_dual_rope_ce_v6_seed42/DZIRIFORMER_DUALROPE_CE_V6_OFFICIAL_CHECK.csv
```

`cmp` must exit with status 0.

### 4. Optional V2 fallback as a separately named artifact

Do not mix this result with the neural architecture result:

```bash
python -m evaluation.track4.Lyes.gated_fusion \
  --checkpoint outputs/dziriformer_dual_rope_ce_v6_seed42/best.pt \
  --output-dir outputs/dziriformer_dual_rope_ce_v6_seed42/v2 \
  --artifact-prefix DZIRIFORMER_DUALROPE_CE_V6_V2 \
  --system-name DziriFormer-DualRoPE-CE-v6-plus-V2 \
  --device mps
```

This creates:

```text
outputs/dziriformer_dual_rope_ce_v6_seed42/v2/
├── DZIRIFORMER_DUALROPE_CE_V6_V2_TEST_VOCALIZED.txt
├── DZIRIFORMER_DUALROPE_CE_V6_V2_SUBMISSION.csv
└── DZIRIFORMER_DUALROPE_CE_V6_V2_MANIFEST.json
```

## Predefined decision gates

The seed-42 result is interpreted without changing thresholds after training:

| Gate | Requirement | Interpretation |
|---|---:|---|
| Minimum architecture win | more than 14,680 correct (`0.9234446751`) | Beats seed-42 HGL-v4 |
| Strong neural win | more than 14,751 correct (`0.9279109266`) | Beats the three-seed HGL neural ensemble |
| Reported-reference target | at least approximately `0.9365` | Reaches the colleague's reported DualRoPE+CRF dev result |

The last comparison is informative, not perfectly controlled: the reported
reference includes a CRF and may differ in implementation details. If CE-v6
beats the current neural controls but remains below `0.9365`, the next clean
experiment is the **same frozen v6 encoder plus CRF**, not simultaneous
architecture changes.

Seeds 43 and 44 should run only if seed 42 passes the minimum architecture
gate. Use separate output directories:

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_dual_rope_ce_v6.json \
  --seed 43 \
  --output-dir outputs/dziriformer_dual_rope_ce_v6_seed43

python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_dual_rope_ce_v6.json \
  --seed 44 \
  --output-dir outputs/dziriformer_dual_rope_ce_v6_seed44
```

## Kaggle descriptions

Use this only after the neural CSV has been generated:

> DziriFormer-DualRoPE-CE-v6: 9.89M-parameter from-scratch Track 4 character
> model with parallel six-layer local-window and four-layer global RoPE
> streams, local-to-global cross-attention, a learned feature-wise fusion gate,
> two full-attention refinement blocks, and a direct 16-class cross-entropy
> head. Spaces are retained as context and excluded from scoring. No pretrained
> model, external data, CRF, or lexical fallback is used.

For the separately generated V2 artifact:

> DziriFormer-DualRoPE-CE-v6 + V2: the same 9.89M-parameter dual-stream RoPE
> neural model, followed by the unchanged confidence-gated training-only
> lexical fallback. The Transformer remains primary; the lexical prior is used
> only on low-confidence disagreements. Submitted separately from the
> neural-only result.

## Track 4 compliance

The encoder, attention projections, token embedding, gate, and output head are
randomly initialized and trained only on the released training split. The
model uses no pretrained parameters, external embeddings, external corpus,
morphological analyzer, or external tokenizer. RoPE is a deterministic
position transform, not a pretrained representation. The architecture is
therefore fully within Track 4.
