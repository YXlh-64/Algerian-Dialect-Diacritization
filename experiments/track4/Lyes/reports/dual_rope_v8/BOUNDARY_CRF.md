# DziriFormer-DualRoPE-BoundaryCRF-v8

## Status and scope

This is a controlled Track 4 seed-42 experiment. Seeds 43/44 and another
multi-seed ensemble are intentionally deferred. No pretrained model, external
corpus, external embedding, analyzer, or tokenizer is used.

## Exact architecture

The encoder is unchanged from `DziriFormer-DualRoPE-CE-v6` and
`DziriFormer-DualRoPE-CRF-v7`:

1. Character vocabulary with spaces preserved as encoder inputs.
2. A 256-dimensional character embedding and RoPE positional encoding.
3. Local stream: six pre-LayerNorm Transformer blocks, eight heads, hidden
   size 256, FFN size 1024, window 16, GELU, dropout 0.15.
4. Global stream: four pre-LayerNorm full-attention Transformer blocks with
   the same dimensions.
5. Cross-attention with local states as queries and global states as keys and
   values.
6. A learned sigmoid fusion gate between the local and cross-attended states.
7. Two full-attention refinement blocks.
8. A linear projection to 16 emission logits per input position.

The only architectural change from `DualRoPE-CRF-v7` is the CRF transition
model:

- `T_within[previous_label, current_label]` is used between consecutive scored
  letters belonging to the same word.
- `T_boundary[previous_label, current_label]` is used when the current scored
  letter is immediately preceded by a space in the original character input.
- The first scored letter uses the normal learned CRF start vector, not
  `T_boundary`.
- The final scored letter uses the normal learned CRF end vector.
- Spaces remain visible to the encoder but never enter CRF loss or Viterbi
  decoding.

This adds exactly 256 trainable parameters to CRF-v7, for 9,890,352 total.

## Motivation

The ordinary CRF uses one transition matrix for two linguistically different
events: transitions inside a word and transitions across words. BoundaryCRF
separates those statistics without changing the neural encoder, emissions,
optimizer, schedule, data, or random seed. This makes the comparison a clean
ablation.

## Training

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_dual_rope_boundary_crf_v8.json \
  --num-workers 0
```

The full run must report `"device": "mps"` and writes:

```text
outputs/dziriformer_dual_rope_boundary_crf_v8_seed42/
  best.pt
  last.pt
  metrics.jsonl
  resolved_config.json
  summary.json
```

## Evaluation and artifact generation

```bash
python -m experiments.track4.Lyes.dual_rope_boundary_crf_v8 \
  --device mps \
  --num-workers 0
```

Artifacts are written below:

```text
outputs/dual_rope_v8/01_boundary_crf_seed42/
  SELECTION.json
  artifacts/
    DZIRIFORMER_DUALROPE_BOUNDARY_CRF_V8_SEED42_NEURAL_SUBMISSION.csv
    DZIRIFORMER_DUALROPE_BOUNDARY_CRF_V8_SEED42_V2_SUBMISSION.csv
    DZIRIFORMER_DUALROPE_BOUNDARY_CRF_V8_SEED42_MANIFEST.json
```

`SELECTION.json` is authoritative:

- accept the architecture only if neural correct is greater than 14,816, the
  standard CRF-v7 seed-42 result;
- recommend the standalone V2 submission only if V2 correct is greater than
  14,977, the current best four-group ensemble;
- otherwise keep the CSVs as documented ablations and do not submit them.

## Ordered follow-up

1. If BoundaryCRF passes its neural gate, evaluate it as a replacement for the
   ordinary CRF expert in the current final ensemble.
2. Build a sentence-disjoint, cross-fitted learned lexical switch. The
   classifier is trained only on neural/lexical disagreements and replaces
   manually selected fallback thresholds.
3. If the learned gate passes, add training-only prefix/suffix priors for OOV
   words behind a low-confidence gate.
4. Run paper ablations only after the competitive system is fixed.
5. Return to seeds 43/44 and multi-seed confirmation when compute time is
   available.

The controlled replacement ensemble command is:

```bash
python -m experiments.track4.Lyes.dual_rope_boundary_crf_v8_ensemble \
  --device mps \
  --num-workers 0
```

It preserves the four equal architecture groups from v7 and changes only the
CRF checkpoint. A canonical `SUBMIT_THIS` CSV is created only if the candidate
strictly exceeds 14,977 released-dev correct letters.

If that gate passes, run the next ordered experiment:

```bash
python -m experiments.track4.Lyes.dual_rope_v8_crossfit_gate \
  --device mps \
  --num-workers 0
```

This experiment creates five balanced, sentence-disjoint folds over released
dev. For each held-out fold, the logistic gate is fitted only on disagreement
examples from the other four folds. The neural checkpoints remain frozen.
The deployment gate is fitted on all dev disagreements only after the
cross-fitted score is computed. It must strictly exceed 14,978 correct letters
before a canonical `SUBMIT_THIS` artifact is created.
