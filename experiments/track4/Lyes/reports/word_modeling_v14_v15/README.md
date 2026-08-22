# Word-Level Modeling Campaign: v14 and v15

## Decision basis

This Track 4 campaign starts from the accepted `DziriFormer-DualRoPE-CRF-v7`
backbone. It does not inherit v13 because v13 improved the standalone model but
failed the pre-registered replacement-ensemble gate.

The leakage-controlled oracle passed on both train-only calibration splits:

| K | Split A recoverable letters / words | Split B recoverable letters / words | Gate |
|---:|---:|---:|---|
| 4 | +1,024 / +970 | +1,035 / +965 | Pass |
| 8 | +1,347 / +1,177 | +1,332 / +1,159 | Pass |

The required threshold was only +20 letters and +10 exact words on each split.
This demonstrates that the v7 CRF already assigns high enough scores to many
gold word sequences, but sentence Viterbi frequently selects another candidate.
The next justified question is therefore candidate selection rather than more
encoder capacity.

## v14: Filtered Word Lattice

```text
sentence characters and spaces
              |
              v
        frozen v7 encoder
              |
              +----------> contextual character states
              |
              v
       frozen v7 emissions + CRF
              |
              v
     K unique paths per known word span
        (K = 4 or K = 8)
              |
              v
 context projection 256 -> 128
 label embeddings 16 x 128
 one pre-LN Transformer block
 4 heads, FFN 256, dropout 0.10
              |
              v
 masked mean pool -> scalar residual
              |
              v
 fixed CRF score + learned residual
              |
              v
 exact sentence-level word-lattice DP
              |
              v
 one 16-class label per scored letter
```

The v7 encoder, emission head, CRF transitions, and candidate set are frozen.
Only the small candidate-composition scorer is trained. Sentence decoding keeps
the original v7 start, end, and cross-word transition scores. No manually tuned
lexical weight is introduced. Uncovered gold words remain errors during
training and evaluation.

Execution:

```bash
.venv/bin/python -B -m experiments.track4.Lyes.filtered_word_lattice_v14 \
  --stage oracle \
  --device mps

.venv/bin/python -B -m experiments.track4.Lyes.filtered_word_lattice_v14 \
  --stage all \
  --device mps
```

## v15: Context Contrastive Shared Encoder

```text
                         shared DualRoPE encoder
                         /                   \
                        /                     \
          complete sentence view       isolated word views
                    context                 isolated
                        \                     /
                         \                   /
                 [context; isolated; |difference|]
                               |
                         sigmoid scalar gate
                               |
          context + gate * Linear(isolated - context)
                               |
                    unchanged v7 emissions + CRF
                               |
                    one label per scored letter
```

The residual projection is initialized to exactly zero, so the initial model is
numerically identical to v7. A training-only ambiguity index marks word
positions with multiple observed labels as context-dependent. Because a high
fusion gate admits more isolated-word residual, the auxiliary
context-dependency probability is `1 - gate`. The objective is CRF negative
log-likelihood plus this complementary gate binary cross-entropy. Coefficients
`{0.1, 0.3, 1.0}` are selected on split A and only the winner is confirmed on
split B.

Execution:

```bash
.venv/bin/python -B -m experiments.track4.Lyes.context_contrastive_v15 \
  --stage all \
  --device mps
```

## Data audit

The audit is review-only and never mutates the dataset. It produced exact CRF
sentence NLL, length-normalized NLL, word gold marginal NLL, v7/v13
disagreements, ambiguity and rare-class evidence, and a deterministic union of
top-50 review queues.

```bash
.venv/bin/python -B -m experiments.track4.Lyes.model_assisted_audit_v2 --device mps
```

Artifacts are under `audits/model_assisted_v2/`; the current review queue has
189 rows.

## Validation

```bash
.venv/bin/python -B -m pytest -q
```

The tests cover exact CRF K-best paths, deterministic deduplication, baseline
candidate inclusion, lattice Viterbi/forward-backward against brute force,
normalized per-letter marginals, frozen v7 gradients, zero-residual v15
equivalence, isolated-word alignment, ambiguity targets, checkpoint round trips,
and locked campaign configurations.

No generated CSV is approved merely because it exists. Acceptance is recorded
in each campaign `SELECTION.json`; no Kaggle upload is automated.

## Final decisions

| Experiment | Result | Decision |
|---|---|---|
| K=4/K=8 oracle | Both K values passed on both splits | Continue to learned word-level models |
| FilteredWordLattice-v14 | Mean +7 letters / +7 exact words | Rejected at calibration; no released-dev run |
| ContextContrastive-v15 calibration | Coefficient 0.3 passed A and B; 11 epochs locked | Continue to released dev |
| ContextContrastive-v15 released dev | +86 neural and +20 V2 letters; V2 Shadda -2 | Rejected by protected-metric gate |

The v15 architecture is useful paper evidence that contrastive sentence/word
views improve total, word, sentence, and OOV accuracy. It is not an approved
final system because the pre-registered Shadda non-regression requirement was
not met. Consequently, the campaign stops before replacement ensembling,
additional seeds, and refitting.

Detailed decisions:

- `outputs/filtered_word_lattice_v14/FINAL_RESULTS.md`
- `outputs/context_contrastive_v15/FINAL_RESULTS.md`
- `outputs/context_contrastive_v15/03_final_seed42/SELECTION.json`
