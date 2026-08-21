# Lyes — Track 4 DziriFormer research pipeline

This directory documents the from-scratch Track 4 architecture campaign from
the ConvLocal baseline through DualRoPE, CRF variants, calibrated ensembles,
SWA, R-Drop, word lattices, and context-contrastive training. No pretrained
model, external corpus, external embedding, or external morphological analyzer
is used.

## Repository layout

| Path | Contents |
|---|---|
| `configs/track4/Lyes/` | Frozen JSON configurations and acceptance gates |
| `models/track4/Lyes/` | DziriFormer, CRF, context, and word-lattice models |
| `training/track4/Lyes/` | Deterministic trainer and R-Drop objectives |
| `evaluation/track4/Lyes/` | Inference, metrics, lexical fusion, and export validation |
| `experiments/track4/Lyes/` | v7-v15 campaign runners and complete reports |
| `tests/track4/Lyes/` | Deterministic unit and integration tests |
| `utils/track4/Lyes/` | Data, labels, checkpoints, configuration, and shared helpers |

The consolidated decision ledger is
[`experiments/track4/Lyes/RESULTS.md`](../../../experiments/track4/Lyes/RESULTS.md).
Full released-dev paper metrics are in
[`experiments/track4/Lyes/results/PAPER_METRICS.md`](../../../experiments/track4/Lyes/results/PAPER_METRICS.md).

## Main strategy sequence

1. ConvLocal, gated, global/local, mixed, and hierarchical baselines.
2. DualRoPE CE v6 with parallel local/global streams and learned fusion.
3. DualRoPE CRF v7 with exact sequence NLL and Viterbi decoding.
4. BoundaryCRF v8 plus cross-fitted lexical arbitration.
5. Calibrated stacking v9 and WordPos/factorized/low-rank CRF v10.
6. Context-boundary v11 and deterministic SWA v12.
7. Emission R-Drop v13, filtered word lattice v14, and context contrastive v15.

Every promotion is controlled by a predeclared acceptance gate. A higher
overall score is not promoted when a protected metric regresses.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The original experiment configurations use the competition's `Data/` layout.
When the dataset is elsewhere, pass `--train-data`, `--dev-data`, and `--vocab`
explicitly; data and checkpoints are not committed.

## Reproduce the core runs

Train DualRoPE CE v6:

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_dual_rope_ce_v6.json \
  --seed 42 \
  --num-workers 0
```

Train DualRoPE CRF v7:

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/model.json \
  --num-workers 0
```

Evaluate the v7 checkpoint:

```bash
python -m evaluation.track4.Lyes.evaluate \
  --campaign-config configs/track4/Lyes/campaign.json \
  --device auto \
  --num-workers 0
```

Re-export the accepted equal-group ensemble after its listed checkpoints exist:

```bash
python -m experiments.track4.Lyes.export_ensemble \
  --config configs/track4/Lyes/campaign.json \
  --stage crf_final \
  --device auto \
  --num-workers 0
```

## Verify

```bash
python -m pytest tests/track4/Lyes -q
python -m compileall -q models/track4/Lyes training/track4/Lyes \
  evaluation/track4/Lyes experiments/track4/Lyes utils/track4/Lyes
```

## Artifact policy

- Commit source, frozen configurations, tests, reports, hashes, and decisions.
- Do not commit datasets, checkpoints, generated predictions, or submissions.
- Never submit an artifact marked rejected or diagnostic in the reports.
- The approved primary submission is the v7 final ensemble documented in
  `experiments/track4/Lyes/reports/dual_rope_v7/FINAL_RESULTS.md`.
