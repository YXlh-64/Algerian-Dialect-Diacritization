# DualRoPE v9 Calibration Campaign

## Decision

Cross-fitted temperature calibration and simplex architecture stacking failed
the robust gate. No Kaggle submission was generated. Under the approved gated
sequence, Low-rank BoundaryCRF, Word-position DualRoPE, and the final train+dev
refit are not started.

## Controlled design

The four frozen production-v7 architecture groups were used:

1. DualRoPE-CRF-v7 seed 42.
2. Mean of existing DualRoPE-CE-v6 seeds 42/43/44.
3. Mean of existing HGL-v4 seeds 42/43/44.
4. Mean of the five legacy seed-42 architectures.

For each of five balanced sentence folds:

- fit four positive temperatures and four nonnegative weights summing to one;
- optimize multiclass NLL on the other four folds with deterministic LBFGS;
- apply the unchanged V2 fallback after stacking;
- evaluate once on the held-out fold.

The predeclared robust gate required at least +10 letters, at least four
improving folds, and zero regressing folds.

## Result

| System | Correct / 15,897 | Accuracy |
|---|---:|---:|
| Production v7 equal-group ensemble + V2 | 14,977 | 0.9421274454 |
| Cross-fitted calibration/stacking + V2 | 14,968 | 0.9415613009 |

Fold correct-letter deltas:

```text
[-7, -4, +2, 0, 0]
```

The candidate lost nine letters, improved only one fold, and regressed two.
It fails every robust acceptance dimension.

The full-dev diagnostic fit assigned weights approximately:

```text
CRF-v7       0.3464
DualRoPE-CE  0.3090
HGL-v4       0.2568
Legacy       0.0877
```

Those values improve NLL on the fit data but do not improve cross-fitted
official accuracy. They must not be used for a submission.

## Artifacts

```text
outputs/dual_rope_v9/01_calibrated_stacking/
  SELECTION.json
  MANIFEST.json
  baseline_paper_metrics.json
  crossfit_paper_metrics.json
  deployment_stacker.json
```

There is intentionally no `SUBMIT_THIS` CSV.
