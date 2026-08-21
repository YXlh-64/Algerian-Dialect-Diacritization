# DualRoPE v8 Final Results

## Decision

`DziriFormer-DualRoPE-BoundaryCRF-v8` passes the controlled neural
architecture gate. Replacing the ordinary CRF-v7 expert in the existing
four-group ensemble also passes, by one released-dev letter. The subsequent
cross-fitted learned lexical gate fails and is rejected. The dependent OOV
affix gate is therefore not run.

No new seed-43/44 models were trained in this iteration. The accepted final
ensemble reuses the already completed CE and HGL seed groups; only the new
BoundaryCRF checkpoint is seed 42.

## Results

All scores use the same 607-sentence released dev split with 15,897 scored
letters.

| System | Change from preceding control | Dev correct | Dev Micro-F1 | Decision |
|---|---|---:|---:|---|
| DualRoPE-CE-v6 seed 42 | Direct independent 16-class CE | 14,764 | 0.9287286909 | Historical control |
| DualRoPE-CRF-v7 seed 42 | One global 16×16 CRF transition matrix | 14,816 | 0.9319997484 | Accepted |
| **DualRoPE-BoundaryCRF-v8 seed 42** | Separate within-word and boundary 16×16 matrices | **14,837** | **0.9333207523** | **Accepted: +21** |
| CRF-v7 seed 42 + fixed V2 | Existing confidence-gated word prior | 14,962 | 0.9411838712 | Historical |
| BoundaryCRF-v8 seed 42 + fixed V2 | Same V2, no gate change | 14,977 | 0.9421274454 | Tie with old final; no standalone submission |
| Final CRF-v7 four-group ensemble + V2 | Previous best local system | 14,977 | 0.9421274454 | Control |
| **Final BoundaryCRF-v8 four-group ensemble + V2** | Replace only CRF-v7 group | **14,978** | **0.9421903504** | **Accepted: +1** |
| BoundaryCRF-v8 ensemble + cross-fitted logistic gate | Replace fixed V2 thresholds | 14,973 | 0.9418758256 | Rejected: −5 |

Relative to direct CE, ordinary CRF removes 52 errors. Boundary conditioning
removes another 21 errors, for 73 fewer errors than CE. The standalone decoder
result is meaningful; the +1 final-ensemble gain is too small to claim robust
superiority until seeds 43/44 are completed.

## BoundaryCRF training record

```json
{
  "best_dev_micro_f1": 0.9333207523432094,
  "best_epoch": 21,
  "device": "mps",
  "elapsed_seconds": 1474.1806738376617,
  "epochs_completed": 25,
  "optimizer_updates": 1900,
  "parameter_count": 9890352,
  "seed": 42
}
```

Checkpoint:

```text
outputs/dziriformer_dual_rope_boundary_crf_v8_seed42/best.pt
SHA-256: 500a7bb547cec1dc1cb924357a638f5d3d552eea54f720997dbf08186845d3b6
```

## Accepted submission

Upload exactly:

```text
outputs/dual_rope_v8/SUBMIT_THIS_DZIRI_FINAL_BOUNDARY_CRF_V8.csv
```

Validation:

```text
Rows including header: 16,439
Prediction rows:       16,438
SHA-256: d7586183a3593e71ad127a1636d4a65c018f97279d8200afb90a5394dcd58f75
Official make_submission.py byte comparison: PASS
```

Kaggle description:

> Track 4 from-scratch BoundaryCRF-v8 ensemble. Replaces the standard
> DualRoPE CRF expert with separate within-word and cross-word transition
> matrices. Equal 1/4 architecture-group probability averaging across
> BoundaryCRF, existing 3-seed DualRoPE-CE, existing 3-seed HGL, and the
> legacy five-model expert; unchanged confidence-gated V2 lexical fallback.
> Dev: 0.942190 (14,978/15,897).

## Files not to submit

- `01_boundary_crf_seed42/..._NEURAL_SUBMISSION.csv` is the neural ablation.
- `01_boundary_crf_seed42/..._V2_SUBMISSION.csv` ties the previous final
  system and failed the strict standalone gate.
- `02_boundary_crf_final_ensemble/..._NEURAL_SUBMISSION.csv` omits V2.
- `03_crossfit_gate/..._SUBMISSION.csv` failed its five-fold gate.

Their artifacts remain immutable evidence, but none is the recommended Kaggle
file.

## Cross-fitted gate finding

The meta-model used five balanced sentence folds, eight standardized
confidence/frequency/position features, deterministic full-batch LBFGS, and a
fixed 0.5 threshold. Each fold trained on 255–274 valid disagreements from the
other four folds; the final deployment fit had only 331 disagreements. This
small and correlated sample did not generalize better than the simpler V2
rules. The result supports retaining V2 rather than tuning another threshold
on dev.

## Ordered next work

1. Submit the accepted v8 CSV and record public/private Kaggle scores.
2. When compute time is available, train BoundaryCRF seeds 43/44 with the exact
   frozen v8 config. Accept a multi-seed BoundaryCRF expert only if the
   three-seed mean improves and at least two seeds beat ordinary CRF controls.
3. Do not resume the OOV affix branch on the rejected learned gate. Revisit it
   only as a separately cross-validated neural fallback with a new explicit
   acceptance contract.
4. For the paper, report CE → CRF → BoundaryCRF as the primary decoder
   ablation and fixed V2 → cross-fitted gate as the lexical arbitration
   ablation.
