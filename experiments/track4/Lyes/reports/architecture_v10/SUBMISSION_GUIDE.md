# Architecture v10 — Kaggle Submission Guide

## Submit this new experiment

```text
outputs/architecture_v10/05_final_train_dev_refit/artifacts/DZIRI_FINAL_TRAIN_DEV_REFIT_V10_V2_SUBMISSION.csv
```

SHA-256:

```text
0fd94726880e5e0f47d2fd86bcd7028e738c48b2b9a89e5c303a0b163f1068e5
```

Kaggle description:

> DziriFinal-TrainDev-Refit-v10 — Track 4 from-scratch DualRoPE character
> Transformer with direct 16-class emissions and linear-chain CRF. The
> previously selected v7 architecture was refit on released train+dev for
> 1,892 update-matched steps with last-epoch selection. The unchanged
> confidence-gated V2 lexical fallback was fit on train+dev. Competition-only
> final refit; no unbiased local dev score is claimed.

## Keep this as the safe comparison

```text
outputs/dual_rope_v7/SUBMIT_THIS_DZIRI_FINAL_V7.csv
```

Known Kaggle score: public `0.94841`, private `0.94257`.

## Do not submit

Do not upload:

```text
outputs/architecture_v10/05_final_train_dev_refit/artifacts/DZIRI_FINAL_TRAIN_DEV_REFIT_V10_NEURAL_SUBMISSION.csv
```

It intentionally omits V2. No WordPos, factorized-emission, low-rank-boundary,
or snapshot CSV was generated because those experiments failed their gates.

Both generated refit CSVs contain 16,438 rows and are byte-identical to the
competition's official `make_submission.py` output.
