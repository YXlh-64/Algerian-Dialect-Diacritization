# Architecture v10 Campaign — Final Results

## Decision

None of the three new seed-42 architectures passed the predeclared acceptance
gate. `DziriFormer-DualRoPE-CRF-v7` therefore remains the selected neural
architecture. It was refit from scratch on train+dev with an update-matched
schedule for one competition-only Kaggle experiment.

The safe released-dev winner remains:

```text
outputs/dual_rope_v7/SUBMIT_THIS_DZIRI_FINAL_V7.csv
```

The new competition-only candidate is:

```text
outputs/architecture_v10/05_final_train_dev_refit/artifacts/DZIRI_FINAL_TRAIN_DEV_REFIT_V10_V2_SUBMISSION.csv
```

Do not use the refit checkpoint in the paper dev table because the released
dev labels are part of its training data.

## Controlled results

All architecture rows use seed 42, the same train/dev split, 25 epochs,
batch size 64, AdamW, OneCycleLR, and 1,900 optimizer updates.

| System | Parameters | Neural correct | Neural F1 | V2 correct | V2 F1 | OOV correct | Exact words | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| DualRoPE-CRF-v7 control | 9,890,096 | **14,816** | **0.9319997** | 14,962 | 0.9411839 | 2,745 | 3,039 | Control |
| best+last snapshot average | 9,890,096 | 14,815 | 0.9319368 | 14,963 | 0.9412468 | — | — | No |
| WordPos-CRF-v10 | 9,907,504 | 14,807 | 0.9314336 | 14,952 | 0.9405548 | 2,745 | 3,027 | No |
| FactorizedEmission-CRF-v10 | 9,888,554 | 14,780 | 0.9297352 | 14,958 | 0.9409323 | 2,744 | 3,027 | No |
| LowRankBoundaryCRF-v10 | 9,890,160 | 14,800 | 0.9309933 | 14,954 | 0.9406806 | **2,752** | **3,042** | No |

The architecture gate required all of:

- at least 14,831 correct neural letters, a +15 gain over v7;
- more than 2,745 correct OOV letters;
- more than 3,039 exact words;
- Shadda binary accuracy no more than 0.001 below v7;
- Tanween binary accuracy no more than 0.0001 below v7.

A Kaggle artifact also required more than the 14,977-correct production
reference after V2. None of the three ablations qualified, so no misleading
architecture-specific Kaggle CSV was generated.

## Paper metrics for the new models

| Model | Variant | WER | CER | Accuracy | Macro F1 | Word acc. | Sentence acc. | Shadda acc. | Tanween acc. | char-BLEU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WordPos | Neural | 0.217829 | 0.038799 | 0.931434 | 0.522380 | 0.782171 | 0.288303 | 0.983708 | 0.999937 | 0.909282 |
| WordPos | V2 | 0.187339 | 0.033344 | 0.940555 | 0.530153 | 0.812661 | 0.344316 | 0.985846 | 0.999874 | 0.922041 |
| Factorized emission | Neural | 0.217829 | 0.039221 | 0.929735 | 0.514262 | 0.782171 | 0.304778 | 0.983393 | 0.999937 | 0.908347 |
| Factorized emission | V2 | 0.186822 | 0.032890 | 0.940932 | 0.528449 | 0.813178 | 0.362438 | 0.986916 | 0.999874 | 0.922281 |
| Low-rank boundary | Neural | **0.213953** | 0.039058 | 0.930993 | 0.448995 | **0.786047** | **0.326194** | 0.982638 | 0.999874 | 0.908793 |
| Low-rank boundary | V2 | 0.188630 | 0.033442 | 0.940681 | 0.526659 | 0.811370 | 0.359143 | 0.985595 | 0.999874 | 0.921510 |

All skeleton mismatch counts are zero. The complete 16-class precision,
recall, F1, confusion matrices, and aligned predictions are in:

```text
outputs/paper_metrics_v1/ALL_MODELS_SUMMARY.csv
outputs/paper_metrics_v1/PER_CLASS_F1.csv
outputs/paper_metrics_v1/models/
```

## What each experiment established

### Snapshot averaging

Equal probability averaging of v7 `best.pt` and `last.pt` gained one V2
letter, from 14,962 to 14,963. This is below the +10 robustness gate and is
not worth another production expert or Kaggle slot.

### Word-position features

Forward/reverse within-word positions and initial/final flags improved
Shadda and Tanween binary accuracy, but lost 9 total letters and 12 exact
words. The explicit positions duplicate information that the local stream,
spaces, RoPE, and CRF already infer.

### Factorized CRF emissions

The base-diacritic and Shadda heads reduced the parameter count by 1,542 and
improved some binary diagnostics, but lost 36 total letters. Enforcing
conditional independence before the CRF is too restrictive for the official
16-class objective.

### Rank-2 boundary residual

The 64-parameter boundary residual is preferable to an unconstrained
replacement matrix as a research direction. It gained seven OOV letters and
three exact words, but lost 16 letters overall. Boundary specialization helps
lexical generalization while slightly damaging common seen-word transitions.

### Final train+dev refit

The selected v7 architecture was trained on 5,471 released train+dev
sentences for 22 epochs and 1,892 updates, closely matching the original
1,900-update budget. The last epoch was selected without dev-based early
stopping. Its V2 lexical prior was also fit on train+dev.

This artifact has no unbiased local score. Kaggle is its only valid
competition evaluation.
