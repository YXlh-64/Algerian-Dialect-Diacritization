# ContextContrastive-v15 Results

`DziriFormer-ContextContrastive-v15` is rejected by its pre-registered
released-dev acceptance gate. It produced clear neural and V2 accuracy gains,
but V2 Shadda accuracy regressed by two letters. No replacement ensemble,
seeds 43/44, or train+dev refit was run.

## Architecture

The accepted v7 DualRoPE encoder is shared between two views: the complete
sentence and separately packed isolated words. The two representations are
combined by a learned per-letter gate:

```text
sentence view ---- context -----------\
                                        +--> context + gate * P(isolated-context)
isolated words --- isolated ----------/                  |
                                                           v
                                                unchanged v7 emissions + CRF
```

The residual projection `P` starts at zero, making initialization exactly
equivalent to v7. A training-only ambiguity target supervises `1 - gate`,
because a context-dependent letter should suppress the isolated-word residual.

## Leakage-controlled calibration

Split A selected the auxiliary BCE coefficient from `{0.1, 0.3, 1.0}`. Only
the winner, `0.3`, was confirmed on split B.

| Split / coefficient | Best epoch | Neural correct | Neural gain | Exact-word gain | V2 correct | V2 gain |
|---|---:|---:|---:|---:|---:|---:|
| A / 0.1 | 15 | 24,837 | +109 | +65 | 25,024 | +22 |
| A / 0.3 | 15 | **24,840** | **+112** | **+68** | **25,029** | **+27** |
| A / 1.0 | 15 | 24,835 | +107 | +66 | 25,028 | +26 |
| B / 0.3 | 6 | **24,840** | **+130** | **+73** | **24,990** | **+56** |

Half-up rounding of epochs 15 and 6 locked the final run to 11 epochs.

## Released-dev result

| Variant | Correct / 15,897 | Micro-F1 | Exact words | Exact sentences | OOV accuracy | Shadda accuracy | Tanween accuracy | Skeleton mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Matched v7 neural | 14,816 | 0.932000 | 3,039 | 180 | 0.866204 | 0.983204 | 0.999874 | 0 |
| v15 neural | **14,902** | **0.937410** | **3,099** | **198** | **0.874408** | **0.983645** | **0.999937** | 0 |
| Matched v7 V2 | 14,962 | 0.941184 | 3,146 | 214 | 0.866204 | **0.985783** | 0.999874 | 0 |
| v15 V2 | **14,982** | **0.942442** | **3,158** | **217** | **0.874408** | 0.985658 | 0.999874 | 0 |

The final count gates passed: neural gained `+86` letters and V2 gained `+20`.
Word, sentence, OOV, Tanween, and skeleton metrics did not regress. However,
V2 Shadda correctness fell from 15,671 to 15,669 of 15,897 letters. The
registered no-regression rule therefore rejects the standalone model.

## Decision

- Standalone accepted: **no**.
- Replacement ensemble evaluated: **no**, because the standalone gate failed.
- Seeds 43/44: **not run**.
- Train+dev refit: **not run**.
- Kaggle submission approved: **none**.
- Paper interpretation: positive context-contrastive accuracy ablation, rejected
  final system because the protected Shadda metric was not preserved.

The generated neural and V2 CSVs under `03_final_seed42/model/artifacts/` are
diagnostic artifacts only. They intentionally do not use a `SUBMIT_THIS`
prefix.

Authoritative decision:
`outputs/context_contrastive_v15/03_final_seed42/SELECTION.json`.
