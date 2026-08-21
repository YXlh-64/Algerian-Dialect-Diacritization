# DualRoPE/CRF v7 Campaign — Final Results

## One-file submission decision

Upload only:

```text
outputs/dual_rope_v7/SUBMIT_THIS_DZIRI_FINAL_V7.csv
```

SHA-256:

```text
51dd19b57e3af6498f8b772725cb3737670ae2d44f83a720af0321bf3c625590
```

The file is byte-identical to:

```text
outputs/dual_rope_v7/04_final_crf_ensemble/artifacts/DZIRI_FINAL_DUALROPE_CRF_ENSEMBLE_V7_V2_SUBMISSION.csv
```

That artifact is independently reproduced by the competition's official
`make_submission.py`.

## Kaggle description

> Four-group equal-probability Track 4 ensemble combining a
> DualRoPE-CRF-v7 expert, three-seed DualRoPE-CE-v6, three-seed HGL-v4, and
> the validated five-model legacy expert. The unchanged confidence-gated V2
> lexical fallback is applied after probability averaging. No pretrained
> models, external data, or manually tuned ensemble weights. Released-dev
> Micro-F1: 0.942127.

## Final candidate table

| Candidate | Neural F1 | Neural correct | V2 F1 | V2 correct | Decision |
|---|---:|---:|---:|---:|---|
| DualRoPE-CE seed 42 | `0.9287286909` | 14,764 | `0.9401773920` | 14,946 | Already submitted |
| v7 seed-42 equal-system ensemble | `0.9316852236` | 14,811 | `0.9408693464` | 14,957 | Superseded |
| DualRoPE-CE three-seed expert | `0.9316223187` | 14,810 | `0.9406177266` | 14,953 | Accepted ablation; superseded |
| Three-seed equal-system ensemble | `0.9315594137` | 14,809 | `0.9401144870` | 14,945 | Rejected |
| DualRoPE-CRF seed 42 | `0.9319997484` | 14,816 | `0.9411838712` | 14,962 | Accepted; superseded |
| **Final CRF four-group ensemble** | **`0.9350820910`** | **14,865** | **`0.9421274454`** | **14,977** | **Submit** |

The final candidate improves:

- the submitted DualRoPE-CE + V2 system by 31 dev letters;
- the old five-model campaign winner by 42 dev letters;
- standalone CRF + V2 by 15 dev letters.

## DualRoPE seed validation

| Seed | Best epoch | Neural F1 | Correct | Runtime | Device |
|---:|---:|---:|---:|---:|---|
| 42 | 21 | `0.9287286909` | 14,764 | 750.26 s | MPS |
| 43 | 25 | `0.9282254513` | 14,756 | 973.23 s | MPS |
| 44 | 22 | `0.9287286909` | 14,764 | 1,001.11 s | MPS |

All three seeds exceed the former HGL three-seed neural ensemble reference.
The architecture is therefore stable, even though the three-seed V2 result
does not beat the best seed-42 equal-system ensemble.

## Controlled CRF result

The CRF experiment changes only the decoder and training objective:

```text
DualRoPE-v6 encoder
        │
        ▼
16 emission logits per scored letter
        │
        ▼
First-order linear-chain CRF
  start transitions: 16
  label transitions: 16 × 16
  end transitions: 16
        │
        ▼
Exact sentence NLL during training
Viterbi path during neural inference
Forward-backward marginals for confidence/fusion
```

Spaces remain encoder inputs, but the CRF packs only scored Arabic-letter
positions. The CRF adds 288 parameters:

| Model | Parameters | Best epoch | Neural F1 | Correct |
|---|---:|---:|---:|---:|
| DualRoPE-CE-v6 | 9,889,808 | 21 | `0.9287286909` | 14,764 |
| **DualRoPE-CRF-v7** | **9,890,096** | **22** | **`0.9319997484`** | **14,816** |

The controlled decoder change gains 52 neural letters.

## Final ensemble

Four system groups receive exactly `1/4` probability weight:

1. DualRoPE-CRF-v7 seed 42;
2. mean probability of DualRoPE-CE-v6 seeds 42/43/44;
3. mean probability of HGL-v4 seeds 42/43/44;
4. mean probability of the validated five-model legacy ensemble.

V2 is applied only after this equal system-level probability average. There
are no manually tuned ensemble weights.

## MPS incident

The first seed-43 attempt hit the MPS high-water mark after epoch 16. Its
complete partial artifacts were preserved under:

```text
outputs/dziriformer_dual_rope_ce_v6_seed43_oom_epoch16/
```

The trainer now calls `torch.mps.empty_cache()` at epoch boundaries. Seed 43
was restarted from seed initialization and completed all 25 epochs without
disabling Apple's memory safety threshold. No checkpoint from the interrupted
run enters any ensemble.

## Artifact index

### Submit

```text
outputs/dual_rope_v7/SUBMIT_THIS_DZIRI_FINAL_V7.csv
```

### Optional scientific ablation only

Standalone CRF + V2:

```text
outputs/dual_rope_v7/03_crf_seed42/artifacts/DZIRIFORMER_DUALROPE_CRF_V7_SEED42_V2_SUBMISSION.csv
```

Description:

> Controlled 9.89M-parameter Track 4 DualRoPE-CRF-v7 model. The complete
> parallel local/global RoPE encoder is unchanged; independent CE decoding is
> replaced by a first-order CRF over packed Arabic-letter positions. The
> unchanged V2 lexical fallback uses CRF marginals and Viterbi labels.
> Released-dev Micro-F1: 0.941184.

### Do not submit

```text
outputs/dual_rope_v7/04_final_crf_ensemble/artifacts/DZIRI_FINAL_DUALROPE_CRF_ENSEMBLE_V7_NEURAL_SUBMISSION.csv
outputs/dual_rope_v7/02_multiseed_equal_system/artifacts/DZIRI_ENSEMBLE_DUALROPE_HGL_V7_MULTI_V2_SUBMISSION.csv
outputs/dual_rope_v7/02a_dual_multiseed/artifacts/DZIRIFORMER_DUALROPE_CE_V7_MULTI_NEURAL_SUBMISSION.csv
outputs/dual_rope_v7/01_seed42_equal_system/artifacts/DZIRI_ENSEMBLE_DUALROPE_HGL_V7_SEED42_NEURAL_SUBMISSION.csv
```

Also never upload `OFFICIAL_CHECK.csv`, `TEST_VOCALIZED.txt`, checkpoint,
manifest, diagnostic, or selection files.

## Validation

- All 60 repository tests pass.
- CRF log-partition and Viterbi decoding match brute-force golden cases.
- Interior space masking is tested.
- CRF forward/backward and marginal normalization are tested.
- Existing checkpoint and fusion behavior remains covered.
- Every listed submission was compared byte-for-byte with the official
  converter.

## Reproduction commands

Train DualRoPE-CE seeds:

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_dual_rope_ce_v6.json \
  --seed 43 \
  --output-dir outputs/dziriformer_dual_rope_ce_v6_seed43 \
  --num-workers 0

python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_dual_rope_ce_v6.json \
  --seed 44 \
  --output-dir outputs/dziriformer_dual_rope_ce_v6_seed44 \
  --num-workers 0
```

Train the controlled CRF:

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/model.json \
  --num-workers 0
```

Re-export the final candidate:

```bash
python -m experiments.track4.Lyes.export_ensemble \
  --stage crf_final \
  --device mps \
  --num-workers 0
```

## Next gated work

Do not start another broad architecture campaign yet. First submit the final
v7 file and record both public and private scores. If further work is needed:

1. train CRF seeds 43/44 only if the final v7 Kaggle result confirms the dev
   gain;
2. average CRF seeds before recomputing the four-group ensemble;
3. replace fixed V2 thresholds only with a DualRoPE-specific five-fold OOF
   gate—never with another dev-tuned static multiplier;
4. run local-only/global-only/no-gate ablations for the paper, not for
   leaderboard selection.
