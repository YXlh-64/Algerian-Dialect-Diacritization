# DziriFormer-Large-11M experiment

## Purpose

This experiment tests whether increasing the Track 4 neural model from
5,085,962 to 11,315,338 parameters improves character-level vocalization.
It evaluates:

1. `DziriFormer-Large-11M`, the large neural model alone.
2. `DziriFusion-Large-11M-v1`, the same checkpoint followed by the
   training-only lexical prior.

## Architecture change

| Parameter | Original model | Large model |
|---|---:|---:|
| Model dimension | 256 | 384 |
| Attention heads | 8 | 12 |
| Transformer blocks | 6 | 6 |
| FFN dimension | 1,024 | 1,536 |
| Dropout | 0.15 | 0.20 |
| Local attention window | 64 | 64 |
| CNN kernels | 3, 5, 7 | 3, 5, 7 |
| Parameters | 5,085,962 | 11,315,338 |

The character vocabulary, CNN frontend, shifted-window attention pattern,
factorized 8-class base-diacritic head, binary shadda head, optimizer, learning
rate, and dataset remain unchanged.

## Training result

```text
Best epoch:        18
Best dev Micro-F1: 0.9147637919
Train F1 at best:  0.9335197546
Stopped at epoch:  28
Final train F1:    0.9568750376
Final dev F1:      0.9141347424
Training time:     1,482.66 seconds on Apple MPS
```

The widening train/dev gap shows that the larger model overfits the current
dataset. Scaling capacity alone did not improve generalization.

## Results

| System | Parameters | Dev Micro-F1 | Correct |
|---|---:|---:|---:|
| Original neural model | 5,085,962 | **0.9174687048** | 14,585 / 15,897 |
| DziriFormer-Large-11M | 11,315,338 | 0.9147637919 | 14,542 / 15,897 |
| Original DziriFusion-v1 | 5,085,962 | **0.9362772850** | 14,884 / 15,897 |
| DziriFusion-Large-11M-v1 | 11,315,338 | 0.9345788514 | 14,857 / 15,897 |

The large neural model is 0.2705 percentage points below the original neural
model. The fused large model is 0.1698 percentage points below the original
DziriFusion-v1.

## Reproduction

Train the large model:

```bash
python -m training.track4.Lyes.train \
  --config configs/track4/Lyes/dziriformer_large_11m.json
```

Generate the neural-only submission:

```bash
python -m evaluation.track4.Lyes.infer \
  --checkpoint outputs/dziriformer_large_11m_seed42/best.pt \
  --input Data/test_data/raw_sentences_test.txt \
  --ids Data/test_data/raw_sentences_test_ids.txt \
  --vocalized-output outputs/dziriformer_large_11m_seed42/DZIRIFORMER_LARGE_11M_TEST_VOCALIZED.txt \
  --submission outputs/dziriformer_large_11m_seed42/DZIRIFORMER_LARGE_11M_SUBMISSION.csv \
  --sample-submission Data/test_data/sample_submission.csv \
  --manifest outputs/dziriformer_large_11m_seed42/DZIRIFORMER_LARGE_11M_MANIFEST.json \
  --system-name DziriFormer-Large-11M
```

Generate the fused large-model submission:

```bash
python -m evaluation.track4.Lyes.dziri_fusion \
  --checkpoint outputs/dziriformer_large_11m_seed42/best.pt \
  --output-dir outputs/dziriformer_large_11m_seed42 \
  --system-name DziriFusion-Large-11M-v1 \
  --artifact-prefix DZIRIFUSION_LARGE_11M_V1
```

## Kaggle submission files

### Neural-only large model

```text
outputs/dziriformer_large_11m_seed42/DZIRIFORMER_LARGE_11M_SUBMISSION.csv
```

SHA-256:

```text
6a75e01c77eb8a4c1800a6352a112366d466585ad158981cfe6a51272baca1fe
```

Kaggle description:

> DziriFormer-Large-11M: 11.32M-parameter from-scratch character model with
> multi-kernel CNN features, six shifted-window Transformer blocks, and
> factorized diacritic/shadda heads. Best dev Micro-F1: 0.91476 at epoch 18.
> No pretrained models or external data.

### Fused large model

```text
outputs/dziriformer_large_11m_seed42/DZIRIFUSION_LARGE_11M_V1_SUBMISSION.csv
```

SHA-256:

```text
bc9aed31b3ebeca175f84101840922687422c06a493d8dacb96730e2f1298ffc
```

Kaggle description:

> DziriFusion-Large-11M-v1: DziriFormer-Large-11M combined with a smoothed
> training-only lexical prior for seen words; unseen words remain neural-only.
> Best dev Micro-F1: 0.93458. No pretrained models or external data.

## Decision

Both files are valid and match the official submission converter. They can be
submitted as capacity-ablation experiments, but neither is expected to beat
the existing 5.1M counterpart based on dev results. The current best local
system remains the original `DziriFusion-v1` at 0.9362772850 dev Micro-F1.
