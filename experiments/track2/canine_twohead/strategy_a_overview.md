# Track 2 — CANINE-S two-head

## Selected model

`canine_s_twohead` uses `google/canine-s` directly on character sequences. Its
classification head factorizes the 16 competition labels into two predictions:

```text
label = 8 * shadda + vowel
```

The two log-probability distributions are recombined into the original 16
classes, so the training and inference interface remains compatible with the
competition submission format.

## Selection evidence

The values below are recorded from `canine.ipynb` on the held-out development
split. Accuracy and micro-F1 are identical here because the metric is exact
character-label accuracy; DER is `1 - accuracy`.

| Approach | Dev accuracy / micro-F1 | DER | Decision |
|---|---:|---:|---|
| Linear CANINE baseline | 0.9406 | 0.0594 | Ablation |
| Factorized two-head CANINE | **0.9406** | **0.0594** | **Selected** |
| Five-fold two-head ensemble | 0.9379 | 0.0621 | Not selected |
| BiLSTM head experiment | 0.3083 | 0.6917 | Not selected |

The two-head model has the same rounded accuracy as the linear baseline but a
slightly better recorded macro-F1 (`0.7040` vs `0.7032`), and is the intended
Track 2 submission implementation. The ensemble is not selected because its
held-out estimate is lower than the single model.

## Reproduction

From the repository root:

```bash
python run_pipeline.py \
  --track track2 \
  --head-type canine_twohead \
  --model canine_s_twohead \
  --data-dir /path/to/data
```

The training script restores the best dev epoch in memory and exports the
model to `working/exports/track2/canine_s_twohead/`. To fit the final model on
train plus dev and create the competition submission:

```bash
python training/track2/canine_twohead/finetune_canine_twohead.py \
  --data-dir /path/to/data \
  --train-on-all \
  --predict-test \
  --submission-path working/exports/track2/canine_s_twohead/submission.csv
```

Weights and generated CSV files stay in `working/`, which is intentionally
gitignored. The branch contains the code and configuration needed to reproduce
them rather than a machine-specific checkpoint.
