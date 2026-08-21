# Evaluation Report - CANINE-S

**Track:** track2 | **Model:** canine_s | **Strategy:** strategy_a

## Summary

| Evaluation split | Metric | Result |
|---|---|---:|
| Training | Final training loss | 0.5646 |
| Validation | Micro-F1 | 0.9452 |
| Validation | Accuracy | 0.9452 |
| Test | F1-score, Kaggle private leaderboard | **0.93235** |



## Model and training setup

- **Backbone:** `google/canine-s`
- **Maximum sequence length:** 512
- **Training epochs:** 10
- **Training batch size:** 8
- **Evaluation batch size:** 16
- **Gradient accumulation:** 2 steps
- **Learning rate:** `2e-5`
- **Weight decay:** `0.01`
- **Warmup ratio:** `0.1`
- **Seed:** 42


## Validation evaluation

### Aggregate metrics


- **Micro-F1:** 0.9452
- **Accuracy:** 0.9452
- **validation loss:** 0.1733

A second, independent character-by-character validation sanity check produced the following values:

- **Accuracy:** 0.9449
- **Micro-F1:** 0.9449
- **Macro-F1:** 0.4504
- **Weighted-F1:** 0.9442
- **Characters evaluated:** 19,160


### Per-class metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Diacritic | 0.9925 | 0.9945 | 0.9935 | 8164 |
| Fatha | 0.9110 | 0.9009 | 0.9059 | 3613 |
| Fathatan | 0.0000 | 0.0000 | 0.0000 | 1 |
| Damma | 0.9226 | 0.9046 | 0.9135 | 922 |
| Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Kasra | 0.9465 | 0.9397 | 0.9431 | 1261 |
| Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Sukoon | 0.9091 | 0.9383 | 0.9235 | 4552 |
| Shadda | 0.0000 | 0.0000 | 0.0000 | 3 |
| Shadda+Fatha | 0.8234 | 0.7266 | 0.7720 | 417 |
| Shadda+Fathatan | 0.0000 | 0.0000 | 0.0000 | 1 |
| Shadda+Damma | 0.6774 | 0.6774 | 0.6774 | 31 |
| Shadda+Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Kasra | 0.8824 | 0.7554 | 0.8140 | 139 |
| Shadda+Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Sukoon | 0.3429 | 0.2143 | 0.2637 | 56 |
| **Accuracy** |  |  | **0.9449** | **19,160** |
| **Macro average** | **0.4630** | **0.4407** | **0.4504** | **19,160** |
| **Weighted average** | **0.9438** | **0.9449** | **0.9442** | **19,160** |


## Test evaluation

- Submission columns: `Id`, `Label`
- Submission rows: 16,438
- Label range: 0-15
- Observed labels: 0, 1, 3, 5, 7, 9, 11, 13, 15

**Kaggle private leaderboard F1-score: 0.93235**