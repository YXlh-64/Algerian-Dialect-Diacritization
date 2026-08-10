# Evaluation Report — p2_ensemble_09483

**Track:** track1 | **Head:** bilstm_cnn_crf | **Strategy:** strategy_a

- Kaggle public score: **0.95048**
- Kaggle private score: **0.94829**
- Dev macro-F1 across all 16 classes: **0.58610**
- Dev macro-F1 across supported classes: **0.78146**
- Dev character accuracy: **0.94395**
- Dev non-space character positions: **15,897**

The experiment trains five independent instances of the same P2 character-level
BiLSTM-CNN-CRF architecture. Architecture and epoch selection use the official
development split. The selected epochs and ensemble hyperparameters are frozen
before the models are refit on train+dev for test inference.

No external training data, pretrained transformer, hand-labeled test data, or
test-label feedback is used. Lexical priors are learned from labeled training
records only.

## Selected seed runs

| Seed | Best epoch | Dev macro-F1 (16) | Dev accuracy | Ensemble weight |
|---|---:|---:|---:|---:|
| 3407 | 9 | 0.53145 | 0.93401 | 0.00 |
| 3408 | 14 | 0.53892 | 0.93766 | 0.00 |
| 3409 | 10 | 0.55182 | 0.93439 | 0.25 |
| 3410 | 17 | 0.55976 | 0.94049 | 0.50 |
| 3411 | 15 | 0.56045 | 0.93609 | 0.25 |

Selected structured-ensemble parameters: lexical strength `0.75`, frequency
adjustment `0.20`, and CRF transition strength `0.25`.

## Per-class development results

| ID | Class | Support | F1 |
|---:|---|---:|---:|
| 0 | No Diacritic | 4,901 | 0.99194 |
| 1 | Fatha | 3,613 | 0.91610 |
| 2 | Fathatan | 1 | 1.00000 |
| 3 | Damma | 922 | 0.94168 |
| 4 | Dammatan | 0 | 0.00000 |
| 5 | Kasra | 1,261 | 0.95886 |
| 6 | Kasratan | 0 | 0.00000 |
| 7 | Sukoon | 4,552 | 0.93178 |
| 8 | Shadda | 3 | 0.66667 |
| 9 | Shadda+Fatha | 417 | 0.82134 |
| 10 | Shadda+Fathatan | 1 | 0.00000 |
| 11 | Shadda+Damma | 31 | 0.82540 |
| 12 | Shadda+Dammatan | 0 | 0.00000 |
| 13 | Shadda+Kasra | 139 | 0.84444 |
| 14 | Shadda+Kasratan | 0 | 0.00000 |
| 15 | Shadda+Sukoon | 56 | 0.47934 |

Classes with no development support receive F1 `0` in the fixed 16-class
macro average. This is why the all-class macro-F1 is lower than both supported-
class macro-F1 and character accuracy.
