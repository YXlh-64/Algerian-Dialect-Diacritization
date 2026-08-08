# Evaluation Report — dziribert_09511

**Track:** track3 | **Head:** bilstm_crf_head | **Strategy:** strategy_a

- Micro-F1 (local dev/test): **0.9511**
- Macro-F1 (local dev/test): **0.8714**
- Characters evaluated: 3228
- Kaggle public score: **0.94695**
- Kaggle private score: **0.94999**

## Per-class metrics

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Diacritic | 0.9950 | 0.9911 | 0.9931 | 1012 |
| Fatha | 0.9171 | 0.9350 | 0.9259 | 769 |
| Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Damma | 0.9755 | 0.9244 | 0.9493 | 172 |
| Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Kasra | 0.9628 | 0.9708 | 0.9668 | 240 |
| Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Sukoon | 0.9407 | 0.9522 | 0.9464 | 900 |
| Shadda | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Fatha | 0.7922 | 0.7176 | 0.7531 | 85 |
| Shadda+Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Damma | 1.0000 | 0.8333 | 0.9091 | 6 |
| Shadda+Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Kasra | 0.8788 | 0.8529 | 0.8657 | 34 |
| Shadda+Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Sukoon | 0.8000 | 0.4000 | 0.5333 | 10 |
| **accuracy** | | | 0.9511 | 3228 |
| **macro avg** | 0.5164 | 0.4736 | 0.4902 | 3228 |
| **weighted avg** | 0.9507 | 0.9511 | 0.9505 | 3228 |
