# Evaluation Report — camelbert_mix_09470

**Track:** track3 | **Head:** bilstm_crf_head | **Strategy:** strategy_a

- Micro-F1 (local dev/test): **0.9470**
- Macro-F1 (local dev/test): **0.8671**
- Characters evaluated: 3228
- Kaggle public score: **0.94597**
- Kaggle private score: **0.95218**

## Per-class metrics

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Diacritic | 0.9950 | 0.9872 | 0.9911 | 1012 |
| Fatha | 0.9120 | 0.9298 | 0.9208 | 769 |
| Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Damma | 0.9209 | 0.9477 | 0.9341 | 172 |
| Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Kasra | 0.9707 | 0.9667 | 0.9687 | 240 |
| Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Sukoon | 0.9378 | 0.9389 | 0.9384 | 900 |
| Shadda | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Fatha | 0.8228 | 0.7647 | 0.7927 | 85 |
| Shadda+Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Damma | 1.0000 | 0.8333 | 0.9091 | 6 |
| Shadda+Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Kasra | 0.9062 | 0.8529 | 0.8788 | 34 |
| Shadda+Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Sukoon | 0.5714 | 0.4000 | 0.4706 | 10 |
| **accuracy** | | | 0.9470 | 3228 |
| **macro avg** | 0.5023 | 0.4763 | 0.4878 | 3228 |
| **weighted avg** | 0.9468 | 0.9470 | 0.9468 | 3228 |
