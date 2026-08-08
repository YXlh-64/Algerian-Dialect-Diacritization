# Evaluation Report — camelbert_mix_09470

**Track:** track3 | **Head:** bilstm_crf_head | **Strategy:** strategy_a

- **DER (Diacritic Error Rate): 0.0530**  (= 1 - micro-F1; micro-F1 = 0.9470)
- Macro-F1, all 16 classes (matches table below): **0.4878**
- Macro-F1, classes with support only (excludes 0-support classes, NOT comparable to the line above): **0.8671**
- Characters evaluated (validation set): 3228

## Per-class metrics (validation set)

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
