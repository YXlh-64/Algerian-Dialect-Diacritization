# Evaluation Report — camelbert_da_09111

**Track:** track3 | **Head:** linear_head | **Strategy:** strategy_a

- **DER (Diacritic Error Rate): 0.0889**  (= 1 - micro-F1; micro-F1 = 0.9111)
- Macro-F1, all 16 classes (matches table below): **0.4732**
- Macro-F1, classes with support only, 9 classes (excludes 0-support classes, NOT comparable to the line above): **0.8412**
- Characters evaluated (validation set): 3228

## Per-class metrics (validation set)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Diacritic | 0.9820 | 0.9713 | 0.9767 | 1012 |
| Fatha | 0.8980 | 0.8700 | 0.8838 | 769 |
| Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Damma | 0.9042 | 0.8779 | 0.8909 | 172 |
| Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Kasra | 0.9307 | 0.8958 | 0.9130 | 240 |
| Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Sukoon | 0.8690 | 0.9067 | 0.8874 | 900 |
| Shadda | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Fatha | 0.7303 | 0.7647 | 0.7471 | 85 |
| Shadda+Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Damma | 0.7500 | 1.0000 | 0.8571 | 6 |
| Shadda+Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Kasra | 0.7561 | 0.9118 | 0.8267 | 34 |
| Shadda+Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Sukoon | 0.7143 | 0.5000 | 0.5882 | 10 |
| **accuracy** | | | 0.9111 | 3228 |
| **macro avg** | 0.4709 | 0.4811 | 0.4732 | 3228 |
| **weighted avg** | 0.9123 | 0.9111 | 0.9113 | 3228 |
