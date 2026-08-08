# Evaluation Report — camelbert_mix_09176

**Track:** track3 | **Head:** linear_head | **Strategy:** strategy_a

- **DER (Diacritic Error Rate): 0.0824**  (= 1 - micro-F1; micro-F1 = 0.9176)
- Macro-F1, all 16 classes (matches table below): **0.4785**
- Macro-F1, classes with support only, 9 classes (excludes 0-support classes, NOT comparable to the line above): **0.8507**
- Characters evaluated (validation set): 3228

## Per-class metrics (validation set)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Diacritic | 0.9840 | 0.9733 | 0.9786 | 1012 |
| Fatha | 0.9004 | 0.8700 | 0.8849 | 769 |
| Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Damma | 0.9325 | 0.8837 | 0.9075 | 172 |
| Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Kasra | 0.9325 | 0.9208 | 0.9266 | 240 |
| Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Sukoon | 0.8775 | 0.9233 | 0.8998 | 900 |
| Shadda | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Fatha | 0.7561 | 0.7294 | 0.7425 | 85 |
| Shadda+Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Damma | 0.8571 | 1.0000 | 0.9231 | 6 |
| Shadda+Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Kasra | 0.7750 | 0.9118 | 0.8378 | 34 |
| Shadda+Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Sukoon | 0.6250 | 0.5000 | 0.5556 | 10 |
| **accuracy** | | | 0.9176 | 3228 |
| **macro avg** | 0.4775 | 0.4820 | 0.4785 | 3228 |
| **weighted avg** | 0.9183 | 0.9176 | 0.9176 | 3228 |
