# Evaluation Report — marbertv2_09074

**Track:** track3 | **Head:** linear_head | **Strategy:** strategy_a

- **DER (Diacritic Error Rate): 0.0926**  (= 1 - micro-F1; micro-F1 = 0.9074)
- Macro-F1, all 16 classes (matches table below): **0.4749**
- Macro-F1, classes with support only, 9 classes (excludes 0-support classes, NOT comparable to the line above): **0.8442**
- Characters evaluated (validation set): 3228

## Per-class metrics (validation set)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Diacritic | 0.9782 | 0.9743 | 0.9762 | 1012 |
| Fatha | 0.8745 | 0.8700 | 0.8722 | 769 |
| Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Damma | 0.9290 | 0.8372 | 0.8807 | 172 |
| Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Kasra | 0.9198 | 0.9083 | 0.9140 | 240 |
| Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Sukoon | 0.8685 | 0.8956 | 0.8818 | 900 |
| Shadda | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Fatha | 0.7805 | 0.7529 | 0.7665 | 85 |
| Shadda+Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Damma | 0.7500 | 1.0000 | 0.8571 | 6 |
| Shadda+Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Kasra | 0.8158 | 0.9118 | 0.8611 | 34 |
| Shadda+Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Sukoon | 0.7143 | 0.5000 | 0.5882 | 10 |
| **accuracy** | | | 0.9074 | 3228 |
| **macro avg** | 0.4769 | 0.4781 | 0.4749 | 3228 |
| **weighted avg** | 0.9078 | 0.9074 | 0.9073 | 3228 |
