# Evaluation Report — alclam_08959

**Track:** track3 | **Head:** linear_head | **Strategy:** strategy_a

- **DER (Diacritic Error Rate): 0.1041**  (= 1 - micro-F1; micro-F1 = 0.8959)
- Macro-F1, all 16 classes (matches table below): **0.4642**
- Macro-F1, classes with support only, 9 classes (excludes 0-support classes, NOT comparable to the line above): **0.8253**
- Characters evaluated (validation set): 3228

## Per-class metrics (validation set)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Diacritic | 0.9761 | 0.9694 | 0.9727 | 1012 |
| Fatha | 0.8847 | 0.8283 | 0.8556 | 769 |
| Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Damma | 0.9006 | 0.8430 | 0.8709 | 172 |
| Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Kasra | 0.9138 | 0.8833 | 0.8983 | 240 |
| Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Sukoon | 0.8378 | 0.9011 | 0.8683 | 900 |
| Shadda | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Fatha | 0.7412 | 0.7412 | 0.7412 | 85 |
| Shadda+Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Damma | 0.6667 | 1.0000 | 0.8000 | 6 |
| Shadda+Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Kasra | 0.8000 | 0.9412 | 0.8649 | 34 |
| Shadda+Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Sukoon | 0.6250 | 0.5000 | 0.5556 | 10 |
| **accuracy** | | | 0.8959 | 3228 |
| **macro avg** | 0.4591 | 0.4755 | 0.4642 | 3228 |
| **weighted avg** | 0.8974 | 0.8959 | 0.8959 | 3228 |
