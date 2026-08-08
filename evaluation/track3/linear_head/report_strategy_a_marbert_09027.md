# Evaluation Report — marbert_09027

**Track:** track3 | **Head:** linear_head | **Strategy:** strategy_a

- **DER (Diacritic Error Rate): 0.0973**  (= 1 - micro-F1; micro-F1 = 0.9027)
- Macro-F1, all 16 classes (matches table below): **0.4666**
- Macro-F1, classes with support only, 9 classes (excludes 0-support classes, NOT comparable to the line above): **0.8295**
- Characters evaluated (validation set): 3228

## Per-class metrics (validation set)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Diacritic | 0.9771 | 0.9704 | 0.9737 | 1012 |
| Fatha | 0.8981 | 0.8479 | 0.8722 | 769 |
| Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Damma | 0.9236 | 0.8430 | 0.8815 | 172 |
| Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Kasra | 0.9068 | 0.8917 | 0.8992 | 240 |
| Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Sukoon | 0.8498 | 0.9056 | 0.8768 | 900 |
| Shadda | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Fatha | 0.7191 | 0.7529 | 0.7356 | 85 |
| Shadda+Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Damma | 0.6667 | 1.0000 | 0.8000 | 6 |
| Shadda+Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Kasra | 0.7750 | 0.9118 | 0.8378 | 34 |
| Shadda+Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Sukoon | 0.7143 | 0.5000 | 0.5882 | 10 |
| **accuracy** | | | 0.9027 | 3228 |
| **macro avg** | 0.4644 | 0.4764 | 0.4666 | 3228 |
| **weighted avg** | 0.9044 | 0.9027 | 0.9029 | 3228 |
