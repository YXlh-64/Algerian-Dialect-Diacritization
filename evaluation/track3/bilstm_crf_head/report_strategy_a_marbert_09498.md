# Evaluation Report — marbert_09498

**Track:** track3 | **Head:** bilstm_crf_head | **Strategy:** strategy_a

- **DER (Diacritic Error Rate): 0.0502**  (= 1 - micro-F1; micro-F1 = 0.9498)
- Macro-F1, all 16 classes (matches table below): **0.4898**
- Macro-F1, classes with support only (excludes 0-support classes, NOT comparable to the line above): **0.8707**
- Characters evaluated (validation set): 3228

## Per-class metrics (validation set)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Diacritic | 0.9950 | 0.9891 | 0.9921 | 1012 |
| Fatha | 0.9157 | 0.9324 | 0.9240 | 769 |
| Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Damma | 0.9581 | 0.9302 | 0.9440 | 172 |
| Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Kasra | 0.9667 | 0.9667 | 0.9667 | 240 |
| Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Sukoon | 0.9375 | 0.9500 | 0.9437 | 900 |
| Shadda | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Fatha | 0.8182 | 0.7412 | 0.7778 | 85 |
| Shadda+Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Damma | 1.0000 | 0.8333 | 0.9091 | 6 |
| Shadda+Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Kasra | 0.9062 | 0.8529 | 0.8788 | 34 |
| Shadda+Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Sukoon | 0.6667 | 0.4000 | 0.5000 | 10 |
| **accuracy** | | | 0.9498 | 3228 |
| **macro avg** | 0.5103 | 0.4747 | 0.4898 | 3228 |
| **weighted avg** | 0.9494 | 0.9498 | 0.9494 | 3228 |
