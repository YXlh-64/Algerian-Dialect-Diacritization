# Evaluation Report — camelbert_da_09483

**Track:** track3 | **Head:** bilstm_crf_head | **Strategy:** strategy_a

- **DER (Diacritic Error Rate): 0.0517**  (= 1 - micro-F1; micro-F1 = 0.9483)
- Macro-F1, all 16 classes (matches table below): **0.4945**
- Macro-F1, classes with support only (excludes 0-support classes, NOT comparable to the line above): **0.8791**
- Characters evaluated (validation set): 3228

## Per-class metrics (validation set)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Diacritic | 0.9950 | 0.9881 | 0.9916 | 1012 |
| Fatha | 0.9106 | 0.9272 | 0.9188 | 769 |
| Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Damma | 0.9419 | 0.9419 | 0.9419 | 172 |
| Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Kasra | 0.9748 | 0.9667 | 0.9707 | 240 |
| Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Sukoon | 0.9309 | 0.9433 | 0.9371 | 900 |
| Shadda | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Fatha | 0.8684 | 0.7765 | 0.8199 | 85 |
| Shadda+Fathatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Damma | 1.0000 | 0.8333 | 0.9091 | 6 |
| Shadda+Dammatan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Kasra | 0.9677 | 0.8824 | 0.9231 | 34 |
| Shadda+Kasratan | 0.0000 | 0.0000 | 0.0000 | 0 |
| Shadda+Sukoon | 0.6667 | 0.4000 | 0.5000 | 10 |
| **accuracy** | | | 0.9483 | 3228 |
| **macro avg** | 0.5160 | 0.4787 | 0.4945 | 3228 |
| **weighted avg** | 0.9481 | 0.9483 | 0.9479 | 3228 |
