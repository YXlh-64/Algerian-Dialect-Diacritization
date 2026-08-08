Evaluation Report — marbert_09498
Track: track3 | Head: bilstm_crf_head | Strategy: strategy_a

Micro-F1 (local dev/test): 0.9498
Macro-F1 (local dev/test): 0.8707
Characters evaluated: 3228

Kaggle public score: 0.94512
Kaggle private score: 0.94975

Per-class metrics:
Class                Precision    Recall  F1-score   Support
------------------------------------------------------------
No Diacritic            0.9950    0.9891    0.9921      1012
Fatha                   0.9157    0.9324    0.9240       769
Fathatan                0.0000    0.0000    0.0000         0
Damma                   0.9581    0.9302    0.9440       172
Dammatan                0.0000    0.0000    0.0000         0
Kasra                   0.9667    0.9667    0.9667       240
Kasratan                0.0000    0.0000    0.0000         0
Sukoon                  0.9375    0.9500    0.9437       900
Shadda                  0.0000    0.0000    0.0000         0
Shadda+Fatha            0.8182    0.7412    0.7778        85
Shadda+Fathatan         0.0000    0.0000    0.0000         0
Shadda+Damma            1.0000    0.8333    0.9091         6
Shadda+Dammatan         0.0000    0.0000    0.0000         0
Shadda+Kasra            0.9062    0.8529    0.8788        34
Shadda+Kasratan         0.0000    0.0000    0.0000         0
Shadda+Sukoon           0.6667    0.4000    0.5000        10
------------------------------------------------------------
accuracy                                    0.9498      3228
macro avg               0.5103    0.4747    0.4898      3228
weighted avg            0.9494    0.9498    0.9494      3228