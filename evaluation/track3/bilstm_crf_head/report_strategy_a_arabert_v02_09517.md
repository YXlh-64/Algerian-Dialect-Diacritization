Evaluation Report — arabert_v02_09517
Track: track3 | Head: bilstm_crf_head | Strategy: strategy_a

Micro-F1 (local dev/test): 0.9517
Macro-F1 (local dev/test): 0.8777
Characters evaluated: 3228

Kaggle public score: 0.94743
Kaggle private score: 0.95413

Per-class metrics:
Class                Precision    Recall  F1-score   Support
------------------------------------------------------------
No Diacritic            0.9950    0.9881    0.9916      1012
Fatha                   0.9186    0.9389    0.9286       769
Fathatan                0.0000    0.0000    0.0000         0
Damma                   0.9474    0.9419    0.9446       172
Dammatan                0.0000    0.0000    0.0000         0
Kasra                   0.9628    0.9708    0.9668       240
Kasratan                0.0000    0.0000    0.0000         0
Sukoon                  0.9403    0.9456    0.9429       900
Shadda                  0.0000    0.0000    0.0000         0
Shadda+Fatha            0.8462    0.7765    0.8098        85
Shadda+Fathatan         0.0000    0.0000    0.0000         0
Shadda+Damma            1.0000    0.8333    0.9091         6
Shadda+Dammatan         0.0000    0.0000    0.0000         0
Shadda+Kasra            0.9667    0.8529    0.9062        34
Shadda+Kasratan         0.0000    0.0000    0.0000         0
Shadda+Sukoon           0.6667    0.4000    0.5000        10
------------------------------------------------------------
accuracy                                    0.9517      3228
macro avg               0.5152    0.4780    0.4937      3228
weighted avg            0.9514    0.9517    0.9513      3228