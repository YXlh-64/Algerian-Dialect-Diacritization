Evaluation Report — camelbert_da_09480
Track: track3 | Head: bilstm_crf_head | Strategy: strategy_a

Micro-F1 (local dev/test): 0.9480
Macro-F1 (local dev/test): 0.8749
Characters evaluated: 3228

Kaggle public score: 0.94512
Kaggle private score: 0.95096

Per-class metrics:
Class                Precision    Recall  F1-score   Support
------------------------------------------------------------
No Diacritic            0.9940    0.9881    0.9911      1012
Fatha                   0.9096    0.9285    0.9189       769
Fathatan                0.0000    0.0000    0.0000         0
Damma                   0.9415    0.9360    0.9388       172
Dammatan                0.0000    0.0000    0.0000         0
Kasra                   0.9748    0.9667    0.9707       240
Kasratan                0.0000    0.0000    0.0000         0
Sukoon                  0.9331    0.9456    0.9393       900
Shadda                  0.0000    0.0000    0.0000         0
Shadda+Fatha            0.8630    0.7412    0.7975        85
Shadda+Fathatan         0.0000    0.0000    0.0000         0
Shadda+Damma            1.0000    0.8333    0.9091         6
Shadda+Dammatan         0.0000    0.0000    0.0000         0
Shadda+Kasra            0.9375    0.8824    0.9091        34
Shadda+Kasratan         0.0000    0.0000    0.0000         0
Shadda+Sukoon           0.6667    0.4000    0.5000        10
------------------------------------------------------------
accuracy                                    0.9480      3228
macro avg               0.5138    0.4764    0.4922      3228
weighted avg            0.9476    0.9480    0.9475      3228