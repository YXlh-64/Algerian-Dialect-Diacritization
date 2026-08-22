# strategy_a — track1 / bilstm_cnn_crf

One focused P2 experiment is currently recorded for this group. It uses no
external pretraining and trains directly on the Algerian data.

Shared code:

- Model: `models/track1/bilstm_cnn_crf/bilstm_cnn_crf_model.py`
- Training: `training/track1/bilstm_cnn_crf/finetune_bilstm_cnn_crf.py`
- Training engine: `training/track1/bilstm_cnn_crf/engine.py`
- Batching and sampling: `training/track1/bilstm_cnn_crf/data.py`
- Evaluation: `evaluation/track1/bilstm_cnn_crf/evaluate_bilstm_cnn_crf.py`
- Shared record/data helpers: `utils/track1/data.py`

| Model | Config | Report | Dev macro-F1 (16) | Public | Private |
|---|---|---|---:|---:|---:|
| p2_ensemble_09483 | `configs/track1/bilstm_cnn_crf/strategy_a_p2_ensemble_09483.yaml` | `evaluation/track1/bilstm_cnn_crf/report_strategy_a_p2_ensemble_09483.md` | 0.58610 | 0.95048 | 0.94829 |

The ensemble contains five random seeds of one architecture, not five distinct
model families. Development labels determine seed epochs, blend weights,
lexical-prior strength, frequency adjustment, and transition strength. The
selected setup is then refit on train+dev before test prediction.
