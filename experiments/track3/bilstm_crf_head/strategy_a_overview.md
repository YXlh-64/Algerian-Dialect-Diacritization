# strategy_a -- track3 / bilstm_crf_head

6 experiment(s) in this group, ranked by DER (Diacritic Error Rate -- lower is better). Shared code:
- Model: `models/track3/bilstm_crf_head/bilstm_crf_head_model.py`
- Fine-tuning script: `training/track3/bilstm_crf_head/finetune_bilstm_crf_head.py`
- Evaluation code: `evaluation/track3/bilstm_crf_head/evaluate_bilstm_crf_head.py`

| Model | Config | Report | Micro-F1 | DER |
|---|---|---|---|---|
| arabert_v02_09517 | `configs/track3/bilstm_crf_head/strategy_a_arabert_v02_09517.yaml` | `evaluation/track3/bilstm_crf_head/report_strategy_a_arabert_v02_09517.md` | 0.9517 | 0.0483 |
| dziribert_09511 | `configs/track3/bilstm_crf_head/strategy_a_dziribert_09511.yaml` | `evaluation/track3/bilstm_crf_head/report_strategy_a_dziribert_09511.md` | 0.9511 | 0.0489 |
| marbert_09498 | `configs/track3/bilstm_crf_head/strategy_a_marbert_09498.yaml` | `evaluation/track3/bilstm_crf_head/report_strategy_a_marbert_09498.md` | 0.9498 | 0.0502 |
| camelbert_da_09483 | `configs/track3/bilstm_crf_head/strategy_a_camelbert_da_09483.yaml` | `evaluation/track3/bilstm_crf_head/report_strategy_a_camelbert_da_09483.md` | 0.9483 | 0.0517 |
| camelbert_da_09480 | `configs/track3/bilstm_crf_head/strategy_a_camelbert_da_09480.yaml` | `evaluation/track3/bilstm_crf_head/report_strategy_a_camelbert_da_09480.md` | 0.9480 | 0.0520 |
| camelbert_mix_09470 | `configs/track3/bilstm_crf_head/strategy_a_camelbert_mix_09470.yaml` | `evaluation/track3/bilstm_crf_head/report_strategy_a_camelbert_mix_09470.md` | 0.9470 | 0.0530 |
