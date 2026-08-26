# Leaderboard

Track 2 validation results are reported separately from the submitted Track 3
competition runs below. The standard CANINE-S report also records a private
leaderboard F1-score of `0.93235`; it is not directly comparable to these dev
split values.

| Model | Track | Head | Strategy | Dev accuracy / micro-F1 | Dev DER |
|---|---|---|---|---:|---:|
| `canine_s` | track2 | canine_s_model | strategy_a | **0.9452** | **0.0548** |
| `canine_s_twohead` | track2 | canine_twohead | strategy_a | **0.9406** | **0.0594** |

## Competition submissions

| Rank | Model | Track | Head | Strategy | Public | Private |
|---|---|---|---|---|---|---|
| 1 | arabert_v02_09517 | track3 | bilstm_crf_head | strategy_a | 0.94743 | 0.95413 |
| 2 | camelbert_mix_09470 | track3 | bilstm_crf_head | strategy_a | 0.94597 | 0.95218 |
| 3 | camelbert_da_09483 | track3 | bilstm_crf_head | strategy_a | 0.94464 | 0.95108 |
| 4 | camelbert_da_09480 | track3 | bilstm_crf_head | strategy_a | 0.94512 | 0.95096 |
| 5 | dziribert_09511 | track3 | bilstm_crf_head | strategy_a | 0.94695 | 0.94999 |
| 6 | marbert_09498 | track3 | bilstm_crf_head | strategy_a | 0.94512 | 0.94975 |
| 7 | arabert_v02_09157 | track3 | linear_head | strategy_a | 0.91276 | 0.91884 |
| 8 | camelbert_mix_09176 | track3 | linear_head | strategy_a | 0.91166 | 0.91848 |
| 9 | marbertv2_09074 | track3 | linear_head | strategy_a | 0.90704 | 0.91519 |
| 10 | camelbert_da_09111 | track3 | linear_head | strategy_a | 0.91300 | 0.91397 |
| 11 | marbert_09027 | track3 | linear_head | strategy_a | 0.90582 | 0.91373 |
| 12 | alclam_08959 | track3 | linear_head | strategy_a | 0.89828 | 0.90850 |
