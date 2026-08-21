# FilteredWordLattice-v14 Results

`DziriFormer-FilteredWordLattice-v14` is rejected at the leakage-controlled
calibration gate. It was not evaluated on the released dev set, no Kaggle CSV
was exported, and no seed 43/44 training was started.

## Oracle

| K | Split A recoverable letters / exact words | Split B recoverable letters / exact words |
|---:|---:|---:|
| 4 | +1,024 / +970 | +1,035 / +965 |
| 8 | +1,347 / +1,177 | +1,332 / +1,159 |

Both K values passed the +20-letter/+10-word oracle gate. K=8 was selected on
split A by neural correct letters, then exact words, then smaller K.

## Learned reranker calibration

| Split | Matched control neural | v14 neural | Letter gain | Exact-word gain | V2 gain | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| A | 24,728 | 24,741 | +13 | +10 | -9 | 4 |
| B | 24,710 | 24,711 | +1 | +4 | +3 | 5 |
| Mean gain | — | — | **+7** | **+7** | — | — |

The registered gate required positive gains on both splits and mean gains of at
least +10 letters and +10 exact words. The mean gains were only +7/+7.

## Conclusion

Candidate generation was not the bottleneck: K=8 covered roughly 96% of gold
words and made more than 1,300 letters recoverable on each split. The small
candidate-composition network could not convert that oracle headroom into a
robust cross-split gain. This separates *candidate availability* from
*learnable candidate selection* and justifies testing v15's letter-level shared
encoder fusion without relaxing or tuning v14's gates.

Authoritative decision:
`outputs/filtered_word_lattice_v14/02_calibration_b/CALIBRATION_SELECTION.json`.
