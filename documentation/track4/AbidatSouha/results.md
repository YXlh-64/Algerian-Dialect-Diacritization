# Results

All dev figures come from `dev_Algerian-DIAC.jsonl` (607 sentences, 15,897
letter positions, 3,870 word tokens). The leaderboard figure comes from the
Kaggle test split, whose labels are not distributed.

**Read the dev numbers as validation, not as a held-out estimate.** The same dev
split drives early stopping and checkpoint selection, so the reported figures are
optimistic by construction. The one genuinely held-out number in this document is
the leaderboard score.

## Headline

| System | Dev micro-F1 | Dev WER | Leaderboard |
|---|---:|---:|---:|
| Lookup baseline (memorisation floor) | — | 26.07 | — |
| **T5 — Transformer-CNN-CRF** | **0.9343** | **20.23** | **0.93977** |

## Baseline — the memorisation floor

Most-frequent-vocalisation lookup from the training lexicon. No model, no
generalisation: every word seen in training gets its majority vocalization,
every unseen word is wrong by definition.

```
lookup baseline: word acc 73.93%  ->  WER 26.07%
   seen_unambig   n= 1897  WER   5.06%
   seen_ambig     n= 1406  WER  24.61%
   oov            n=  567  WER 100.00%
```

Bucket shares of dev tokens: 49.0% seen-unambiguous, 36.3% seen-ambiguous,
14.7% OOV. Any model has to beat 26.07% WER, and the interesting question is
*which bucket* the improvement comes from.

## T5 — Transformer-CNN-CRF

Full model, all switches on. 40 epochs, seed 0, CPU, 4,618 s. 1,910,460
parameters. Best checkpoint at epoch 36.

```
BEST  LB(microF1) 0.9343 | DER_let  6.57 | DER_dia  9.12
      WER 20.23 (un  8.49  amb 25.46  oov 46.56) | macroF1 81.91
```

| Metric | Value |
|---|---:|
| micro-F1 (leaderboard metric) | 0.9343 |
| DER, letters only | 6.57 |
| DER, gold-diacritic-bearing positions | 9.12 |
| WER | 20.23 |
| WER seen-unambiguous | 8.49 |
| WER seen-ambiguous | 25.46 |
| WER OOV | 46.56 |
| macro-F1 over the 9 live classes | 81.91 |

**Leaderboard: 0.93977.**

### Training curve

| Epoch | 00 | 02 | 06 | 13 | 19 | 26 | 31 | 36 | 39 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev micro-F1 | 0.7893 | 0.8863 | 0.9095 | 0.9248 | 0.9309 | 0.9333 | 0.9339 | **0.9343** | 0.9343 |
| dev DER_let | 21.07 | 11.37 | 9.05 | 7.52 | 6.91 | 6.67 | 6.61 | **6.57** | 6.57 |

The run converged. Between epochs 29 and 39, DER_let moved 6.62 → 6.57 while
training loss moved 5.28 → 5.15, with the cosine schedule driving the learning
rate to zero. Dev was still improving very slightly rather than degrading, so the
model was capacity- or regularisation-limited at the end, not overfitting.
Increasing `epochs` alone would not help.

### Dev underestimated the leaderboard

Dev micro-F1 0.9343 against a leaderboard score of 0.93977 — the held-out score
came in **0.0055 higher** than the validation estimate. That direction is
unusual: selection on dev normally makes dev optimistic. It suggests the
selection did not meaningfully overfit dev, and that improvements measured on dev
should transfer rather than evaporate.

It is also the strongest available evidence about *which* metric the leaderboard
uses. Dev macro-F1 was 81.91 (0.8191) — 0.12 away from the leaderboard score,
while micro-F1 is within 0.006 of it.

## Error analysis — where the gain comes from

The same buckets, model against baseline:

| Bucket | n | Lookup WER | T5 WER | Difference |
|---|---:|---:|---:|---|
| seen_unambig | 1,897 | **5.06** | 8.49 | model **3.43 worse** |
| seen_ambig | 1,406 | **24.61** | 25.46 | model **0.85 worse** |
| oov | 567 | 100.00 | **46.56** | model **53.44 better** |
| **all** | 3,870 | 26.07 | **20.23** | model 5.84 better |

The arithmetic reconciles exactly — 0.0849·1897 + 0.2546·1406 + 0.4656·567 = 783
errors out of 3,870 = 20.23% — confirming both systems are scored over the same
tokens.

Two findings follow, and they are the most useful things in this document:

**All of the gain is OOV.** On the 14.7% of tokens never seen in training, the
model cuts word error from 100% to 46.6%. That is the entire margin over the
baseline, and it is exactly what a character-level model is for.

**On seen words the model loses to a lookup table.** Words seen in training with
a single vocalization are handled *better* by naive memorisation than by the
trained network — 5.06% against 8.49%. On ambiguous words the model is also no
better than picking the most frequent reading, meaning the contextual
disambiguation the architecture is built for is not yet contributing measurably.

This asymmetry is what motivates the V2 confidence-gated lexical fallback: where
the model is unsure about a word the lexicon knows, defer to the lexicon.

## Not yet measured

Stated explicitly so that nothing in this document is mistaken for a result:

| Item | Status |
|---|---|
| **T1 plain Transformer** | Not completed. Only epoch 0 measured (dev micro-F1 0.5895, DER_let 41.05). The required track-4 baseline comparison is therefore **outstanding**. |
| **Seed ensemble (§13)** | Not run. `EnsembleConfig(seeds=(0,1,2))` costs roughly 2 × 77 min beyond the seed-0 model already trained. |
| **V2 lexical fallback (§16)** | Threshold search never run against the trained model. The defaults in `LexicalFallbackConfig` are grid midpoints and are **not tuned**. |
| **Ablation ladder** | Not run. No component in [architecture.md](architecture.md) has an isolated measured contribution — the rationales given there are motivations, not results. |

## Reproducing

```bash
PYTHONPATH=. python experiments/track4/AbidatSouha/predict_run.py
```

Environment for the numbers above: torch 2.11.0, CPU, Apple Silicon.

**One deliberate difference from the original notebook.** Checkpoint selection
and early stopping now use `TrainingConfig.select_metric`, defaulting to
`macro_f1`; the notebook selected on `der_letters`. A rerun will therefore not
reproduce the exact checkpoint behind the 0.93977 submission. Set
`select_metric="der_letters"` to reproduce the original behaviour.
