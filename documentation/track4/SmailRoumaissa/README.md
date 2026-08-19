# Algerian Arabic Diacritization

Character-level **Transformer + CNN + CRF** tagger with an entropy-gated lexical-prior fallback for Algerian Arabic diacritization.

## Task

Character-level sequence tagging: predict one of **16 diacritic classes** for every Arabic letter in a sentence (space positions are masked out with `label = -100` and excluded from loss/metrics).

## Dataset

| Split | Size | Format |
|---|---|---|
| Train | 4,864 sentences | JSONL: `{sent_id, chars, labels}` |
| Dev | 607 sentences | JSONL: `{sent_id, chars, labels}` |
| Test | 608 sentences | Raw text only (no labels) |

- **Vocabulary**: fixed 43-token character vocabulary (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, space, plus 38 Arabic/loanword letters, including Algerian-specific letters پ چ ڤ ). Loaded from `vocab.json`, ids 0–42.
- **Preprocessing**: `<BOS>`/`<EOS>` prepended/appended per sentence; an `is_letter` mask separates letters from spaces (used for per-word CRF segmentation and metric masking).
- **Augmentation**: `char_dropout_prob=0.10` at train time only — randomly replaces input letters with `<UNK>` to discourage the model from memorizing spellings rather than learning diacritic patterns.
- **Batching**: dynamic padding via a custom `collate` function.
- **Tokenization for inference**: `tokenize_raw()` normalizes whitespace but deliberately does **not** filter by Arabic Unicode range, since Algerian loanword letters fall outside the standard range and a regex whitelist would silently drop them.

## Project structure

```
configs/track4/SmailRoumaissa/
    model_config.py      # ModelConfig dataclass (dim, layers, heads, dropout…)
    training_config.py   # TrainingConfig, LexicalFusionConfig, CalibrationConfig
    paths.py             # DataPaths auto-detection under /kaggle/input

models/track4/SmailRoumaissa/
    cnn.py               # DepthwiseConv1d, MultiKernelCNNFrontend (kernels 3/5/7, gated fusion)
    transformer.py       # RelativePositionBias, RelativeMultiHeadAttention, TransformerBlock, Backbone
    crf.py               # ChainCRF (from-scratch linear-chain CRF, Viterbi decode)
    heads.py             # DecomposedHead (8-class base + 2-class shadda), PerWordCRFHead
    tagger.py            # TransformerCNNCRFTagger (backbone + CRF head)

utils/track4/SmailRoumaissa/
    device.py            # get_device()
    constants.py         # SPACE, NUM_CLASSES (16), DIACRITIC_MARKS, CLASS_NAMES
    data.py               # Vocab, DiacritizationDataset, collate
    render.py             # render_sentence() — interleaves diacritic marks into text

evaluation/track4/SmailRoumaissa/
    lexical_prior.py     # LexicalPrior, entropy(), fuse_sentence()
    calibration.py        # fit_temperature() — post-hoc temperature scaling (LBFGS)
    metrics.py             # MicroF1Accumulator (= pooled accuracy for single-label)
    inference.py            # tokenize_raw, predict_log_probs, decode_crf, run_inference,
                            # evaluate_lexical_on_dev, collect_dev_predictions

training/track4/SmailRoumaissa/
    trainer.py            # train_model(), evaluate(), evaluate_lexical_fused()

experiments/track4/SmailRoumaissa/
    train_run.py          # Full train → dev-sweep → return best config
    predict_run.py        # Inference → model_output.txt → submission.csv

tests/track4/SmailRoumaissa/     # (placeholder for unit tests)

documentation/track4/SmailRoumaissa/
    README.md             # This file
```

## Model architecture — Transformer + CNN + CRF

```
input chars → embedding
            → MultiKernelCNNFrontend (kernels 3/5/7, gated fusion)
            → 6× Relative-Position Transformer blocks (pre-norm)
            → DecomposedHead (8-class base + 2-class shadda)
            → PerWordCRFHead (per-word linear-chain CRF, Viterbi decode)
```

| Component | Details |
|---|---|
| **MultiKernelCNNFrontend** | Depthwise conv1d, kernels = {3, 5, 7}; branch outputs concatenated → projected to `2×dim` → sigmoid-gated fusion → residual + LayerNorm |
| **RelativePositionBias** | T5-style learned relative-position biases, replacing absolute positional embeddings, to better capture distance-based diacritic dependencies |
| **RelativeMultiHeadAttention** | Custom multi-head attention using the relative-position bias above |
| **Transformer blocks** | 6 layers, pre-norm, 8 heads, feed-forward dim 1024 |
| **DecomposedHead** | Splits the 16 joint classes into an 8-class base-vowel head + 2-class shadda head, recombined into final logits |
| **ChainCRF** | From-scratch linear-chain CRF — learned transition matrix, forward-algorithm training loss, Viterbi decoding |
| **PerWordCRFHead** | Applies the CRF independently per word (segmented on `is_letter`/space boundaries) rather than over the whole sentence |

## Training hyperparameters

| Hyperparameter | Value |
|---|---|
| `dim` (hidden size) | 256 |
| `n_layers` | 6 |
| `n_heads` | 8 |
| `ff_dim` | 1024 |
| `batch_size` | 64 |
| `max_epochs` | 80 (early stopping, patience 15) |
| `dropout` | 0.30 |
| `weight_decay` | 0.05 (matrix weights only; biases/norms/embeddings excluded) |
| `char_dropout_prob` | 0.10 |
| LR schedule | Cosine decay with warmup |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Early-stopping metric | Dev Micro-F1 (neural-only decode) |

Regularization (dropout, weight decay, char dropout) was deliberately increased from an earlier, lighter setting (dropout 0.15→0.30, weight_decay 0.01→0.05) because dev Micro-F1 was plateauing/noisy while train loss kept falling — a sign of overfitting on the relatively small ~4.9k-sentence training set.

## Lexical prior & calibration

- **Temperature scaling**: fit post-training on frozen dev logits via LBFGS, minimizing cross-entropy. Two **independent** temperatures are learned — one for the base-vowel head, one for the shadda head — since they have different natural confidence levels. This doesn't change predicted classes or accuracy on its own; it makes softmax entropy a meaningful confidence signal for the gate below.
- **Lexical prior**: `(word, position) → label` counts collected from the **training set only**, Laplace-smoothed into a probability distribution. Never sees dev/test labels.
- **Entropy gate** (fusion rule):

  `weight(t) = max_strength · σ((entropy(t) − entropy_threshold) / gate_temperature)`

  The lexical distribution is blended into the calibrated neural log-probs proportionally to how uncertain the model is at that position — confident predictions pass through untouched, uncertain ones lean on the lexical fallback. Fused log-probs are re-normalized and passed through the same CRF Viterbi decode.
- **Gate hyperparameters**: grid-searched on dev only (`entropy_threshold × gate_temperature × max_strength`, 5×3×3 = 45 combos), never touching test. Selected: `entropy_threshold=0.75`, `gate_temperature=0.5`, `max_strength=3.0`.

## Pipeline overview

```
raw text
  → tokenize_raw()                     (whitespace normalization, keep all letters incl. loanwords)
  → model.marginal_log_probs()         (calibrated per-head temperatures)
  → fuse_sentence()                    (entropy-gated lexical prior, dev-selected gate params)
  → decode_crf()                       (per-word Viterbi)
  → render_sentence()                  (diacritics inserted back into text)
  → model_output.txt → make_submission.py → submission.csv
```

## Evaluation metric

**Micro-F1**, computed directly as pooled accuracy — mathematically equivalent here since every character has exactly one gold class (single-label multi-class setting).

## Results

### Training dynamics

Trained up to 80 epochs (patience 15) on 4,864 train / 607 dev sentences:

| Epoch | Train loss | Dev Micro-F1 (neural) | Dev Micro-F1 (+lexical) |
|---|---|---|---|
| 1 | 6.8922 | 0.74089 | 0.87853 |
| 5 | 2.1627 | 0.88746 | 0.91200 |
| 10 | 1.7120 | 0.90973 | 0.92584 |
| 20 | 1.2963 | 0.92672 | 0.93326 |
| 30 | 1.0895 | 0.93364 | 0.93842 |
| 34 | 1.0175 | 0.93414 | 0.93842 |
| 35 | 1.0073 | 0.93445 | — |

- **Fast early gains, slow late gains**: neural-only Micro-F1 jumps from 0.741→0.887 in the first 5 epochs (dominant/frequent diacritics learned quickly), then improves more slowly from epoch ~10 onward (0.910→0.934 over 25 epochs), consistent with the heavier regularization trading faster convergence for less overfitting.
- **Lexical fusion helps at every logged epoch**, and the gap to neural-only narrows as training progresses (+13.8 F1 points at epoch 1 → ~+0.4 by epoch 34) — exactly the expected behavior of an entropy-gated fallback: it corrects an uncertain early model a lot, then steps aside once the model is confident.

### Dev-set hyperparameter sweep

Calibrated neural-only dev Micro-F1: **0.93716**. Sweeping the fusion gate (45 combos) found a flat optimum — the top-10 combos all landed between 0.94188 and 0.94213, reassuring given dev is only 607 sentences. Selected config (`entropy_threshold=0.75, gate_temperature=0.5, max_strength=3.0`) gave **dev Micro-F1 = 0.94213** (+0.5 pts over neural-only).

### Class support and imbalance

The gold class distribution on dev is heavily skewed: five classes (None, Sukoon, Fatha, Kasra, Damma) account for the vast majority of characters, while tanwin-related classes (Fathatan, Dammatan, Kasratan and their Shadda-combined forms) occur rarely or not at all. Since Micro-F1 is pooled accuracy, overall score is effectively driven by the five frequent classes — the confusion matrix (built in the notebook) is the more informative view for the rare classes.

### Final scores

| Split | Micro-F1 |
|---|---|
| Dev (neural only) | 0.93716 |
| Dev (neural + lexical) | 0.94213 |
| Public test | 0.94610 |
| Private test | 0.94427 |

Public and private scores are close to and consistent with the dev fused score (gap of 0.002–0.004), with no sign of the dev-set gate search overfitting to dev noise. The public/private gap itself (0.00183) is small, indicating stable generalization to the hidden test split.

## Usage

```bash
# Training (runs train + dev grid sweep)
python -m experiments.train_run

# Inference (requires trained model in /kaggle/working/runs/crfcnn/)
python -m experiments.predict_run
```

## Class labels

| ID | Name | ID | Name |
|---|---|---|---|
| 0 | None | 8 | Shadda |
| 1 | Fatha | 9 | Shadda + Fatha |
| 2 | Fathatan | 10 | Shadda + Fathatan |
| 3 | Damma | 11 | Shadda + Damma |
| 4 | Dammatan | 12 | Shadda + Dammatan |
| 5 | Kasra | 13 | Shadda + Kasra |
| 6 | Kasratan | 14 | Shadda + Kasratan |
| 7 | Sukoon | 15 | Shadda + Sukoon |