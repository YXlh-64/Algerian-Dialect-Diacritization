# Track 1 — BiLSTM Taggers for Algerian Arabic Diacritization

**Track:** Track 1 — BiLSTM-based tagger family (BiLSTM-CNN, BiLSTM-CRF, BiLSTM-CNN-CRF)
---

## 1. Overview

This track implements, trains, and benchmarks recurrent (BiLSTM-based) sequence taggers for
Algerian Arabic diacritization: assigning one of 16 diacritic classes to every character of an
undiacritized input sentence. Three required base architectures are trained as a controlled
ablation, plus 8 additional experiments extending the strongest base architecture along different
axes (class imbalance, long-range dependencies, small-data robustness, checkpoint variance,
inference-time robustness).

## 2. Pipeline

```
raw JSONL (train/dev, {"sent_id","chars","labels","input","target"})
        │
        ▼
utils/track1/soundous/data_utils.py     -- length-bucketed Dataset/DataLoader, mask-aware padding
        │
        ▼
models/track1/soundous/{layers,tagger,experimental_taggers}.py
        -- CharEmbedding -> [CharCNNHighway]? -> BiLSTMEncoder -> Linear -> [CRF]?
        │
        ▼
training/track1/soundous/{train_loop,experiment_trainers}.py
        -- AdamW + cosine-annealing-warm-restarts + grad clipping + early stopping on dev DER
        │
        ▼
evaluation/track1/soundous/{metrics,inference,evaluate_all_experiments}.py
        -- CER/DER/WER/Accuracy/WordAcc/SentAcc (global micro-average, matches P3 methodology)
        -- inference + official make_submission.py -> per-experiment submission.csv
```

## 3. Repository layout

```
configs/track1/soundous/        Hyperparameters + data/output paths (JSON, one file per experiment)
models/track1/soundous/         Model definitions (shared layers, base tagger, experimental taggers)
utils/track1/soundous/          Vocab/label loading, diacritic <-> class-id mapping, Dataset/DataLoader, paths
training/track1/soundous/       Generic training loop (run_training) + per-experiment training loops
experiments/track1/soundous/    One run_*.py script per experiment (calls training/ + saves a checkpoint)
evaluation/track1/soundous/     Metrics, inference/submission generation, evaluate_all_experiments.py
tests/track1/soundous/          Unit tests (diacritic round-trip, data pipeline, CRF sanity checks)
documentation/track1/soundous/  This README:(
```

## 4. Experimental setup

**Data.** `train_Algerian-DIAC.jsonl` / `dev_Algerian-DIAC.jsonl`, flat JSONL, character-level
`{chars, labels}` pairs, 16-class diacritic scheme (see `class_labels.txt`). Test set
(`raw_sentences_test.txt` + `raw_sentences_test_ids.txt`) has no released labels; submissions are
scored by Kaggle against a held-out reference via `evaluate_diacritization.py` (P3).

**Base architecture capacity** (shared across all 3 base architectures and most experiments, so
differences are attributable to the architectural change, not model capacity —
`configs/track1/soundous/model_common.json`):

| emb_dim | cnn_out_dim | lstm_hidden | lstm_layers | dropout | batch_size |
|---|---|---|---|---|---|
| 128 | 128 | 256 | 2 | 0.3 | 64 |

**Optimization** (shared, `configs/track1/soundous/base.json`): AdamW, lr=1e-3, weight_decay=1e-5,
gradient clipping at max-norm 5.0, label smoothing 0.05 (non-CRF variants only), cosine-annealing-
warm-restarts (T0=8, T_mult=2), early stopping on dev DER with patience 6, max 40 epochs.

**Evaluation.** CER, DER, WER, Accuracy, WordAcc, SentAcc — global micro-average, matching P3's
methodology exactly (`evaluation/track1/soundous/metrics.py`).

## 5. Experiments

| # | Experiment | Script | Targets |
|---|---|---|---|
| Base | BiLSTM-CNN | `experiments/track1/soundous/run_base_architectures.py` | CNN-only ablation |
| Base | BiLSTM-CRF | `experiments/track1/soundous/run_base_architectures.py` | CRF-only ablation |
| Base | BiLSTM-CNN-CRF | `experiments/track1/soundous/run_base_architectures.py` | 
| 1 | Focal-style emission reweighting | `run_focal.py` | rare diacritic classes |
| 2 | Auxiliary has-diacritic head | `run_multitask.py` | denser gradient signal, small-data regularization |
| 3 | Self-attention (BiLSTM + attention) | `run_attention.py` | long-range dependency bottleneck |
| 4 | Consistency-regularized augmentation | `run_consistency.py` | small-data (~6k sentences) robustness |
| 5 | Stochastic Weight Averaging | `run_swa.py` | checkpoint variance, flatter minima |
| 6 | Multi-seed ensemble | `run_ensemble.py` | variance reduction via decorrelated errors |
| 7 | Length-based curriculum learning | `run_curriculum.py` | training stability/convergence speed |
| 8 | Test-time multi-offset chunk averaging | *(inference-only, see `evaluate_all_experiments.py`)* | long-sentence chunking robustness |

## 6. How to reproduce

```bash
pip install -r requirements.txt

# edit configs/track1/soundous/paths.json to point data_root at the 8 dataset files, or rely on

python experiments/track1/soundous/run_all.py                       # trains everything
python evaluation/track1/soundous/evaluate_all_experiments.py       # evaluates + generates all submissions
```


Unit tests: `pytest tests/track1/soundous/ -v`

## 7. Results

| Rank | Experiment | DER | WER | CER | Accuracy | WordAcc | SentAcc |
|---|---|---|---|---|---|---|---|
| 1 | **exp_ensemble** (6, 3 seeds) | **4.74%** | 18.55% | 4.74% | 95.26% | 81.45% | 35.09% |
| 2 | exp_attention (3) | 4.86% | 19.15% | 4.86% | 95.14% | 80.85% | 34.93% |
| 3 | exp_curriculum (7) | 4.86% | 18.81% | 4.86% | 95.14% | 81.19% | 35.09% |
| 4 | exp_consistency (4) | 4.88% | 19.07% | 4.88% | 95.12% | 80.93% | 33.77% |
| 5 | bilstm_cnn_crf (P2 baseline) | 4.90% | 19.12% | 4.90% | 95.10% | 80.88% | 35.26% |
| 5 | exp_tta (8, on bilstm_cnn_crf) | 4.90% | 19.12% | 4.90% | 95.10% | 80.88% | 35.26% |
| 7 | exp_multitask (2) | 4.95% | 19.04% | 4.95% | 95.05% | 80.96% | 33.94% |
| 8 | exp_swa (5) | 4.99% | 19.30% | 4.99% | 95.01% | 80.70% | 34.27% |
| 9 | bilstm_cnn | 5.01% | 19.30% | 5.01% | 94.99% | 80.70% | 35.91% |
| 9 | exp_focal (1) | 5.01% | 19.48% | 5.01% | 94.99% | 80.52% | 33.77% |
| 11 | bilstm_crf | 5.45% | 20.59% | 5.45% | 94.55% | 79.41% | 31.80% |


**Best model (dev):** `exp_ensemble` — 3-seed ensemble of BiLSTM-CNN-CRF, averaged emissions with a
single CRF decode (6) — DER 4.74%, ~0.16pp ahead of the next best (`exp_attention` /
`exp_curriculum` at 4.86%), and ~0.44pp ahead of the BiLSTM-CNN-CRF baseline.

