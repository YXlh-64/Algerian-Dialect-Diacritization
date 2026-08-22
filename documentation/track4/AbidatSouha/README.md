# Track 4 — Algerian Dialect Diacritization (AbidatSouha)

A character-level **Transformer + CNN + CRF** sequence tagger for Algerian Arabic
diacritization, built from scratch: the encoder, the attention, the relative
position bias and the linear-chain CRF are all hand-written. No pretrained
weights, no `nn.Transformer*`, no `torchcrf`. The only third-party imports are
`torch` and `numpy`.

## Documents

| Document | Contents |
|---|---|
| [architecture.md](architecture.md) | End-to-end component chart, tensor shapes, parameter budget |
| [features.md](features.md) | The six morphological input streams and the data that motivates them |
| [cnn-frontend.md](cnn-frontend.md) | Depthwise-separable convolution front-end |
| [transformer.md](transformer.md) | The encoder in depth — RMSNorm, SwiGLU, T5 relative bias, attention |
| [output-head.md](output-head.md) | Factorized head, interaction table, character prior, auxiliary head |
| [crf.md](crf.md) | Linear-chain CRF with split intra-/inter-word transitions |
| [results.md](results.md) | Baselines, measured results, error analysis, what remains unrun |

## Task

Character-level sequence tagging. Every Arabic letter gets exactly one of **16
diacritic classes**; space positions are not scored. The label scheme is
factorizable:

```
label = 8 * shadda + base
base  = 0 none | 1 fatha | 2 fathatan | 3 damma | 4 dammatan | 5 kasra | 6 kasratan | 7 sukoon
```

Output is length-preserving: one label per input character, so the letter
skeleton of the input is never altered.

## Dataset

| Split | Sentences | Positions | Letters | Spaces | Mean len | Max len |
|---|---:|---:|---:|---:|---:|---:|
| train | 4,864 | 160,583 | 133,032 | 27,551 (17.2%) | 33.0 | 274 |
| dev | 607 | 19,160 | 15,897 | 3,263 (17.0%) | 31.6 | 220 |
| test | 608 | 19,817 | 16,438 | 3,379 (17.1%) | 32.6 | 143 |

- **Vocabulary**: 43 characters, loaded from `vocab.json`. Ids 0–42, with
  `<PAD>` = 0 and `<UNK>` = 1.
- **Train lexicon**: 7,687 distinct surface forms, of which 1,012 are ambiguous
  (more than one vocalization observed). This drives the evaluation buckets.
- **Test labels are not distributed** — the test split is scored only by the
  leaderboard.
- **Augmentation**: `char_dropout = 0.08` at training time replaces input
  characters with `<UNK>`, approximating the 14.7% OOV rate seen on dev.

Only 133,032 labelled letter positions exist. Every design decision in
[architecture.md](architecture.md) follows from that number being small.

## Code structure

The pipeline is split by responsibility, one directory per concern, all under
`track4/AbidatSouha/`:

```
configs/track4/AbidatSouha/
    model_config.py      ModelConfig (architecture + ablation switches), PLAIN_BASELINE
    training_config.py   TrainingConfig, EnsembleConfig, LexicalFallbackConfig
    paths.py             DataPaths, find_data_paths() — locates inputs locally or on Kaggle

models/track4/AbidatSouha/
    layers.py            RMSNorm, SwiGLU, SinPos
    transformer.py       T5RelBias, MHSA, EncoderLayer
    cnn.py               ConvFrontEnd
    crf.py               CRF, is_intra_mask
    tagger.py            DiacModel — the assembled model

utils/track4/AbidatSouha/
    constants.py         label scheme, diacritic marks, sun/mater letter sets
    features.py          word_ids, featurize — the six morphological streams
    data.py              DiacData, collate, build_char_prior
    device.py            get_device() — cuda > mps > cpu
    seed.py              set_seed
    render.py            render() — interleaves marks back into text

training/track4/AbidatSouha/
    trainer.py           train_model — warmup+cosine schedule, early stopping

evaluation/track4/AbidatSouha/
    metrics.py           evaluate, fmt, letters_microf1
    baselines.py         lookup_baseline — the memorisation floor
    inference.py         load_test_set, ensemble_predict, predict_with_confidence
    lexical_fallback.py  parse_vocalized_word, word_spans, gated_labels
    submission.py        write_submission + the sample_submission.csv id check

experiments/track4/AbidatSouha/
    train_run.py         baseline -> T1 -> T5 -> seed ensemble -> threshold search
    predict_run.py       submission.csv and submission_v2.csv

tests/track4/AbidatSouha/
    test_crf.py          CRF vs brute-force enumeration; padding invariance
    test_submission.py   submission Id format against sample_submission.csv
```

Dependencies run strictly one way:

```
configs  ->  utils  ->  models  ->  evaluation  ->  training  ->  experiments
```

`configs` imports nothing from the project; `experiments` is the only layer that
wires everything together and the only one with side effects.

## Running it

From the repository root, so that absolute imports resolve:

```bash
# full pipeline: train, freeze the fallback thresholds, write both submissions
PYTHONPATH=. python experiments/track4/AbidatSouha/predict_run.py

# training only
PYTHONPATH=. python -c "from experiments.track4.AbidatSouha.train_run import run_training; run_training()"

# tests (standalone, no pytest required)
PYTHONPATH=. python tests/track4/AbidatSouha/test_crf.py
PYTHONPATH=. python tests/track4/AbidatSouha/test_submission.py
```

`find_data_paths()` searches `/kaggle/input` first, then `Data/`, `../Data`,
`../../Data`, `../../../Data`, so the same code runs unmodified on Kaggle and on
a local checkout.

Outputs (`submission.csv`, `submission_v2.csv`, `test_vocalized*.txt`) are
excluded by the repository `.gitignore`. Scores are recorded in
[results.md](results.md) instead.
