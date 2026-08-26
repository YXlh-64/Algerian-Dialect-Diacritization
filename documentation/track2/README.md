# Track 2 — CANINE-S character-level diacritization

Track 2 contains the CANINE-S implementations consolidated on the
`Track2/Combined` branch. Both pipelines use `google/canine-s` as an
encoder-only, character-level backbone and predict one of the 16 official
diacritic classes for each input character. Spaces are retained as context but
are ignored by the loss and evaluation metrics.

## Implementations

| Implementation | Model key | Head | Purpose |
|---|---|---|---|
| `canine_s_model` | `canine_s` | Standard 16-class classifier | Direct CANINE-S baseline |
| `canine_twohead` | `canine_s_twohead` | Factorized shadda/vowel head | Predicts 2 shadda classes × 8 vowel classes |

The standard classifier is based on
[`models/track2/canine_s_model/canine_s_model.py`](../../models/track2/canine_s_model/canine_s_model.py).
The factorized implementation is in
[`models/track2/canine_twohead/canine_twohead_model.py`](../../models/track2/canine_twohead/canine_twohead_model.py).

## Repository structure

```text
configs/track2/
    canine_s_model/canine_s_strategy_a.yaml
    canine_twohead/strategy_a_canine_s_twohead_09406.yaml

models/track2/
    canine_s_model/canine_s_model.py
    canine_twohead/canine_twohead_model.py

training/track2/
    canine_s_model/finetune_canine_s_model.py  # dispatcher entry point
    canine_s_model/train_canine_s.py           # direct entry point
    canine_twohead/finetune_canine_twohead.py

evaluation/track2/
    canine_s_model/evaluate_canine_s.py
    canine_twohead/evaluate_canine_twohead.py

utils/track2/
    canine_s_model/data_utils.py
    canine_twohead/data_utils.py

experiments/track2/
    canine_s_model/strategy_a_overview.md
    canine_twohead/strategy_a_overview.md
```

`run_pipeline.py` discovers both training pipelines through the convention
`training/<track>/<head_type>/finetune_<head_type>.py`.

## Data and label alignment

The training and development data are expected under a dataset root with:

```text
data/
    train_data/*.jsonl
    dev_data/*.jsonl
    test_data/                         # needed for submissions
```

Each JSONL record contains an input character sequence and one label per
character. CANINE adds special tokens around each sequence; the utilities add
ignored labels for those special tokens, truncate characters and labels
together, and mask spaces with `-100`. This keeps the two implementations'
loss and metric scope consistent.

## Running the pipelines

From the repository root, use a dataset directory whose final component is
`data` when calling the shared dispatcher:

```bash
# Standard 16-class CANINE-S
python run_pipeline.py \
  --track track2 \
  --head-type canine_s_model \
  --model canine_s \
  --data-dir /path/to/data

# Factorized two-head CANINE-S
python run_pipeline.py \
  --track track2 \
  --head-type canine_twohead \
  --model canine_s_twohead \
  --data-dir /path/to/data
```

The direct standard-classifier entry point is also available:

```bash
python training/track2/canine_s_model/train_canine_s.py \
  --active-model canine_s \
  --data-dir /path/to/data
```

To evaluate an exported model without training:

```bash
python -m evaluation.track2.canine_s_model.evaluate_canine_s \
  --model-dir working/exports/track2/canine_s_model \
  --data-dir /path/to/data
```

## Outputs

Generated files are kept outside the source layout:

```text
working/checkpoints/track2/canine_s_model/
working/exports/track2/canine_s_model/
working/exports/track2/canine_s_twohead/
```

Exports may include model weights, tokenizer files, training configuration,
development metrics, evaluation reports, and submission CSV files. Datasets,
checkpoints, and generated submissions should not be committed.

## Recorded results

The recorded Strategy-A results are documented in the experiment summaries:

- Standard CANINE-S: validation accuracy/micro-F1 `0.9452`; private leaderboard
  F1 `0.93235` is reported separately.
- Factorized two-head CANINE-S: held-out development accuracy/micro-F1
  `0.9406`, DER `0.0594`.

These values are inherited records from the source branches. The combined
branch was checked structurally only; no training or experiment was run while
consolidating it.

## Verification

The consolidation was verified with Python syntax compilation, YAML parsing,
package imports, label-alignment checks using a dummy tokenizer, and
`run_pipeline.py --dry-run` for both Track 2 combinations. These checks do not
download a model, load the competition data, or execute training.
