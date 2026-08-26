# Track 2 — CANINE-S standard classifier

## Selected model

`canine_s` fine-tunes `google/canine-s` with the standard 16-class token
classification head. It is the direct CANINE-S baseline for Track 2, while
`canine_s_twohead` is the factorized shadda/vowel variant.

## Recorded results

The recorded validation result is micro-F1/accuracy **0.9452**. The associated
evaluation report also records a private leaderboard F1-score of **0.93235**;
these are kept separate because the validation and competition splits are not
the same.

## Reproduction

From the repository root:

```bash
python run_pipeline.py \
  --track track2 \
  --head-type canine_s_model \
  --model canine_s \
  --data-dir /path/to/data
```

The dispatcher-compatible entry point is
`training/track2/canine_s_model/finetune_canine_s_model.py`. The original
`train_canine_s.py` entry point remains available for direct use.
