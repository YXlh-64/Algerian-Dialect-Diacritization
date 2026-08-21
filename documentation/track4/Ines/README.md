# Track 4 — Ines — Dual-Stream CRF Head

This package is the file-based version of Ines's Track 4 notebook work on
branch `Track4/Ines/Dual_Stream_CRF_Head`.

## Structure

```text
configs/track4/Ines/
    dscat_base_v1.yaml

models/track4/Ines/
    dual_stream_crf_head_model.py

training/track4/Ines/
    finetune_dual_stream_crf_head.py

evaluation/track4/Ines/
    evaluate_dual_stream_crf_head.py

experiments/track4/Ines/
    run.py

documentation/track4/Ines/
    README.md
```

## Architecture

`Track4DualStreamCRF` is trained from scratch and contains:

1. a character embedding layer;
2. a six-layer local RoPE attention stream with a fixed attention window;
3. a four-layer global RoPE attention stream;
4. cross-attention from the local stream to the global stream;
5. an adaptive fusion gate;
6. two full-attention refinement layers;
7. a 16-class emission projection and first-order linear-chain CRF.

The CRF operates only on letter positions. Spaces and padding remain encoder
context but are excluded from loss and metrics. Their decoded placeholder is
label `0`, never the character vocabulary's space ID.

## Configuration

The registered Strategy-A values are recorded in
`configs/track4/Ines/dscat_base_v1.yaml`. The training `Config` dataclass uses
the same defaults and permits explicit CLI overrides for the data root, epoch
count, and random seed.

## Training and submission generation

From the repository root:

```bash
python -m experiments.track4.Ines.run \
  --data-root /path/to/algerian-arabic-diacritization \
  --epochs 25 \
  --seed 42
```

The pipeline trains with AdamW and OneCycleLR, selects the checkpoint with the
best dev Micro-F1, predicts the test sentences, and invokes the competition's
`make_submission.py` using the current Python interpreter.

Default Kaggle artifacts:

```text
/kaggle/working/dscat_best.pt
/kaggle/working/submission.txt
/kaggle/working/submission.csv
```

## Standalone evaluation

```bash
python -m evaluation.track4.Ines.evaluate_dual_stream_crf_head \
  --checkpoint /path/to/dscat_best.pt \
  --data-dir /path/to/algerian-arabic-diacritization
```

Evaluation reports Micro-F1, Macro-F1, per-class F1, and a confusion matrix
over letter positions only. DER, DER*, WER, and WER* helpers are also included.

## Verification

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q configs models training evaluation experiments tests
```

Full training and submission generation require the competition dataset and
are not claimed by these structural/unit checks alone.
