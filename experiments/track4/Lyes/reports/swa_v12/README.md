# SWA-v12 controlled experiment

This area separates two materially different operations:

1. `checkpoint-weight-probe` averages the existing `best.pt` and `last.pt`
   weights without additional training. It is a cheap diagnostic and is **not
   SWA**.
2. `swa-tail` resumes the selected CRF checkpoint, preserves its AdamW moments,
   replaces the exhausted OneCycle schedule with a fixed learning rate equal to
   the selected checkpoint's recorded learning rate, saves one model-only
   checkpoint per epoch, and evaluates each arithmetic prefix mean of the tail
   weights.

The learning rate therefore introduces no newly tuned multiplier:

- CRF-v7 starts from epoch 22 at `2.1052428265465288e-05`.
- BoundaryCRF-v8 starts from epoch 21 at `3.680753068922983e-05`.

Both models contain LayerNorm but no BatchNorm, so averaged weights need no
running-statistics refresh. RoPE inverse-frequency buffers are non-persistent
and reconstructed from the checkpoint model configuration.

## Cheap probe

```bash
python -m experiments.track4.Lyes.swa_v12 \
  --mode checkpoint-weight-probe \
  --system crf_v7 \
  --device cpu

python -m experiments.track4.Lyes.swa_v12 \
  --mode checkpoint-weight-probe \
  --system boundary_crf_v8 \
  --device cpu
```

## True SWA tail

Run the eight-epoch primary protocol on MPS:

```bash
python -m experiments.track4.Lyes.swa_v12 \
  --mode swa-tail \
  --system crf_v7 \
  --target-tail-epochs 8 \
  --device mps

python -m experiments.track4.Lyes.swa_v12 \
  --mode swa-tail \
  --system boundary_crf_v8 \
  --target-tail-epochs 8 \
  --device mps
```

The commands resume from `outputs/swa_v12/01_swa_tail/<system>/resume.pt`.
Only extend an accepted run, or a run where epoch 8 is the best prefix and the
last three prefix V2-correct counts are non-decreasing, to twelve epochs:

```bash
python -m experiments.track4.Lyes.swa_v12 \
  --mode swa-tail \
  --system boundary_crf_v8 \
  --target-tail-epochs 12 \
  --device mps
```

## Acceptance rule

Compare V2 predictions with the matching source checkpoint. Accept only if:

- correct dev letters improve by at least 10;
- exact-word accuracy does not regress;
- exact-sentence accuracy does not regress;
- Shadda presence accuracy does not regress;
- Tanween presence accuracy does not regress.

The CRF-v7 threshold is 14,972 V2-correct dev letters. The BoundaryCRF-v8
threshold is 14,987. Passing CRF-v7's experiment threshold does not make it a
production candidate unless it also exceeds the current 14,977-letter system.

If a best SWA prefix passes, export its V2 submission with a distinct name:

```bash
python -m evaluation.track4.Lyes.gated_fusion \
  --checkpoint outputs/swa_v12/01_swa_tail/boundary_crf_v8/best_swa.pt \
  --output-dir outputs/swa_v12/01_swa_tail/boundary_crf_v8/artifacts \
  --artifact-prefix DZIRIFORMER_DUALROPE_BOUNDARY_CRF_V8_SWA_TAIL_V12 \
  --system-name DziriFormer-DualRoPE-BoundaryCRF-v8-SWA-tail-v12 \
  --device mps
```

## Cost

The original MPS runs took about 59 seconds per epoch. Eight to twelve tail
epochs should take roughly 8–12 minutes per architecture plus prefix-average
evaluation overhead. Model-only snapshots are approximately 38 MiB each. With
snapshots, the resumable optimizer/model/mean state, and the best/current
averages, budget approximately 1.1 GiB for both eight-epoch runs or 1.4 GiB if
both are extended to twelve epochs.
