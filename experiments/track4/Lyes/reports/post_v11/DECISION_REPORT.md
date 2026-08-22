# Post-v11 Direction Decision Report

Date: 2026-07-31

## Executive decision

The project has not exhausted every possible modelling direction, but the
specialized CRF branch has reached a falsifiable stopping point. The equal
v7+v11 diversity probe failed, v11's context gate collapsed to an almost
always-on residual, and no-training checkpoint averages also failed. The
training corpus is structurally clean, with only two safe duplicate removals.

The next and only approved short experiment is an eight-epoch true SWA tail on
BoundaryCRF-v8. It is orthogonal to the failed decoder variants, requires no
new inference component, and has a predeclared acceptance gate. V11 seeds
43/44, another specialized CRF, a cleaned-data campaign, and CRF-v7 SWA are all
deferred or rejected.

## Consolidated workstream results

| Workstream | Result | Decision |
|---|---|---|
| Equal v7+v11 ensemble probe | 14,963 V2-correct, 14 below v7 and 24 below acceptance | Reject; do not submit |
| V11 seeds 43/44 | V11 failed neural by 11 and V2 by 22; gate mean 0.997834 | Do not train |
| Specialized CRF frontier | Only ordinary CRF produced a strong replicated decoder gain; later refinements failed | Close branch |
| Train-only data audit | Two redundant records, no conflicting full-sentence skeletons or hard defects | Document; defer retraining |
| Checkpoint averaging/SWA | Existing best+last weight means lost 2 and 11 V2 letters | Reject probes; run one true Boundary-v8 SWA tail |

## 1. Equal architecture-level v7 plus v11 probe

The probe used five fixed architecture-level experts, each contributing 20%:

1. DualRoPE-CRF-v7 seed 42.
2. Mean of DualRoPE-CE-v6 seeds 42/43/44.
3. Mean of HGL-v4 seeds 42/43/44.
4. Mean of the legacy Base/J16/GL/Mixed/Hier group.
5. ContextLowRankBoundaryCRF-v11 seed 42.

The unchanged V2 fallback was applied only after probability averaging. No
weight or threshold was tuned.

| Metric | Existing v7 | Equal-five probe | Delta |
|---|---:|---:|---:|
| Neural correct | 14,865 | 14,858 | -7 |
| Neural Micro-F1 | 0.935082 | 0.934642 | -0.000440 |
| V2 correct | 14,977 | 14,963 | -14 |
| V2 Micro-F1 | 0.942127 | 0.941247 | -0.000881 |

Acceptance required 14,987 V2-correct letters. The probe is rejected and both
generated CSV files are explicitly non-submittable. It also lost 10 OOV and 4
seen-word letters, while adding 12 Fatha-to-Sukoon errors.

Evidence:

- `outputs/context_boundary_v11/03_equal_group_probe/SELECTION.json`
- `outputs/context_boundary_v11/03_equal_group_probe/diagnostics.json`
- `outputs/context_boundary_v11/03_equal_group_probe/artifacts/DZIRI_FINAL_V7_PLUS_CONTEXT_V11_EQUAL5_PROBE_MANIFEST.json`
- `track4/v11_ensemble_probe.py`
- `configs/track4/Lyes/context_boundary_v11/ensemble_probe.json`

## 2. V11 seed-expansion decision

V11 seed 42 produced 14,820 neural-correct and 14,956 V2-correct letters. It
missed the predeclared neural requirement by 11 and the production V2 reference
by 22. Its OOV and exact-word gains were real but insufficient.

The intended contextual selection mechanism also failed:

| Gate diagnostic | Value |
|---|---:|
| All-position mean | 0.997834 |
| Standard deviation | 0.008483 |
| Word-initial mean | 0.999302 |
| Within-word mean | 0.997455 |
| Boundary-minus-within difference | 0.001848 |
| Residual/shared transition-norm ratio | 1.33x |

The model therefore applies a comparatively large low-rank residual almost
everywhere. This is a mechanism failure rather than sufficient evidence of an
unlucky seed. Seeds 43/44 must not be trained unless a revised seed-42 design
passes every gate and beats forced-open and forced-closed gate ablations.

## 3. Specialized CRF stopping decision

The meaningful decoder result remains independent CE to ordinary CRF:

- CRF-v7 gained 52 neural letters over DualRoPE-CE and reached the current best
  Kaggle ensemble result.
- BoundaryCRF-v8 gained another 21 standalone neural letters and 15 standalone
  V2 letters, but its final ensemble gained only one local letter and regressed
  on Kaggle.
- Cross-fitted lexical gating, temperature/simplex stacking, explicit word
  position, factorized emissions, static rank-2 boundary transitions, and the
  context-gated rank-2 transition all failed their controlled gates.

The CRF-design branch is closed unless a future proposal is motivated by a new
measured error cluster and satisfies all of:

1. At least 14,831 neural-correct letters at seed 42.
2. At least 14,987 V2/final-correct letters.
3. Higher three-seed mean than its matched control, with at least two positive
   paired seed deltas.
4. No material OOV, exact-word, Shadda, Tanween, alignment, or skeleton
   regression.
5. No released-dev tuning of weights or thresholds.

A paired seed replication of BoundaryCRF-v8 remains scientifically defensible
for the paper, but it is deferred because it is not the immediate leaderboard
priority.

## 4. Train-only data-quality audit

The audit examined 4,864 training sentences and 133,032 scored letters without
modifying the source data or deriving corrections from dev.

| Finding | Count |
|---|---:|
| Exact duplicate groups | 2 |
| Safely removable redundant records | 2 |
| Conflicting full-sentence skeleton groups | 0 |
| Hard alignment/label/space/rendering defects | 0 |
| Ambiguous word skeletons | 1,012 |
| Proposed retained records | 4,862/4,864 |

The 2,772 non-NFC rendered targets are caused by the intentional
Shadda-before-vowel serialization and reproduce the released labels exactly;
they are not corruption. Word-level vocalization variation is substantial but
context-sensitive, so it is not safe automatic correction material. No
per-sentence training-loss artifact exists, so the audit makes no high-loss
sample claim.

Simple deduplication is low leverage. If tested later, retrain one unchanged
seed-42 control after removing only the two redundant copies, and accept only
at +10 dev letters with no greater than 0.2 percentage-point regression in OOV,
Shadda, or Tanween accuracy.

Evidence:

- `audits/data_quality_v1/REPORT.md`
- `audits/data_quality_v1/summary.json`
- `audits/data_quality_v1/clean_experiment_manifest.csv`
- `audits/data_quality_v1/manifest.json`
- `track4/data_quality_audit.py`
- `tests/test_data_quality_audit.py`

## 5. Checkpoint averaging and true SWA

The no-training arithmetic weight averages of existing best and last
checkpoints are not SWA and both failed:

| Probe | Baseline V2 | Averaged V2 | Delta | Decision |
|---|---:|---:|---:|---|
| CRF-v7 best+last weights | 14,962 | 14,960 | -2 | Reject |
| Boundary-v8 best+last weights | 14,977 | 14,966 | -11 | Reject |

The previous CRF-v7 best+last probability average gained only one letter, also
below the +10 gate. These failures do not invalidate true SWA, which samples
multiple checkpoints under a deliberate fixed-learning-rate tail.

The approved SWA protocol starts from BoundaryCRF-v8 epoch 21, preserves its
AdamW moments, and replaces the exhausted OneCycle schedule with the exact
checkpoint-recorded learning rate `3.680753068922983e-05`. It trains eight
deterministic tail epochs and evaluates every arithmetic prefix weight mean.
The model contains no BatchNorm, so no running-statistics refresh is needed.

Evidence and implementation:

- `experiments/swa_v12/README.md`
- `experiments/swa_v12/CHECKPOINT_WEIGHT_PROBE_RESULTS.md`
- `configs/track4/Lyes/swa_v12/campaign.json`
- `track4/swa_v12.py`
- `tests/test_swa_v12.py`
- `outputs/swa_v12/00_checkpoint_weight_probe/`

## Approved next execution

Run exactly one MPS experiment:

```bash
.venv/bin/python -B -m experiments.track4.Lyes.swa_v12 \
  --mode swa-tail \
  --system boundary_crf_v8 \
  --target-tail-epochs 8 \
  --device mps
```

Accept only if the best prefix reaches at least 14,987 V2-correct letters and
word accuracy, sentence accuracy, Shadda accuracy, and Tanween accuracy do not
regress. Extend to 12 epochs only if the eight-epoch run is accepted, or epoch
8 is the best prefix and the final three prefix counts are non-decreasing.

If it passes, export the accepted checkpoint with the documented distinct SWA
artifact name, then evaluate it as an equal architecture-level replacement in
the final ensemble before considering Kaggle. If it fails, stop SWA, skip the
CRF-v7 tail, keep v7 as the production system, and move to an orthogonal
training objective such as predeclared R-Drop CRF-marginal consistency rather
than another CRF transition parameterization.

## Submission safety

Do not submit any artifact from:

- `outputs/context_boundary_v11/03_equal_group_probe/`
- `outputs/swa_v12/00_checkpoint_weight_probe/`

No approved new Kaggle submission exists from this report. The current best
submission remains the v7 final ensemble until a future candidate passes its
local acceptance protocol.

## Validation

- Full repository test suite: 121 passed.
- Data-audit independent rerun: byte-identical artifacts.
- Ensemble submission/official-check CSV pairs: byte-identical, 16,439 lines.
- SWA implementation: deterministic averaging tests and Python compilation
  passed.
