# DziriFormer-DualRoPE-CRF-EmissionRDrop-v13

## Purpose

This controlled Track 4 experiment changes only the training objective of the
successful `DziriFormer-DualRoPE-CRF-v7` architecture. Two independent dropout
passes share all weights. Their CRF negative log-likelihoods are averaged and a
symmetric KL penalty aligns their normalized 16-class emission distributions.
The CRF and Viterbi decoder remain unchanged.

```text
characters + spaces
        |
        v
unchanged DualRoPE local/global encoder
        |
        +--------------------+
        | dropout pass A     | dropout pass B
        v                    v
   16 emissions A       16 emissions B
        |                    |
        v                    v
  shared linear CRF     shared linear CRF
        |                    |
        +------ emission symmetric KL
        |
        v
0.5 * (CRF_NLL_A + CRF_NLL_B) + lambda * emission_symmetric_KL
```

No pretrained model, external corpus, external tokenizer, or external analyzer
is used. The experiment remains within Track 4.

The initially registered exact CRF-marginal variant was implemented and
numerically validated, then aborted before completing one candidate epoch. On
MPS it exceeded seven minutes per epoch because autograd retained two encoder
graphs plus two full differentiable CRF forward-backward graphs. Emission
R-Drop is the standard controlled alternative. Training uses micro-batches of
32 with two-step gradient accumulation, preserving effective batch size 64
without the MPS memory spike. Aborted artifacts remain under
`outputs/rdrop_v13/` and are never considered by the campaign.

## Execution

CPU validation:

```bash
.venv/bin/python -B -m pytest -q tests/test_rdrop_v13.py
```

Full gated MPS campaign:

```bash
.venv/bin/python -B -m experiments.track4.Lyes.rdrop_v13 \
  --config configs/track4/Lyes/rdrop_v13/campaign.json \
  --stage all \
  --device mps
```

The controller resumes completed runs and fails closed on partial directories.
It does not upload to Kaggle. A generated CSV is submit-worthy only when both
`standalone_accepted` and `ensemble_accepted` are true in
`outputs/rdrop_v13/03_final_seed42/SELECTION.json`.

## Pre-registered gates

- Split A: candidate neural gain at least 5 letters, V2 non-regression, no
  protected-metric regression, and zero skeleton mismatches.
- Split B: positive neural gain on both splits and mean gain at least 5.
- Released dev: at least +15 neural letters and +10 V2 letters over a matched
  fixed-epoch control, with no protected-metric regression.
- Final equal-architecture ensemble: at least 14,987 correct dev letters.

The completed campaign accepted v13 as a standalone ablation but rejected its
replacement ensemble. No v13 seeds 43/44 or tuned ensemble weights are allowed.
The subsequent word-modeling campaign therefore returns to the accepted v7
backbone rather than inheriting v13.
