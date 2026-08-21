# R-Drop v13 Results

This report closes the pre-registered `DziriFormer-DualRoPE-CRF-EmissionRDrop-v13`
campaign. The experiment regularizes two dropout-perturbed forward passes with
symmetric KL divergence over normalized 16-class CRF emissions. The v7 encoder,
linear CRF, label scheme, and V2 fallback are otherwise unchanged.

## Decision

| Decision | Result |
|---|---:|
| Selected coefficient | `1.0` |
| Locked final epochs | `22` |
| Standalone accepted | **Yes** |
| Final ensemble accepted | **No** |
| Overall campaign accepted | **No** |

The standalone model passed every declared gate, but replacing the v7 CRF
expert in the four-group ensemble produced only 14,978 V2-correct dev letters,
below the pre-registered 14,987 requirement. No v13 CSV is therefore approved
for Kaggle submission under this campaign protocol.

## Train-only calibration

| Split | Candidate | Neural gain | V2 gain | Best epoch | Decision |
|---|---:|---:|---:|---:|---|
| A | lambda 0.1 | +83 | +35 | - | Pass |
| A | lambda 0.3 | +82 | +34 | - | Pass |
| A | lambda 1.0 | **+128** | **+45** | 23 | Selected |
| B | lambda 1.0 | **+64** | **+32** | 21 | Confirmed |

Both 973-sentence calibration folds were held out from their matching control
and candidate training data. The mean neural gain was 96 correct letters. No
protected V2 metric regressed on either split.

## Locked released-dev result

| System | Neural correct | Neural F1 | V2 correct | V2 F1 |
|---|---:|---:|---:|---:|
| Fixed-epoch control | 14,829 | 0.932818 | 14,954 | 0.940681 |
| EmissionRDrop-v13 | **14,845** | **0.933824** | **14,991** | **0.943008** |
| Replacement ensemble | 14,860 | 0.934768 | 14,978 | 0.942190 |

Standalone gains were +16 neural and +37 V2 correct letters. The V2 candidate
also recorded word accuracy 0.818605, sentence accuracy 0.365733, OOV accuracy
0.876302, Shadda accuracy 0.986601, Tanween accuracy 0.999874, and zero skeleton
mismatches. Word, sentence, OOV, Shadda, and Tanween accuracy all satisfied the
non-regression gate.

## Reproducibility

- Device: `mps`
- Candidate checkpoint:
  `outputs/rdrop_emission_v13/03_final_seed42/rdrop_lambda_1/best.pt`
- Candidate checkpoint SHA-256:
  `26d2a9bd3a5a33c55a1760ef20997270c1bf53bb43ba3adb034fbee91ab7f977`
- Rejected ensemble V2 submission SHA-256:
  `5118117192a0aa7969d6d0389925ae91d10453ca239fcd459b0ef5346314cda7`
- Authoritative decision:
  `outputs/rdrop_emission_v13/03_final_seed42/SELECTION.json`

No v13 seeds 43/44 will be trained, and no post-hoc ensemble weight will be
tuned. v7 remains the approved backbone; v13 is retained as a positive
standalone regularization ablation and negative ensemble-replacement result.
