# Pre-HGL/HGL Campaign — Final Results

## Final decision

The released-dev winner is **DziriEnsemble-Uniform-v3 + V2**:

| Metric | Result |
|---|---:|
| Neural Micro-F1 | `0.9277851167` |
| Neural correct | `14,749 / 15,897` |
| V2 Micro-F1 | **`0.9394854375`** |
| V2 correct | **`14,935 / 15,897`** |
| Test rows | `16,438` |
| Submission SHA-256 | `25674bd7c2cfeb6506b4efdf6eff958f6835b11d39e8686be774ed11eda5961f` |

Kaggle-ready file:

```text
outputs/pre_hgl_v5/08_final/DZIRI_FINAL_CAMPAIGN_V5_SUBMISSION.csv
```

The final CSV is byte-identical to the selected uniform-ensemble V2 CSV,
which was independently regenerated with the competition's
`make_submission.py`.

## Final candidate comparison

| Candidate | Neural F1 | Neural correct | Fused F1 | Fused correct | Decision |
|---|---:|---:|---:|---:|---|
| Current five-model uniform + V2 | `0.9277851167` | 14,749 | **`0.9394854375`** | **14,935** | Selected |
| Expanded accepted-architecture ensemble + V2 | `0.9274705919` | 14,744 | `0.9390451028` | 14,928 | Rejected: -7 |
| HGL three-seed ensemble + V2 | **`0.9279109266`** | **14,751** | `0.9376611939` | 14,906 | Neural win; fused -29 |
| Five-fold OOF lexical gate | — | — | — | — | Deferred at 17/25 trainings |

The HGL ensemble is the strongest neural system, improving the current
five-model neural ensemble by two letters. However, applying the unchanged V2
fallback reduces its result to 14,906 correct. The final selector therefore
correctly retains the simpler uniform ensemble + V2.

## Per-seed architecture results

All runs below report the exported artifact evaluation on the released
15,897-letter dev split.

| Architecture | Seed | Params | Best epoch | Neural F1 | Neural correct | V2 F1 | V2 correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| HierMixed-v4 | 42 | 6,420,491 | 32 | `0.9231301503` | 14,675 | `0.9362772850` | 14,884 |
| HierMixed-v4 | 43 | 6,420,491 | 32 | `0.9225640058` | 14,666 | `0.9354595207` | 14,871 |
| HierMixed-v4 | 44 | 6,420,491 | 48 | `0.9231930553` | 14,676 | `0.9354595207` | 14,871 |
| Direct16-v3 | 42 | 5,087,504 | 34 | `0.9178461345` | 14,591 | `0.9339498018` | 14,847 |
| Direct16-v3 | 43 | 5,087,504 | 37 | `0.9182864691` | 14,598 | `0.9334465622` | 14,839 |
| Direct16-v3 | 44 | 5,087,504 | 27 | `0.9182235642` | 14,597 | `0.9344530415` | 14,855 |
| GL-Curriculum-v4 | 42 | 5,090,314 | 49 | `0.9181606592` | 14,596 | `0.9336352771` | 14,842 |
| GL-Curriculum-v4 | 43 | 5,090,314 | 38 | `0.9177832295` | 14,590 | `0.9357111405` | 14,875 |
| GL-Curriculum-v4 | 44 | 5,090,314 | 46 | `0.9179090394` | 14,592 | `0.9362143801` | 14,883 |
| HGL-v4 | 42 | 6,426,385 | 49 | `0.9234446751` | 14,680 | `0.9363401900` | 14,885 |
| HGL-v4 | 43 | 6,426,385 | 51 | `0.9235704850` | 14,682 | `0.9360885702` | 14,881 |
| HGL-v4 | 44 | 6,426,385 | 43 | `0.9237591998` | 14,685 | `0.9352708058` | 14,868 |

The seed-42 HGL training summary recorded one additional correct letter
relative to artifact inference. This is an MPS batch-shape numerical boundary
case. Gate conclusions remain unchanged, and the table uses artifact-level
scores consistently.

## Multi-seed decisions

| Architecture | Neural mean ± sample SD | V2 mean ± sample SD | Acceptance |
|---|---:|---:|---|
| HierMixed-v4 | `0.9229624038 ± 0.0003464535` | `0.9357321088 ± 0.0004721365` | Accepted |
| Direct16-v3 | `0.9181187226 ± 0.0002381543` | `0.9339498018 ± 0.0005032396` | Accepted |
| GL-Curriculum-v4 | `0.9179509761 ± 0.0001921778` | `0.9351869325 ± 0.0013671278` | Accepted |
| HGL-v4 | **`0.9235914533 ± 0.0001583073`** | `0.9358998553 ± 0.0005591114` | Accepted vs HierMixed neural control |

HGL beats the matched HierMixed control on all three seeds:

| Seed | HierMixed correct | HGL correct | Delta |
|---:|---:|---:|---:|
| 42 | 14,675 | 14,680 | +5 |
| 43 | 14,666 | 14,682 | +16 |
| 44 | 14,676 | 14,685 | +9 |

This satisfies both HGL acceptance conditions: its three-seed mean is higher,
and at least two seeds improve. In fact, all three improve.

## Diagnostic comparison

| System | OOV acc. | Seen acc. | Shadda acc. | Fatha→Sukoon | Sukoon→Fatha |
|---|---:|---:|---:|---:|---:|
| Expanded ensemble, neural | `0.856106` | `0.945239` | `0.630603` | 288 | 241 |
| Expanded ensemble, V2 | `0.856106` | `0.959695` | `0.704791` | 237 | 220 |
| HGL ensemble, neural | `0.855475` | `0.945946` | `0.649150` | 245 | 259 |
| HGL ensemble, V2 | `0.855475` | `0.958124` | `0.709428` | 230 | 223 |

HGL's main architectural gain is not OOV coverage; it improves contextual
neural accuracy and Shadda handling. The unchanged V2 thresholds favor the
older ensemble's seen-word behavior and are not calibrated to HGL.

## OOF gate status

The OOF logistic gate is explicitly **not a valid result**. It was deferred
after 17 of 25 architecture-fold trainings because sustained MPS thermal
throttling expanded runtime beyond the project deadline.

- Complete outer folds: 0, 1, and 2.
- Fold 3 complete models: Base and J16.
- Interrupted partial run: fold 3 GL.
- Fold 4: not started.
- Logistic gate fitted: no.
- OOF/dev score reported: no.

The durable state is recorded in:

```text
outputs/pre_hgl_v5/06_oof_gate/DEFERRED.json
```

## Submission descriptions

### Final — `DZIRI_FINAL_CAMPAIGN_V5_SUBMISSION.csv`

Equal probability ensemble of five from-scratch Track 4 character systems:
Base, J16, GL, Mixed, and Hier. The unchanged confidence-gated V2 lexical
fallback is applied only after neural probability averaging. Selected on the
released dev split at 0.939485 Micro-F1.

### HGL ensemble — `DZIRIFORMER_HGL_V4_ENSEMBLE_V2_SUBMISSION.csv`

Three-seed ensemble of the 6.43M-parameter HGL architecture combining a
hierarchical word-character encoder, periodic full character attention, a
direct 16-class head, and linear blank-hint curriculum. The architecture is a
stable neural improvement, while its unchanged V2 fusion reaches 0.937661
dev Micro-F1.

### HGL neural — `DZIRIFORMER_HGL_V4_ENSEMBLE_NEURAL_SUBMISSION.csv`

Neural-only three-seed HGL probability ensemble with no lexical fallback.
This is the campaign's strongest neural result at 0.927911 dev Micro-F1.

## Track compliance

Every completed system remains Track 4:

- encoders are trained from scratch;
- no pretrained model or embedding is used;
- no external corpus, analyzer, morphological lexicon, or tokenizer is used;
- word vectors are composed from released character inputs;
- lexical statistics use released training labels only.
