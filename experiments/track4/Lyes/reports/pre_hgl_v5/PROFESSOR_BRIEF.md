# Professor Brief — DziriFormer Pre-HGL/HGL Campaign

## Scope

All systems are Track 4 models trained from scratch using only the released
competition train split. No pretrained encoder, external embedding, external
corpus, morphological analyzer, or external tokenizer is used.

The campaign separates two questions:

1. Which neural inductive biases improve character-label prediction?
2. Does the existing confidence-gated lexical fallback preserve those gains?

The answer is now different for each question: HGL is a stable neural
improvement, but the old V2 fallback is not the best fusion policy for it.

## 1. DziriEnsemble-Uniform-v3

**Previous architecture.** Five independently trained systems were available:
Base, J16, GL, Mixed, and Hier.

**Exact change.** Normalize each model's 16-class probabilities, average them
arithmetically with equal `1/5` weight, then apply the unchanged V2 lexical
fallback once to the averaged distribution.

**Motivation.** Measure complementary errors without learned or hand-selected
ensemble weights.

**Result.** Neural `0.927785` (14,749 correct); V2 `0.939485`
(14,935 correct). This is the final released-dev winner.

**Conclusion.** Equal probability averaging is a strong, robust baseline and
does not need model-specific weights.

**Focused feedback question.** Should this ensemble remain the paper's
production result while HGL is presented as the strongest neural ablation?

## 2. DziriFormer-HierMixed-v4

**Previous architecture.** Hier-v4 uses local/shifted-local character
attention plus a full word-level Transformer and a factorized base/Shadda
head.

**Exact change.** Set `global_attention_every=3`, producing:

```text
local, shifted-local, full, shifted-local, local, full
```

All other hierarchy and factorized-head settings remain unchanged.

**Motivation.** Test whether sentence-wide character syntax complements
word-level global context and local morphology.

**Result.** Three-seed neural mean `0.922962 ± 0.000346`, higher than the
original Hier-v4 control. All seed gates passed.

**Conclusion.** Periodic full character attention provides a repeatable
improvement and is justified for HGL.

**Focused feedback question.** Is a fixed full-attention interval of three an
acceptable controlled architectural choice, or should the paper add a small
interval ablation?

## 3. DziriFormer-Direct16-v3

**Previous architecture.** The original model uses separate eight-class base
and binary Shadda heads, combining their probabilities into 16 labels.

**Exact change.** Replace both factorized heads and the J16 gate with one
direct 16-class linear head optimized by one unweighted official-label
cross-entropy.

**Motivation.** Remove auxiliary-loss and mixture assumptions, allowing the
official classes to be modeled jointly.

**Result.** Three-seed neural mean `0.918119 ± 0.000238`; all seeds beat the
14,585-correct original control.

**Conclusion.** The direct head is a small but stable improvement on the base
encoder and is justified as an HGL component.

**Focused feedback question.** Given that Direct16 improves Micro-F1 but has
variable Shadda accuracy, should the paper retain the factorized head mainly
as an interpretability ablation?

## 4. DziriFormer-GL-Curriculum-v4

**Previous architecture.** GL-v3 samples a uniform hint-masking level during
training, but validation and test always use blank hints.

**Exact change.** At epoch `e` of 60, force an entire example's hints blank
with probability `(e-1)/59`; otherwise retain the original uniform masking
schedule. Run all 60 epochs and select only on blank-hint dev F1.

**Motivation.** Progressively eliminate the train/inference mismatch without
introducing a hand-weighted lexical interpolation.

**Result.** Three-seed neural mean `0.917951 ± 0.000192`; all three seeds beat
the original 14,585-correct neural control.

**Conclusion.** Curriculum training is stable but its standalone improvement
is small. Its strongest value is as a regularizer inside HGL.

**Focused feedback question.** Is the linear blank probability sufficiently
principled, or should a future experiment compare linear and cosine schedules?

## 5. DziriEnsemble-Expanded-v5

**Previous architecture.** The uniform ensemble treats the original five
seed-42 models as one equal-probability pool.

**Exact change.** First average seeds within each accepted new architecture,
then treat the current uniform ensemble, HierMixed, Direct16, and
GL-Curriculum as four equal architecture-level experts.

**Motivation.** Prevent three-seed architectures from receiving triple weight
while testing whether accepted ablations add useful diversity.

**Result.** Neural `0.927471` (14,744 correct); V2 `0.939045`
(14,928 correct), seven fused letters below the current uniform ensemble.

**Conclusion.** Passing a standalone gate does not guarantee useful ensemble
diversity. The expanded ensemble is rejected.

**Focused feedback question.** Is it preferable to report this negative result
as evidence against uncalibrated ensemble expansion?

## 6. DziriEnsemble-OOFGate-v3

**Proposed change.** Learn a standardized logistic neural-versus-lexical
switch only from leakage-free out-of-fold disagreements, replacing manually
selected interpolation or thresholds.

**Motivation.** Directly address the concern that fixed lexical weighting can
override context-sensitive neural predictions.

**Execution status.** Deferred after 17/25 surrogate trainings because MPS
thermal throttling made the experiment incompatible with the deadline. No
gate was fitted and no score is claimed.

**Conclusion.** The partial artifacts are retained only for possible future
resumption; they are not evidence for or against the method.

**Focused feedback question.** Is completing the remaining eight OOF
trainings worth prioritizing later, or should the next gate be trained on a
smaller, architecture-specific calibration split?

## 7. DziriFormer-HGL-v4

**Previous architecture.** HierMixed provides hierarchy and mixed attention;
Direct16 provides the official-label head; GL-Curriculum provides progressive
blank-hint training.

**Exact change.** Combine:

- character CNN with six Transformer blocks;
- attention schedule
  `local, shifted-local, full, shifted-local, local, full`;
- character-composed word vectors;
- learned gated word pooling;
- two full word-level Transformer layers;
- gated word-to-character broadcast;
- one direct 16-class head;
- 60-epoch linear blank-hint curriculum.

The model has 6,426,385 parameters.

**Motivation.** Combine only components that passed their independent
multi-seed gates, then test whether hierarchy, global syntax, direct class
modeling, and guided regularization interact positively.

**Result.** HGL beats matched HierMixed on every seed:

| Seed | HierMixed correct | HGL correct | Delta |
|---:|---:|---:|---:|
| 42 | 14,675 | 14,680 | +5 |
| 43 | 14,666 | 14,682 | +16 |
| 44 | 14,676 | 14,685 | +9 |

HGL neural mean is `0.923591 ± 0.000158`. Its three-seed neural ensemble
reaches `0.927911` (14,751 correct), the best neural campaign result.

After unchanged V2 fusion, however, the ensemble falls to `0.937661`
(14,906 correct), 29 letters below the final uniform + V2 system.

**Conclusion.** HGL is accepted as an architectural improvement. The fusion
policy, not the neural encoder, is now the limiting component.

**Focused feedback question.** Should the next experiment recalibrate a
confidence gate specifically for HGL, using neural entropy/margin and lexical
confidence on a held-out calibration partition, rather than applying V2
thresholds tuned for the older model family?

## Recommended paper framing

- Report **Uniform Ensemble + V2** as the best complete released-dev system.
- Report **HGL** as the strongest and most stable neural architecture.
- State explicitly that HGL's neural gain does not transfer through the old
  lexical gate.
- Report the expanded ensemble as a negative ablation.
- Mark OOF gating as deferred, with no result claimed.
