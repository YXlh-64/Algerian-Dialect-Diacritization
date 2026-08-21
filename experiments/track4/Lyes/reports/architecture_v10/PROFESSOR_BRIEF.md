# Architecture v10 — Professor Brief

## Baseline retained

The control is `DziriFormer-DualRoPE-CRF-v7`: parallel six-layer
windowed-RoPE and four-layer full-attention-RoPE streams, local-to-global
cross-attention, an adaptive feature-wise fusion gate, two full-attention
refinement blocks, direct 16-class emissions, and a first-order CRF over
scored letters. It obtains 14,816/15,897 neural dev letters.

## 1. Best/last snapshot averaging

- **Change:** equal arithmetic mean of CRF marginal probabilities from
  `best.pt` and `last.pt`; no retraining and no fitted weight.
- **Motivation:** reduce checkpoint noise without adding a new architecture.
- **Result:** V2 moved from 14,962 to 14,963 correct.
- **Conclusion:** rejected as a one-letter, non-robust gain.
- **Feedback question:** should snapshot diversity be revisited only after
  saving multiple checkpoints around the optimum rather than just best/last?

## 2. WordPos-CRF-v10

- **Change:** added learned position-from-word-start, position-from-word-end,
  word-initial, and word-final embeddings. Everything after the input
  embedding remained identical to v7.
- **Motivation:** expose morphology-relevant boundary position directly
  instead of requiring attention to infer it from spaces.
- **Result:** 14,807 neural and 14,952 V2; OOV tied v7, exact words −12.
  Shadda and Tanween binary accuracies improved.
- **Conclusion:** rejected. Explicit position features appear redundant with
  spaces, local RoPE attention, and CRF transitions.
- **Feedback question:** is a boundary auxiliary objective preferable to
  injecting boundary embeddings into every encoder layer?

## 3. FactorizedEmission-CRF-v10

- **Change:** replaced the 16-way emission projection with an eight-way base
  diacritic head and a binary Shadda head. Their normalized log-probabilities
  compose the 16 CRF emissions. CRF NLL is the sole loss; no auxiliary
  weighting is used.
- **Motivation:** preserve the interpretable reformulation while allowing the
  CRF to model sequence structure.
- **Result:** 14,780 neural and 14,958 V2; OOV −1 and exact words −12.
- **Conclusion:** rejected. The independence constraint removes useful
  joint-label context before sequence decoding.
- **Feedback question:** would a residual direct-16 emission branch retain
  interpretability without imposing hard independence?

## 4. LowRankBoundaryCRF-v10

- **Change:** at cross-word transitions, used
  `T_boundary = T_shared + U·V` with rank two. This adds 64 parameters instead
  of BoundaryCRF-v8's unconstrained 256-parameter replacement matrix.
- **Motivation:** capture a small boundary-specific correction while keeping
  common transition statistics shared.
- **Result:** 14,800 neural and 14,954 V2. It gained seven OOV letters and
  three exact words, but lost 16 total letters.
- **Conclusion:** rejected for production but retained as the strongest
  research signal. Boundary conditioning helps OOV/word structure, while the
  current residual still perturbs frequent transitions too much.
- **Feedback question:** should the residual be regularized or gated by
  encoder confidence so it activates only on uncertain word starts?

## 5. Final train+dev refit

- **Selection:** none of the new architectures passed all gates, so the
  standard v7 CRF was selected.
- **Training:** 5,471 train+dev sentences, 22 epochs, 1,892 updates, last
  epoch selection; the original run used 1,900 updates.
- **Fusion:** the unchanged V2 fallback was fit on train+dev.
- **Scientific status:** competition-only. It must not appear as a dev-scored
  paper result because dev labels were used for training.
- **Feedback question:** if Kaggle improves, should the paper report this only
  as a final competition refit while retaining all ablations on untouched dev?

## Main recommendation

Keep direct 16-class CRF emissions. If another architectural experiment is
allowed, combine the low-rank boundary residual with a learned confidence
gate and an explicit regularizer toward zero; do not revisit static lexical
weights or unconditional boundary replacement.
