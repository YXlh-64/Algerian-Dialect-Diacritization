# Literature Review and Track 4 Research Roadmap

Last updated: 2026-07-31

## Scope and evidence standard

This review asks one question: what should be tested after the controlled
DualRoPE, CRF, BoundaryCRF, calibration, word-position, factorized-emission,
low-rank-boundary, and final-refit experiments?

The primary roadmap is restricted to Track 4:

- every encoder is initialized and trained from scratch;
- no pretrained model, pretrained embedding, external tokenizer, external
  corpus, or external morphological analyzer is used;
- training labels and lexical statistics come only from the released data;
- every architectural claim is evaluated on the untouched released dev split;
- the train+dev refit is competition-only and is never used as paper dev
  evidence.

Results from other Arabic varieties are not directly comparable because
corpora, label inventories, cleaning rules, and metric conventions differ.
They are used to motivate experiments, not to claim a cross-corpus SOTA.

## What the literature says

### Data quality and reference ambiguity

Mohamed and Mubarak (EMNLP 2025) identify corpus analysis, corpus refinement,
multi-reference evaluation, and high-quality augmentation as major sources of
Arabic diacritization progress. Their reported 3.12% and 2.70% WER results
are on MSA WikiNews benchmarks, not Algerian Arabic, but the central lesson is
directly relevant: once model errors are small, annotation inconsistency can
be as important as architecture.

Source: [Advancing Arabic Diacritization: Improved Datasets, Benchmarking,
and State-of-the-Art Models](https://aclanthology.org/2025.emnlp-main.846/).

### Character and word context should interact

Deep Diacritization models character and word levels separately, reconnects
them using cross-level attention, and reports a 5.34% WER on Tashkeela. LAMAD
also combines word and character context. Our Hier and HierMixed gains support
the same general claim, although the unchanged V2 fallback was not calibrated
to those neural distributions.

Sources:

- [Deep Diacritization](https://aclanthology.org/2020.wanlp-1.4/)
- [LAMAD](https://aclanthology.org/2021.findings-emnlp.317/)

### Morphological and syntactic signals can help

Morphologically informed character models add segmentation markers and report
strong MSA, Classical, Moroccan, and Tunisian results. Multitask restoration
work jointly learns segmentation, POS, and syntactic diacritization. The
published implementations depend on extra annotations or analyzers, so copying
them directly would not be a clean Track 4 experiment. A compliant analogue
must derive auxiliary targets from the released text itself, such as
whitespace boundaries, within-word position, or consistency under affix
masking.

Sources:

- [Arabic Diacritization Using Morphologically Informed Character-Level
  Model](https://aclanthology.org/2024.lrec-main.128/)
- [A Multitask Learning Approach for Diacritic
  Restoration](https://aclanthology.org/2020.acl-main.732/)

### Sequence decoding is valuable, but static transitions are limited

Our ordinary CRF recovered 52 letters over direct CE and BoundaryCRF recovered
another 21. This is internally stronger evidence than a generic citation.
Semi-Markov CRFs provide a literature-backed route from letter transitions to
whole segments, but unrestricted semi-Markov inference is quadratic. Filtered
Semi-Markov CRF shows that candidate filtering can reduce the search space.

Source: [Filtered Semi-Markov
CRF](https://aclanthology.org/2023.findings-emnlp.17/).

### Context-free and contextual predictions contain different information

Context-Contrastive Partial Diacritization processes a word both with and
without sentence context, using disagreements to identify context-dependent
diacritics. Our task requires full diacritization, but the two-view mechanism
suggests a compliant learned expert: one shared encoder evaluates the sentence
and isolated word, and a learned gate decides how much contextual evidence is
needed at each letter.

Source: [A Context-Contrastive Inference Approach to Partial
Diacritization](https://aclanthology.org/2024.arabicnlp-1.8/).

### Regularization and averaging are cheap compared with new ensembles

R-Drop makes two dropout-perturbed forward passes agree through symmetric KL
regularization. Stochastic Weight Averaging averages weights sampled along a
constant or cyclical learning-rate tail. Both target generalization without
changing inference architecture. Born-Again Networks show that a student with
the same architecture can improve by learning from a teacher distribution,
which motivates distilling the validated ensemble into one DualRoPE model.

Sources:

- [R-Drop](https://openreview.net/forum?id=bw5Arp3O3eY)
- [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407)
- [Born Again Neural Networks](https://arxiv.org/abs/1805.04770)

### Convolution remains a plausible local inductive bias

Conformer combines attention for global interactions with depthwise
convolution for local structure. DualRoPE removed the original CNN and still
won decisively, so a full Conformer rewrite is not justified. A controlled
local-stream-only convolution module is testable because it preserves the
successful global stream and decoder.

Source: [Conformer: Convolution-augmented Transformer for Speech
Recognition](https://arxiv.org/abs/2005.08100).

## Ranked implementation plan

### 1. Training-label quality audit — highest leverage, low compute

Build a train-only report that groups identical undiacritized sentences and
words, measures label-sequence disagreement, flags impossible or very rare
transitions, and ranks examples by disagreement and current ensemble loss.

Implementation contract:

1. Never use dev correctness to decide which training examples to delete.
2. Produce an immutable CSV with sentence ID, raw skeleton, competing targets,
   frequencies, model loss, and recommended action.
3. Retrain the unchanged DualRoPE-CRF-v7 on: original data; exact-consensus
   deduplication; and reviewed high-confidence corrections.
4. Accept only if dev accuracy, WER, and sentence accuracy improve without a
   Shadda or Tanween regression.

Why first: it requires no new model and directly addresses the strongest 2025
Arabic-diacritization evidence.

### 2. Context-conditioned low-rank BoundaryCRF — highest architectural value

Status: implemented as `DziriFormer-DualRoPE-ContextLowRankBoundaryCRF-v11`;
seed-42 training is pending. See
[`experiments/context_boundary_v11/README.md`](context_boundary_v11/README.md).

The full BoundaryCRF improved overall accuracy; the static rank-2 residual
improved OOV and exact-word counts but lost common seen-word letters. Replace
the binary choice with an end-to-end learned contextual strength:

```text
hidden state h_i
      │
      ▼
g_i = sigmoid(w^T h_i + b)
      │
      ▼
T_i = T_shared + g_i · U V^T
```

The boundary indicator remains an input to the gate, but it is not the sole
condition. Initialize `U`, `V`, and the gate contribution at zero so the model
starts exactly as CRF-v7. Use rank 2 first because the controlled v10 test
already measured it. There is no manually selected interpolation multiplier.

Acceptance gate against CRF-v7 seed 42:

- neural correct letters greater than 14,831;
- OOV correct letters greater than 2,745;
- exact words greater than 3,039;
- no Shadda drop beyond 0.001;
- reproduce with seeds 43/44 only after seed 42 passes.

### 3. SWA tail on CRF-v7 and BoundaryCRF-v8 — low compute

Continue each selected checkpoint for 8–12 epochs with a fixed small learning
rate, save one checkpoint per epoch, and average weights rather than
probabilities. Evaluate every prefix average without tuning on Kaggle.

This is different from the failed best+last probability average: SWA averages
multiple weights sampled from a deliberate constant-LR tail.

Acceptance gate: at least +10 dev letters over the matching checkpoint and no
word/sentence-accuracy regression.

### 4. R-Drop CRF marginal consistency — medium compute

For each batch, run the same model twice with independent dropout masks. Keep
the mean of the two exact CRF NLL terms and add symmetric KL between
forward-backward marginals at scored positions. Inference stays unchanged.

Predeclare a small coefficient grid on a train-only calibration split, then
run one locked configuration on released dev. This avoids selecting a loss
weight on the official dev set. Expect roughly two times training compute.

### 5. Filtered word-lattice decoder — medium/high risk, high novelty

Exploit known spaces to form words. For every word, retain only the top-K
label sequences produced by beam search over CRF emissions, score the whole
candidate with a word-composition network, and run sentence-level dynamic
programming over candidates. This makes the segment set linear in the number
of words times K instead of enumerating every span.

Primary objective: improve exact word and sentence accuracy without
sacrificing official letter accuracy. Start with `K=4` and `K=8` as declared
complexity ablations, not dev-tuned hidden choices.

### 6. Context-contrastive shared encoder — medium/high compute

Evaluate each training word in two views:

1. full sentence context;
2. isolated word with boundary tokens.

Use the same encoder weights for both views. A learned per-letter gate fuses
the two distributions, and an auxiliary disagreement target marks positions
where the gold label cannot be predicted reliably from the isolated word.
This provides a neural, contextual replacement for static lexical fallback.

### 7. Local-stream Conformer module — high compute

Keep global RoPE, cross-attention, fusion, refinement, and CRF unchanged.
Insert a depthwise convolution module only inside each local block. Compare
against DualRoPE-CRF-v7 at matched parameter count and optimizer updates.

### 8. Ensemble-to-student distillation — high compute, deployment value

Use the four-group v7 ensemble as a frozen teacher. Generate soft
distributions for train sentences without adding new text. Train a fresh
DualRoPE-CRF student with hard labels plus an annealed teacher term that
reaches zero late in training. The final model remains a single Track 4
network.

## Deferred or rejected directions

| Direction | Decision | Evidence |
|---|---|---|
| Increase width to 11M | Reject | Worse than the 5.09M control; capacity was not the bottleneck. |
| Static lexical multiplier | Reject | Unjustified and can override contextual predictions. |
| Another dev-fitted logistic gate | Reject | Cross-fitted gate and v9 stacking both failed. |
| Explicit word positions alone | Reject | Lost 9 neural letters and 12 exact words. |
| Factorized CRF emissions alone | Reject | Improved diagnostics but lost 36 neural letters. |
| Static rank-2 boundary residual | Reject as final; retain insight | Gained OOV/exact words but lost 16 total letters. |
| Full external morphological analyzer | Outside primary Track 4 plan | Requires an external model/resource. |
| AraBERT/CAMeLBERT | Track 3, not Track 4 | Pretrained Arabic Transformer. |
| ByT5/CANINE | Track 2, not Track 4 | Pretrained/tokenizer-free LLM family. |
| BiLSTM SOTA reproduction | Track 1 | Valuable paper reference, invalid as the main Track 4 submission. |
| More train+dev refits | Competition-only | No unbiased dev comparison remains after refitting. |

## Recommended execution order

```text
train-only label-quality audit
        │
        ├─ clean-data CRF-v7 control
        │
        ▼
context-conditioned low-rank BoundaryCRF seed 42
        │
        ├─ pass ──> seeds 43/44 later
        │
        ▼
SWA tail on CRF-v7 and BoundaryCRF-v8
        │
        ▼
R-Drop CRF seed 42
        │
        ▼
filtered word-lattice decoder
        │
        ▼
context-contrastive shared encoder
        │
        ▼
local-stream Conformer / ensemble distillation
```

The immediate implementation recommendation is the train-only label-quality
audit followed by the context-conditioned low-rank BoundaryCRF. Together they
test the two strongest unresolved hypotheses: annotation noise and the need
for context-dependent transition specialization.
