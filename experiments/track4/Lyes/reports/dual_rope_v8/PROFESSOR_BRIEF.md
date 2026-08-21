# Professor Brief — BoundaryCRF-v8

## What changed

The encoder, data, seed, optimizer, OneCycle schedule, batch size, epoch
budget, direct 16-class emission head, and CRF objective are unchanged from
DualRoPE-CRF-v7. The only model change is a second 16×16 CRF transition
matrix:

- the original matrix handles consecutive scored letters within a word;
- the new matrix handles the transition into a scored letter immediately
  following a space;
- sentence starts still use the separate learned start vector.

This adds 256 parameters: 9,890,096 → 9,890,352.

## Why

A single CRF transition table conflates two different contexts: orthographic
transitions inside a word and label transitions across a word boundary. The
boundary-conditioned version tests whether separating these statistics helps
without modifying the Transformer or adding external linguistic resources.

## Result

| Controlled model | Correct / 15,897 | Micro-F1 |
|---|---:|---:|
| Direct CE-v6 | 14,764 | 0.9287286909 |
| Standard CRF-v7 | 14,816 | 0.9319997484 |
| **BoundaryCRF-v8** | **14,837** | **0.9333207523** |

Boundary conditioning adds 21 correct letters over standard CRF and 73 over
direct CE at seed 42. This supports the architectural hypothesis.

In the existing equal four-group ensemble, replacing only CRF-v7 with
BoundaryCRF-v8 changes the final V2 result from 14,977 to 14,978 correct
letters. This is accepted by the predeclared gate but should be described as a
small candidate gain, not robust evidence, until other seeds are run.

## Lexical-gate follow-up

We also replaced the fixed V2 rules with a logistic neural-versus-lexical
switch:

- five balanced sentence-disjoint folds;
- training only on disagreements where exactly one expert is correct;
- neural confidence, margin, entropy, architecture disagreement, lexical
  confidence, word frequency, character position, and word length;
- deterministic standardized LBFGS;
- fixed 0.5 decision threshold.

Cross-fitted performance was 14,973, five letters below fixed V2. The learned
gate is rejected. Only 331 valid full-dev training disagreements existed, so
the eight correlated features appear data-limited. We did not proceed to the
dependent OOV affix gate.

## Focused feedback questions

1. Is the CE → standard CRF → boundary-conditioned CRF sequence sufficiently
   controlled for the decoder ablation section?
2. Should the +1 ensemble result remain only a candidate until BoundaryCRF
   seeds 43/44 confirm the seed-42 gain?
3. For future lexical arbitration, is it preferable to keep the interpretable
   V2 fallback, or revisit a regularized learned gate only after collecting
   more disagreement examples?
