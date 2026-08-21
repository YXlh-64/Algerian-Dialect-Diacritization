# Existing-checkpoint weight probe results

This is an equal arithmetic average of the existing `best.pt` and `last.pt`
model weights. No continuation training occurred, so these results must not be
reported as stochastic weight averaging.

| System | Source V2 correct | Averaged neural correct | Averaged V2 correct | Gain | Regressions | Decision |
|---|---:|---:|---:|---:|---|---|
| CRF-v7 | 14,962 | 14,814 | 14,960 | -2 | None of the protected aggregate metrics | Reject |
| BoundaryCRF-v8 | 14,977 | 14,827 | 14,966 | -11 | Exact-word and Shadda accuracy | Reject |

The earlier **probability** average of CRF-v7 best/last reached 14,963 (+1).
The new **weight** average reaches 14,960 (-2). Neither is evidence for or
against a true SWA tail, because neither sampled multiple models under a
deliberate fixed learning-rate continuation.

No Kaggle submission should be generated from either checkpoint-weight probe.
