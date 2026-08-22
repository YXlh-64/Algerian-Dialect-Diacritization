# Linear-chain CRF

A hand-written conditional random field over the label sequence, with **two**
transition matrices selected per position: intra-word and inter-word.

Implementation: [`models/track4/AbidatSouha/crf.py`](../../../models/track4/AbidatSouha/crf.py).
Enabled by `ModelConfig.use_crf`; the split by `ModelConfig.split_crf`.

## Why a CRF at all

The output head produces independent per-position scores. Decoding them with
`argmax` assumes each character's diacritic is conditionally independent of its
neighbours given the encoder state. Arabic diacritization violates this
constantly: vowel sequences within a word follow templatic patterns, sukoon
cannot follow sukoon in most positions, and a geminated consonant constrains
what may come next.

A CRF makes the decoder score whole sequences:

```
score(y) = start[y₀] + Σₜ emission[t, yₜ] + Σₜ transition[yₜ₋₁, yₜ] + end[y_last]
```

Training maximises the log-likelihood of the gold sequence against all `16^T`
alternatives, computed exactly by the forward algorithm. Inference takes the
single highest-scoring sequence by Viterbi, not the position-wise argmax.

## Why the transitions are split

Word-final positions in this dialect emit essentially only
`{sukoon 52%, bare 46%, fatha 1.4%}` — the dialect has no case system, so the
final-vowel inventory collapses to almost nothing.

The consequence: the diacritic on the first letter of a word tells you very
little about the last letter of the previous word, and vice versa. Transitions
*across* a word boundary are close to independent, while transitions *inside* a
word are highly structured.

Forcing both through one 16×16 matrix averages two genuinely different
distributions and blurs both. So:

```python
def _trans(self, is_intra):
    return torch.where(is_intra[:, None, None], self.intra, self.inter)
```

`is_intra_mask(wid)` marks, for every position `t`, whether the transition
`t-1 → t` stays inside a single word:

```python
prev = torch.cat([wid[:, :1], wid[:, :-1]], dim=1)
return (wid == prev) & (wid >= 0)
```

Spaces carry `wid = -1` and padding `wid = -2`, so any transition touching a
space or a pad is inter-word by construction.

## Parameters

| Tensor | Shape | Count |
|---|---|---:|
| `intra` | (16,16) | 256 |
| `inter` | (16,16) | 256 |
| `start` | (16,) | 16 |
| `end` | (16,) | 16 |
| **Total** | | **544** |

0.03% of the model — and, as noted below, its dominant runtime cost.

## The three algorithms

**`_logZ` — forward algorithm.** Log-sum-exp recursion over all label sequences,
giving the partition function. Masked positions carry the accumulator forward
unchanged (`torch.where(mask[:, t], nxt, a)`), so padding contributes nothing.

**`_score` — gold sequence score.** The score of the reference labelling.
`tags.clamp(min=0)` neutralises the `-100` ignore-index before indexing, and
masked steps are zeroed by multiplication rather than skipped, keeping the
operation batched. The end transition indexes position `mask.sum(1) - 1`, the
true final position of each sequence.

**`nll` = `_logZ − _score`**, averaged over the batch. This is the training loss
whenever `use_crf` is on; `label_smoothing` is then unused, as it applies only to
the cross-entropy path.

**`decode` — Viterbi.** Max-product forward pass storing backpointers, then a
backward walk from `argmax(a + end)` at each sequence's true length.

## Correctness

[`tests/track4/AbidatSouha/test_crf.py`](../../../tests/track4/AbidatSouha/test_crf.py)
validates the implementation against **brute-force enumeration** of all `K^T`
label sequences, with K and T small enough for exhaustive enumeration:

| Test | What it checks |
|---|---|
| `test_crf_split_transitions` | `logZ` and Viterbi against enumeration, split transitions |
| `test_crf_shared_transitions` | the same with `split=False`, the ablation path |
| `test_crf_ignores_padding` | a padded batch scores its prefix identically to the same sequence unpadded |

The padding test matters more than it looks: every training batch is padded, so
that path executes constantly, and a masking error there would be invisible in
the loss curve while quietly corrupting short sequences.

## Runtime cost

`_logZ`, `_score` and `decode` are all sequential `for t in range(1, T)` Python
loops — a genuine dependency chain, since step `t` needs step `t-1`. With
sequences up to 274 characters that is hundreds of small kernel launches per
batch, and the loops run during training *and* during every dev evaluation.

This makes the CRF the dominant wall-clock cost of the model despite being
0.03% of its parameters, and it is the main reason a 40-epoch run takes
roughly 77 minutes on CPU. It also benefits far less from GPU acceleration than
the encoder does: the matmuls parallelise, the recursion does not.

## Ensembling interacts cleanly

`ensemble_predict` averages per-model `log_softmax` emissions before decoding.
Because `log_softmax` subtracts a per-position constant, and every candidate path
passes through exactly one label per position, the subtraction shifts every path
score by the same total — the Viterbi argmax is unchanged. Averaging in
log-probability space is therefore safe, and a single-model "ensemble" reproduces
the plain decode exactly.
