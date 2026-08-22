# Output head

Four components turn the encoder's `(B,T,192)` hidden states into `(B,T,16)`
emission scores. All four exist for the same reason: **133,032 labelled
positions is not many, and the 16 classes are extremely unbalanced.**

Implementation: `DiacModel.emissions` in
[`models/track4/AbidatSouha/tagger.py`](../../../models/track4/AbidatSouha/tagger.py).

## The class imbalance problem

Label frequencies over the 160,583 training positions:

| Label | Class | Count | Share |
|---:|---|---:|---:|
| 0 | none | 68,371 | 42.58% |
| 1 | fatha | 29,686 | 18.49% |
| 2 | fathatan | 27 | 0.02% |
| 3 | damma | 7,674 | 4.78% |
| 4 | dammatan | 1 | 0.00% |
| 5 | kasra | 10,521 | 6.55% |
| 6 | kasratan | 0 | 0.00% |
| 7 | sukoon | 38,988 | 24.28% |
| 8 | shadda | 18 | 0.01% |
| 9 | shadda+fatha | 3,311 | 2.06% |
| 10 | shadda+fathatan | 0 | 0.00% |
| 11 | shadda+damma | 350 | 0.22% |
| 12 | shadda+dammatan | 1 | 0.00% |
| 13 | shadda+kasra | 1,108 | 0.69% |
| 14 | shadda+kasratan | 0 | 0.00% |
| 15 | shadda+sukoon | 527 | 0.33% |

Four classes span five orders of magnitude, three never occur at all, and the
seven rarest together account for 47 positions — 0.03%. A flat
`Linear(192 → 16)` treats these as 16 unrelated categories and has no way to
learn `shadda+damma` from 350 examples.

Every excluded class is a tanween form or bare shadda. Tanween marks Modern
Standard Arabic case endings, and this dialect has no case system — the absence
is linguistic, not a sampling artefact.

## 1. Factorized head

The label scheme is exactly separable:

```
label = 8 * shadda + base
```

So instead of one 16-way decision, the head makes two:

```python
ls = F.log_softmax(self.h_shadda(h), -1)   # (B,T,2)   gemination
lb = F.log_softmax(self.h_base(h),   -1)   # (B,T,8)   short vowel
em = ls.unsqueeze(-1) + lb.unsqueeze(-2)   # (B,T,2,8) log-space outer sum
```

`Linear(192→2)` = 386 parameters, `Linear(192→8)` = 1,544.

**Why this helps.** `shadda+kasra` (label 13) has 1,108 examples of its own.
Under the factorization its *vowel* decision is trained by every position whose
base is kasra — 11,629 of them, labels 5 and 13 together — and its *gemination*
decision by all 5,315 shadda-bearing positions, labels 8–15 together.
Statistical strength is shared across the factorization rather than partitioned
by it: a class with 1,108 examples gets gradient from ten times that many.

## 2. Interaction table

A plain outer sum assumes shadda and base are conditionally independent given
`h`. They are not.

Over the 133,032 **letter** positions: shadda appears on 5,315 (4.0%), base
`none` on 40,838 (30.7%). Independence predicts about **1,632** positions
carrying shadda with no vowel. The actual count is **18**.

That is a factor of 90, and it is systematic — a geminated consonant in this
dialect essentially always carries a vowel. An additive head cannot represent
it, because no choice of the two marginals produces a joint that suppresses one
specific cell.

```python
em = em + self.inter_tab      # (2,8) broadcast over batch and time
```

A context-independent `2×8` table — **16 parameters** — restores full 16-way
expressivity while keeping the parameter sharing. It is initialised to zeros, so
training starts from the purely factorized model and learns the deviations.

## 3. Per-character logit prior

```python
em = em + self.prior[ids]     # (43,16) gathered per input character
```

Initialised from smoothed training log-frequencies
(`build_char_prior`, add-one smoothing), then trained as a normal parameter.
688 parameters.

**Why.** The identity of a character is enormously predictive on its own — alif
is bare 99.2% of the time, and every character has a skewed label distribution.
Without the prior, part of the encoder's capacity goes into re-deriving the
unigram distribution. With it, the encoder learns the *residual*: how context
shifts the label away from what the character alone would suggest.

It stays learnable rather than frozen so training can correct the initialisation
where context systematically overrides the unigram statistics.

## 4. Auxiliary diacritic head

```python
tgt  = (lab > 0)                              # is this position marked at all?
loss = main_loss + 0.3 * cross_entropy(self.aux(h), tgt)
```

A second `Linear(192→2)` — 386 parameters — trained on the binary question "does
this character bear any diacritic?", weighted 0.3.

**Why.** The binary question is far better conditioned than the 16-way one:
both of its classes are abundant, so its gradient is dense and stable from the
first step. It shapes the shared representation `h` early in training, when the
16-way head is still dominated by the majority classes. It is a training-time
regulariser only — `self.aux` is never consulted at inference and contributes
nothing to emissions.

## Cost

| Component | Parameters |
|---|---:|
| Base head | 1,544 |
| Character prior | 688 |
| Shadda head | 386 |
| Auxiliary head | 386 |
| Interaction table | 16 |
| **Total** | **3,020** |

0.16% of the model. A flat `Linear(192→16)` would cost 3,088 — the entire
factorized head with all its machinery is *smaller* than the naive alternative
it replaces.
