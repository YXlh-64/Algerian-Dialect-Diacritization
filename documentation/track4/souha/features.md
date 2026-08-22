# Morphological input streams

Six integer features per character, each with its own embedding table, all
summed onto the character embedding before the encoder sees anything.

Implementation: [`utils/track4/souha/features.py`](../../../utils/track4/souha/features.py).

## Why hand-built features at all

The training set has 133,032 labelled letter positions. That is enough to learn
statistical regularities, but not enough to rediscover the morphology of Arabic
from scratch *and* learn the task on top of it. Each stream below encodes a fact
that is already measurable in the data, so encoder capacity is spent on what
remains uncertain rather than on re-deriving what is known.

They cost 5,376 parameters — 0.3% of the model. Even a small gain justifies
them; the risk is not size but redundancy with what attention would learn anyway.

## The six streams

| # | Stream | Size | Values |
|---|---|---:|---|
| 0 | `pos_in_word` | 5 | 0 initial · 1 medial · 2 final · 3 isolated · 4 space |
| 1 | `dist_start` | 5 | distance from word start, capped at 3 · 4 space |
| 2 | `dist_end` | 5 | distance from word end, capped at 3 · 4 space |
| 3 | `wlen` | 7 | word length − 1, capped at 5 · 6 space |
| 4 | `mater` | 3 | 0 no · 1 mater lectionis · 2 space |
| 5 | `sun` | 3 | 0 n/a · 1 `al-`+sun letter · 2 `al-`+moon letter |

Each size is exactly one more than the highest value the stream can emit, which
is what `nn.Embedding` needs as its table height. The sizes are declared in
`ModelConfig.feat_sizes` and must stay in step with `featurize()`.

## What each stream encodes

### Position streams (0–3)

Arabic diacritization is strongly positional. Word-final positions in this
dialect carry essentially only `{sukoon 52%, bare 46%, fatha 1.4%}` — the
dialect has no case system, so the rich final-vowel inventory of Modern Standard
Arabic simply does not occur. Telling the model *where in the word* a character
sits, and *how far from each end*, hands it that distribution directly.

`dist_start` and `dist_end` are capped at 3 because the informative signal is
"near the edge", not the exact index; capping keeps the tables small and stops
long words from generating sparse, rarely-updated embedding rows.

### `mater` (4) — matres lectionis

The letters `ا و ي ى` are *matres lectionis*: historically consonants, used in
writing to spell long vowels. A letter that already represents a long vowel does
not take a short-vowel mark on top of it. In this dataset **alif is bare 99.2%
of the time**.

This is close to a deterministic rule, and it covers a large fraction of
positions. Flagging it means the model never has to spend attention deciding
that an alif is probably unmarked.

### `sun` (5) — sun and moon letters

The definite article is `ال`. Before a **sun letter**, the `ل` assimilates and
the following consonant doubles — written as a shadda:

| Word | Reading | Third letter | `sun` value |
|---|---|---|---|
| الشمس | *ash-shams* ("the sun") | ش — sun letter | 1 |
| القمر | *al-qamar* ("the moon") | ق — moon letter | 2 |

In the training data that position carries shadda **84.0%** of the time after a
sun letter, and **0.1%** after a moon letter. That is nearly a decision
procedure, and it depends on membership in a fixed 14-letter set that the model
would otherwise have to infer from a handful of examples per letter.

The stream is set on exactly one position per word — the character immediately
after `ال` — and only when the word is at least three characters long.
Everything else is 0. Note that set membership and stream value are different
things: `س` in الشمس *is* a sun letter, but its stream value is 0 because it is
not the position the rule applies to.

## Worked example

`featurize(list("الشمس القمر"))`:

```
char    pos_in_word  dist_start  dist_end  wlen  mater  sun
ا                 0           0         3     4      1    0
ل                 1           1         3     4      0    0
ش                 1           2         2     4      0    1   ← al- + sun letter
م                 1           3         1     4      0    0
س                 2           3         0     4      0    0
SPACE             4           4         4     6      2    0
ا                 0           0         3     4      1    0
ل                 1           1         3     4      0    0
ق                 1           2         2     4      0    2   ← al- + moon letter
م                 1           3         1     4      0    0
ر                 2           3         0     4      0    0
```

## Word identity

`word_ids()` produces a parallel array assigning each character its word index,
with `-1` for spaces. It is not an embedded feature — it is consumed by two
other components:

- the **same-word attention bias** in [transformer.md](transformer.md), which
  needs to know which pairs of positions belong to the same word;
- **`is_intra_mask`** in [crf.md](crf.md), which selects the intra-word or
  inter-word transition matrix per position.

In `collate`, padded positions get `wid = -2` so they can never be considered
same-word as anything, including each other.
