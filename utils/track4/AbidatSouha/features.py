"""Morphological feature streams (notebook §2).

Six cheap linguistic features that encode facts already measured in the dataset,
so the encoder does not have to spend capacity rediscovering them:

- **alif is bare 99.2% of the time** -> `mater lectionis` flag
- **word-final is 52% sukoon / 46% bare** (the dialect has no case system)
  -> `position-in-word`, `distance-from-end`
- **`al-` + sun letter carries shadda 84.0% of the time vs 0.1% after a moon
  letter** -> `sun` stream
"""

import collections

from utils.track4.AbidatSouha.constants import MATER, SUN


def word_ids(chars):
    "word index per position; spaces get -1"
    out, w = [], 0
    for c in chars:
        if c == " ":
            out.append(-1); w += 1
        else:
            out.append(w)
    return out


def featurize(chars):
    n = len(chars)
    wid = word_ids(chars)
    spans = collections.defaultdict(list)
    for i, w in enumerate(wid):
        if w >= 0:
            spans[w].append(i)
    pos_in_word = [4] * n   # 0 init | 1 med | 2 fin | 3 iso | 4 space
    dist_start  = [4] * n   # 0..3, 4 = space
    dist_end    = [4] * n
    wlen        = [6] * n   # 0..5, 6 = space
    mater       = [2] * n   # 0 no | 1 yes | 2 space
    sun         = [0] * n   # 0 n/a | 1 al-+sun | 2 al-+moon
    for w, idx in spans.items():
        L = len(idx)
        for k, i in enumerate(idx):
            pos_in_word[i] = 3 if L == 1 else (0 if k == 0 else (2 if k == L - 1 else 1))
            dist_start[i]  = min(k, 3)
            dist_end[i]    = min(L - 1 - k, 3)
            wlen[i]        = min(L - 1, 5)
            mater[i]       = 1 if chars[i] in MATER else 0
        if L >= 3 and chars[idx[0]] == "ا" and chars[idx[1]] == "ل":
            sun[idx[2]] = 1 if chars[idx[2]] in SUN else 2
    return pos_in_word, dist_start, dist_end, wlen, mater, sun
