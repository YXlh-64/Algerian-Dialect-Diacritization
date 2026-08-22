"""Memorisation floor (notebook §8, "Memorisation floor").

Most-frequent-vocalisation lookup from the training lexicon: 73.93% word
accuracy (WER 26.07%) on dev. Every model has to beat this, and the interesting
question is *which bucket* the improvement comes from -- by construction the
lookup scores 100% WER on OOV words, so that bucket is where a character model
has to earn its keep.
"""

import collections


def lookup_baseline(data):
    tot = cor = 0
    per = collections.defaultdict(lambda: [0, 0])
    for r in data.dev:
        for wi, wo in zip(r["input"].split(), r["target"].split()):
            tot += 1
            ok = int(wi in data.lex and data.lex[wi].most_common(1)[0][0] == wo)
            cor += ok
            k = ("oov" if wi not in data.lex
                 else "seen_ambig" if wi in data.ambiguous else "seen_unambig")
            per[k][0] += 1; per[k][1] += ok
    print(f"lookup baseline: word acc {100*cor/tot:.2f}%  ->  WER {100*(1-cor/tot):.2f}%")
    for k, (n, c) in per.items():
        print(f"   {k:14s} n={n:5d}  WER {100*(1-c/n):6.2f}%")
    return 100 * (1 - cor / tot)
