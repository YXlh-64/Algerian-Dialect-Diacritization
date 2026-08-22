"""Evaluation metrics (notebook §8).

Six numbers, because a single aggregate figure is misleading on this dataset:

- `micro_f1`  -- the official leaderboard score, in [0, 1].
- `der_all` -- every position. **Inflated**: 17.2% of positions are spaces with
  a guaranteed label.
- `der_letters` -- Arabic letters only. **The honest headline figure**. The
  notebook selected checkpoints on this; the port selects on
  `train_cfg.select_metric` (default `macro_f1`).
- `der_diacritic` -- gold-diacritic-bearing positions only. Excludes the
  near-deterministic alif / waw / ya cases.
- `wer` -- a word is correct only if *every* character in it is correct.
- `wer_seen_unambig / wer_seen_ambig / wer_oov` -- 49.0% / 36.3% / 14.7% of dev
  tokens. This split tells you whether a change improved memorisation,
  disambiguation, or generalisation.
- `macro_f1` over the nine live classes.

One signature change from the notebook: `evaluate` took the whole `Cfg` and read
`cfg.device` off it; here the device is passed directly.
"""

import collections

import numpy as np
import torch

from models.track4.souha.crf import is_intra_mask
from utils.track4.souha.constants import LIVE, NUM_CLASSES
from utils.track4.souha.data import collate


# Which keys of evaluate()'s result improve when they go *up*. Everything else
# it returns is an error rate, so lower is better. Used by train_model to decide
# the direction of checkpoint selection and early stopping.
HIGHER_IS_BETTER = {"micro_f1", "macro_f1"}


@torch.no_grad()
def evaluate(model, data, recs, device, batch_size=64):
    model.eval()
    dv = device
    gold_all, pred_all, ch_all = [], [], []
    for i in range(0, len(recs), batch_size):
        b = recs[i:i + batch_size]
        ids, feats, lab, mask, wid = collate(b, unk=data.unk)
        ids, feats, mask, wid = ids.to(dv), feats.to(dv), mask.to(dv), wid.to(dv)
        em, _ = model.emissions(ids, feats, mask, wid)
        pred = (model.crf.decode(em, mask, is_intra_mask(wid))
                if model.crf is not None else em.argmax(-1)).cpu()
        for j, r in enumerate(b):
            n = len(r["ids"])
            gold_all.append(r["labels"]); pred_all.append(pred[j, :n].tolist())
            ch_all.append(r["chars"])

    tot = cor = ltot = lcor = dtot = dcor = 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for g, p, ch in zip(gold_all, pred_all, ch_all):
        for gi, pi, c in zip(g, p, ch):
            tot += 1; cor += (gi == pi)
            if c != " ":
                ltot += 1; lcor += (gi == pi); cm[gi, pi] += 1
                if gi != 0:
                    dtot += 1; dcor += (gi == pi)

    buckets = collections.defaultdict(lambda: [0, 0])
    wtot = wcor = 0
    for g, p, ch in zip(gold_all, pred_all, ch_all):
        words, cg, cp, cc = [], [], [], []
        for gi, pi, c in zip(g, p, ch):
            if c == " ":
                if cc: words.append((cc, cg, cp)); cg, cp, cc = [], [], []
            else:
                cc.append(c); cg.append(gi); cp.append(pi)
        if cc: words.append((cc, cg, cp))
        for wc, wg, wp in words:
            surf, ok = "".join(wc), int(wg == wp)
            wtot += 1; wcor += ok
            k = ("oov" if surf not in data.lex
                 else "seen_ambig" if surf in data.ambiguous else "seen_unambig")
            buckets[k][0] += 1; buckets[k][1] += ok

    f1s = []
    for c in LIVE:
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        if tp + fn == 0: continue
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1s.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))

    B = lambda k: 100 * (1 - buckets[k][1] / max(buckets[k][0], 1))
    # micro_f1 on the official [0, 1] scale (1.0 = perfect), computed exactly as
    # stated in the competition metric: TP/FP/FN pooled over all 16 classes
    # across every scored (letter) position, before precision/recall/F1.
    # For a single-label multi-class problem this reduces to plain accuracy
    # over scored positions, which is what we compute directly here.
    return dict(micro_f1=lcor / ltot,                 # == official leaderboard score, in [0, 1]
                der_all=100 * (1 - cor / tot),
                der_letters=100 * (1 - lcor / ltot),
                der_diacritic=100 * (1 - dcor / dtot),
                wer=100 * (1 - wcor / wtot),
                wer_seen_unambig=B("seen_unambig"),
                wer_seen_ambig=B("seen_ambig"),
                wer_oov=B("oov"),
                macro_f1=100 * float(np.mean(f1s)),
                confusion=cm)


def fmt(m):
    return (f"LB(microF1) {m['micro_f1']:.4f} | DER_let {m['der_letters']:5.2f} | "
            f"DER_dia {m['der_diacritic']:5.2f} | WER {m['wer']:5.2f} "
            f"(un {m['wer_seen_unambig']:5.2f} amb {m['wer_seen_ambig']:5.2f} "
            f"oov {m['wer_oov']:5.2f}) | F1 {m['macro_f1']:5.2f}")


def letters_microf1(recs, pred_labels_list):
    """Leaderboard metric for a list of predictions (notebook §16).

    Same quantity as evaluate()'s `micro_f1`, but taken over already-computed
    label lists instead of running the model.
    """
    tot = cor = 0
    for r, pred in zip(recs, pred_labels_list):
        for gi, pi, c in zip(r["labels"], pred, r["chars"]):
            if c != " ":
                tot += 1; cor += (gi == pi)
    return cor / tot
