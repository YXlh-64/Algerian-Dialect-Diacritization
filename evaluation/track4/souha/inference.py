"""Test-set loading and ensemble prediction (notebook §13, §14, §16).

`ensemble_predict` returns labels only; `predict_with_confidence` also returns
the per-position probability of the chosen label, which is what the V2 lexical
fallback gates on.

Signature change from the notebook: both took the whole `Cfg` to read
`cfg.device`; here the device is passed directly.
"""

import re

import torch
import torch.nn.functional as F

from models.track4.souha.crf import is_intra_mask
from utils.track4.souha.data import collate
from utils.track4.souha.features import featurize, word_ids


def load_test_set(paths, data):
    """Read the raw test file exactly as make_submission.py sees it (§14).

    Whitespace is normalised the same way, because the submission Id is
    `{sent_id}_{index}` with `index` counted over the whitespace-normalised
    line. Labels are filled with zeros: the test set is unlabelled, and they
    exist only so `collate` has a tensor to build.
    """
    sent_ids = [l.strip() for l in open(paths.raw_test_ids, encoding="utf-8")]
    raw_lines = [re.sub(r"\s+", " ", l.strip())
                 for l in open(paths.raw_test, encoding="utf-8")]
    assert len(sent_ids) == len(raw_lines), "ids / sentences length mismatch"

    test_enc = []
    for line in raw_lines:
        chars = list(line)
        test_enc.append(dict(ids=[data.vocab.get(c, data.unk) for c in chars],
                             labels=[0] * len(chars), feats=featurize(chars),
                             wid=word_ids(chars), chars=chars))
    return sent_ids, test_enc


@torch.no_grad()
def ensemble_predict(models, data, recs, device, batch_size=64):
    "Average per-model log-probabilities, then decode with the first model's CRF."
    for m in models: m.eval()
    preds = []
    for i in range(0, len(recs), batch_size):
        b = recs[i:i + batch_size]
        ids, feats, lab, mask, wid = collate(b, unk=data.unk)
        ids, feats, mask, wid = (ids.to(device), feats.to(device),
                                 mask.to(device), wid.to(device))
        acc = None
        for m in models:
            em, _ = m.emissions(ids, feats, mask, wid)
            em = F.log_softmax(em, dim=-1)
            acc = em if acc is None else acc + em
        acc = acc / len(models)
        out = (models[0].crf.decode(acc, mask, is_intra_mask(wid))
               if models[0].crf is not None else acc.argmax(-1)).cpu()
        for j, r in enumerate(b):
            preds.append(out[j, :len(r["ids"])].tolist())
    return preds


@torch.no_grad()
def predict_with_confidence(models, data, recs, device, batch_size=64):
    "Ensemble-averaged softmax probabilities -> (labels, per-position confidence)."
    for m in models: m.eval()
    out_labels, out_conf = [], []
    for i in range(0, len(recs), batch_size):
        b = recs[i:i + batch_size]
        ids, feats, lab, mask, wid = collate(b, unk=data.unk)
        ids, feats, mask, wid = (ids.to(device), feats.to(device),
                                 mask.to(device), wid.to(device))
        acc = None
        for m in models:
            em, _ = m.emissions(ids, feats, mask, wid)
            p = F.softmax(em, dim=-1)
            acc = p if acc is None else acc + p
        probs = acc / len(models)
        labels = (models[0].crf.decode(torch.log(probs.clamp_min(1e-9)), mask, is_intra_mask(wid))
                  if models[0].crf is not None else probs.argmax(-1))
        conf = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        for j, r in enumerate(b):
            n = len(r["ids"])
            out_labels.append(labels[j, :n].cpu().tolist())
            out_conf.append(conf[j, :n].cpu().tolist())
    return out_labels, out_conf
