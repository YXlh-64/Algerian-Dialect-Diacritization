"""Training loop (notebook §9).

Linear warmup for `warmup` optimiser steps, then cosine decay to zero. The dev
metric named by `train_cfg.select_metric` is both the early-stopping signal and
the checkpoint-selection metric; the best state is restored before returning, so
the returned model is not the last-epoch one.

The notebook selected on `der_letters`; the default here is `macro_f1`, so a
run will not reproduce the notebook's checkpoint choice unless you set
`select_metric="der_letters"`.

Signature change from the notebook: `train_model(cfg, data, tr_enc, dv_enc)`
read architecture, optimisation and device off one `Cfg`. Those are now three
arguments, matching training/track4/SmailRoumaissa/trainer.py.

Note the returned metrics are dev metrics, and dev is what was selected on --
they are validation numbers, not a held-out estimate.
"""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.track4.souha.model_config import ModelConfig
from configs.track4.souha.training_config import TrainingConfig
from evaluation.track4.souha.metrics import HIGHER_IS_BETTER, evaluate, fmt
from models.track4.souha.crf import is_intra_mask
from models.track4.souha.tagger import DiacModel
from utils.track4.souha.data import build_char_prior, collate
from utils.track4.souha.seed import set_seed


def train_model(data, tr_enc, dv_enc, model_cfg: ModelConfig,
                train_cfg: TrainingConfig, device: str, verbose: bool = True):
    set_seed(train_cfg.seed)
    prior = build_char_prior(data, tr_enc) if model_cfg.char_prior else None
    model = DiacModel(model_cfg, len(data.vocab), prior).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)
    total = math.ceil(len(tr_enc) / train_cfg.batch_size) * train_cfg.epochs

    def lr_scale(s):
        if s < train_cfg.warmup:
            return s / max(train_cfg.warmup, 1)
        q = (s - train_cfg.warmup) / max(total - train_cfg.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(q, 1.0)))

    higher = train_cfg.select_metric in HIGHER_IS_BETTER
    best, best_state, bad, step = (-1e9 if higher else 1e9), None, 0, 0
    t0 = time.time()
    for ep in range(train_cfg.epochs):
        model.train()
        idx = list(range(len(tr_enc))); random.shuffle(idx)
        tl = nb = 0
        for i in range(0, len(idx), train_cfg.batch_size):
            b = [tr_enc[j] for j in idx[i:i + train_cfg.batch_size]]
            ids, feats, lab, mask, wid = collate(b, train_cfg.char_dropout,
                                                 data.unk, train=True)
            ids, feats, lab = ids.to(device), feats.to(device), lab.to(device)
            mask, wid = mask.to(device), wid.to(device)
            em, h = model.emissions(ids, feats, mask, wid)
            if model.crf is not None:
                loss = model.crf.nll(em, lab, mask, is_intra_mask(wid))
            else:
                loss = F.cross_entropy(em.reshape(-1, model_cfg.num_classes),
                                       lab.reshape(-1),
                                       ignore_index=-100,
                                       label_smoothing=train_cfg.label_smoothing)
            if model.aux is not None:
                tgt = torch.where(lab >= 0, (lab > 0).long(), torch.full_like(lab, -100))
                loss = loss + train_cfg.aux_weight * F.cross_entropy(
                    model.aux(h).reshape(-1, 2), tgt.reshape(-1), ignore_index=-100)
            for g in opt.param_groups:
                g["lr"] = train_cfg.lr * lr_scale(step)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step(); step += 1; tl += float(loss); nb += 1

        m = evaluate(model, data, dv_enc, device)
        if verbose:
            print(f"ep{ep:02d} loss {tl/nb:7.3f} | {fmt(m)}")
        score = m[train_cfg.select_metric]
        improved = (score > best + 1e-4) if higher else (score < best - 1e-4)
        if improved:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= train_cfg.patience:
                if verbose: print(f"early stop at epoch {ep}")
                break
    if best_state:
        model.load_state_dict(best_state)
    m = evaluate(model, data, dv_enc, device)
    if verbose:
        print(f"BEST  {fmt(m)}  | {n_par:,} params | {time.time()-t0:.0f}s")
    return model, m, n_par
