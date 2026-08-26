"""Full training loops for every  experiment. Each returns (model, history) or (models, history)
for the ensemble, and is meant to be followed by evaluation.track1.soundousndous.inference.generate_submission
in the calling script (see experiments/track1/soundousndous/run_*.py).
"""
import os

import torch
import torch.nn.functional as F

from utils.track1.soundous.seed_utils import set_seed, SEED
from utils.track1.soundous.data_utils import make_loader, make_curriculum_loader
from evaluation.track1.soundous.metrics import evaluate_predictions
from models.track1.soundous.tagger import build_model, DiacritizationTagger
from models.track1.soundous.experimental_taggers import MultiTaskDiacritizationTagger, AttnDiacritizationTagger
from training.track1.soundous.train_loop import TrainConfig, run_training


# ==================================================================================================
# Focal-style emission reweighting for class imbalance (rare diacritic classes).
# Hybrid loss: L = L_CRF + lambda * focal(emissions). True per-path focal reweighting of a CRF's
# NLL isn't well-defined, so the focal term supervises the emission layer directly, alongside the
# CRF term that supervises path-level consistency.
# ==================================================================================================
def focal_loss(emissions, labels, mask, gamma=2.0, class_weights=None):
    logp = F.log_softmax(emissions, dim=-1)
    p = logp.exp()
    logp_t = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    p_t = p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    focal_term = (1 - p_t).clamp(min=1e-6) ** gamma
    loss = -focal_term * logp_t
    if class_weights is not None:
        loss = loss * class_weights[labels]
    mask_f = mask.float()
    return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def compute_class_weights(train_rows, num_classes, label2idx, no_diac_idx, device):
    import numpy as np
    counts = np.zeros(num_classes)
    for row in train_rows:
        labels = row["labels"]
        if labels and isinstance(labels[0], str):
            labels = [label2idx.get(l, no_diac_idx) for l in labels]
        for l in labels:
            counts[l] += 1
    counts = np.maximum(counts, 1)
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_focal(vocab_size, num_classes, pad_idx, device, model_kwargs, train_rows,
                 label2idx, no_diac_idx, train_loader, dev_loader, out_dir,
                 epochs=40, gamma=2.0, lam=0.3, **cfg_overrides):
    class_weights = compute_class_weights(train_rows, num_classes, label2idx, no_diac_idx, device)
    model = build_model("bilstm_cnn_crf", vocab_size, num_classes, pad_idx, device, **model_kwargs)
    full_kwargs = dict(vocab_size=vocab_size, num_classes=num_classes, pad_idx=pad_idx,
                        use_cnn=True, use_crf=True, **model_kwargs)
    cfg = TrainConfig(arch_name="bilstm_cnn_crf_focal", epochs=epochs, model_kwargs=full_kwargs, **cfg_overrides)

    def compute_batch_loss(model, batch, device):
        char_ids = batch["char_ids"].to(device); labels = batch["labels"].to(device)
        mask = batch["mask"].to(device); lengths = batch["lengths"]
        emissions = model._encode(char_ids, mask, lengths)
        main_loss = model.compute_loss(emissions, labels, mask, label_smoothing=cfg.label_smoothing)
        aux_loss = focal_loss(emissions, labels, mask, gamma=gamma, class_weights=class_weights)
        return main_loss + lam * aux_loss

    return run_training(model, cfg, train_loader, dev_loader, out_dir, device, compute_batch_loss=compute_batch_loss)


# ==================================================================================================
# Auxiliary "has-diacritic" multi-task head.
# ==================================================================================================
def train_multitask(vocab_size, num_classes, pad_idx, device, model_kwargs, train_loader, dev_loader,
                     out_dir, epochs=40, aux_weight=0.3, no_diac_idx=0, **cfg_overrides):
    set_seed(SEED)
    os.makedirs(out_dir, exist_ok=True)
    full_kwargs = dict(vocab_size=vocab_size, num_classes=num_classes, pad_idx=pad_idx,
                        use_cnn=True, use_crf=True, aux_weight=aux_weight, **model_kwargs)
    model = MultiTaskDiacritizationTagger(**full_kwargs).to(device)
    # cfg.model_kwargs must be the FULL kwargs (incl. use_cnn/use_crf/aux_weight) so the checkpoint
    # metadata alone is enough to reconstruct this exact model class later (see evaluate_all_experiments.py)
    train_cfg = TrainConfig(arch_name="bilstm_cnn_crf_multitask", epochs=epochs,
                             model_kwargs=full_kwargs, **cfg_overrides)

    def compute_batch_loss(model, batch, device):
        char_ids = batch["char_ids"].to(device); labels = batch["labels"].to(device)
        mask = batch["mask"].to(device); lengths = batch["lengths"]
        total_loss, _, _ = model.forward_multitask(char_ids, mask, lengths, labels,
                                                     no_diac_idx=no_diac_idx,
                                                     label_smoothing=train_cfg.label_smoothing)
        return total_loss

    return run_training(model, train_cfg, train_loader, dev_loader, out_dir, device,
                         compute_batch_loss=compute_batch_loss)


# ==================================================================================================
# Self-attention between BiLSTM and CRF. AttnDiacritizationTagger already conforms to the
# standard interface (._encode overridden, .compute_loss/.decode inherited), so it's a drop-in for
# the plain run_training loop -- no custom loss callback needed.
# ==================================================================================================
def train_attention(vocab_size, num_classes, pad_idx, device, model_kwargs, train_loader, dev_loader,
                     out_dir, epochs=40, num_heads=4, **cfg_overrides):
    set_seed(SEED)
    full_kwargs = dict(vocab_size=vocab_size, num_classes=num_classes, pad_idx=pad_idx,
                        use_cnn=True, use_crf=True, num_heads=num_heads, **model_kwargs)
    model = AttnDiacritizationTagger(**full_kwargs).to(device)
    train_cfg = TrainConfig(arch_name="bilstm_cnn_crf_attention", epochs=epochs,
                             model_kwargs=full_kwargs, **cfg_overrides)
    return run_training(model, train_cfg, train_loader, dev_loader, out_dir, device)


# ==================================================================================================
# Consistency-regularized data augmentation (random char dropout + symmetric KL, UDA/R-Drop
# style) -- small-data (~6k sentences) generalization / robustness.
# ==================================================================================================
def augment_char_dropout(char_ids, mask, unk_idx, pad_idx, dropout_p=0.08):
    rand = torch.rand_like(char_ids, dtype=torch.float)
    replace_mask = (rand < dropout_p) & mask & (char_ids != pad_idx)
    augmented = char_ids.clone()
    augmented[replace_mask] = unk_idx
    return augmented


def train_consistency(vocab_size, num_classes, pad_idx, unk_idx, device, model_kwargs,
                       train_loader, dev_loader, out_dir, epochs=40, kl_weight=1.0,
                       dropout_p=0.08, **cfg_overrides):
    model = build_model("bilstm_cnn_crf", vocab_size, num_classes, pad_idx, device, **model_kwargs)
    full_kwargs = dict(vocab_size=vocab_size, num_classes=num_classes, pad_idx=pad_idx,
                        use_cnn=True, use_crf=True, **model_kwargs)
    cfg = TrainConfig(arch_name="bilstm_cnn_crf_consistency", epochs=epochs,
                       model_kwargs=full_kwargs, **cfg_overrides)

    def compute_batch_loss(model, batch, device):
        char_ids = batch["char_ids"].to(device); labels = batch["labels"].to(device)
        mask = batch["mask"].to(device); lengths = batch["lengths"]

        emissions_clean = model._encode(char_ids, mask, lengths)
        char_ids_aug = augment_char_dropout(char_ids, mask, unk_idx, pad_idx, dropout_p=dropout_p)
        emissions_aug = model._encode(char_ids_aug, mask, lengths)

        sup_loss = model.compute_loss(emissions_clean, labels, mask, label_smoothing=cfg.label_smoothing)

        logp_clean = F.log_softmax(emissions_clean, dim=-1)
        logp_aug = F.log_softmax(emissions_aug, dim=-1)
        kl = 0.5 * (F.kl_div(logp_aug, logp_clean.exp(), reduction="none").sum(-1)
                    + F.kl_div(logp_clean, logp_aug.exp(), reduction="none").sum(-1))
        mask_f = mask.float()
        kl = (kl * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        return sup_loss + kl_weight * kl

    return run_training(model, cfg, train_loader, dev_loader, out_dir, device, compute_batch_loss=compute_batch_loss)


# ==================================================================================================
# Stochastic Weight Averaging -- average weight snapshots taken at each cosine-restart cycle's
# low-LR point, then evaluate the averaged model on dev (piggybacks on the §8 LR schedule).
# ==================================================================================================
def train_swa(vocab_size, num_classes, pad_idx, device, model_kwargs, train_loader, dev_loader,
              out_dir, epochs=40, swa_start_cycle=1, t0=8, t_mult=2, lr=1e-3, weight_decay=1e-5,
              grad_clip=5.0, label_smoothing=0.05, **_ignored_cfg_overrides):
    # **_ignored_cfg_overrides: SWA has no early-stopping/patience concept (it always runs the
    # full `epochs` schedule to collect cosine-restart snapshots), so any 'patience' key present
    # in configs/track1/soundousndous/swa.json (or passed by a caller reusing a shared config dict) is
    # accepted and silently ignored rather than raising a TypeError.
    set_seed(SEED)
    os.makedirs(out_dir, exist_ok=True)
    model = build_model("bilstm_cnn_crf", vocab_size, num_classes, pad_idx, device, **model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=t_mult)

    swa_state, swa_count, cycle_idx = None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            char_ids = batch["char_ids"].to(device); labels = batch["labels"].to(device)
            mask = batch["mask"].to(device); lengths = batch["lengths"]
            optimizer.zero_grad()
            emissions = model._encode(char_ids, mask, lengths)
            loss = model.compute_loss(emissions, labels, mask, label_smoothing=label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step(epoch - 1)

        if epoch % t0 == 0:
            cycle_idx += 1
            if cycle_idx >= swa_start_cycle:
                sd = model.state_dict()
                if swa_state is None:
                    swa_state = {k: v.clone().float() for k, v in sd.items()}
                else:
                    for k in swa_state:
                        swa_state[k] += sd[k].float()
                swa_count += 1
                print(f"  [SWA] snapshot at epoch {epoch} (cycle {cycle_idx}), total={swa_count}")

    if swa_count > 0:
        for k in swa_state:
            swa_state[k] /= swa_count
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in swa_state.items()})

    # evaluate the averaged model on dev (SWA's own loop doesn't get this "for free" like run_training does)
    model.eval()
    dev_preds, dev_golds, dev_chars = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            char_ids = batch["char_ids"].to(device); labels = batch["labels"].to(device)
            mask = batch["mask"].to(device); lengths = batch["lengths"]
            emissions = model._encode(char_ids, mask, lengths)
            preds = model.crf.decode(emissions, mask) if model.use_crf else emissions.argmax(-1)
            for i, L in enumerate(lengths.tolist()):
                p = preds[i][:L] if model.use_crf else preds[i, :L].tolist()
                dev_preds.append(p); dev_golds.append(labels[i, :L].tolist())
                dev_chars.append(batch["chars"][i][:L])
    metrics = evaluate_predictions(dev_chars, dev_golds, dev_preds)
    history = {"dev_metrics": [metrics], "swa_snapshots": swa_count}
    print(f"[SWA] final dev DER={metrics['DER']:.4f} WER={metrics['WER']:.4f} (averaged {swa_count} snapshots)")

    full_kwargs = dict(vocab_size=vocab_size, num_classes=num_classes, pad_idx=pad_idx,
                        use_cnn=True, use_crf=True, **model_kwargs)
    torch.save({
        "state_dict": model.state_dict(),
        "arch_name": "bilstm_cnn_crf_swa",
        "model_class": "DiacritizationTagger",
        "model_kwargs": full_kwargs,
    }, os.path.join(out_dir, "best_model.pt"))
    import json
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return model, history


# ==================================================================================================
# Multi-seed ensembling. Trains N independent seeds (each via run_training, so each gets its
# own checkpoint/history saved normally); combining/decoding happens at inference time -- see
# evaluation.track1.soundousndous.inference.generate_submission_ensemble.
# ==================================================================================================
def train_multi_seed(vocab_size, num_classes, pad_idx, device, model_kwargs, seeds,
                      train_rows, char2idx, label2idx, no_diac_idx, dev_loader, out_dir,
                      batch_size=64, epochs=40, **cfg_overrides):
    models, histories = [], []
    for seed in seeds:
        set_seed(seed)
        seed_train_loader = make_loader(train_rows, char2idx, label2idx, pad_idx,
                                         char2idx["<UNK>"], no_diac_idx, batch_size, shuffle=True)
        seed_out_dir = os.path.join(out_dir, f"seed_{seed}")
        model = build_model("bilstm_cnn_crf", vocab_size, num_classes, pad_idx, device, **model_kwargs)
        full_kwargs = dict(vocab_size=vocab_size, num_classes=num_classes, pad_idx=pad_idx,
                            use_cnn=True, use_crf=True, **model_kwargs)
        cfg = TrainConfig(arch_name=f"bilstm_cnn_crf_seed{seed}", epochs=epochs,
                           model_kwargs=full_kwargs, **cfg_overrides)
        m, h = run_training(model, cfg, seed_train_loader, dev_loader, seed_out_dir, device)
        models.append(m); histories.append(h)
    return models, histories


# ==================================================================================================
# Length-based curriculum learning -- shortest-first for the first few epochs.
# ==================================================================================================
def train_curriculum(vocab_size, num_classes, pad_idx, unk_idx, device, model_kwargs,
                      train_rows, char2idx, label2idx, no_diac_idx, dev_loader, out_dir,
                      batch_size=64, epochs=40, curriculum_epochs=5, **cfg_overrides):
    model = build_model("bilstm_cnn_crf", vocab_size, num_classes, pad_idx, device, **model_kwargs)
    full_kwargs = dict(vocab_size=vocab_size, num_classes=num_classes, pad_idx=pad_idx,
                        use_cnn=True, use_crf=True, **model_kwargs)
    cfg = TrainConfig(arch_name="bilstm_cnn_crf_curriculum", epochs=epochs,
                       model_kwargs=full_kwargs, **cfg_overrides)

    train_loader, sampler = make_curriculum_loader(train_rows, char2idx, label2idx, pad_idx, unk_idx,
                                                     no_diac_idx, batch_size, curriculum_epochs)

    def on_epoch_start(epoch):
        sampler.set_epoch(epoch - 1)  # 0-indexed inside the sampler

    return run_training(model, cfg, train_loader, dev_loader, out_dir, device, on_epoch_start=on_epoch_start)
