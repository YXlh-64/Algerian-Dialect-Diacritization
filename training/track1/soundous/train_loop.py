"""Generic training loop: AdamW + cosine-annealing-warm-restarts + grad clipping + early stopping
on dev DER. 
A custom `compute_batch_loss(model, batch)` callback lets an experiment swap in a different loss
(e.g. focal-reweighted, consistency-regularized) without duplicating the rest of the loop. A custom
`on_epoch_start(epoch)` callback lets curriculum learning update its sampler between epochs.
"""
import json
import os
from dataclasses import dataclass, field

import torch

from utils.track1.soundous.seed_utils import set_seed, SEED
from evaluation.track1.soundous.metrics import evaluate_predictions


@dataclass
class TrainConfig:
    arch_name: str
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    label_smoothing: float = 0.05
    patience: int = 6
    t0: int = 8
    t_mult: int = 2
    model_kwargs: dict = field(default_factory=dict)


def run_training(model, cfg: TrainConfig, train_loader, dev_loader, out_dir, device,
                  compute_batch_loss=None, on_epoch_start=None):
    set_seed(SEED)
    os.makedirs(out_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.t0, T_mult=cfg.t_mult)

    history = {"train_loss": [], "dev_loss": [], "dev_metrics": []}
    best_der, best_state, epochs_no_improve = float("inf"), None, 0

    for epoch in range(1, cfg.epochs + 1):
        if on_epoch_start is not None:
            on_epoch_start(epoch)

        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            char_ids = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]

            optimizer.zero_grad()
            if compute_batch_loss is not None:
                loss = compute_batch_loss(model, batch, device)
            else:
                emissions = model._encode(char_ids, mask, lengths)
                loss = model.compute_loss(emissions, labels, mask, label_smoothing=cfg.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step(epoch - 1)
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        dev_loss, dev_preds, dev_golds, dev_chars = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                char_ids = batch["char_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["mask"].to(device)
                lengths = batch["lengths"]

                emissions = model._encode(char_ids, mask, lengths)
                loss = model.compute_loss(emissions, labels, mask, label_smoothing=0.0)
                dev_loss += loss.item()

                preds = model.crf.decode(emissions, mask) if model.use_crf else emissions.argmax(-1)
                for i, L in enumerate(lengths.tolist()):
                    p = preds[i][:L] if model.use_crf else preds[i, :L].tolist()
                    dev_preds.append(p)
                    dev_golds.append(labels[i, :L].tolist())
                    dev_chars.append(batch["chars"][i][:L])
        dev_loss /= max(len(dev_loader), 1)

        metrics = evaluate_predictions(dev_chars, dev_golds, dev_preds)
        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_metrics"].append(metrics)

        improved = metrics["DER"] < best_der
        if improved:
            best_der = metrics["DER"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"[{cfg.arch_name}] epoch {epoch:02d} | train_loss {train_loss:.4f} | "
              f"dev_loss {dev_loss:.4f} | DER {metrics['DER']:.4f} | WER {metrics['WER']:.4f} | "
              f"Acc {metrics['Accuracy']:.4f} | {'*' if improved else ''}")

        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (no dev-DER improvement for {cfg.patience} epochs).")
            break

    model.load_state_dict(best_state)
    torch.save({
        "state_dict": best_state,
        "arch_name": cfg.arch_name,
        "model_class": type(model).__name__,   # lets evaluate_all_experiments.py reconstruct the right class
        "model_kwargs": cfg.model_kwargs,       # exact kwargs used to build this model
        "cfg": cfg.__dict__,
    }, os.path.join(out_dir, "best_model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved best checkpoint (dev DER={best_der:.4f}) -> {out_dir}/best_model.pt")
    return model, history


def train_model(arch_name, model_kwargs, train_loader, dev_loader, out_dir, device,
                 vocab_size, num_classes, pad_idx, epochs=40, **cfg_overrides):
    """Thin convenience wrapper: builds a base DiacritizationTagger via models.track1.soundous.tagger,
    then calls run_training. Used for the 3 required base architectures (§10)."""
    from models.track1.soundous.tagger import build_model, ARCHITECTURES
    model = build_model(arch_name, vocab_size, num_classes, pad_idx, device, **model_kwargs)
    # store the FULL kwargs (incl. use_cnn/use_crf/vocab_size/etc) so the checkpoint alone is
    # enough to reconstruct this exact model later, e.g. in evaluate_all_experiments.py
    full_kwargs = dict(vocab_size=vocab_size, num_classes=num_classes, pad_idx=pad_idx,
                        **ARCHITECTURES[arch_name], **model_kwargs)
    cfg = TrainConfig(arch_name=arch_name, epochs=epochs, model_kwargs=full_kwargs, **cfg_overrides)
    return run_training(model, cfg, train_loader, dev_loader, out_dir, device)
