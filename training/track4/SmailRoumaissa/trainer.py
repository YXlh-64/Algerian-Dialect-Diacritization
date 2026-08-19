import json
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from configs.track4.SmailRoumaissa.model_config import ModelConfig
from configs.track4.SmailRoumaissa.training_config import TrainingConfig
from utils.track4.SmailRoumaissa.data import Vocab, DiacritizationDataset, collate
from utils.track4.SmailRoumaissa.constants import NUM_CLASSES
from models.track4.SmailRoumaissa.tagger import build_model
from evaluation.track4.SmailRoumaissa.lexical_prior import LexicalPrior, fuse_sentence
from evaluation.track4.SmailRoumaissa.calibration import fit_temperature
from evaluation.track4.SmailRoumaissa.metrics import MicroF1Accumulator


def cosine_warmup_lr(step, total_steps, warmup_frac, base_lr):
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc = MicroF1Accumulator()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        is_letter = batch["is_letter"].to(device)
        labels = batch["labels"]
        preds = model.decode(input_ids, attn_mask, is_letter).cpu()
        mask = is_letter.cpu() & (labels != -100)
        for b in range(preds.size(0)):
            m = mask[b]
            acc.update(preds[b][m].tolist(), labels[b][m].tolist())
    return acc.score


@torch.no_grad()
def evaluate_lexical_fused(model, loader, lexical, *, base_temperature=1.0, shadda_temperature=1.0,
                            entropy_threshold=1.0, gate_temperature=0.3, max_strength=2.0,
                            device="cpu"):
    model.eval()
    acc = MicroF1Accumulator()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        is_letter = batch["is_letter"].to(device)
        labels = batch["labels"]
        chars_batch = batch["chars"]

        log_probs = model.marginal_log_probs(input_ids, attn_mask,
                                              base_temperature=base_temperature,
                                              shadda_temperature=shadda_temperature)
        fused_log_probs = log_probs.clone()
        for b, chars in enumerate(chars_batch):
            L = len(chars)
            if L == 0:
                continue
            sent_log_probs = log_probs[b, 1:1 + L].cpu()
            fused = fuse_sentence(sent_log_probs, chars, lexical,
                                   entropy_threshold=entropy_threshold,
                                   gate_temperature=gate_temperature,
                                   max_strength=max_strength)
            fused_log_probs[b, 1:1 + L] = fused.to(device)

        preds = model.decode_from_log_probs(fused_log_probs, is_letter).cpu()
        mask = is_letter.cpu() & (labels != -100)
        for b in range(preds.size(0)):
            m = mask[b]
            acc.update(preds[b][m].tolist(), labels[b][m].tolist())
    return acc.score


def train_model(train_path, dev_path, vocab_path, out_dir, *,
                model_cfg: Optional[ModelConfig] = None,
                train_cfg: Optional[TrainingConfig] = None,
                entropy_threshold=1.0, gate_temperature=0.3, max_strength=2.0,
                device="cpu"):
    if model_cfg is None:
        model_cfg = ModelConfig()
    if train_cfg is None:
        train_cfg = TrainingConfig()

    torch.manual_seed(train_cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocab.load(vocab_path)
    train_ds = DiacritizationDataset(train_path, vocab, char_dropout_prob=train_cfg.char_dropout_prob)
    dev_ds = DiacritizationDataset(dev_path, vocab)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                               collate_fn=lambda b: collate(b, vocab.pad_id))
    dev_loader = DataLoader(dev_ds, batch_size=train_cfg.batch_size, shuffle=False,
                             collate_fn=lambda b: collate(b, vocab.pad_id))
    print(f"train examples: {len(train_ds)}  dev examples: {len(dev_ds)}  vocab size: {len(vocab)}")

    lexical_for_eval = LexicalPrior().fit(train_path)

    backbone_kwargs = dict(dim=model_cfg.dim, n_layers=model_cfg.n_layers,
                           n_heads=model_cfg.n_heads, ff_dim=model_cfg.ff_dim,
                           dropout=model_cfg.dropout)
    model = build_model(len(vocab), vocab.pad_id, **backbone_kwargs).to(device)

    # Standard AdamW split: decay 2D+ matrix weights, exclude biases/LayerNorm/embeddings
    # from weight decay so a larger `weight_decay` regularizes without shrinking norm
    # scale/shift or embedding rows toward zero.
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "norm" in name.lower() or "emb" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
    )

    total_steps = train_cfg.max_epochs * len(train_loader)
    step = 0
    best_f1, best_epoch, bad_epochs = -1.0, -1, 0

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            cur_lr = cosine_warmup_lr(step, total_steps, train_cfg.warmup_frac, train_cfg.lr)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            is_letter = batch["is_letter"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = model.loss(input_ids, attn_mask, is_letter, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            running_loss += loss.item()
            step += 1

        dev_f1 = evaluate(model, dev_loader, device)
        dev_fused_f1 = evaluate_lexical_fused(
            model, dev_loader, lexical_for_eval,
            entropy_threshold=entropy_threshold, gate_temperature=gate_temperature,
            max_strength=max_strength, device=device,
        )
        print(f"epoch {epoch:3d} | train_loss {running_loss/len(train_loader):.4f} | "
              f"dev_micro_f1 (neural) {dev_f1:.5f} | dev_micro_f1 (+lexical) {dev_fused_f1:.5f}")

        if dev_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = dev_f1, epoch, 0
            torch.save({"model_state": model.state_dict()}, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= train_cfg.patience:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch}, dev_micro_f1 {best_f1:.5f})")
                break

    print(f"best dev_micro_f1 = {best_f1:.5f} at epoch {best_epoch}")

    # Post-hoc temperature calibration on dev
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    all_base_logits, all_shadda_logits, all_labels = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            is_letter = batch["is_letter"].to(device)
            labels = batch["labels"].to(device)
            h = model.backbone(input_ids, attn_mask)
            base_logits, shadda_logits = model.crf_head.emit.raw_head_logits(h)
            mask = is_letter & (labels != -100)
            all_base_logits.append(base_logits[mask].cpu())
            all_shadda_logits.append(shadda_logits[mask].cpu())
            all_labels.append(labels[mask].cpu())
    all_base_logits = torch.cat(all_base_logits, 0)
    all_shadda_logits = torch.cat(all_shadda_logits, 0)
    all_labels = torch.cat(all_labels, 0)

    # Flat class id l (0..15) decomposes as base_idx = l // 2, shadda_idx = l % 2
    # -- this matches exactly how DecomposedHead.raw_logits recombines the two
    # sub-heads (base.unsqueeze(-1) + shadda.unsqueeze(-2)).reshape(...,16), so
    # it's the correct label split for calibrating each head independently.
    base_labels = all_labels // 2
    shadda_labels = all_labels % 2

    base_temperature = fit_temperature(all_base_logits, base_labels)
    shadda_temperature = fit_temperature(all_shadda_logits, shadda_labels)
    print(f"fitted base_temperature = {base_temperature:.4f}  shadda_temperature = {shadda_temperature:.4f}")

    config = {"model_kind": "crfcnn", "backbone_kwargs": backbone_kwargs,
              "base_temperature": base_temperature, "shadda_temperature": shadda_temperature,
              "best_dev_micro_f1": best_f1, "best_epoch": best_epoch,
              "vocab_path": vocab_path}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return model, vocab, config
