# Shared model.py for track3/linear_head (6 of 6 experiments — auto-consolidated, identical after stripping header/noise). Excluded (differ for real reasons): none

# ## 6. Model — Pretrained Backbone + Classification Head

class Track3Diacritizer(nn.Module):
    '''Pretrained backbone -> per-character feature -> classification head.'''

    def __init__(self, backbone_name: str, char_vocab_size: int,
                 num_classes: int = 16, char_emb_dim: int = 32,
                 n_concat_layers: int = 4, head_hidden_dim: int = 512,
                 head_dropout: float = 0.15, use_deep_head: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.use_deep_head = use_deep_head
        self.n_concat_layers = n_concat_layers
        hidden = self.backbone.config.hidden_size

        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(head_dropout)

        backbone_feat_dim = hidden * n_concat_layers if use_deep_head else hidden
        fused_dim = backbone_feat_dim + char_emb_dim

        if use_deep_head:
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, head_hidden_dim), nn.GELU(), nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, head_hidden_dim), nn.GELU(), nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, input_ids, attention_mask, char_ids, token_idx_per_char):
        if self.use_deep_head:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
            # concat last N layers (embeddings + each transformer layer are in hidden_states)
            seq_out = torch.cat(out.hidden_states[-self.n_concat_layers:], dim=-1)  # (B, T, H*N)
        else:
            seq_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        Hc = seq_out.size(-1)
        idx = token_idx_per_char.unsqueeze(-1).expand(-1, -1, Hc)
        gathered = torch.gather(seq_out, 1, idx)                 # (B, C, Hc)
        char_emb = self.char_embedding(char_ids)                 # (B, C, E)
        fused = self.dropout(torch.cat([gathered, char_emb], dim=-1))
        return self.classifier(fused)                            # (B, C, num_classes)


# ## 8. Train `ACTIVE_MODEL`

tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME, use_fast=True)
aligner = CharAligner(tokenizer, CFG.max_subword_len)

train_ds = DiacritizationDataset(train_records, aligner, CHAR2ID)
dev_ds = DiacritizationDataset(dev_records, aligner, CHAR2ID)

_collate = lambda b: collate_fn(b, tokenizer.pad_token_id or 0)
train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, collate_fn=_collate)
dev_loader = DataLoader(dev_ds, batch_size=CFG.eval_batch_size, shuffle=False, collate_fn=_collate)
loss_fn = build_loss_fn(CFG, DEVICE)



def check_truncation(records: List[dict], aligner: "CharAligner", name: str = "split") -> Tuple[int, int]:
    '''Flags characters that fall past CFG.max_subword_len subword tokens.
    Those characters get token_idx_per_char == -1 from CharAligner, and
    collate_fn / _predict_chars silently remap that to token index 0
    ([CLS]) instead of erroring -- i.e. they would be classified from a
    meaningless vector with no visible failure. Run this once before you
    trust any score.'''
    n_trunc_sent = 0
    n_trunc_chars = 0
    max_len_seen = 0
    for r in records:
        chars = r["chars"]
        enc = aligner.encode(chars)
        dropped = sum(1 for t in enc["token_idx_per_char"] if t == -1)
        if dropped > 0:
            n_trunc_sent += 1
            n_trunc_chars += dropped
        max_len_seen = max(max_len_seen, len(enc["input_ids"]))
    print(f"[{name:9s}] longest sentence -> {max_len_seen} subword tokens "
          f"(cap={aligner.max_len}) | {n_trunc_sent}/{len(records)} sentences affected, "
          f"{n_trunc_chars} characters would silently misalign")
    return n_trunc_sent, n_trunc_chars

for _name, _recs in [("train", train_records), ("dev", dev_records), ("dev_test", dev_test_records)]:
    check_truncation(_recs, aligner, _name)

# KAGGLE_TEST was NOT length-filtered the way train/dev were during data prep,
# so it is the most likely place a sentence actually exceeds max_subword_len.
# Check it here too, before trusting any submission built from this model.
with open(PATHS.raw_test_txt, "r", encoding="utf-8") as f:
    _kaggle_test_sentences_check = [l.rstrip("\n") for l in f if l.strip()]
_n_sent_kt = _n_chars_kt = 0
for _s in _kaggle_test_sentences_check:
    _chars = list(clean_arabic_text(_s))
    _enc = aligner.encode(_chars)
    _d = sum(1 for t in _enc["token_idx_per_char"] if t == -1)
    if _d > 0:
        _n_sent_kt += 1
        _n_chars_kt += _d
print(f"[kaggle_test] {_n_sent_kt}/{len(_kaggle_test_sentences_check)} sentences affected, "
      f"{_n_chars_kt} characters would silently misalign at submission time")

if _n_chars_kt > 0:
    print("\n>>> ACTION NEEDED: raise CFG.max_subword_len (check "
          "backbone.config.max_position_embeddings for the ceiling) or "
          "chunk+stitch long sentences before trusting this run's submission.")



def micro_f1_from_counts(logits, labels) -> float:
    preds = logits.argmax(-1)
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == labels[mask]).float().mean().item()  # == micro-F1 for single-label multiclass


def run_train_epoch(model, loader, optimizer, scheduler, loss_fn, cfg, ckpt, epoch, global_step):
    model.train()
    epoch_loss, epoch_f1, n_batches = 0.0, 0.0, 0
    t0 = time.time()
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits1 = model(batch["input_ids"], batch["attention_mask"],
                         batch["char_ids"], batch["token_idx_per_char"])
        loss = loss_fn(logits1.reshape(-1, cfg.num_classes), batch["char_labels"].reshape(-1))

        if cfg.use_rdrop:
            logits2 = model(batch["input_ids"], batch["attention_mask"],
                             batch["char_ids"], batch["token_idx_per_char"])
            loss2 = loss_fn(logits2.reshape(-1, cfg.num_classes), batch["char_labels"].reshape(-1))
            mask = (batch["char_labels"] != IGNORE_INDEX)
            kl = rdrop_kl(logits1, logits2, mask)
            loss = (loss + loss2) / 2 + cfg.rdrop_alpha * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1
        epoch_loss += loss.item()
        epoch_f1 += micro_f1_from_counts(logits1.detach(), batch["char_labels"])
        n_batches += 1

        if global_step % cfg.checkpoint_every_steps == 0:
            ckpt.save(model, optimizer, scheduler, epoch, global_step, best_dev_score=-1.0)

        if (time.time() - t0) / 60 > cfg.max_train_minutes:
            print("Time budget reached mid-epoch; checkpointing and stopping.")
            ckpt.save(model, optimizer, scheduler, epoch, global_step, best_dev_score=-1.0)
            return epoch_loss / n_batches, epoch_f1 / n_batches, global_step, True

    return epoch_loss / n_batches, epoch_f1 / n_batches, global_step, False


@torch.no_grad()
def run_eval(model, loader, loss_fn, cfg):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"],
                        batch["char_ids"], batch["token_idx_per_char"])
        loss = loss_fn(logits.reshape(-1, cfg.num_classes), batch["char_labels"].reshape(-1))
        total_loss += loss.item(); n_batches += 1
        all_logits.append(logits.cpu()); all_labels.append(batch["char_labels"].cpu())
    return total_loss / n_batches, all_logits, all_labels



if CFG.k_folds == 1:
    model = Track3Diacritizer(
        BACKBONE_NAME, len(CHAR2ID), CFG.num_classes, CFG.char_emb_dim,
        n_concat_layers=CFG.n_concat_layers, head_hidden_dim=CFG.head_hidden_dim,
        head_dropout=CFG.head_dropout, use_deep_head=CFG.use_deep_head,
    ).to(DEVICE)
    optimizer = build_layerwise_optimizer(model, CFG)
    total_steps = len(train_loader) * CFG.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(CFG.warmup_ratio * total_steps), num_training_steps=total_steps)
    ckpt = CheckpointManager(PATHS.checkpoint_dir / RUN_ID)

    state = ckpt.load_latest(model, optimizer, scheduler)
    start_epoch, global_step, best_dev_score = state["epoch"], state["global_step"], state["best_dev_score"]
    print(f"Starting from epoch {start_epoch}, global_step {global_step}, best_dev_score {best_dev_score:.4f}")

    patience_counter = 0
    best_dev_loss = float("inf")
    stopped_for_time = False

    for epoch in range(start_epoch, CFG.epochs):
        train_loss, train_f1, global_step, stopped_for_time = run_train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, CFG, ckpt, epoch, global_step)
        if stopped_for_time:
            break

        dev_loss, dev_logits, dev_labels = run_eval(model, dev_loader, loss_fn, CFG)
        dev_f1 = np.mean([micro_f1_from_counts(l, y) for l, y in zip(dev_logits, dev_labels)])
        gap = train_f1 - dev_f1

        overfit_flag = " <- dev_loss rising (overfitting signal)" if dev_loss > best_dev_loss else ""
        print(f"Epoch {epoch+1}/{CFG.epochs} | train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"| dev_loss={dev_loss:.4f} dev_f1={dev_f1:.4f} | train-dev gap={gap:+.4f}{overfit_flag}")

        # Best-checkpoint selection always uses dev_f1 (what the competition scores).
        is_best_f1 = dev_f1 > best_dev_score
        if is_best_f1:
            best_dev_score = dev_f1

        # Early-stopping *trigger* uses whichever metric CFG.early_stop_metric picks --
        # dev_loss tends to signal overfitting a few epochs before dev_f1 visibly
        # plateaus (see Section 9 discussion), so it's the recommended default.
        if CFG.early_stop_metric == "dev_loss":
            improved = dev_loss < best_dev_loss
        else:
            improved = is_best_f1

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt.save(model, optimizer, scheduler, epoch + 1, global_step, best_dev_score, is_best=is_best_f1)

        if patience_counter >= CFG.early_stop_patience:
            print(f"Early stopping ({CFG.early_stop_metric} plateaued for {CFG.early_stop_patience} epochs).")
            break

    print(f"Training loop finished. Best dev micro-F1: {best_dev_score:.4f} | Best dev_loss: {best_dev_loss:.4f}")
    if stopped_for_time:
        print("NOTE: stopped due to time budget mid-epoch — re-run this cell/notebook to resume.")
else:
    print(f"CFG.k_folds = {CFG.k_folds} (> 1) -> skipping Section 8's single-model training. "
          f"Section 8b will train the fold models instead.")
