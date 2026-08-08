# Shared model.py for track3/bilstm_crf_head (6 of 6 experiments — auto-consolidated, identical after stripping header/noise). Excluded (differ for real reasons): none

# ## 6. Model — Pretrained Backbone + Char BiLSTM-CRF Head

class LayerPool(nn.Module):
    '''Learned softmax mix over the last `num_layers` backbone hidden-state
    layers (ELMo-style) instead of hard-coding "use the last layer" or a
    fixed concat -- the model learns which depths matter.'''

    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states_tuple):
        stacked = torch.stack(hidden_states_tuple, dim=0)       # (L,B,T,H)
        w = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (stacked * w).sum(dim=0)                          # (B,T,H)



class CRF(nn.Module):
    '''Standard linear-chain CRF: learned start/end/transition scores over
    `num_tags`, batch-first API. `forward` returns per-example
    log-likelihood (turn into a loss with `-llh`); `decode` runs Viterbi and
    returns a list of python lists (one variable-length label sequence per
    batch element). Plain PyTorch, no external dependency.'''

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask):
        emissions = emissions.transpose(0, 1)       # (T,B,C)
        tags = tags.transpose(0, 1)                 # (T,B)
        mask = mask.transpose(0, 1).float()          # (T,B)
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        return numerator - denominator               # (B,) log-likelihood

    def decode(self, emissions, mask):
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1).float()
        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, mask):
        T, B = tags.shape
        arangeB = torch.arange(B, device=emissions.device)
        score = self.start_transitions[tags[0]] + emissions[0, arangeB, tags[0]]
        for i in range(1, T):
            score = score + self.transitions[tags[i - 1], tags[i]] * mask[i] \
                          + emissions[i, arangeB, tags[i]] * mask[i]
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, arangeB]
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions, mask):
        T, B, C = emissions.shape
        score = self.start_transitions + emissions[0]       # (B,C)
        for i in range(1, T):
            broadcast_score = score.unsqueeze(2)              # (B,C,1)
            broadcast_emission = emissions[i].unsqueeze(1)     # (B,1,C)
            next_score = broadcast_score + self.transitions.unsqueeze(0) + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)    # (B,C)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, mask):
        T, B, C = emissions.shape
        score = self.start_transitions + emissions[0]
        history = []
        for i in range(1, T):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions.unsqueeze(0) + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)
        score = score + self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1

        best_tags_list = []
        for b in range(B):
            _, best_last_tag = score[b].max(dim=0)
            best_tags = [best_last_tag.item()]
            end = int(seq_ends[b].item())
            for hist in reversed(history[:end]):
                best_last_tag = hist[b][best_tags[-1]]
                best_tags.append(int(best_last_tag.item()))
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list



class Track3BiLSTMCRF(nn.Module):
    def __init__(self, backbone_name: str, char_vocab_size: int, num_labels: int,
                 char_emb_dim: int = 64, n_pool_layers: int = 4, lstm_hidden_dim: int = 384,
                 num_lstm_layers: int = 2, dropout: float = 0.3, use_crf: bool = True,
                 aux_loss_weight: float = 0.3, class_weights: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = True, freeze_n_layers: int = 0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        bert_hidden = self.backbone.config.hidden_size
        self.n_pool_layers = n_pool_layers
        self.layer_pool = LayerPool(n_pool_layers)

        if freeze_embeddings:
            for p in self.backbone.get_input_embeddings().parameters():
                p.requires_grad = False
        if freeze_n_layers > 0:
            for layer in self.backbone.encoder.layer[:freeze_n_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.word_final_proj = nn.Linear(1, 16)

        combined_dim = bert_hidden + char_emb_dim + 16
        self.input_proj = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, lstm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            lstm_hidden_dim, lstm_hidden_dim // 2, num_layers=num_lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_dim, num_labels)

        self.use_crf = use_crf
        self.crf = CRF(num_labels)
        self.aux_loss_weight = aux_loss_weight
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def _encode(self, input_ids, attention_mask, token_idx_per_char, char_ids, is_word_final):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
        hs = outputs.hidden_states[-self.n_pool_layers:]
        pooled = self.layer_pool(hs)                           # (B,T,H)

        H = pooled.size(-1)
        idx = token_idx_per_char.unsqueeze(-1).expand(-1, -1, H)
        gathered = torch.gather(pooled, 1, idx)                  # (B,Tc,H)

        char_emb = self.char_embedding(char_ids)
        wf = self.word_final_proj(is_word_final.unsqueeze(-1))
        combined = torch.cat([gathered, char_emb, wf], dim=-1)

        x = self.input_proj(combined)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        return self.classifier(x)                                # emissions (B,Tc,num_labels)

    def forward(self, input_ids, attention_mask, char_ids, token_idx_per_char,
                is_word_final, char_mask, labels=None):
        emissions = self._encode(input_ids, attention_mask, token_idx_per_char, char_ids, is_word_final)

        if labels is None:
            if self.use_crf:
                return self.crf.decode(emissions, mask=char_mask)
            preds = emissions.argmax(dim=-1)
            lengths = char_mask.sum(dim=1)
            return [preds[i, :int(lengths[i])].tolist() for i in range(preds.size(0))]

        if self.use_crf:
            llh = self.crf(emissions, labels, mask=char_mask)
            seq_loss = (-llh).mean()
        else:
            logits_flat = emissions.reshape(-1, emissions.size(-1))
            labels_flat = labels.reshape(-1)
            mask_flat = char_mask.reshape(-1)
            seq_loss = F.cross_entropy(logits_flat[mask_flat], labels_flat[mask_flat],
                                        weight=self.class_weights, reduction="mean")

        aux_loss = emissions.new_zeros(())
        if self.aux_loss_weight > 0:
            logits_flat = emissions.reshape(-1, emissions.size(-1))
            labels_flat = labels.reshape(-1)
            mask_flat = char_mask.reshape(-1)
            aux_loss = F.cross_entropy(logits_flat[mask_flat], labels_flat[mask_flat],
                                        weight=self.class_weights, reduction="mean")
        return (seq_loss + self.aux_loss_weight * aux_loss).view(1), emissions



def majority_vote_decode(models: List[nn.Module], input_ids, attention_mask, char_ids,
                          token_idx_per_char, is_word_final, char_mask) -> List[List[int]]:
    '''Ensembles multiple models sharing one tokenizer (e.g. k-fold members)
    by majority-voting each model's own decoded label sequence per character
    position -- defined here (not in Section 9) so Section 14's
    cross-backbone ensemble can use it without needing Section 9 to have run.'''
    all_decoded = [m(input_ids, attention_mask, char_ids, token_idx_per_char,
                      is_word_final, char_mask, labels=None) for m in models]
    B = len(all_decoded[0])
    voted = []
    for b in range(B):
        length = len(all_decoded[0][b])
        seq = []
        for pos in range(length):
            votes = [all_decoded[m][b][pos] for m in range(len(models))]
            vals, counts = np.unique(votes, return_counts=True)
            seq.append(int(vals[np.argmax(counts)]))  # ties -> lowest class id, deterministic
        voted.append(seq)
    return voted


# ## 8. Train `ACTIVE_MODEL`

tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME, use_fast=True)
aligner = CharAligner(tokenizer, CFG.max_subword_len)

train_ds = DiacritizationDataset(train_records, aligner, CHAR2ID, CFG.space_label)
dev_ds = DiacritizationDataset(dev_records, aligner, CHAR2ID, CFG.space_label)

_collate = lambda b: collate_fn(b, tokenizer.pad_token_id or 0)
train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, collate_fn=_collate)
dev_loader = DataLoader(dev_ds, batch_size=CFG.eval_batch_size, shuffle=False, collate_fn=_collate)

class_weights = compute_class_weights(train_records, CFG, DEVICE)
print("Auxiliary-loss class weights:", class_weights.tolist())



def _to_device(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


def masked_accuracy_from_emissions(emissions, labels, char_mask, space_label) -> float:
    '''Fast approximate accuracy from raw emissions (no Viterbi decode) --
    used as the per-batch training-progress signal only.'''
    preds = emissions.argmax(-1)
    valid = char_mask & (labels != space_label)
    if valid.sum() == 0:
        return 0.0
    return (preds[valid] == labels[valid]).float().mean().item()


def build_model(cfg: TrainConfig, class_weights: torch.Tensor) -> "Track3BiLSTMCRF":
    return Track3BiLSTMCRF(
        cfg.backbone, len(CHAR2ID), cfg.num_labels, char_emb_dim=cfg.char_emb_dim,
        n_pool_layers=cfg.n_pool_layers, lstm_hidden_dim=cfg.lstm_hidden_dim,
        num_lstm_layers=cfg.num_lstm_layers, dropout=cfg.head_dropout, use_crf=cfg.use_crf,
        aux_loss_weight=cfg.aux_loss_weight, class_weights=class_weights,
        freeze_embeddings=cfg.freeze_embeddings, freeze_n_layers=cfg.freeze_n_layers,
    ).to(DEVICE)


def run_train_epoch(model, loader, optimizer, scheduler, cfg, ckpt, epoch, global_step):
    model.train()
    epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
    t0 = time.time()
    for batch in loader:
        batch = _to_device(batch)
        loss1, emissions1 = model(batch["input_ids"], batch["attention_mask"], batch["char_ids"],
                                   batch["token_idx_per_char"], batch["is_word_final"],
                                   batch["char_mask"], labels=batch["labels"])
        loss1 = loss1.mean()

        if cfg.use_rdrop:
            loss2, emissions2 = model(batch["input_ids"], batch["attention_mask"], batch["char_ids"],
                                       batch["token_idx_per_char"], batch["is_word_final"],
                                       batch["char_mask"], labels=batch["labels"])
            loss2 = loss2.mean()
            kl = rdrop_kl(emissions1, emissions2, batch["char_mask"])
            loss = (loss1 + loss2) / 2 + cfg.rdrop_alpha * kl
        else:
            loss = loss1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1
        epoch_loss += loss.item()
        epoch_acc += masked_accuracy_from_emissions(emissions1.detach(), batch["labels"],
                                                      batch["char_mask"], cfg.space_label)
        n_batches += 1

        if global_step % cfg.checkpoint_every_steps == 0:
            ckpt.save(model, optimizer, scheduler, epoch, global_step, best_dev_score=-1.0)

        if (time.time() - t0) / 60 > cfg.max_train_minutes:
            print("Time budget reached mid-epoch; checkpointing and stopping.")
            ckpt.save(model, optimizer, scheduler, epoch, global_step, best_dev_score=-1.0)
            return epoch_loss / n_batches, epoch_acc / n_batches, global_step, True

    return epoch_loss / n_batches, epoch_acc / n_batches, global_step, False


@torch.no_grad()
def run_eval(model, loader, cfg):
    '''Returns (avg_loss, decode_accuracy). Decode accuracy uses the model's
    real inference path (CRF Viterbi or argmax decode), not the fast
    emissions-argmax approximation used mid-training -- so this number is
    what checkpoint selection and early stopping actually optimize for.'''
    model.eval()
    total_loss, n_batches = 0.0, 0
    y_true, y_pred = [], []
    for batch in loader:
        batch_gpu = _to_device(batch)
        loss, _ = model(batch_gpu["input_ids"], batch_gpu["attention_mask"], batch_gpu["char_ids"],
                         batch_gpu["token_idx_per_char"], batch_gpu["is_word_final"],
                         batch_gpu["char_mask"], labels=batch_gpu["labels"])
        total_loss += loss.mean().item(); n_batches += 1

        decoded = model(batch_gpu["input_ids"], batch_gpu["attention_mask"], batch_gpu["char_ids"],
                         batch_gpu["token_idx_per_char"], batch_gpu["is_word_final"],
                         batch_gpu["char_mask"], labels=None)
        labels_cpu = batch["labels"]
        for i, seq in enumerate(decoded):
            true_seq = labels_cpu[i, :len(seq)].tolist()
            for t, p in zip(true_seq, seq):
                if t != cfg.space_label:
                    y_true.append(t); y_pred.append(p)

    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0
    return total_loss / n_batches, acc



if CFG.k_folds == 1:
    model = build_model(CFG, class_weights)
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
        train_loss, train_acc, global_step, stopped_for_time = run_train_epoch(
            model, train_loader, optimizer, scheduler, CFG, ckpt, epoch, global_step)
        if stopped_for_time:
            break

        dev_loss, dev_acc = run_eval(model, dev_loader, CFG)
        gap = train_acc - dev_acc

        overfit_flag = " <- dev_loss rising (overfitting signal)" if dev_loss > best_dev_loss else ""
        print(f"Epoch {epoch+1}/{CFG.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} | train-dev gap={gap:+.4f}{overfit_flag}")

        is_best_f1 = dev_acc > best_dev_score
        if is_best_f1:
            best_dev_score = dev_acc

        if CFG.early_stop_metric == "dev_loss":
            improved = dev_loss < best_dev_loss
        else:
            improved = is_best_f1
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss

        patience_counter = 0 if improved else patience_counter + 1
        ckpt.save(model, optimizer, scheduler, epoch + 1, global_step, best_dev_score, is_best=is_best_f1)

        if patience_counter >= CFG.early_stop_patience:
            print(f"Early stopping ({CFG.early_stop_metric} plateaued for {CFG.early_stop_patience} epochs).")
            break

    print(f"Training loop finished. Best dev accuracy: {best_dev_score:.4f} | Best dev_loss: {best_dev_loss:.4f}")
    if stopped_for_time:
        print("NOTE: stopped due to time budget mid-epoch — re-run this cell/notebook to resume.")
else:
    print(f"CFG.k_folds = {CFG.k_folds} (> 1) -> skipping Section 8's single-model training. "
          f"Section 8b will train the fold models instead.")
