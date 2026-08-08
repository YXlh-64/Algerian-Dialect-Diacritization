# Auto-split from Raw Experiments/Bilstm head/camelbert_da-0.9483/camelbert-da-crf (1).ipynb for camelbert_da_09483 (track3/bilstm_crf_head)
# NOTE: shared imports/setup live in training/track3/bilstm_crf_head/camelbert_da_09483_train.py
# Sanity-check imports here before relying on this file standalone.

# --- Imports (copied from the notebook's preamble so this file has its
#     basic dependencies resolved; full setup still lives in training/) ---
import os, sys, json, re, glob, random, shutil, time, zipfile, unicodedata
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import transformers
import sklearn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
import subprocess

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


