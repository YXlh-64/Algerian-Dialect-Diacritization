# Shared model.py for track3/linear_head (6 of 6 experiments — auto-consolidated, identical after stripping header/noise). Excluded (differ for real reasons): none

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
import sympy
import sympy.printing  # noqa: F401  -- registers sympy.printing before torch needs it
import transformers
import importlib
import sympy.printing  # noqa: F401
import sklearn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
from collections import defaultdict, Counter
import subprocess

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
