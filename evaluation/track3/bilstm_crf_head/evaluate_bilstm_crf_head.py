# Shared evaluate.py for track3/bilstm_crf_head (5 of 6 experiments — auto-consolidated, identical after stripping header/noise). Excluded (differ for real reasons): camelbert_da_09483

# --- Imports (copied from the notebook's preamble so this file has its
#     basic dependencies resolved; full setup still lives in training/) ---
import os, sys, json, re, glob, random, shutil, time, zipfile, unicodedata, copy
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

# Repo root is 4 levels up from this file: evaluation/track3/bilstm_crf_head/<this file>.
# Bootstrapped here too (not just in training/) so this module is independently
# importable, e.g. directly from run_pipeline.py or a notebook.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from models.track3.bilstm_crf_head.bilstm_crf_head_model import majority_vote_decode

# Recomputed independently rather than imported from training/ -- avoids a
# training<->evaluation circular import. Deterministic, so duplication here
# is safe (same convention already used for the import preamble).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_device(batch):
    '''Duplicated from training/track3/bilstm_crf_head/finetune_bilstm_crf_head.py
    (Section 7 "Training Utilities") -- Evaluator.evaluate() below needs it in
    THIS module's namespace, since Python resolves a method's free variables
    against the module it's defined in, not the caller's globals.'''
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ## 9. Final Local Evaluation on `DEV_TEST`

class Evaluator:
    def __init__(self, class_names: List[str], space_label: int):
        self.class_names = class_names
        self.space_label = space_label

    @torch.no_grad()
    def evaluate(self, models: List[nn.Module], loader) -> Dict[str, Any]:
        for m in models:
            m.eval()
        y_true, y_pred = [], []
        for batch in loader:
            batch_gpu = _to_device(batch)
            decoded = majority_vote_decode(models, batch_gpu["input_ids"], batch_gpu["attention_mask"],
                                            batch_gpu["char_ids"], batch_gpu["token_idx_per_char"],
                                            batch_gpu["is_word_final"], batch_gpu["char_mask"])
            labels_cpu = batch["labels"]
            for i, seq in enumerate(decoded):
                true_seq = labels_cpu[i, :len(seq)].tolist()
                for t, p in zip(true_seq, seq):
                    if t != self.space_label:
                        y_true.append(t); y_pred.append(p)

        micro_f1 = f1_score(y_true, y_pred, average="micro")
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        report = classification_report(y_true, y_pred, target_names=self.class_names,
                                        labels=list(range(len(self.class_names))),
                                        output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        return {"micro_f1": micro_f1, "macro_f1": macro_f1,
                "classification_report": report, "confusion_matrix": cm.tolist(),
                "n_chars_evaluated": len(y_true), "n_models_ensembled": len(models)}

    def plot_confusion(self, cm: List[List[int]], normalize: bool = True):
        cm = np.array(cm, dtype=float)
        if normalize:
            cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(self.class_names))); ax.set_xticklabels(self.class_names, rotation=90)
        ax.set_yticks(range(len(self.class_names))); ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion matrix (row-normalized)")
        plt.colorbar(im, fraction=0.046)
        plt.tight_layout(); plt.show()
