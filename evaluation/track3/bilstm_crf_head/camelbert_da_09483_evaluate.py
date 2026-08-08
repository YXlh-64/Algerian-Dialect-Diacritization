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


if CFG.k_folds > 1 and FOLD_CHECKPOINT_DIRS:
    MODELS_FOR_INFERENCE = []
    for fold_dir in FOLD_CHECKPOINT_DIRS:
        fm = build_model(CFG, class_weights)
        CheckpointManager(fold_dir).load_best(fm)
        MODELS_FOR_INFERENCE.append(fm)
    print(f"Loaded {len(MODELS_FOR_INFERENCE)} fold models for ensemble evaluation.")
else:
    ckpt.load_best(model)   # best checkpoint by dev accuracy, from Section 8
    MODELS_FOR_INFERENCE = [model]

dev_test_ds = DiacritizationDataset(dev_test_records, aligner, CHAR2ID, CFG.space_label)
dev_test_loader = DataLoader(dev_test_ds, batch_size=CFG.eval_batch_size, shuffle=False, collate_fn=_collate)

evaluator = Evaluator(CLASS_NAMES, CFG.space_label)
DEV_TEST_REPORT = evaluator.evaluate(MODELS_FOR_INFERENCE, dev_test_loader)
DEV_TEST_SCORE = DEV_TEST_REPORT["micro_f1"]
print(f"DEV_TEST micro-F1 (Kaggle-score proxy): {DEV_TEST_SCORE:.4f}  "
      f"(ensembling {DEV_TEST_REPORT['n_models_ensembled']} model(s))")
print(f"DEV_TEST macro-F1: {DEV_TEST_REPORT['macro_f1']:.4f}")
evaluator.plot_confusion(DEV_TEST_REPORT["confusion_matrix"])



@torch.no_grad()
def _predict_chars(models: List[nn.Module], chars: List[str]) -> List[int]:
    '''Single-sentence prediction used by DER/WER (Section 5's helper) and
    self-training pseudo-labeling. Majority-votes across models exactly as
    Evaluator does for batches.'''
    if not chars:
        return []
    n = len(chars)
    enc = aligner.encode(chars)
    input_ids = torch.tensor([enc["input_ids"]], device=DEVICE)
    attn = torch.ones_like(input_ids)
    toks = torch.tensor([[t if t >= 0 else 0 for t in enc["token_idx_per_char"][:n]]], device=DEVICE)
    char_ids = torch.tensor([[CHAR2ID.get(c, CHAR2ID.get('<UNK>', 1)) for c in chars]], device=DEVICE)
    wf = torch.tensor([compute_is_word_final(chars)], device=DEVICE)
    mask = torch.ones((1, n), dtype=torch.bool, device=DEVICE)
    decoded = majority_vote_decode(models, input_ids, attn, char_ids, toks, wf, mask)
    return decoded[0]

WORD_METRICS = word_level_metrics_from_predict_fn(
    lambda chars: _predict_chars(MODELS_FOR_INFERENCE, chars), dev_test_records)
DEV_TEST_REPORT.update(WORD_METRICS)

print(f"DER       : {WORD_METRICS['DER']:.4f}   (sanity check, should ~= 1 - micro_f1 = "
      f"{1 - DEV_TEST_SCORE:.4f})")
print(f"DER*      : {WORD_METRICS['DER_star']:.4f}   (excluding each word's last letter)")
print(f"WER       : {WORD_METRICS['WER']:.4f}   (fraction of words with >=1 wrong letter)")
print(f"WER*      : {WORD_METRICS['WER_star']:.4f}   (excluding each word's last letter)")


