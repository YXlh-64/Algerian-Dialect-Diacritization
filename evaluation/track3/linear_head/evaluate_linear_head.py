# Shared evaluate.py for track3/linear_head (6 of 6 experiments — auto-consolidated, identical after stripping header/noise). Excluded (differ for real reasons): none

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

# Recomputed/duplicated independently rather than imported from training/ --
# avoids a training<->evaluation circular import. Evaluator.evaluate() below
# needs these in THIS module's namespace, since Python resolves a method's
# free variables against the module it's defined in, not the caller's
# globals. DEVICE and IGNORE_INDEX are deterministic, so duplication is
# safe; word_level_metrics_from_predict_fn is duplicated verbatim from
# training/track3/linear_head/finetune_linear_head.py (Section 5) -- it's
# pure (only touches SPACE_CHAR + stdlib), so both copies always agree.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX = -100
SPACE_CHAR = " "


def word_level_metrics_from_predict_fn(predict_fn, records: List[dict]) -> Dict[str, Any]:
    '''Word-level DER/WER plus sentence exact-match, per-class DER (char
    error rate per diacritic class), and the most common (true, predicted)
    confusion pairs -- for a richer error analysis than DER/WER alone.'''
    total_chars = char_errors = 0
    total_chars_star = char_errors_star = 0
    total_words = word_errors = 0
    total_words_star = word_errors_star = 0
    n_sent = n_sent_exact = 0
    class_total: Dict[int, int] = defaultdict(int)
    class_errors: Dict[int, int] = defaultdict(int)
    confusion_pairs: Counter = Counter()

    for rec in records:
        chars, labels = rec["chars"], rec["labels"]
        preds = predict_fn(chars)

        n_sent += 1
        sent_ok = True
        words, cur = [], []
        for i, c in enumerate(chars):
            if c == SPACE_CHAR:
                if cur:
                    words.append(cur)
                cur = []
                continue
            t, p = labels[i], preds[i]
            cur.append((p, t))
            class_total[t] += 1
            if p != t:
                class_errors[t] += 1
                confusion_pairs[(t, p)] += 1
                sent_ok = False
        if cur:
            words.append(cur)
        if sent_ok:
            n_sent_exact += 1

        for word in words:
            if not word:
                continue
            n = len(word)
            errs = [p != t for p, t in word]

            total_chars += n
            char_errors += sum(errs)
            total_words += 1
            word_errors += int(any(errs))

            if n > 1:
                total_chars_star += n - 1
                char_errors_star += sum(errs[:-1])
                total_words_star += 1
                word_errors_star += int(any(errs[:-1]))

    per_class_der = {c: class_errors[c] / class_total[c]
                      for c in class_total if class_total[c] > 0}

    return {
        "DER": char_errors / max(total_chars, 1),
        "DER_star": char_errors_star / max(total_chars_star, 1),
        "WER": word_errors / max(total_words, 1),
        "WER_star": word_errors_star / max(total_words_star, 1),
        "sentence_exact_match": n_sent_exact / max(n_sent, 1),
        "per_class_der": per_class_der,
        "top_confusions": confusion_pairs.most_common(15),
        "n_chars": total_chars, "n_words": total_words, "n_sentences": n_sent,
    }


# ## 9. Final Local Evaluation on `DEV_TEST`
class Evaluator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    @torch.no_grad()
    def evaluate(self, models: List[nn.Module], loader,
                 predict_fn=None, word_metric_records: Optional[List[dict]] = None) -> Dict[str, Any]:
        for m in models:
            m.eval()
        y_true, y_pred = [], []
        for batch in loader:
            batch_gpu = {k: v.to(DEVICE) for k, v in batch.items()}
            probs_sum = None
            for m in models:
                logits = m(batch_gpu["input_ids"], batch_gpu["attention_mask"],
                           batch_gpu["char_ids"], batch_gpu["token_idx_per_char"])
                probs = torch.softmax(logits, dim=-1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            preds = (probs_sum / len(models)).argmax(-1).cpu()
            labels = batch["char_labels"]
            mask = labels != IGNORE_INDEX
            y_true.extend(labels[mask].tolist())
            y_pred.extend(preds[mask].tolist())

        n_classes = len(self.class_names)
        present_labels = sorted(set(y_true))

        micro_f1 = f1_score(y_true, y_pred, average="micro")
        macro_f1_all16 = f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)))
        macro_f1_present = f1_score(y_true, y_pred, average="macro", labels=present_labels)

        report = classification_report(y_true, y_pred, target_names=self.class_names,
                                        labels=list(range(n_classes)),
                                        output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

        result = {
            "micro_f1": micro_f1,
            "macro_f1_all16": macro_f1_all16,               # kept for continuity with earlier exports
            "macro_f1_present_classes": macro_f1_present,    # honest number: excludes 0-support classes
            "n_present_classes": len(present_labels),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "n_chars_evaluated": len(y_true),
            "n_models_ensembled": len(models),
        }
        if predict_fn is not None and word_metric_records is not None:
            result.update(word_level_metrics_from_predict_fn(predict_fn, word_metric_records))
        return result

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


def print_eval_report(report: Dict[str, Any], class_names: List[str], name: str = "DEV_TEST") -> None:
    '''One readable printout pulling from the same report dict that gets
    saved/exported -- so the notebook output and the exported JSON never
    disagree again.'''
    print(f"=== {name} ===")
    print(f"Micro-F1 (competition metric)                : {report['micro_f1']:.4f}")
    print(f"Macro-F1 (all 16 classes, legacy/comparable)  : {report['macro_f1_all16']:.4f}")
    print(f"Macro-F1 ({report['n_present_classes']} classes actually present)        : "
          f"{report['macro_f1_present_classes']:.4f}")
    if "DER" in report:
        print(f"DER  / DER* (excl. word-final letter)        : {report['DER']:.4f} / {report['DER_star']:.4f}")
        print(f"WER  / WER* (excl. word-final letter)        : {report['WER']:.4f} / {report['WER_star']:.4f}")
        print(f"Sentence exact-match accuracy                : {report['sentence_exact_match']:.4f}")
    print(f"Characters evaluated                         : {report['n_chars_evaluated']}")
    print(f"Models ensembled                             : {report['n_models_ensembled']}")
    if "per_class_der" in report:
        print("\nPer-class DER (char error rate), worst first:")
        for cid, der in sorted(report["per_class_der"].items(), key=lambda x: -x[1]):
            print(f"  {class_names[cid]:20s} DER={der:.4f}")
    if "top_confusions" in report:
        print("\nTop confusions (true -> predicted : count):")
        for (t, p), n in report["top_confusions"]:
            print(f"  {class_names[t]:18s} -> {class_names[p]:18s} : {n}")
    print()



