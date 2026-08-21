# training/track4/Ines/finetune_dual_stream_crf_head.py
#
# Training + inference script for Track 4 / Ines, split out
# of the Kaggle notebook. Follows the same repo-root import pattern as
# training/track3/*/finetune_*.py.

import os
import sys
import json
import glob
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Repo-root cross-module imports (mirrors training/track3/*/finetune_*.py) ---
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from models.track4.Ines.dual_stream_crf_head_model import Track4DualStreamCRF
from evaluation.track4.Ines.evaluate_dual_stream_crf_head import (
    Evaluator, word_level_metrics_from_predict_fn,
)


@dataclass
class Config:
    # paths -- adjust to the actual Kaggle dataset mount point
    data_root: str = "/kaggle/input/algerian-arabic-diacritization"
    train_glob: str = "train_data/*.json*"
    dev_glob: str = "dev_data/*.json*"
    test_glob: str = "test_data/*.json*"
    vocab_path: str = "vocab.json"
    out_dir: str = "/kaggle/working"

    # data
    max_seq_len: int = 512
    num_labels: int = 16

    # model
    dim: int = 256
    n_heads: int = 8
    local_layers: int = 6
    global_layers: int = 4
    final_layers: int = 2
    local_window: int = 16
    dropout: float = 0.15

    # optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 25
    early_stopping_patience: int = 10
    grad_clip: float = 1.0
    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Vocab / dataset / collate
# ---------------------------------------------------------------------------
def load_vocab(vocab_path):
    with open(vocab_path, encoding="utf-8") as f:
        raw = json.load(f)
    # accept either a flat {char: id} dict, or a {"char2id": {...}} wrapper
    char2id = raw["char2id"] if "char2id" in raw else raw
    char2id = {c: int(i) for c, i in char2id.items()}
    if " " not in char2id:
        raise ValueError("vocab.json must map the space character to an id")
    return char2id


class DiacriticsDataset(Dataset):
    '''One example = one sentence: list of chars + list of int labels (0-15).'''

    def __init__(self, glob_pattern, char2id, max_seq_len):
        self.char2id = char2id
        self.max_seq_len = max_seq_len
        self.records = []
        for fp in sorted(glob.glob(glob_pattern)):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # supports both JSON-lines and a single JSON array per file
                    if line.startswith("["):
                        self.records.extend(json.loads(line))
                    else:
                        self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        chars = rec["chars"][: self.max_seq_len]
        labels = rec.get("labels", [0] * len(chars))[: self.max_seq_len]
        ids = [self.char2id.get(c, self.char2id[" "]) for c in chars]
        return {
            "ids": ids,
            "labels": labels,
            "chars": chars,
            "sent_id": rec.get("sent_id", str(idx)),
        }


def make_collate_fn(pad_id):
    def collate_fn(batch):
        max_len = max(len(b["ids"]) for b in batch)
        B = len(batch)
        ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        labels = torch.zeros((B, max_len), dtype=torch.long)
        pad_mask = torch.ones((B, max_len), dtype=torch.bool)   # True = PAD
        is_space = torch.zeros((B, max_len), dtype=torch.bool)  # True = space char
        sent_ids, all_chars = [], []
        for i, b in enumerate(batch):
            L = len(b["ids"])
            ids[i, :L] = torch.tensor(b["ids"], dtype=torch.long)
            labels[i, :L] = torch.tensor(b["labels"], dtype=torch.long)
            pad_mask[i, :L] = False
            is_space[i, :L] = torch.tensor([c == " " for c in b["chars"]], dtype=torch.bool)
            sent_ids.append(b["sent_id"])
            all_chars.append(b["chars"])
        return {
            "ids": ids, "labels": labels, "pad_mask": pad_mask, "is_space": is_space,
            "sent_ids": sent_ids, "chars": all_chars,
        }
    return collate_fn


# ---------------------------------------------------------------------------
# Fast per-epoch eval (letter-level F1/acc) -- kept here since it needs to
# run every epoch inside the training loop. The richer end-of-run report
# (confusion matrix, per-class F1, DER/WER) uses evaluation.Evaluator instead.
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, num_labels):
    model.eval()
    tp = torch.zeros(num_labels)
    fp = torch.zeros(num_labels)
    fn = torch.zeros(num_labels)
    correct = total = 0
    for batch in loader:
        ids = batch["ids"].to(device)
        labels = batch["labels"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        is_space = batch["is_space"].to(device)

        preds = model.predict(ids, pad_mask, is_space)
        letter_mask = (~pad_mask) & (~is_space)
        p = preds[letter_mask].cpu()
        g = labels[letter_mask].cpu()

        correct += (p == g).sum().item()
        total += g.numel()
        for c in range(num_labels):
            tp[c] += ((p == c) & (g == c)).sum().item()
            fp[c] += ((p == c) & (g != c)).sum().item()
            fn[c] += ((p != c) & (g == c)).sum().item()

    acc = correct / max(total, 1)
    precision = tp.sum() / max((tp + fp).sum().item(), 1)
    recall = tp.sum() / max((tp + fn).sum().item(), 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-9)).item()
    return f1, acc


def build_dataloaders(cfg, char2id, pad_id):
    collate_fn = make_collate_fn(pad_id)
    train_ds = DiacriticsDataset(os.path.join(cfg.data_root, cfg.train_glob), char2id, cfg.max_seq_len)
    dev_ds = DiacriticsDataset(os.path.join(cfg.data_root, cfg.dev_glob), char2id, cfg.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn, drop_last=False)
    return train_loader, dev_loader


def train(cfg, char2id):
    device = cfg.device
    pad_id = len(char2id)
    vocab_size = len(char2id) + 1
    train_loader, dev_loader = build_dataloaders(cfg, char2id, pad_id)

    model = Track4DualStreamCRF(
        vocab_size=vocab_size, num_labels=cfg.num_labels, dim=cfg.dim, n_heads=cfg.n_heads,
        local_layers=cfg.local_layers, global_layers=cfg.global_layers, final_layers=cfg.final_layers,
        local_window=cfg.local_window, dropout=cfg.dropout, max_seq_len=cfg.max_seq_len,
        unscored_label_id=0,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, steps_per_epoch=steps_per_epoch, epochs=cfg.max_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_f1 = -1.0
    best_path = os.path.join(cfg.out_dir, "dscat_best.pt")
    patience_left = cfg.early_stopping_patience

    for epoch in range(cfg.max_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            ids = batch["ids"].to(device)
            labels = batch["labels"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            is_space = batch["is_space"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = model.loss(ids, labels, pad_mask, is_space)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        f1, acc = evaluate(model, dev_loader, device, cfg.num_labels)
        print(f"[epoch {epoch}] train_loss={total_loss/steps_per_epoch:.4f} dev_f1={f1:.4f} dev_letter_acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_left = cfg.early_stopping_patience
            torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__, "f1": f1}, best_path)
            print(f"  -> new best (f1={f1:.4f}), checkpoint saved to {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  -> early stopping at epoch {epoch} (no improvement for {cfg.early_stopping_patience} epochs)")
                break

    return best_path


# ---------------------------------------------------------------------------
# Test-set inference + submission
# ---------------------------------------------------------------------------
def load_test_records(test_dir):
    """The competition's test set is NOT in the JSON/JSONL format used by
    train/dev -- it's a pair of plain-text files:
        raw_sentences_test.txt      (one undiacritized sentence per line)
        raw_sentences_test_ids.txt  (matching sentence id per line, same order)
    DiacriticsDataset's glob-based JSON loader finds nothing in this folder,
    which is why an earlier attempt at this returned an empty test set."""
    raw_path = os.path.join(test_dir, "raw_sentences_test.txt")
    ids_path = os.path.join(test_dir, "raw_sentences_test_ids.txt")
    with open(raw_path, encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f]
    with open(ids_path, encoding="utf-8") as f:
        id_lines = [l.strip() for l in f]
    if len(raw_lines) != len(id_lines):
        raise ValueError(
            f"raw_sentences_test.txt has {len(raw_lines)} lines but "
            f"raw_sentences_test_ids.txt has {len(id_lines)} lines -- they must match 1:1."
        )
    records = []
    for sid, line in zip(id_lines, raw_lines):
        chars = list(line)
        records.append({"chars": chars, "labels": [0] * len(chars), "sent_id": sid})
    return records


class TestSentencesDataset(Dataset):
    '''Same __getitem__ contract as DiacriticsDataset, built from raw records instead of a JSON glob.'''

    def __init__(self, records, char2id, max_seq_len):
        self.records = records
        self.char2id = char2id
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        chars = rec["chars"][: self.max_seq_len]
        labels = rec["labels"][: self.max_seq_len]
        ids = [self.char2id.get(c, self.char2id[" "]) for c in chars]
        return {"ids": ids, "labels": labels, "chars": chars, "sent_id": rec["sent_id"]}


# Inverse of the DIACRITICS / SHADDA_COMBOS scheme in test_data/make_submission.py,
# so run_inference's output is fully diacritized text -- not raw label ids.
LABEL_TO_DIACRITIC = {
    0: "",
    1: "\u064E",         # fatha
    2: "\u064B",         # fathatan
    3: "\u064F",         # damma
    4: "\u064C",         # dammatan
    5: "\u0650",         # kasra
    6: "\u064D",         # kasratan
    7: "\u0652",         # sukoon
    8: "\u0651",         # shadda
    9: "\u0651\u064E",  # shadda + fatha
    10: "\u0651\u064B", # shadda + fathatan
    11: "\u0651\u064F", # shadda + damma
    12: "\u0651\u064C", # shadda + dammatan
    13: "\u0651\u0650", # shadda + kasra
    14: "\u0651\u064D", # shadda + kasratan
    15: "\u0651\u0652", # shadda + sukoon
}


def run_inference(cfg, char2id, checkpoint_path):
    device = cfg.device
    pad_id = len(char2id)
    vocab_size = len(char2id) + 1
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Track4DualStreamCRF(
        vocab_size=vocab_size, num_labels=cfg.num_labels, dim=cfg.dim, n_heads=cfg.n_heads,
        local_layers=cfg.local_layers, global_layers=cfg.global_layers, final_layers=cfg.final_layers,
        local_window=cfg.local_window, dropout=cfg.dropout, max_seq_len=cfg.max_seq_len,
        unscored_label_id=0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    collate_fn = make_collate_fn(pad_id)
    test_dir = os.path.join(cfg.data_root, "test_data")
    test_records = load_test_records(test_dir)
    test_ds = TestSentencesDataset(test_records, char2id, cfg.max_seq_len)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    print("test sentences loaded:", len(test_ds))

    diacritized = {}
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["ids"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            is_space = batch["is_space"].to(device)
            preds = model.predict(ids, pad_mask, is_space)

            for i, sent_id in enumerate(batch["sent_ids"]):
                L = (~pad_mask[i]).sum().item()
                chars = batch["chars"][i]
                pred_labels = preds[i, :L].tolist()
                # rebuild the fully diacritized sentence, character by character
                diacritized[sent_id] = "".join(
                    ch + LABEL_TO_DIACRITIC[lab] for ch, lab in zip(chars, pred_labels)
                )

    out_path = os.path.join(cfg.out_dir, "submission.txt")
    with open(out_path, "w", encoding="utf-8") as f_out:
        # write in the SAME order as raw_sentences_test.txt -- make_submission.py zips by line position
        for rec in test_records:
            f_out.write(diacritized[rec["sent_id"]] + "\n")

    print("wrote", out_path, "-", len(test_records), "lines")
    return out_path, test_dir


# ---------------------------------------------------------------------------
# Data-root auto-discovery (Kaggle mounts each dataset under a slug that
# doesn't always match the competition name)
# ---------------------------------------------------------------------------
def find_data_root(search_root="/kaggle/input", vocab_filename="vocab.json"):
    if not os.path.isdir(search_root):
        raise FileNotFoundError(f"{search_root} does not exist -- is a dataset attached to this notebook?")
    for dirpath, _, filenames in os.walk(search_root):
        if vocab_filename in filenames:
            return dirpath
    available = os.listdir(search_root)
    raise FileNotFoundError(
        f"Could not find {vocab_filename} under {search_root}. "
        f"Datasets currently attached: {available}. "
        f"Open the 'Input' panel on the right of the notebook to confirm the exact folder name, "
        f"then set cfg.data_root manually if this still fails."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--data-root", type=str, default=None,
                          help="Override auto-discovered data root")
    _parser.add_argument("--epochs", type=int, default=None,
                          help="Override cfg.max_epochs")
    _parser.add_argument("--seed", type=int, default=None,
                          help="Override the deterministic random seed")
    _args = _parser.parse_args(argv)

    cfg = Config()
    cfg.data_root = _args.data_root or find_data_root()
    if _args.epochs:
        cfg.max_epochs = _args.epochs
    if _args.seed is not None:
        cfg.seed = _args.seed
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("device:", cfg.device)
    print("using data_root:", cfg.data_root)
    print("contents:", os.listdir(cfg.data_root))

    char2id = load_vocab(os.path.join(cfg.data_root, cfg.vocab_path))
    best_ckpt = train(cfg, char2id)
    submission_txt_path, test_dir = run_inference(cfg, char2id, best_ckpt)

    # Hand off to the organizers' own script, exactly as it expects:
    #   fully-diacritized text (submission.txt)  ->  Id,Label CSV (submission.csv)
    import subprocess
    submission_csv_path = os.path.join(cfg.out_dir, "submission.csv")
    result = subprocess.run(
        [
            sys.executable, os.path.join(test_dir, "make_submission.py"),
            "--ids", os.path.join(test_dir, "raw_sentences_test_ids.txt"),
            "--input", os.path.join(test_dir, "raw_sentences_test.txt"),
            "--pred", submission_txt_path,
            "--out", submission_csv_path,
        ],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("make_submission.py failed -- see stderr above")

    import pandas as pd
    sub = pd.read_csv(submission_csv_path)
    print("submission rows:", len(sub))
    print(sub.head())
    return submission_csv_path


if __name__ == "__main__":
    main()
