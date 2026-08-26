"""Shared setup for every run_*.py script"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json

from utils.track1.soundousndous.paths import resolve_paths, print_paths
from utils.track1.soundousndous.vocab_utils import load_vocab, load_class_labels, NUM_CLASSES, NO_DIAC_IDX
from utils.track1.soundousndous.data_utils import read_jsonl, make_loader
from utils.track1.soundousndous.seed_utils import get_device, set_seed, SEED

CONFIGS_DIR = os.path.join(_REPO_ROOT, "configs", "track1", "Sou")


def load_config(name):
    with open(os.path.join(CONFIGS_DIR, f"{name}.json")) as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)
    return cfg


def load_everything(batch_size=None):
    set_seed(SEED)
    device = get_device()
    paths = resolve_paths()
    print_paths(paths)

    char2idx, idx2char, pad_idx, unk_idx, vocab_size = load_vocab(paths["vocab_path"])
    class_labels, label2idx, idx2label = load_class_labels(paths["labels_path"])

    train_rows = read_jsonl(paths["train_jsonl"])
    dev_rows = read_jsonl(paths["dev_jsonl"])
    print(f"train: {len(train_rows)} | dev: {len(dev_rows)}")

    model_common = load_config("model_common")
    bs = batch_size or model_common["batch_size"]
    train_loader = make_loader(train_rows, char2idx, label2idx, pad_idx, unk_idx, NO_DIAC_IDX, bs, shuffle=True)
    dev_loader = make_loader(dev_rows, char2idx, label2idx, pad_idx, unk_idx, NO_DIAC_IDX, bs, shuffle=False)

    model_kwargs = {k: v for k, v in model_common.items() if k != "batch_size"}

    return dict(
        device=device, paths=paths, char2idx=char2idx, idx2char=idx2char, pad_idx=pad_idx,
        unk_idx=unk_idx, vocab_size=vocab_size, class_labels=class_labels, label2idx=label2idx,
        idx2label=idx2label, num_classes=NUM_CLASSES, no_diac_idx=NO_DIAC_IDX,
        train_rows=train_rows, dev_rows=dev_rows, train_loader=train_loader, dev_loader=dev_loader,
        model_kwargs=model_kwargs, batch_size=bs,
    )
