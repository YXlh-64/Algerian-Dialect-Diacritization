"""Loads vocab.json and class_labels.txt exactly as the released Kaggle dataset formats them.

vocab.json: flat {char: id} map, 43 entries, with <PAD>=0 <UNK>=1 <BOS>=2 <EOS>=3 already reserved.
class_labels.txt: a documentation TABLE (not a plain list) -- "ID  Class name  Mark(s)  Description".
"""
import json
import re
from typing import Dict, Tuple

NUM_CLASSES = 16
NO_DIAC_IDX = 0


def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str], int, int, int]:
    with open(vocab_path, encoding="utf-8") as f:
        char2idx = json.load(f)
    idx2char = {v: k for k, v in char2idx.items()}
    assert "<PAD>" in char2idx and "<UNK>" in char2idx, "vocab.json must reserve <PAD>/<UNK>"
    pad_idx, unk_idx = char2idx["<PAD>"], char2idx["<UNK>"]
    return char2idx, idx2char, pad_idx, unk_idx, len(char2idx)


def _parse_class_labels_table(labels_path: str) -> Dict[int, str]:
    names = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\s*(\d+)\s+(\S.*?)\s{2,}", line)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
    return names


def load_class_labels(labels_path: str):
    idx2name = _parse_class_labels_table(labels_path)
    class_labels = [idx2name.get(i, f"CLASS_{i}") for i in range(NUM_CLASSES)]
    label2idx = {name: i for i, name in enumerate(class_labels)}
    idx2label = {i: name for i, name in enumerate(class_labels)}

    expected_fragment = {0: "No Diacritic", 1: "Fatha", 8: "Shadda", 15: "Shadda"}
    for cid, frag in expected_fragment.items():
        assert frag.lower() in class_labels[cid].lower(), (
            f"class_labels.txt table order disagrees with the expected id scheme at id {cid} "
            f"(got {class_labels[cid]!r}) -- reconcile before training."
        )
    return class_labels, label2idx, idx2label
