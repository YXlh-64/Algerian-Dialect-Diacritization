"""Sanity checks for the Dataset/DataLoader pipeline -- padding, masking, and that `chars` survives
collation
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.track1.soundous.data_utils import DiacritizationDataset, make_collate_fn

CHAR2IDX = {"<PAD>": 0, "<UNK>": 1, " ": 2, "a": 3, "b": 4, "c": 5}
LABEL2IDX = {"NO_DIAC": 0, "FATHA": 1}
PAD_IDX, UNK_IDX = 0, 1

ROWS = [
    {"sent_id": "1", "chars": ["a", "b", " ", "c"], "labels": [0, 1, 0, 0]},
    {"sent_id": "2", "chars": ["a"], "labels": [1]},
]


def test_collate_shapes_and_mask():
    ds = DiacritizationDataset(ROWS, CHAR2IDX, LABEL2IDX, UNK_IDX)
    collate_fn = make_collate_fn(PAD_IDX)
    batch = collate_fn([ds[0], ds[1]])

    assert batch["char_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    assert batch["mask"].shape == (2, 4)
    # longer sequence first (sorted descending by length)
    assert batch["lengths"].tolist() == [4, 1]
    # second row padded: only first position valid
    assert batch["mask"][1].tolist() == [True, False, False, False]


def test_chars_survive_collation():
    ds = DiacritizationDataset(ROWS, CHAR2IDX, LABEL2IDX, UNK_IDX)
    collate_fn = make_collate_fn(PAD_IDX)
    batch = collate_fn([ds[0], ds[1]])
    assert batch["chars"][0] == ["a", "b", " ", "c"]
    assert batch["chars"][1] == ["a"]


def test_unknown_char_maps_to_unk():
    ds = DiacritizationDataset(
        [{"sent_id": "3", "chars": ["z"], "labels": [0]}], CHAR2IDX, LABEL2IDX, UNK_IDX
    )
    item = ds[0]
    assert item["char_ids"].tolist() == [UNK_IDX]
