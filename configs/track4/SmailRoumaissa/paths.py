import glob
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataPaths:
    train: Optional[str] = None
    dev: Optional[str] = None
    vocab: Optional[str] = None
    labels: Optional[str] = None
    raw_test: Optional[str] = None
    raw_test_ids: Optional[str] = None
    sample_submission: Optional[str] = None
    make_submission_py: Optional[str] = None

    def missing(self):
        return [k for k, v in self.__dict__.items() if v is None]


def _find_one(patterns, roots=("/kaggle/input",)):
    for root in roots:
        for pat in patterns:
            hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
            if hits:
                return hits[0]
    return None


def find_data_paths(roots=("/kaggle/input",)) -> DataPaths:
    paths = DataPaths(
        train=_find_one(["*train*.jsonl"], roots),
        dev=_find_one(["*dev*.jsonl"], roots),
        vocab=_find_one(["vocab.json"], roots),
        labels=_find_one(["class_labels.txt"], roots),
        raw_test=_find_one(["raw_sentences_test.txt"], roots),
        raw_test_ids=_find_one(["raw_sentences_test_ids.txt"], roots),
        sample_submission=_find_one(["sample_submission.csv"], roots),
        make_submission_py=_find_one(["make_submission.py"], roots),
    )
    return paths
