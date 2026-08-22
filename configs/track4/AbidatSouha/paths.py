import glob
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# Searched in order, first hit wins. "/kaggle/input" first so a Kaggle run needs
# no change; the relative roots cover a local checkout, where the dataset lives
# outside the repo (Algerian_Vocalisation_Project/Data) and the repo itself sits
# at Algerian_Vocalisation_Project/github/Algerian-Dialect-Diacritization.
# Relative roots resolve against the current working directory, so run scripts
# from the repo root.
DEFAULT_ROOTS: Tuple[str, ...] = (
    "/kaggle/input",
    "Data",
    "../Data",
    "../../Data",
    "../../../Data",
)


@dataclass
class DataPaths:
    """Resolved locations of the competition files.

    Field names match configs/track4/SmailRoumaissa/paths.py so evaluation and
    inference code is interchangeable between the two track-4 submissions.
    """

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


def _find_one(patterns, roots=DEFAULT_ROOTS):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(root, "**", pat), recursive=True))
            if hits:
                return hits[0]
    return None


def find_data_paths(roots=DEFAULT_ROOTS) -> DataPaths:
    """Locate every input file by name, wherever the dataset happens to sit.

    Replaces the notebook's DATA_DIR probe (§1), which tried a fixed list of
    relative directories and then joined hard-coded subpaths onto it.
    """
    return DataPaths(
        train=_find_one(["*train*.jsonl"], roots),
        dev=_find_one(["*dev*.jsonl"], roots),
        vocab=_find_one(["vocab.json"], roots),
        labels=_find_one(["class_labels.txt"], roots),
        raw_test=_find_one(["raw_sentences_test.txt"], roots),
        raw_test_ids=_find_one(["raw_sentences_test_ids.txt"], roots),
        sample_submission=_find_one(["sample_submission.csv"], roots),
        make_submission_py=_find_one(["make_submission.py"], roots),
    )
