"""Submission format checks (notebook §14).

The notebook asserted the Id set against sample_submission.csv inline, at the
moment of writing the file. Here it is a test, so the format can be checked
without a trained model -- dummy labels are enough, because only the Ids are
under test.

Runs under pytest, or standalone with `python tests/track4/souha/test_submission.py`
from the repo root. Skipped when the dataset is not reachable.
"""

import csv

from configs.track4.souha.paths import find_data_paths
from evaluation.track4.souha.inference import load_test_set
from evaluation.track4.souha.submission import build_rows, verify_against_sample
from utils.track4.souha.data import DiacData

try:
    import pytest
except ImportError:                     # standalone run, pytest not installed
    pytest = None

_CACHE = {}


def _skip(msg):
    if pytest is not None:
        pytest.skip(msg, allow_module_level=True)
    raise SystemExit(f"SKIP: {msg}")


def _fixture():
    "Load the test set once and reuse it across the checks below."
    if not _CACHE:
        paths = find_data_paths()
        if paths.missing():
            _skip(f"dataset not found; missing {paths.missing()}")
        data = DiacData(paths)
        sent_ids, test_enc = load_test_set(paths, data)
        _CACHE.update(paths=paths, sent_ids=sent_ids, test_enc=test_enc)
    return _CACHE


def _dummy_rows(f):
    dummy = [[0] * len(t["ids"]) for t in f["test_enc"]]
    return build_rows(f["sent_ids"], f["test_enc"], dummy)


def test_ids_match_sample_submission():
    f = _fixture()
    ordered = verify_against_sample(_dummy_rows(f), f["paths"].sample_submission,
                                    verbose=False)
    want = [r[0] for r in list(csv.reader(open(f["paths"].sample_submission,
                                               encoding="utf-8")))[1:]]
    assert [r[0] for r in ordered] == want, "row order does not match the sample"


def test_no_space_positions_emitted():
    f = _fixture()
    n_letters = sum(1 for t in f["test_enc"] for c in t["chars"] if c != " ")
    assert len(_dummy_rows(f)) == n_letters, "one row per non-space character expected"


def test_ids_index_into_the_raw_line():
    """`{sent_id}_{index}` must index the whitespace-normalised line."""
    f = _fixture()
    by_sent = {}
    for rid, _ in _dummy_rows(f):
        sid, idx = rid.rsplit("_", 1)
        by_sent.setdefault(sid, []).append(int(idx))
    for sid, t in zip(f["sent_ids"], f["test_enc"]):
        expected = [i for i, c in enumerate(t["chars"]) if c != " "]
        assert by_sent[sid] == expected, f"indices wrong for sentence {sid}"


if __name__ == "__main__":
    test_ids_match_sample_submission(); print("  [PASS] ids match sample_submission.csv")
    test_no_space_positions_emitted(); print("  [PASS] no space positions emitted")
    test_ids_index_into_the_raw_line(); print("  [PASS] ids index the raw line")
