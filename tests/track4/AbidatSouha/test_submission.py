"""Submission format checks (notebook §14).

The notebook asserted the Id set against sample_submission.csv inline, at the
moment of writing the file. Here it is a test, so the format can be checked
without a trained model -- dummy labels are enough, because only the Ids are
under test.

Runs under pytest, or standalone with `python tests/track4/AbidatSouha/test_submission.py`
from the repo root. Skipped when the dataset is not reachable.
"""

import csv
from pathlib import Path

from configs.track4.AbidatSouha.paths import find_data_paths
from evaluation.track4.AbidatSouha.inference import load_test_set
from evaluation.track4.AbidatSouha.submission import build_rows, verify_against_sample
from utils.track4.AbidatSouha.data import DiacData

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
    with Path(f["paths"].sample_submission).open(encoding="utf-8") as stream:
        want = [row[0] for row in list(csv.reader(stream))[1:]]
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


def test_sample_validation_rejects_wrong_row_count(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("Id,Label\n000001_0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row count"):
        verify_against_sample([], sample, verbose=False)


def test_sample_validation_rejects_wrong_ids(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("Id,Label\n000001_0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Id set"):
        verify_against_sample([("000001_1", 0)], sample, verbose=False)


if __name__ == "__main__":
    test_ids_match_sample_submission(); print("  [PASS] ids match sample_submission.csv")
    test_no_space_positions_emitted(); print("  [PASS] no space positions emitted")
    test_ids_index_into_the_raw_line(); print("  [PASS] ids index the raw line")
