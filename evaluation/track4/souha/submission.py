"""Submission files (notebook §14 and §16).

The `Id` format is `{sent_id}_{index}`, where `index` is the position in the
whitespace-normalised raw test line, **space positions omitted**. That matches
Data/test_data/make_submission.py exactly. `verify_against_sample` checks the
row set against sample_submission.csv before anything is written, so a mismatch
fails locally rather than on the leaderboard.

`test_vocalized.txt` is written alongside the CSV so the organisers' own script
can independently re-derive it:

    python make_submission.py --ids raw_sentences_test_ids.txt \\
        --input raw_sentences_test.txt --pred test_vocalized.txt \\
        --out submission_check.csv

Note the repo .gitignore excludes *.csv and *.txt, so these outputs stay local
by design -- report the scores in documentation/track4/souha/README.md instead.
"""

import collections
import csv
import os

from utils.track4.souha.render import render


def force_space_labels(test_enc, preds):
    "Force label 0 on spaces -- never scored, but keeps the output clean."
    for chars, p in zip((t["chars"] for t in test_enc), preds):
        for k, c in enumerate(chars):
            if c == " ":
                p[k] = 0
    return preds


def write_vocalized(path, test_enc, preds):
    with open(path, "w", encoding="utf-8") as f:
        for t, p in zip(test_enc, preds):
            f.write(render(t["chars"], p) + "\n")
    return path


def build_rows(sent_ids, test_enc, preds):
    rows = []
    for sid, t, p in zip(sent_ids, test_enc, preds):
        for idx, (c, l) in enumerate(zip(t["chars"], p)):
            if c == " ":
                continue
            rows.append((f"{sid}_{idx}", int(l)))
    return rows


def verify_against_sample(rows, sample_path, verbose=True):
    "Assert the Id set matches sample_submission.csv, and adopt its row order."
    if not sample_path or not os.path.exists(sample_path):
        return rows
    want = [r[0] for r in list(csv.reader(open(sample_path, encoding="utf-8")))[1:]]
    got = [r[0] for r in rows]
    assert len(want) == len(got), f"row count: expected {len(want)}, got {len(got)}"
    assert set(want) == set(got), "Id set does not match sample_submission.csv"
    order = {i: k for k, i in enumerate(want)}
    rows = sorted(rows, key=lambda r: order[r[0]])
    if verbose:
        print(f"verified against sample_submission.csv: {len(rows)} rows, ids match")
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Label"])
        w.writerows(rows)
    return path


def print_label_distribution(rows):
    d = collections.Counter(l for _, l in rows)
    print("\nlabel distribution of predictions:")
    for k in sorted(d):
        print(f"  {k:2d}: {100*d[k]/len(rows):5.2f}%")


def write_submission(sent_ids, test_enc, preds, sample_path=None,
                     sub_path="submission.csv", voc_path="test_vocalized.txt",
                     verbose=True):
    """Full §14 flow: zero the spaces, write both files, verify, report."""
    preds = force_space_labels(test_enc, preds)
    write_vocalized(voc_path, test_enc, preds)
    rows = build_rows(sent_ids, test_enc, preds)
    rows = verify_against_sample(rows, sample_path, verbose)
    write_csv(sub_path, rows)
    if verbose:
        print("wrote", os.path.abspath(sub_path))
        print("wrote", os.path.abspath(voc_path))
        print_label_distribution(rows)
    return rows
