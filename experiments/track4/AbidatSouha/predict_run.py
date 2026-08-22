"""Inference and submission files (notebook §14, §15, §16).

Writes two submissions: `submission.csv` from the raw ensemble, and
`submission_v2.csv` after the confidence-gated lexical fallback. Both are
verified against sample_submission.csv before being written.

Takes the trained models as arguments, the same way
experiments/track4/SmailRoumaissa/predict_run.py does -- run train_run first
and pass its result in.
"""

import os

from configs.track4.AbidatSouha.paths import find_data_paths
from configs.track4.AbidatSouha.training_config import LexicalFallbackConfig
from evaluation.track4.AbidatSouha.inference import (
    ensemble_predict,
    load_test_set,
    predict_with_confidence,
)
from evaluation.track4.AbidatSouha.lexical_fallback import apply_fallback
from evaluation.track4.AbidatSouha.submission import force_space_labels, write_submission
from utils.track4.AbidatSouha.device import get_device
from utils.track4.AbidatSouha.render import render


def preview(sent_ids, test_enc, preds, n=20):
    "Print the first n vocalized test sentences (§15)."
    print(f"\n{'sent_id':<10}{'input (undiacritized)':<45}predicted vocalization")
    print("-" * 110)
    for sid, t, p in list(zip(sent_ids, test_enc, preds))[:n]:
        print(f"{sid:<10}{''.join(t['chars']):<45}{render(t['chars'], p)}")


def run_prediction(models, data, fallback_cfg: LexicalFallbackConfig = None,
                   device: str = None, out_dir: str = ".", n_show: int = 20):
    device = device or get_device()
    paths = find_data_paths()
    missing = paths.missing()
    if missing:
        print("MISSING:", missing)
        return None

    fallback_cfg = fallback_cfg or LexicalFallbackConfig()
    sent_ids, test_enc = load_test_set(paths, data)

    # ---- §14 raw ensemble submission ---------------------------------------
    preds = ensemble_predict(models, data, test_enc, device)
    rows = write_submission(sent_ids, test_enc, preds, paths.sample_submission,
                            os.path.join(out_dir, "submission.csv"),
                            os.path.join(out_dir, "test_vocalized.txt"))

    # ---- §15 eyeball the output --------------------------------------------
    if n_show:
        preview(sent_ids, test_enc, preds, n_show)

    # ---- §16 confidence-gated lexical fallback ------------------------------
    labels, conf = predict_with_confidence(models, data, test_enc, device)
    preds_v2 = force_space_labels(test_enc,
                                  apply_fallback(test_enc, labels, conf, data, fallback_cfg))

    # counted over letters only, after both sides have had spaces zeroed; the
    # notebook compared before zeroing v2, which also counted space positions
    n_letters = n_changed = 0
    for t, a, b in zip(test_enc, preds, preds_v2):
        for c, x, y in zip(t["chars"], a, b):
            if c != " ":
                n_letters += 1
                n_changed += (x != y)
    print(f"\nV2 changed {n_changed} of {n_letters} test letter predictions "
          f"({100*n_changed/n_letters:.2f}%)")

    rows_v2 = write_submission(sent_ids, test_enc, preds_v2, paths.sample_submission,
                               os.path.join(out_dir, "submission_v2.csv"),
                               os.path.join(out_dir, "test_vocalized_v2.txt"))
    return dict(sent_ids=sent_ids, test_enc=test_enc,
                preds=preds, preds_v2=preds_v2, rows=rows, rows_v2=rows_v2)


if __name__ == "__main__":
    from experiments.track4.AbidatSouha.train_run import run_training

    result = run_training()
    if result:
        run_prediction(result["models"], result["data"], result["fallback_cfg"],
                       result["device"])
