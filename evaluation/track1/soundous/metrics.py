
import numpy as np
import pandas as pd


def compute_all_metrics(gold_seqs, pred_seqs):
    """Character-level metrics only. WER/WordAcc need word-boundary info -> use evaluate_predictions."""
    assert len(gold_seqs) == len(pred_seqs)
    total_chars, wrong_chars = 0, 0
    total_sents, wrong_sents = 0, 0

    for gold, pred in zip(gold_seqs, pred_seqs):
        L = min(len(gold), len(pred))
        sent_wrong = len(gold) != len(pred)
        for i in range(L):
            total_chars += 1
            if gold[i] != pred[i]:
                wrong_chars += 1
                sent_wrong = True
        total_sents += 1
        if sent_wrong:
            wrong_sents += 1

    cer = wrong_chars / max(total_chars, 1)
    sent_err = wrong_sents / max(total_sents, 1)
    return {
        "CER": cer, "DER": cer,  
        "WER": None, "Accuracy": 1 - cer, "WordAcc": None, "SentAcc": 1 - sent_err,
    }


def evaluate_predictions(chars_seqs, gold_seqs, pred_seqs, space_char=" "):
    """Full evaluator (adds WER/WordAcc using `chars` for word-boundary/space positions). This is
    what training loops and the final test-set evaluation script both use."""
    metrics = compute_all_metrics(gold_seqs, pred_seqs)

    total_words, wrong_words = 0, 0
    for chars, gold, pred in zip(chars_seqs, gold_seqs, pred_seqs):
        L = min(len(chars), len(gold), len(pred))
        word_ok = True
        for i in range(L):
            if chars[i] == space_char:
                total_words += 1
                if not word_ok:
                    wrong_words += 1
                word_ok = True
            elif gold[i] != pred[i]:
                word_ok = False
        total_words += 1
        if not word_ok:
            wrong_words += 1

    wer = wrong_words / max(total_words, 1)
    metrics["WER"] = wer
    metrics["WordAcc"] = 1 - wer
    return metrics


def per_class_report(gold_seqs, pred_seqs, class_labels):
    C = len(class_labels)
    confusion = np.zeros((C, C), dtype=np.int64)
    for gold, pred in zip(gold_seqs, pred_seqs):
        for g, p in zip(gold, pred):
            confusion[g, p] += 1

    rows = []
    for c in range(C):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        support = confusion[c, :].sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append({"class": class_labels[c], "support": int(support),
                     "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows).sort_values("support", ascending=False), confusion
