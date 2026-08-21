from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from models.track4.Ines.dual_stream_crf_head_model import (
    Track4DualStreamCRF, majority_vote_decode,
)


class Evaluator:
    '''class_names: the 16 scored diacritic class names.
    Space and pad positions are excluded from every metric below via the
    dataloader's (pad_mask, is_space) tensors -- they're structural signals,
    not scored diacritic classes.'''

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)

    @torch.no_grad()
    def evaluate(self, models: List[torch.nn.Module], loader) -> Dict:
        device = next(models[0].parameters()).device
        y_true: List[int] = []
        y_pred: List[int] = []

        for m in models:
            m.eval()

        for batch in loader:
            ids = batch["ids"].to(device)
            labels = batch["labels"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            is_space = batch["is_space"].to(device)

            if len(models) == 1:
                preds = models[0].predict(ids, pad_mask, is_space)
            else:
                preds = majority_vote_decode(models, ids, pad_mask, is_space)

            letter_mask = (~pad_mask) & (~is_space)
            y_true.extend(labels[letter_mask].cpu().tolist())
            y_pred.extend(preds[letter_mask].cpu().tolist())

        if not y_true:
            raise ValueError("Evaluator.evaluate(): no letter positions found in loader.")

        micro_f1 = f1_score(y_true, y_pred, labels=list(range(self.num_classes)), average="micro")
        macro_f1 = f1_score(y_true, y_pred, labels=list(range(self.num_classes)), average="macro")
        precision, recall, per_class_f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(self.num_classes)), zero_division=0)
        cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        report_str = classification_report(
            y_true, y_pred, labels=list(range(self.num_classes)),
            target_names=self.class_names, zero_division=0)

        return {
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "per_class_f1": {name: float(f) for name, f in zip(self.class_names, per_class_f1)},
            "confusion_matrix": cm,
            "classification_report": report_str,
            "n_models_ensembled": len(models),
            "n_chars_evaluated": len(y_true),
        }

    def plot_confusion(self, confusion_matrix: np.ndarray):
        fig, ax = plt.subplots(figsize=(8, 7))
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(confusion_matrix, np.clip(row_sums, 1, None))
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels(self.class_names, rotation=75, ha="right")
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix (row-normalized)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig


def word_level_metrics_from_predict_fn(predict_fn, records: List[dict]) -> Dict:
    '''DER / DER* / WER / WER*, track-agnostic given a predict_fn(chars) -> List[int].
    Same definition used in track3's evaluate_bilstm_crf_head.py.'''
    SPACE_CHAR = " "
    total_chars = char_errors = 0
    total_chars_star = char_errors_star = 0
    total_words = word_errors = 0
    total_words_star = word_errors_star = 0

    for rec in records:
        chars, labels = rec["chars"], rec["labels"]
        preds = predict_fn(chars)

        words, cur = [], []
        for i, c in enumerate(chars):
            if c == SPACE_CHAR:
                if cur:
                    words.append(cur)
                cur = []
            else:
                cur.append((preds[i], labels[i]))
        if cur:
            words.append(cur)

        for word in words:
            if not word:
                continue
            n = len(word)
            errs = [p != t for p, t in word]

            total_chars += n
            char_errors += sum(errs)
            total_words += 1
            word_errors += int(any(errs))

            if n > 1:
                total_chars_star += n - 1
                char_errors_star += sum(errs[:-1])
                total_words_star += 1
                word_errors_star += int(any(errs[:-1]))

    return {
        "DER": char_errors / max(total_chars, 1),
        "DER_star": char_errors_star / max(total_chars_star, 1),
        "WER": word_errors / max(total_words, 1),
        "WER_star": word_errors_star / max(total_words_star, 1),
        "n_chars": total_chars, "n_words": total_words,
    }


# ---------------------------------------------------------------------------
# Standalone CLI: evaluate a saved checkpoint without re-running training.
#   python evaluation/track4/Ines/evaluate_dual_stream_crf_head.py \
#       --checkpoint working/dscat_best.pt
# ---------------------------------------------------------------------------
def _find_data_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    for candidate in [Path("./data"), Path("/kaggle/input")]:
        if candidate.exists():
            if (candidate / "vocab.json").exists():
                return candidate
            for sub in candidate.rglob("vocab.json"):
                return sub.parent
    raise FileNotFoundError("Could not auto-locate a data/ directory; pass --data-dir explicitly.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a Track 4 dual_stream_crf_head checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    from training.track4.Ines.finetune_dual_stream_crf_head import (
        Config, load_vocab, build_dataloaders, make_collate_fn,
    )

    data_root = _find_data_root(args.data_dir)
    print(f"Using data root: {data_root}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("cfg", {})
    print(f"Loaded checkpoint: {args.checkpoint} (dev f1 at save time: {ckpt.get('f1')})")

    cfg = Config(**{**Config().__dict__, **saved_cfg})
    cfg.data_root = str(data_root)
    char2id = load_vocab(str(data_root / "vocab.json"))
    pad_id = len(char2id)
    vocab_size = len(char2id) + 1
    model = Track4DualStreamCRF(
        vocab_size=vocab_size, num_labels=cfg.num_labels, dim=cfg.dim, n_heads=cfg.n_heads,
        local_layers=cfg.local_layers, global_layers=cfg.global_layers, final_layers=cfg.final_layers,
        local_window=cfg.local_window, dropout=cfg.dropout, max_seq_len=cfg.max_seq_len,
        unscored_label_id=0,
    ).to(cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, dev_loader = build_dataloaders(cfg, char2id, pad_id)
    class_names = [str(i) for i in range(cfg.num_labels)]  # swap in real diacritic names if you have them
    evaluator = Evaluator(class_names)
    report = evaluator.evaluate([model], dev_loader)

    print(f"micro_f1={report['micro_f1']:.4f}  macro_f1={report['macro_f1']:.4f}  "
          f"n_chars={report['n_chars_evaluated']}")
    print(report["classification_report"])


if __name__ == "__main__":
    main()
