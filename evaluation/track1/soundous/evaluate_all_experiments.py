"""Evaluates every trained Track 1 experiment and reports metrics + generates test-set submissions
for all of them in one run.

For each experiment found under outputs/checkpoints/<exp_name>/best_model.pt:
  1. Reconstructs the exact model (class + kwargs stored in the checkpoint) and loads its weights.
  2. Re-evaluates it on the DEV set (fresh forward pass -- CER/DER/WER/Accuracy/WordAcc/SentAcc,
     via the same evaluate_predictions() used during training) so this report is self-contained
     and doesn't just trust each experiment's own history.json.
  3. Runs inference on the released TEST set (raw_sentences_test.txt) and produces an official
     submission.csv via the organizers' make_submission.py, saved under
     outputs/submissions/<exp_name>/.
  4. Writes a per-class precision/recall/F1 report for that experiment.

The multi-seed ensemble (exp_ensemble/seed_*/) and TTA (applied to the base bilstm_cnn_crf
checkpoint) are handled as special cases since they don't fit the "one checkpoint -> one model"
pattern.

At the end, writes track1_experiment_log.csv (one row per experiment) and prints a summary table
sorted by dev DER 
Run: python evaluation/track1/soundous/evaluate_all_experiments.py
"""
import glob
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import torch

from utils.track1.soundous.paths import resolve_paths, print_paths
from utils.track1.soundous.vocab_utils import load_vocab, load_class_labels, NUM_CLASSES, NO_DIAC_IDX
from utils.track1.soundous.data_utils import read_jsonl, make_loader
from utils.track1.soundous.seed_utils import get_device, set_seed, SEED
from evaluation.track1.soundous.metrics import evaluate_predictions, per_class_report
from evaluation.track1.soundous.inference import (
    generate_submission, generate_submission_ensemble, generate_submission_tta,
)
from models.track1.soundous.tagger import DiacritizationTagger
from models.track1.soundous.experimental_taggers import MultiTaskDiacritizationTagger, AttnDiacritizationTagger

MODEL_CLASS_REGISTRY = {
    "DiacritizationTagger": DiacritizationTagger,
    "MultiTaskDiacritizationTagger": MultiTaskDiacritizationTagger,
    "AttnDiacritizationTagger": AttnDiacritizationTagger,
}


def load_checkpoint_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cls = MODEL_CLASS_REGISTRY[ckpt["model_class"]]
    model = cls(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("arch_name", os.path.basename(os.path.dirname(ckpt_path)))


@torch.no_grad()
def evaluate_on_dev(model, dev_loader, device):
    dev_preds, dev_golds, dev_chars = [], [], []
    for batch in dev_loader:
        char_ids = batch["char_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"]
        emissions = model._encode(char_ids, mask, lengths)
        preds = model.crf.decode(emissions, mask) if model.use_crf else emissions.argmax(-1)
        for i, L in enumerate(lengths.tolist()):
            p = preds[i][:L] if model.use_crf else preds[i, :L].tolist()
            dev_preds.append(p)
            dev_golds.append(labels[i, :L].tolist())
            dev_chars.append(batch["chars"][i][:L])
    metrics = evaluate_predictions(dev_chars, dev_golds, dev_preds)
    return metrics, dev_golds, dev_preds


@torch.no_grad()
def evaluate_ensemble_on_dev(models, dev_loader, device):
    from evaluation.track1.soundous.inference import ensemble_decode
    dev_preds, dev_golds, dev_chars = [], [], []
    for batch in dev_loader:
        char_ids = batch["char_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"]
        preds = ensemble_decode(models, char_ids, mask, lengths)
        for i, L in enumerate(lengths.tolist()):
            dev_preds.append(preds[i][:L])
            dev_golds.append(labels[i, :L].tolist())
            dev_chars.append(batch["chars"][i][:L])
    return evaluate_predictions(dev_chars, dev_golds, dev_preds), dev_golds, dev_preds


@torch.no_grad()
def evaluate_tta_on_dev(model, dev_rows, char2idx, unk_idx, device, num_classes, max_len=300, num_offsets=3):
    from evaluation.track1.soundous.inference import tta_infer_long_sentence
    dev_preds, dev_golds, dev_chars = [], [], []
    for row in dev_rows:
        chars = row["chars"]
        labels = row["labels"]
        if labels and isinstance(labels[0], str):
            labels = [0] * len(chars)  # defensive; dev labels are ints in this dataset
        avg_emissions = tta_infer_long_sentence(model, chars, char2idx, unk_idx, device,
                                                 num_classes, max_len, num_offsets).unsqueeze(0)
        mask = torch.ones(1, avg_emissions.size(1), dtype=torch.bool, device=device)
        path = model.crf.decode(avg_emissions, mask)[0] if model.use_crf else avg_emissions.argmax(-1)[0].tolist()
        dev_preds.append(path)
        dev_golds.append(labels)
        dev_chars.append(chars)
    return evaluate_predictions(dev_chars, dev_golds, dev_preds), dev_golds, dev_preds


def log_row(log, name, description, metrics, submission_csv=None, extra=None):
    row = {
        "Experiment": name, "Description": description,
        "DER": metrics["DER"], "WER": metrics["WER"], "CER": metrics["CER"],
        "Accuracy": metrics["Accuracy"], "WordAcc": metrics["WordAcc"], "SentAcc": metrics["SentAcc"],
        "submission_csv": submission_csv,
    }
    if extra:
        row.update(extra)
    log.append(row)


def main():
    set_seed(SEED)
    device = get_device()
    paths = resolve_paths()
    print_paths(paths)

    char2idx, idx2char, pad_idx, unk_idx, vocab_size = load_vocab(paths["vocab_path"])
    class_labels, label2idx, idx2label = load_class_labels(paths["labels_path"])
    dev_rows = read_jsonl(paths["dev_jsonl"])
    dev_loader = make_loader(dev_rows, char2idx, label2idx, pad_idx, unk_idx, NO_DIAC_IDX,
                              batch_size=64, shuffle=False)

    ckpt_root = paths["checkpoints_dir"]
    log = []
    per_class_dir = os.path.join(paths["output_dir"], "per_class_reports")
    os.makedirs(per_class_dir, exist_ok=True)

    DESCRIPTIONS = {
        "bilstm_cnn": "BiLSTM-CNN (no CRF)",
        "bilstm_crf": "BiLSTM-CRF (no CNN)",
        "bilstm_cnn_crf": "BiLSTM-CNN-CRF (P2 baseline architecture)",
        "exp_focal": "BiLSTM-CNN-CRF + focal-style emission reweighting (\u00a712.1)",
        "exp_multitask": "BiLSTM-CNN-CRF + auxiliary has-diacritic head (\u00a712.2)",
        "exp_attention": "BiLSTM-CNN-CRF + self-attention block (\u00a712.3)",
        "exp_consistency": "BiLSTM-CNN-CRF + consistency-regularized augmentation (\u00a712.4)",
        "exp_swa": "BiLSTM-CNN-CRF + Stochastic Weight Averaging (\u00a712.5)",
        "exp_curriculum": "BiLSTM-CNN-CRF + length-based curriculum learning (\u00a712.7)",
    }

    # ---- single-checkpoint experiments (everything except the ensemble) ----
    for exp_name, desc in DESCRIPTIONS.items():
        ckpt_path = os.path.join(ckpt_root, exp_name, "best_model.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[skip] {exp_name}: no checkpoint at {ckpt_path} (train it first).")
            continue

        print(f"\n=== Evaluating {exp_name} ===")
        model, _ = load_checkpoint_model(ckpt_path, device)
        metrics, dev_golds, dev_preds = evaluate_on_dev(model, dev_loader, device)
        print(f"  dev DER={metrics['DER']:.4f} WER={metrics['WER']:.4f} SentAcc={metrics['SentAcc']:.4f}")

        report_df, _ = per_class_report(dev_golds, dev_preds, class_labels)
        report_df.to_csv(os.path.join(per_class_dir, f"{exp_name}_per_class.csv"), index=False)

        _, sub_csv = generate_submission(model, exp_name, paths, char2idx, unk_idx, device)
        log_row(log, exp_name, desc, metrics, submission_csv=sub_csv)

    # ---- §12.6 multi-seed ensemble ----
    ensemble_dir = os.path.join(ckpt_root, "exp_ensemble")
    seed_ckpts = sorted(glob.glob(os.path.join(ensemble_dir, "seed_*", "best_model.pt")))
    if seed_ckpts:
        print(f"\n=== Evaluating exp_ensemble ({len(seed_ckpts)} seeds) ===")
        seed_models = [load_checkpoint_model(p, device)[0] for p in seed_ckpts]
        metrics, dev_golds, dev_preds = evaluate_ensemble_on_dev(seed_models, dev_loader, device)
        print(f"  dev DER={metrics['DER']:.4f} WER={metrics['WER']:.4f} SentAcc={metrics['SentAcc']:.4f}")

        report_df, _ = per_class_report(dev_golds, dev_preds, class_labels)
        report_df.to_csv(os.path.join(per_class_dir, "exp_ensemble_per_class.csv"), index=False)

        _, sub_csv = generate_submission_ensemble(seed_models, "exp_ensemble", paths, char2idx, unk_idx, device)
        log_row(log, "exp_ensemble",
                f"Multi-seed ensemble ({len(seed_ckpts)} seeds, averaged emissions, single CRF decode) (\u00a712.6)",
                metrics, submission_csv=sub_csv, extra={"n_seeds": len(seed_ckpts)})
    else:
        print(f"[skip] exp_ensemble: no seed checkpoints under {ensemble_dir} (train it first).")

    # ---- §12.8 TTA -- applied to the base bilstm_cnn_crf checkpoint, no training of its own ----
    tta_base_ckpt = os.path.join(ckpt_root, "bilstm_cnn_crf", "best_model.pt")
    if os.path.isfile(tta_base_ckpt):
        print("\n=== Evaluating exp_tta (multi-offset chunk-averaged inference on bilstm_cnn_crf) ===")
        model, _ = load_checkpoint_model(tta_base_ckpt, device)
        metrics, dev_golds, dev_preds = evaluate_tta_on_dev(model, dev_rows, char2idx, unk_idx, device, NUM_CLASSES)
        print(f"  dev DER={metrics['DER']:.4f} WER={metrics['WER']:.4f} SentAcc={metrics['SentAcc']:.4f}")

        report_df, _ = per_class_report(dev_golds, dev_preds, class_labels)
        report_df.to_csv(os.path.join(per_class_dir, "exp_tta_per_class.csv"), index=False)

        _, sub_csv = generate_submission_tta(model, "exp_tta", paths, char2idx, unk_idx, device, NUM_CLASSES)
        log_row(log, "exp_tta",
                "Base bilstm_cnn_crf + test-time multi-offset chunk averaging (\u00a712.8)",
                metrics, submission_csv=sub_csv)
    else:
        print(f"[skip] exp_tta: no base checkpoint at {tta_base_ckpt} (train bilstm_cnn_crf first).")

    # ---- final report ----
    if not log:
        print("\nNo checkpoints found anywhere under", ckpt_root, "-- nothing to report. Train something first.")
        return

    log_df = pd.DataFrame(log).sort_values("DER").reset_index(drop=True)
    out_csv = os.path.join(paths["output_dir"], "track1_experiment_log.csv")
    log_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 100)
    print("TRACK 1 -- FINAL EXPERIMENT REPORT (sorted by dev DER)")
    print("=" * 100)
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(log_df[["Experiment", "DER", "WER", "CER", "Accuracy", "WordAcc", "SentAcc"]].to_string(index=False))
    print(f"\nSaved -> {out_csv}")
    print(f"Per-class reports -> {per_class_dir}/")
    print(f"Test-set submissions -> {paths['submissions_dir']}/<experiment>/<experiment>_submission.csv")
    print("\nPaste the table above into documentation/track1/soundous/README.md's Results section.")


if __name__ == "__main__":
    main()
