
import os
import subprocess
import sys

import pandas as pd
import torch

from utils.track1.soundous.diacritics import clean_and_tokenize, chunk_chars, reconstruct_vocalized


@torch.no_grad()
def infer_file(model, input_path, output_path, char2idx, unk_idx, device, p1_module=None):
    model.eval()
    with open(input_path, encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f]

    vocalized_lines = []
    for line in raw_lines:
        if not line.strip():
            vocalized_lines.append("")
            continue
        chars = clean_and_tokenize(line, p1_module=p1_module)
        chunk_labels = []
        for chunk in chunk_chars(chars):
            ids = torch.tensor([[char2idx.get(c, unk_idx) for c in chunk]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            lengths = torch.tensor([len(chunk)])
            preds = model.decode(ids, mask, lengths)[0]
            chunk_labels.extend(preds)
        vocalized_lines.append(reconstruct_vocalized(chars, chunk_labels))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocalized_lines) + "\n")
    print(f"Wrote {len(vocalized_lines)} vocalized lines -> {output_path}")
    return vocalized_lines


def run_make_submission(make_submission_py, ids_path, input_path, pred_path, out_csv):
    cmd = [sys.executable, make_submission_py, "--ids", ids_path, "--input", input_path,
           "--pred", pred_path, "--out", out_csv]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"make_submission.py failed for {pred_path} -- see stderr above.")
    return out_csv


def generate_submission(model, exp_name, paths, char2idx, unk_idx, device, p1_module=None):
    """paths: dict with keys raw_test_txt, raw_test_ids_txt, make_submission_py, sample_submission,
    submissions_dir. Returns (voc_txt_path, submission_csv_path)."""
    out_dir = os.path.join(paths["submissions_dir"], exp_name)
    os.makedirs(out_dir, exist_ok=True)
    voc_txt = os.path.join(out_dir, f"{exp_name}_voc.txt")
    sub_csv = os.path.join(out_dir, f"{exp_name}_submission.csv")

    vocalized = infer_file(model, paths["raw_test_txt"], voc_txt, char2idx, unk_idx, device, p1_module)

    with open(paths["raw_test_ids_txt"], encoding="utf-8") as f:
        n_ids = sum(1 for _ in f)
    with open(paths["raw_test_txt"], encoding="utf-8") as f:
        n_inputs = sum(1 for _ in f)
    assert len(vocalized) == n_ids == n_inputs, (
        f"[{exp_name}] line count mismatch: preds={len(vocalized)}, ids={n_ids}, inputs={n_inputs}"
    )

    run_make_submission(paths["make_submission_py"], paths["raw_test_ids_txt"],
                         paths["raw_test_txt"], voc_txt, sub_csv)

    sub_df = pd.read_csv(sub_csv)
    sample_df = pd.read_csv(paths["sample_submission"])
    assert list(sub_df.columns) == list(sample_df.columns) == ["Id", "Label"], f"schema mismatch for {exp_name}"
    print(f"[{exp_name}] {len(sub_df)} rows -> {sub_csv}")
    return voc_txt, sub_csv


@torch.no_grad()
def ensemble_decode(models, char_ids, mask, lengths):
    for m in models:
        m.eval()
    all_emissions = [m._encode(char_ids, mask, lengths) for m in models]
    avg_emissions = torch.stack(all_emissions, dim=0).mean(dim=0)
    crf_model = next(m for m in models if m.use_crf)
    return crf_model.crf.decode(avg_emissions, mask)


@torch.no_grad()
def infer_file_ensemble(models, input_path, output_path, char2idx, unk_idx, device, p1_module=None):
    for m in models:
        m.eval()
    with open(input_path, encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f]

    vocalized_lines = []
    for line in raw_lines:
        if not line.strip():
            vocalized_lines.append("")
            continue
        chars = clean_and_tokenize(line, p1_module=p1_module)
        chunk_labels = []
        for chunk in chunk_chars(chars):
            ids = torch.tensor([[char2idx.get(c, unk_idx) for c in chunk]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            lengths = torch.tensor([len(chunk)])
            preds = ensemble_decode(models, ids, mask, lengths)[0]
            chunk_labels.extend(preds)
        vocalized_lines.append(reconstruct_vocalized(chars, chunk_labels))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocalized_lines) + "\n")
    print(f"Wrote {len(vocalized_lines)} ensembled vocalized lines -> {output_path}")
    return vocalized_lines


def generate_submission_ensemble(models, exp_name, paths, char2idx, unk_idx, device, p1_module=None):
    out_dir = os.path.join(paths["submissions_dir"], exp_name)
    os.makedirs(out_dir, exist_ok=True)
    voc_txt = os.path.join(out_dir, f"{exp_name}_voc.txt")
    sub_csv = os.path.join(out_dir, f"{exp_name}_submission.csv")

    vocalized = infer_file_ensemble(models, paths["raw_test_txt"], voc_txt, char2idx, unk_idx, device, p1_module)
    with open(paths["raw_test_ids_txt"], encoding="utf-8") as f:
        n_ids = sum(1 for _ in f)
    assert len(vocalized) == n_ids

    run_make_submission(paths["make_submission_py"], paths["raw_test_ids_txt"],
                         paths["raw_test_txt"], voc_txt, sub_csv)
    print(f"[{exp_name}] ensemble submission -> {sub_csv}")
    return voc_txt, sub_csv


NUM_CLASSES_DEFAULT = 16


def _multi_chunk_offsets(chars, max_len, num_offsets=3):
    variants = []
    for k in range(num_offsets):
        shift = (k * max_len) // num_offsets
        shifted = [" "] * shift + chars
        variants.append((shift, chunk_chars(shifted, max_len=max_len)))
    return variants


@torch.no_grad()
def tta_infer_long_sentence(model, chars, char2idx, unk_idx, device, num_classes,
                             max_len=300, num_offsets=3):
    if len(chars) <= max_len:
        ids = torch.tensor([[char2idx.get(c, unk_idx) for c in chars]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        return model._encode(ids, mask, torch.tensor([len(chars)]))[0]

    T = len(chars)
    summed = torch.zeros(T, num_classes, device=device)
    counts = torch.zeros(T, device=device)
    for shift, chunks in _multi_chunk_offsets(chars, max_len, num_offsets):
        pos = -shift
        for chunk in chunks:
            ids = torch.tensor([[char2idx.get(c, unk_idx) for c in chunk]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            emissions = model._encode(ids, mask, torch.tensor([len(chunk)]))[0]
            for j in range(len(chunk)):
                orig_pos = pos + j
                if 0 <= orig_pos < T:
                    summed[orig_pos] += emissions[j]
                    counts[orig_pos] += 1
            pos += len(chunk)
    return summed / counts.clamp(min=1).unsqueeze(-1)


@torch.no_grad()
def infer_file_tta(model, input_path, output_path, char2idx, unk_idx, device, num_classes,
                    p1_module=None, max_len=300, num_offsets=3):
    model.eval()
    with open(input_path, encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f]

    vocalized_lines = []
    for line in raw_lines:
        if not line.strip():
            vocalized_lines.append("")
            continue
        chars = clean_and_tokenize(line, p1_module=p1_module)
        avg_emissions = tta_infer_long_sentence(model, chars, char2idx, unk_idx, device,
                                                 num_classes, max_len, num_offsets).unsqueeze(0)
        mask = torch.ones(1, avg_emissions.size(1), dtype=torch.bool, device=device)
        path = model.crf.decode(avg_emissions, mask)[0] if model.use_crf else avg_emissions.argmax(-1)[0].tolist()
        vocalized_lines.append(reconstruct_vocalized(chars, path))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocalized_lines) + "\n")
    print(f"Wrote {len(vocalized_lines)} TTA vocalized lines -> {output_path}")
    return vocalized_lines


def generate_submission_tta(model, exp_name, paths, char2idx, unk_idx, device, num_classes, p1_module=None):
    out_dir = os.path.join(paths["submissions_dir"], exp_name)
    os.makedirs(out_dir, exist_ok=True)
    voc_txt = os.path.join(out_dir, f"{exp_name}_voc.txt")
    sub_csv = os.path.join(out_dir, f"{exp_name}_submission.csv")

    vocalized = infer_file_tta(model, paths["raw_test_txt"], voc_txt, char2idx, unk_idx, device,
                                num_classes, p1_module)
    with open(paths["raw_test_ids_txt"], encoding="utf-8") as f:
        n_ids = sum(1 for _ in f)
    assert len(vocalized) == n_ids

    run_make_submission(paths["make_submission_py"], paths["raw_test_ids_txt"],
                         paths["raw_test_txt"], voc_txt, sub_csv)
    print(f"[{exp_name}] TTA submission -> {sub_csv}")
    return voc_txt, sub_csv
