import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from utils.track4.SmailRoumaissa.constants import SPACE
from utils.track4.SmailRoumaissa.render import render_sentence
from utils.track4.SmailRoumaissa.data import Vocab
from evaluation.track4.SmailRoumaissa.lexical_prior import LexicalPrior, fuse_sentence
from evaluation.track4.SmailRoumaissa.metrics import MicroF1Accumulator


def tokenize_raw(sentence: str) -> List[str]:
    """Keep every non-whitespace character, in order; only normalize
    whitespace (runs collapse to a single space, ends stripped). Do NOT
    filter by an Arabic Unicode range -- Algerian Arabic uses loanword
    letters (e.g. پ چ ڤ گ ژ) outside the standard ranges, and a regex
    whitelist silently drops them, which breaks make_submission.py's
    skeleton check."""
    chars, prev_space = [], True
    for ch in sentence.strip():
        if ch.isspace():
            if not prev_space and chars:
                chars.append(" "); prev_space = True
        else:
            chars.append(ch); prev_space = False
    while chars and chars[-1] == " ":
        chars.pop()
    return chars


def make_is_letter(chars, device):
    return torch.tensor([[False] + [c != " " for c in chars] + [False]],
                        dtype=torch.bool, device=device)


@torch.no_grad()
def predict_log_probs(model, vocab, chars, base_temperature, shadda_temperature, device):
    ids = [vocab.bos_id] + vocab.encode(chars) + [vocab.eos_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
    log_probs = model.marginal_log_probs(input_ids, attn_mask,
                                          base_temperature=base_temperature,
                                          shadda_temperature=shadda_temperature)[0]
    return log_probs[1:-1]


@torch.no_grad()
def decode_crf(model, chars, log_probs, device):
    is_letter = make_is_letter(chars, device)
    padded = torch.zeros(1, is_letter.size(1), log_probs.size(-1), device=device)
    padded[0, 1:-1] = log_probs
    preds = model.decode_from_log_probs(padded, is_letter)[0]
    return preds[1:-1]


def run_inference(model, vocab, base_temperature, shadda_temperature, raw_sentences_path, out_path,
                   lexical=None, entropy_threshold=1.0, gate_temperature=0.3, max_strength=2.0,
                   device="cpu"):
    model.eval()
    raw_lines = Path(raw_sentences_path).read_text(encoding="utf-8").splitlines()
    out_lines = []
    for line in raw_lines:
        chars = tokenize_raw(line)
        if not chars:
            out_lines.append("")
            continue
        log_probs = predict_log_probs(model, vocab, chars, base_temperature, shadda_temperature, device)
        if lexical is not None:
            log_probs = fuse_sentence(log_probs, chars, lexical,
                                       entropy_threshold=entropy_threshold,
                                       gate_temperature=gate_temperature,
                                       max_strength=max_strength)
        labels = decode_crf(model, chars, log_probs, device).tolist()
        out_lines.append(render_sentence(chars, labels))
    Path(out_path).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(out_lines)} diacritized sentences to {out_path}")


@torch.no_grad()
def evaluate_lexical_on_dev(model, vocab, dev_path, base_temperature, shadda_temperature, lexical, *,
                             entropy_threshold=1.0, gate_temperature=0.3, max_strength=2.0,
                             device="cpu"):
    model.eval()
    acc_neural = MicroF1Accumulator()
    acc_fused = MicroF1Accumulator()

    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chars, labels = rec["chars"], rec["labels"]
            if not chars:
                continue
            gold = [l for c, l in zip(chars, labels) if c != SPACE]

            log_probs = predict_log_probs(model, vocab, chars, base_temperature, shadda_temperature, device)

            neural_preds = decode_crf(model, chars, log_probs, device).tolist()
            neural_preds = [p for c, p in zip(chars, neural_preds) if c != SPACE]
            acc_neural.update(neural_preds, gold)

            fused_log_probs = fuse_sentence(log_probs, chars, lexical,
                                             entropy_threshold=entropy_threshold,
                                             gate_temperature=gate_temperature,
                                             max_strength=max_strength)
            fused_preds = decode_crf(model, chars, fused_log_probs, device).tolist()
            fused_preds = [p for c, p in zip(chars, fused_preds) if c != SPACE]
            acc_fused.update(fused_preds, gold)

    return acc_neural.score, acc_fused.score


@torch.no_grad()
def collect_dev_predictions(model, vocab, dev_path, base_temperature, shadda_temperature, lexical, *,
                             entropy_threshold, gate_temperature, max_strength, device="cpu"):
    """Same pass as evaluate_lexical_on_dev, but returns the raw (pred, gold)
    pairs instead of just a score, so we can build a confusion matrix."""
    model.eval()
    all_preds, all_gold = [], []

    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chars, labels = rec["chars"], rec["labels"]
            if not chars:
                continue
            gold = [l for c, l in zip(chars, labels) if c != SPACE]

            log_probs = predict_log_probs(model, vocab, chars, base_temperature, shadda_temperature, device)
            fused_log_probs = fuse_sentence(log_probs, chars, lexical,
                                             entropy_threshold=entropy_threshold,
                                             gate_temperature=gate_temperature,
                                             max_strength=max_strength)
            preds = decode_crf(model, chars, fused_log_probs, device).tolist()
            preds = [p for c, p in zip(chars, preds) if c != SPACE]

            all_preds.extend(preds)
            all_gold.extend(gold)

    return all_preds, all_gold
