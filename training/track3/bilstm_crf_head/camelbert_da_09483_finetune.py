# Auto-split from Raw Experiments/Bilstm head/camelbert_da-0.9483/camelbert-da-crf (1).ipynb for camelbert_da_09483 (track3/bilstm_crf_head)
# NOTE: shared imports/setup live in training/track3/bilstm_crf_head/camelbert_da_09483_train.py
# Sanity-check imports here before relying on this file standalone.

# --- Environment & setup (preamble cells before first ## section) ---


# ## 1. Environment & Reproducibility

import os, sys, json, re, glob, random, shutil, time, zipfile, unicodedata
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:
    os.system("pip install -q torch --index-url https://download.pytorch.org/whl/cu121")
    import torch

try:
    import transformers
except ImportError:
    os.system("pip install -q transformers")
    import transformers

try:
    import sklearn
except ImportError:
    os.system("pip install -q scikit-learn")
    import sklearn

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))



def seed_everything(seed: int) -> None:
    '''Full determinism across python/numpy/torch (CUDA kernels remain
    only approximately deterministic, which is expected/acceptable here).'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

SEED = 42
seed_everything(SEED)


# ## 2. Configuration

# ---------------------------------------------------------------------------
# 2.1  Path resolution (auto-detects Kaggle vs local layout)
# ---------------------------------------------------------------------------
def _first_existing(*candidates: str) -> Optional[Path]:
    for c in candidates:
        matches = glob.glob(c, recursive=True)
        if matches:
            return Path(matches[0])
    return None

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
_DATASET_DIR: Optional[Path] = None
if KAGGLE_INPUT_ROOT.exists():
    # search at any depth for a folder that contains both train_data/ and dev_data/
    # (handles both flat layouts and nested ones like .../Data/train_data)
    for candidate in KAGGLE_INPUT_ROOT.rglob("train_data"):
        parent = candidate.parent
        if (parent / "dev_data").exists():
            _DATASET_DIR = parent
            break
_DATASET_DIR = _DATASET_DIR or Path("./data")  # local fallback for dry-runs
print(f"Resolved dataset root: {_DATASET_DIR}")

@dataclass(frozen=True)
class Paths:
    root: Path = _DATASET_DIR
    train_jsonl: Path = field(default_factory=lambda: _first_existing(
        str(_DATASET_DIR / "train_data" / "*.jsonl")))
    dev_jsonl: Path = field(default_factory=lambda: _first_existing(
        str(_DATASET_DIR / "dev_data" / "*.jsonl")))
    raw_test_txt: Path = field(default_factory=lambda: _first_existing(
        str(_DATASET_DIR / "test_data" / "raw_sentences_test.txt"),   # exact match first
        str(_DATASET_DIR / "test_data" / "raw_sentences_test_[0-9]*.txt")))
    raw_test_ids: Path = field(default_factory=lambda: _first_existing(
        str(_DATASET_DIR / "test_data" / "raw_sentences_test_ids*.txt")))
    make_submission_py: Path = field(default_factory=lambda: _first_existing(
        str(_DATASET_DIR / "test_data" / "make_submission.py")))
    vocab_json: Path = field(default_factory=lambda: _first_existing(
        str(_DATASET_DIR / "vocab.json")))
    class_labels_txt: Path = field(default_factory=lambda: _first_existing(
        str(_DATASET_DIR / "class_labels.txt")))
    work_dir: Path = Path("/kaggle/working" if Path("/kaggle/working").exists() else "./working")
    checkpoint_dir: Path = field(init=False)
    export_dir: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "checkpoint_dir", self.work_dir / "checkpoints")
        object.__setattr__(self, "export_dir", self.work_dir / "exports")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

PATHS = Paths()
for f in ["train_jsonl", "dev_jsonl", "raw_test_txt", "raw_test_ids",
          "make_submission_py", "vocab_json", "class_labels_txt"]:
    val = getattr(PATHS, f)
    status = "OK" if val is not None else "NOT FOUND"
    print(f"{f:20s}: {val}  [{status}]")



# ---------------------------------------------------------------------------
# 2.2  Model registry — Track 3-legal backbones only (subword-tokenized,
# Arabic-pretrained). Add/remove entries here; nothing else changes.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, str] = {
    "camelbert_da":   "CAMeL-Lab/bert-base-arabic-camelbert-da",
    "camelbert_mix":  "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "arabert_v02":    "aubmindlab/bert-base-arabertv02",
    "marbert":        "UBC-NLP/MARBERT",
    "dziribert":      "alger-ia/dziribert",
}

# <<< CHANGE THIS to switch which model this Kaggle session trains >>>
ACTIVE_MODEL: str = "camelbert_da"

assert ACTIVE_MODEL in MODEL_REGISTRY, f"{ACTIVE_MODEL} not in MODEL_REGISTRY"
BACKBONE_NAME = MODEL_REGISTRY[ACTIVE_MODEL]
RUN_ID = ACTIVE_MODEL
print(f"Active model  : {ACTIVE_MODEL}")
print(f"HF checkpoint : {BACKBONE_NAME}")



# ---------------------------------------------------------------------------
# 2.3  Hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    run_id: str = RUN_ID
    backbone: str = BACKBONE_NAME
    num_classes: int = 16                 # the 16 scored diacritic classes
    space_label: int = 16                 # extra internal class for the space
                                           # character -- gives the BiLSTM/CRF an
                                           # explicit word-boundary signal in the
                                           # sequence itself; always stripped back
                                           # out before scoring (never touches the
                                           # 16-class metrics)
    num_labels: int = 17                  # num_classes + 1 (space)
    char_emb_dim: int = 64
    n_pool_layers: int = 4                # learned softmax mix over the top N
                                           # backbone hidden-state layers (ELMo-style)
                                           # instead of a fixed concat
    lstm_hidden_dim: int = 384            # BiLSTM total hidden size (192/direction)
    num_lstm_layers: int = 2
    head_dropout: float = 0.30
    use_crf: bool = True                  # False -> plain weighted-CE ablation,
                                           # same head minus structured decoding
    aux_loss_weight: float = 0.3          # weight of the auxiliary per-position
                                           # weighted CE loss, added to the CRF NLL
                                           # (CRF's own NLL doesn't support per-class
                                           # weighting directly, so the auxiliary
                                           # loss is what carries the rare-class
                                           # up-weighting into the gradient)
    freeze_embeddings: bool = True        # freeze the backbone's input embedding
                                           # table (light regularization)
    freeze_n_layers: int = 0              # additionally freeze the first N
                                           # backbone transformer layers
    max_subword_len: int = 256            # cap for *training* batches only --
                                           # inference never truncates (Section 11
                                           # recursively splits at space boundaries
                                           # instead; see MAX_LEN derivation below)
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 25
    lr_head: float = 1e-3
    lr_backbone_top: float = 2e-5
    layerwise_decay: float = 0.9
    weight_decay: float = 0.08
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    use_focal_loss: bool = True           # applies to the auxiliary CE loss only
                                           # (CRF NLL itself can't use focal weighting)
    focal_gamma: float = 2.0
    rare_class_ids: Tuple[int, ...] = (9, 10, 11, 12, 13, 14, 15)
    rare_class_weight: float = 3.0
    class_weight_cap: float = 10.0        # inverse-frequency class weights,
                                           # clipped so the rarest classes don't
                                           # dominate the auxiliary loss entirely
    use_rdrop: bool = True                # applied to the auxiliary logits only
                                           # (well-defined for a softmax output;
                                           # not applied to the CRF path itself)
    rdrop_alpha: float = 1.0
    early_stop_metric: str = "dev_loss"
    early_stop_patience: int = 2
    seed: int = SEED
    dev_split_for_test: float = 0.20      # kept from this team's prior notebook --
                                           # the source notebook selected its best
                                           # checkpoint AND reported its final score
                                           # from the same dev set, which risks a
                                           # mild optimistic bias. This held-out
                                           # DEV_TEST split (never touched during
                                           # training decisions) avoids that.
    max_train_minutes: int = 480
    checkpoint_every_steps: int = 500
    self_training_enabled: bool = True
    self_training_conf_threshold: float = 0.90
    self_training_epochs: int = 4
    k_folds: int = 5
    export_include_weights: bool = True    # embed best.pt in the exported zip --
                                            # required if you plan to ensemble this
                                            # run together with others trained in
                                            # separate Kaggle sessions

CFG = TrainConfig()

# To A/B a variant against a previous run without overwriting its
# checkpoint/export, override the run id here, e.g.:
#   CFG.run_id = f"{ACTIVE_MODEL}_v2"
RUN_ID = CFG.run_id

print(json.dumps(asdict(CFG), indent=2, default=str))


# ## 3. Data Loading & Normalization

# ---------------------------------------------------------------------------
# 3.1  Diacritics constants (verbatim from the team's preprocessing notebook)
# ---------------------------------------------------------------------------
DIACRITICS = {
    'FATHAH': '\u064E', 'DAMMAH': '\u064F', 'KASRAH': '\u0650', 'SUKOON': '\u0652',
    'SHADDA': '\u0651', 'FATHATAN': '\u064B', 'DAMMATAN': '\u064C', 'KASRATAN': '\u064D',
    'SUPERSCRIPT_ALEF': '\u0670', 'MADDAH': '\u0653', 'HAMZA_ABOVE': '\u0654', 'HAMZA_BELOW': '\u0655',
}
DIACRITIC_CHARS = set(DIACRITICS.values())
DIACRITIC_STR = ''.join(sorted(DIACRITIC_CHARS))
SHADDA = '\u0651'
SHORT_VOWELS_STR = '\u064E\u064F\u0650\u064B\u064C\u064D\u0670'
ARABIC_LETTERS_STR = (
    '\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631'
    '\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0641\u0642\u0643\u0644\u0645\u0646\u0647'
    '\u0648\u0649\u064A\u0629\u0671'
)
ARABIC_LETTERS_SET = set(ARABIC_LETTERS_STR)
TATWEEL = '\u0640'
SPACE_CHAR = " "

def normalize_shadda_vowel_order(text: str) -> str:
    '''Enforce Letter+Shadda+Vowel encoding (some sources reverse this).'''
    return re.sub(f'([{SHORT_VOWELS_STR}])({SHADDA})', r'\2\1', text)

def fix_consecutive_diacritics(text: str) -> str:
    '''Collapse only *runs of the same* diacritic; never merges distinct ones
    (naively collapsing would destroy valid combos like Shadda+Fatha).'''
    for diac in DIACRITIC_CHARS:
        text = re.sub(re.escape(diac) + r'{2,}', diac, text)
    return text

def clean_arabic_text(text: str, preserve_latin: bool = True) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = unicodedata.normalize('NFC', text)
    text = normalize_shadda_vowel_order(text)
    text = text.replace(TATWEEL, '')
    if not preserve_latin:
        text = re.sub(r'[a-zA-Z\u00C0-\u024F]+', ' ', text)
    text = fix_consecutive_diacritics(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def strip_diacritics(text: str) -> str:
    return text.translate(str.maketrans('', '', DIACRITIC_STR))

# quick self-test, mirrors the team's unit tests
assert normalize_shadda_vowel_order('\u0643\u064E\u0651') == '\u0643\u0651\u064E'
assert fix_consecutive_diacritics('\u0651\u064E') == '\u0651\u064E'
print("Reused preprocessing utilities loaded and verified.")



# ---------------------------------------------------------------------------
# 3.2  JSONL / vocab loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

with open(PATHS.vocab_json, "r", encoding="utf-8") as f:
    CHAR2ID: Dict[str, int] = json.load(f)
ID2CHAR = {v: k for k, v in CHAR2ID.items()}

# Robust class_labels.txt parsing: file has 3 whitespace-padded columns
# (short name / diacritic mark / description) glued together. Split on
# runs of 2+ spaces to recover the columns properly.
with open(PATHS.class_labels_txt, "r", encoding="utf-8") as f:
    raw_lines = [l.rstrip("\n") for l in f]

id_rows = []
for line in raw_lines:
    stripped = line.strip()
    if not stripped:
        continue
    m = re.match(r"^(\d+)\s*[:\t,]?\s*(.+)$", stripped)
    if m:
        idx = int(m.group(1))
        cols = re.split(r"\s{2,}", m.group(2).strip())
        name = cols[0].strip()
        mark = cols[1].strip() if len(cols) > 1 else ""
        desc = cols[2].strip() if len(cols) > 2 else ""
        id_rows.append((idx, name, mark, desc))

if len(id_rows) == CFG.num_classes:
    id_rows.sort(key=lambda x: x[0])
    CLASS_NAMES = [name for _, name, _, _ in id_rows]
    CLASS_MARKS = [mark for _, _, mark, _ in id_rows]
    CLASS_DESCRIPTIONS = [desc for _, _, _, desc in id_rows]
else:
    print(f"Could not confidently parse {CFG.num_classes} classes (got {len(id_rows)} candidate rows).")
    print(f"Raw file has {len(raw_lines)} lines total. Full contents:")
    for i, l in enumerate(raw_lines):
        print(f"  [{i}] {l!r}")
    raise AssertionError(f"expected {CFG.num_classes} classes, parsed {len(id_rows)}")

train_records_raw = load_jsonl(PATHS.train_jsonl)
dev_records_raw = load_jsonl(PATHS.dev_jsonl)

print(f"train_data records: {len(train_records_raw)}")
print(f"dev_data  records : {len(dev_records_raw)}")
print(f"vocab size         : {len(CHAR2ID)}")
print()
print(f"{'ID':<3} {'Class':<18} {'Mark':<4} Description")
print("-" * 60)
for i, (name, mark, desc) in enumerate(zip(CLASS_NAMES, CLASS_MARKS, CLASS_DESCRIPTIONS)):
    print(f"{i:<3} {name:<18} {mark:<4} {desc}")



CLASS_ID_TO_DIACRITIC = {i: ("" if mark in ("(none)", "") else mark)
                          for i, mark in enumerate(CLASS_MARKS)}

print("Shadda-combination class codepoints (classes 9-15):")
for i in range(9, 16):
    mark = CLASS_ID_TO_DIACRITIC[i]
    codepoints = " ".join(f"U+{ord(c):04X}" for c in mark)
    print(f"  class {i:2d} ({CLASS_NAMES[i]:<16}): {codepoints}")

def _reconstruct(chars, labels):
    return "".join(c if c == SPACE_CHAR else c + CLASS_ID_TO_DIACRITIC.get(lab, "")
                    for c, lab in zip(chars, labels))

_sample = train_records_raw[:200]
_has_target = _sample and "target" in _sample[0]
if not _has_target:
    print("\n[SKIP] No 'target' field found in records -- cannot cross-check "
          "reconstruction against ground-truth text this way. Verify "
          "CLASS_ID_TO_DIACRITIC manually against a few known examples instead.")
else:
    n_match, mismatches = 0, []
    for r in _sample:
        recon = _reconstruct(r["chars"], r["labels"])
        if recon == r["target"]:
            n_match += 1
        elif len(mismatches) < 3:
            mismatches.append((r.get("sent_id", "?"), recon, r["target"]))
    print(f"\nReconstruction match: {n_match}/{len(_sample)} sampled records")
    if mismatches:
        print("First few mismatches:")
        for sid, recon, target in mismatches:
            print(f"  sent_id={sid}\n    reconstructed: {recon}\n    target       : {target}")
    assert n_match / len(_sample) > 0.98, (
        "Reconstruction mismatch rate too high -- check CLASS_MARKS ordering "
        "before trusting anything downstream (see printed examples above)."
    )
    print("Sanity check passed.")



# ---------------------------------------------------------------------------
# 3.3  DEV -> DEV / DEV_TEST split (deterministic, stratified by sentence length
# bucket so both partitions have comparable difficulty distribution)
# ---------------------------------------------------------------------------
def stratified_holdout(records: List[dict], holdout_frac: float, seed: int,
                        n_buckets: int = 5) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    lengths = [len(r["chars"]) for r in records]
    order = np.argsort(lengths)
    buckets = np.array_split(order, n_buckets)
    dev_idx, test_idx = [], []
    for b in buckets:
        b = list(b)
        rng.shuffle(b)
        n_test = max(1, int(len(b) * holdout_frac))
        test_idx.extend(b[:n_test])
        dev_idx.extend(b[n_test:])
    dev = [records[i] for i in dev_idx]
    test = [records[i] for i in test_idx]
    return dev, test

train_records = train_records_raw
dev_records, dev_test_records = stratified_holdout(
    dev_records_raw, CFG.dev_split_for_test, CFG.seed
)
print(f"TRAIN     : {len(train_records)} sentences")
print(f"DEV       : {len(dev_records)} sentences  (used for validation/early stopping)")
print(f"DEV_TEST  : {len(dev_test_records)} sentences  (held out, scored once at the end)")


# ## 4. Exploratory Data Analysis

class DiacritizationEDA:
    '''Runs a fixed battery of dataset diagnostics and renders them.
    Kept separate from the modeling code so it can be re-run standalone.'''

    def __init__(self, records: List[dict], class_names: List[str]):
        self.records = records
        self.class_names = class_names

    def label_distribution(self) -> pd.DataFrame:
        counts = np.zeros(len(self.class_names), dtype=np.int64)
        for r in self.records:
            for ch, lab in zip(r["chars"], r["labels"]):
                if ch != SPACE_CHAR:
                    counts[lab] += 1
        df = pd.DataFrame({"class_id": range(len(self.class_names)),
                            "class_name": self.class_names,
                            "count": counts})
        df["pct"] = 100 * df["count"] / df["count"].sum()
        return df.sort_values("count", ascending=False)

    def sentence_length_stats(self) -> pd.DataFrame:
        lens = [len(r["chars"]) for r in self.records]
        return pd.DataFrame({"n_chars": lens}).describe().T

    def plot(self, label_df: pd.DataFrame, len_df_source: List[int]):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].bar(label_df["class_name"], label_df["pct"])
        axes[0].set_title("Diacritic class distribution (% of labeled chars)")
        axes[0].tick_params(axis='x', rotation=75)
        axes[1].hist(len_df_source, bins=40)
        axes[1].set_title("Sentence length (characters)")
        axes[1].set_xlabel("n_chars")
        plt.tight_layout()
        plt.show()

def tokenizer_char_coverage(tokenizer, records: List[dict], sample_n: int = 500) -> pd.DataFrame:
    '''For a sample of sentences, checks whether each Algerian-specific
    letter round-trips through the tokenizer without collapsing to <unk>.'''
    watch_chars = ["پ", "ڤ"]
    rows = []
    sample = records[:sample_n]
    for wc in watch_chars:
        n_present = sum(wc in "".join(r["chars"]) for r in sample)
        n_unk = 0
        for r in sample:
            text = "".join(r["chars"])
            if wc in text:
                enc = tokenizer(text, add_special_tokens=False)
                if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in enc["input_ids"]:
                    n_unk += 1
        rows.append({"char": wc, "sentences_containing": n_present, "sentences_with_unk_token": n_unk})
    return pd.DataFrame(rows)

eda = DiacritizationEDA(train_records, CLASS_NAMES)
label_df = eda.label_distribution()
print(eda.sentence_length_stats())
eda.plot(label_df, [len(r["chars"]) for r in train_records])
label_df



# Tokenizer coverage diagnostic for the ACTIVE backbone specifically
_probe_tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME, use_fast=True)
coverage_df = tokenizer_char_coverage(_probe_tokenizer, train_records)
print(f"Tokenizer coverage check — backbone: {BACKBONE_NAME}")
coverage_df


# ## 5. Character ↔ Subword Alignment and Dataset

class CharAligner:
    '''Maps every character index of a sentence to the subword token index
    that covers it, using a fast tokenizer's offset mapping.'''

    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode(self, chars: List[str]) -> Dict[str, list]:
        text = "".join(chars)
        enc = self.tokenizer(text, return_offsets_mapping=True,
                              truncation=True, max_length=self.max_len)
        offsets = enc["offset_mapping"]
        char_to_token = {}
        for tok_idx, (start, end) in enumerate(offsets):
            if start == end:
                continue  # special token
            for c in range(start, end):
                char_to_token[c] = tok_idx
        token_idx_per_char = [char_to_token.get(i, -1) for i in range(len(chars))]
        return {"input_ids": enc["input_ids"], "token_idx_per_char": token_idx_per_char}


def compute_is_word_final(chars: List[str]) -> List[float]:
    out = []
    for i, c in enumerate(chars):
        if c == SPACE_CHAR:
            out.append(0.0)
        else:
            boundary = (i == len(chars) - 1) or (chars[i + 1] == SPACE_CHAR)
            out.append(1.0 if boundary else 0.0)
    return out


class DiacritizationDataset(Dataset):
    '''Wraps JSONL records ({'chars','labels',...}) into aligned tensors.
    `labels` may be omitted (has_labels=False) for unlabeled inference use.'''

    def __init__(self, records: List[dict], aligner: CharAligner, char2id: Dict[str, int],
                 space_label: int, has_labels: bool = True):
        self.records = records
        self.aligner = aligner
        self.char2id = char2id
        self.unk_id = char2id.get("<UNK>", 1)
        self.space_label = space_label
        self.has_labels = has_labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        chars = rec["chars"]
        n = len(chars)
        enc = self.aligner.encode(chars)
        char_ids = [self.char2id.get(c, self.unk_id) for c in chars]
        is_word_final = compute_is_word_final(chars)

        ext_labels = None
        if self.has_labels:
            labels = rec["labels"]
            ext_labels = [self.space_label if chars[i] == SPACE_CHAR else int(labels[i])
                          for i in range(n)]

        return {
            "input_ids": enc["input_ids"],
            "token_idx_per_char": enc["token_idx_per_char"][:n],
            "char_ids": char_ids,
            "is_word_final": is_word_final,
            "labels": ext_labels,
            "chars": chars,
            "sent_id": rec.get("sent_id", idx),
        }


def collate_fn(batch, pad_token_id: int):
    B = len(batch)
    max_tok = max(len(b["input_ids"]) for b in batch)
    max_char = max(len(b["chars"]) for b in batch)
    has_labels = all(b["labels"] is not None for b in batch)

    input_ids = torch.full((B, max_tok), pad_token_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_tok), dtype=torch.long)
    char_ids = torch.zeros((B, max_char), dtype=torch.long)
    token_idx_per_char = torch.zeros((B, max_char), dtype=torch.long)
    is_word_final = torch.zeros((B, max_char), dtype=torch.float)
    char_mask = torch.zeros((B, max_char), dtype=torch.bool)
    labels = torch.zeros((B, max_char), dtype=torch.long)

    meta = []
    for i, b in enumerate(batch):
        L = len(b["input_ids"]); input_ids[i, :L] = torch.tensor(b["input_ids"]); attn_mask[i, :L] = 1
        C = len(b["chars"])
        char_ids[i, :C] = torch.tensor(b["char_ids"])
        toks = [t if t >= 0 else 0 for t in b["token_idx_per_char"]]
        token_idx_per_char[i, :C] = torch.tensor(toks)
        is_word_final[i, :C] = torch.tensor(b["is_word_final"])
        char_mask[i, :C] = True
        if has_labels:
            labels[i, :C] = torch.tensor(b["labels"])
        meta.append({"sent_id": b["sent_id"], "chars": b["chars"]})

    return {"input_ids": input_ids, "attention_mask": attn_mask, "char_ids": char_ids,
            "token_idx_per_char": token_idx_per_char, "is_word_final": is_word_final,
            "char_mask": char_mask, "labels": labels if has_labels else None, "meta": meta}



def partition_indices_for_length(chars: List[str], tokenizer, max_tokens: int) -> List[Tuple[int, int]]:
    '''Returns (start, end) index ranges over chars that together cover the whole
    sentence, splitting recursively at a near-middle space whenever a fragment's
    token length would exceed max_tokens.'''
    n = len(chars)
    text = "".join(chars)
    if len(tokenizer.encode(text, add_special_tokens=True)) <= max_tokens:
        return [(0, n)]
    space_positions = [i for i in range(n) if chars[i] == SPACE_CHAR]
    if not space_positions:
        mid = n // 2
        return [(0, mid), (mid, n)]
    mid_target = n // 2
    mid_pos = min(space_positions, key=lambda p: abs(p - mid_target))
    left_ranges = partition_indices_for_length(chars[:mid_pos], tokenizer, max_tokens)
    right_ranges = partition_indices_for_length(chars[mid_pos:], tokenizer, max_tokens)
    right_shifted = [(s + mid_pos, e + mid_pos) for s, e in right_ranges]
    return left_ranges + right_shifted



def word_level_metrics_from_predict_fn(predict_fn, records: List[dict]) -> Dict[str, Any]:
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
            # single-letter words contribute nothing to the *-word count,
            # matching the standard literature convention (nothing to exclude)

    return {
        "DER": char_errors / max(total_chars, 1),
        "DER_star": char_errors_star / max(total_chars_star, 1),
        "WER": word_errors / max(total_words, 1),
        "WER_star": word_errors_star / max(total_words_star, 1),
        "n_chars": total_chars, "n_words": total_words,
    }


# ## 7. Training Utilities

def compute_class_weights(records: List[dict], cfg: TrainConfig, device: str) -> torch.Tensor:
    '''Inverse-frequency weights over the 16 diacritic classes (space excluded
    -- it's a structural signal, not a class we want balanced), capped so the
    rarest classes don't dominate, then extra-boosted on the Shadda-combo
    classes that stayed weak across every backbone tried so far.'''
    counts = np.zeros(cfg.num_classes, dtype=np.float64)
    for r in records:
        for ch, lab in zip(r["chars"], r["labels"]):
            if ch != SPACE_CHAR:
                counts[lab] += 1
    counts = np.clip(counts, 1, None)
    inv_freq = counts.sum() / (cfg.num_classes * counts)
    inv_freq = np.clip(inv_freq, None, cfg.class_weight_cap)

    weights = np.ones(cfg.num_labels, dtype=np.float64)  # +1 slot for the space label
    weights[:cfg.num_classes] = inv_freq
    for c in cfg.rare_class_ids:
        weights[c] *= cfg.rare_class_weight
    weights[cfg.space_label] = 1.0  # space is trivial to predict; don't up-weight it
    return torch.tensor(weights, dtype=torch.float32, device=device)


def rdrop_kl(logits1: torch.Tensor, logits2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    '''Bidirectional KL between two dropout-perturbed forward passes of the
    same batch, restricted to real (non-padding) positions.'''
    p1, p2 = torch.log_softmax(logits1, -1), torch.log_softmax(logits2, -1)
    kl = (F.kl_div(p1, p2, log_target=True, reduction="none").sum(-1)
          + F.kl_div(p2, p1, log_target=True, reduction="none").sum(-1)) / 2
    kl = kl[mask]
    return kl.mean() if mask.any() else torch.tensor(0.0, device=logits1.device)


def build_layerwise_optimizer(model: Track3BiLSTMCRF, cfg: TrainConfig):
    '''Discriminative fine-tuning: deeper (earlier) transformer layers get
    progressively smaller learning rates than the BiLSTM/CRF head. Head
    params are anything not prefixed "backbone." -- this covers
    char_embedding/word_final_proj/input_proj/bilstm/classifier/crf
    automatically without needing to name them individually.'''
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_layers = getattr(model.backbone.config, "num_hidden_layers", 12)
    groups = []
    head_params = [p for n, p in named_params if not n.startswith("backbone.")]
    groups.append({"params": head_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay})
    for layer_idx in range(n_layers):
        lr = cfg.lr_backbone_top * (cfg.layerwise_decay ** (n_layers - 1 - layer_idx))
        layer_params = [p for n, p in named_params if f"backbone.encoder.layer.{layer_idx}." in n]
        if layer_params:
            groups.append({"params": layer_params, "lr": lr, "weight_decay": cfg.weight_decay})
    embed_params = [p for n, p in named_params if n.startswith("backbone.embeddings")]
    if embed_params:
        groups.append({"params": embed_params,
                        "lr": cfg.lr_backbone_top * (cfg.layerwise_decay ** n_layers),
                        "weight_decay": cfg.weight_decay})
    return torch.optim.AdamW(groups)


class CheckpointManager:
    '''Saves/restores full training state so a Kaggle session restart loses
    at most `checkpoint_every_steps` steps of progress, never the whole run.
    Writes are atomic (temp file + rename) and disk-space-aware to avoid the
    "unexpected pos" RuntimeError that happens when the disk fills up mid-save.'''

    def __init__(self, run_dir: Path, save_optimizer: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.run_dir / "latest.pt"
        self.best_path = self.run_dir / "best.pt"
        # Optimizer/scheduler state roughly doubles/triples file size. Skip it
        # when this manager is only used for short-lived self-training runs
        # that never need to resume mid-epoch.
        self.save_optimizer = save_optimizer

    @staticmethod
    def _free_gb(path: Path) -> float:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)

    def _atomic_torch_save(self, state: dict, dest: Path):
        # Write to a temp file in the same directory, flush+fsync, then rename.
        # This guarantees dest is either the old valid file or the new valid
        # file -- never a half-written corrupt one -- and it lets us detect a
        # disk-full failure *before* clobbering the previous good checkpoint.
        free_gb = self._free_gb(dest.parent)
        if free_gb < 1.0:
            print(f"    [ckpt] WARNING: only {free_gb:.2f} GB free on disk before saving "
                  f"{dest.name}; attempting cleanup of stale checkpoints first.")
            self._cleanup_other_run_dirs()

        tmp_path = dest.with_suffix(".pt.tmp")
        try:
            with open(tmp_path, "wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, dest)  # atomic on POSIX
        except (RuntimeError, OSError) as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            print(f"    [ckpt] save failed ({e}); free space was {free_gb:.2f} GB. "
                  f"Skipping this checkpoint and continuing training.")

    def _cleanup_other_run_dirs(self):
        # Free space by dropping "latest.pt" (resumable-but-not-best state)
        # from sibling run directories under the same checkpoint root. Keeps
        # every "best.pt" so nothing you actually need is lost.
        root = self.run_dir.parent
        if not root.exists():
            return
        for sibling in root.iterdir():
            if sibling == self.run_dir or not sibling.is_dir():
                continue
            latest = sibling / "latest.pt"
            if latest.exists():
                try:
                    latest.unlink()
                    print(f"    [ckpt] freed space: removed {latest}")
                except OSError:
                    pass

    def save(self, model, optimizer, scheduler, epoch: int, global_step: int,
              best_dev_score: float, is_best: bool = False):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if self.save_optimizer else None,
            "scheduler": (scheduler.state_dict() if (scheduler and self.save_optimizer) else None),
            "epoch": epoch, "global_step": global_step, "best_dev_score": best_dev_score,
        }
        if is_best:
            self._atomic_torch_save(state, self.best_path)
        # "latest.pt" is only needed for mid-epoch resume; skip it entirely when
        # optimizer state isn't being kept (self-training runs), since there's
        # nothing meaningful to resume without the optimizer anyway.
        if self.save_optimizer:
            self._atomic_torch_save(state, self.ckpt_path)

    def load_latest(self, model, optimizer, scheduler):
        if not self.ckpt_path.exists():
            return {"epoch": 0, "global_step": 0, "best_dev_score": -1.0}
        state = torch.load(self.ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        print(f"Resumed from checkpoint: epoch={state['epoch']} step={state['global_step']} "
              f"best_dev_score={state['best_dev_score']:.4f}")
        return state

    def load_best(self, model):
        state = torch.load(self.best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        return state

# ## 8. Train `ACTIVE_MODEL`

tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME, use_fast=True)
aligner = CharAligner(tokenizer, CFG.max_subword_len)

train_ds = DiacritizationDataset(train_records, aligner, CHAR2ID, CFG.space_label)
dev_ds = DiacritizationDataset(dev_records, aligner, CHAR2ID, CFG.space_label)

_collate = lambda b: collate_fn(b, tokenizer.pad_token_id or 0)
train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, collate_fn=_collate)
dev_loader = DataLoader(dev_ds, batch_size=CFG.eval_batch_size, shuffle=False, collate_fn=_collate)

class_weights = compute_class_weights(train_records, CFG, DEVICE)
print("Auxiliary-loss class weights:", class_weights.tolist())



def _to_device(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


def masked_accuracy_from_emissions(emissions, labels, char_mask, space_label) -> float:
    '''Fast approximate accuracy from raw emissions (no Viterbi decode) --
    used as the per-batch training-progress signal only.'''
    preds = emissions.argmax(-1)
    valid = char_mask & (labels != space_label)
    if valid.sum() == 0:
        return 0.0
    return (preds[valid] == labels[valid]).float().mean().item()


def build_model(cfg: TrainConfig, class_weights: torch.Tensor) -> "Track3BiLSTMCRF":
    return Track3BiLSTMCRF(
        cfg.backbone, len(CHAR2ID), cfg.num_labels, char_emb_dim=cfg.char_emb_dim,
        n_pool_layers=cfg.n_pool_layers, lstm_hidden_dim=cfg.lstm_hidden_dim,
        num_lstm_layers=cfg.num_lstm_layers, dropout=cfg.head_dropout, use_crf=cfg.use_crf,
        aux_loss_weight=cfg.aux_loss_weight, class_weights=class_weights,
        freeze_embeddings=cfg.freeze_embeddings, freeze_n_layers=cfg.freeze_n_layers,
    ).to(DEVICE)


def run_train_epoch(model, loader, optimizer, scheduler, cfg, ckpt, epoch, global_step):
    model.train()
    epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
    t0 = time.time()
    for batch in loader:
        batch = _to_device(batch)
        loss1, emissions1 = model(batch["input_ids"], batch["attention_mask"], batch["char_ids"],
                                   batch["token_idx_per_char"], batch["is_word_final"],
                                   batch["char_mask"], labels=batch["labels"])
        loss1 = loss1.mean()

        if cfg.use_rdrop:
            loss2, emissions2 = model(batch["input_ids"], batch["attention_mask"], batch["char_ids"],
                                       batch["token_idx_per_char"], batch["is_word_final"],
                                       batch["char_mask"], labels=batch["labels"])
            loss2 = loss2.mean()
            kl = rdrop_kl(emissions1, emissions2, batch["char_mask"])
            loss = (loss1 + loss2) / 2 + cfg.rdrop_alpha * kl
        else:
            loss = loss1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1
        epoch_loss += loss.item()
        epoch_acc += masked_accuracy_from_emissions(emissions1.detach(), batch["labels"],
                                                      batch["char_mask"], cfg.space_label)
        n_batches += 1

        if global_step % cfg.checkpoint_every_steps == 0:
            ckpt.save(model, optimizer, scheduler, epoch, global_step, best_dev_score=-1.0)

        if (time.time() - t0) / 60 > cfg.max_train_minutes:
            print("Time budget reached mid-epoch; checkpointing and stopping.")
            ckpt.save(model, optimizer, scheduler, epoch, global_step, best_dev_score=-1.0)
            return epoch_loss / n_batches, epoch_acc / n_batches, global_step, True

    return epoch_loss / n_batches, epoch_acc / n_batches, global_step, False


@torch.no_grad()
def run_eval(model, loader, cfg):
    '''Returns (avg_loss, decode_accuracy). Decode accuracy uses the model's
    real inference path (CRF Viterbi or argmax decode), not the fast
    emissions-argmax approximation used mid-training -- so this number is
    what checkpoint selection and early stopping actually optimize for.'''
    model.eval()
    total_loss, n_batches = 0.0, 0
    y_true, y_pred = [], []
    for batch in loader:
        batch_gpu = _to_device(batch)
        loss, _ = model(batch_gpu["input_ids"], batch_gpu["attention_mask"], batch_gpu["char_ids"],
                         batch_gpu["token_idx_per_char"], batch_gpu["is_word_final"],
                         batch_gpu["char_mask"], labels=batch_gpu["labels"])
        total_loss += loss.mean().item(); n_batches += 1

        decoded = model(batch_gpu["input_ids"], batch_gpu["attention_mask"], batch_gpu["char_ids"],
                         batch_gpu["token_idx_per_char"], batch_gpu["is_word_final"],
                         batch_gpu["char_mask"], labels=None)
        labels_cpu = batch["labels"]
        for i, seq in enumerate(decoded):
            true_seq = labels_cpu[i, :len(seq)].tolist()
            for t, p in zip(true_seq, seq):
                if t != cfg.space_label:
                    y_true.append(t); y_pred.append(p)

    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0
    return total_loss / n_batches, acc



if CFG.k_folds == 1:
    model = build_model(CFG, class_weights)
    optimizer = build_layerwise_optimizer(model, CFG)
    total_steps = len(train_loader) * CFG.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(CFG.warmup_ratio * total_steps), num_training_steps=total_steps)
    ckpt = CheckpointManager(PATHS.checkpoint_dir / RUN_ID)

    state = ckpt.load_latest(model, optimizer, scheduler)
    start_epoch, global_step, best_dev_score = state["epoch"], state["global_step"], state["best_dev_score"]
    print(f"Starting from epoch {start_epoch}, global_step {global_step}, best_dev_score {best_dev_score:.4f}")

    patience_counter = 0
    best_dev_loss = float("inf")
    stopped_for_time = False

    for epoch in range(start_epoch, CFG.epochs):
        train_loss, train_acc, global_step, stopped_for_time = run_train_epoch(
            model, train_loader, optimizer, scheduler, CFG, ckpt, epoch, global_step)
        if stopped_for_time:
            break

        dev_loss, dev_acc = run_eval(model, dev_loader, CFG)
        gap = train_acc - dev_acc

        overfit_flag = " <- dev_loss rising (overfitting signal)" if dev_loss > best_dev_loss else ""
        print(f"Epoch {epoch+1}/{CFG.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} | train-dev gap={gap:+.4f}{overfit_flag}")

        is_best_f1 = dev_acc > best_dev_score
        if is_best_f1:
            best_dev_score = dev_acc

        if CFG.early_stop_metric == "dev_loss":
            improved = dev_loss < best_dev_loss
        else:
            improved = is_best_f1
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss

        patience_counter = 0 if improved else patience_counter + 1
        ckpt.save(model, optimizer, scheduler, epoch + 1, global_step, best_dev_score, is_best=is_best_f1)

        if patience_counter >= CFG.early_stop_patience:
            print(f"Early stopping ({CFG.early_stop_metric} plateaued for {CFG.early_stop_patience} epochs).")
            break

    print(f"Training loop finished. Best dev accuracy: {best_dev_score:.4f} | Best dev_loss: {best_dev_loss:.4f}")
    if stopped_for_time:
        print("NOTE: stopped due to time budget mid-epoch — re-run this cell/notebook to resume.")
else:
    print(f"CFG.k_folds = {CFG.k_folds} (> 1) -> skipping Section 8's single-model training. "
          f"Section 8b will train the fold models instead.")


# ## 8b. K-Fold Cross-Validation (optional, alternative to Section 8)

FOLD_CHECKPOINT_DIRS: List[Path] = []
FOLD_TRAIN_RECORDS: List[List[dict]] = []   # stored so Section 10 can continue-train each fold
FOLD_VAL_RECORDS: List[List[dict]] = []

if CFG.k_folds > 1:
    pool = train_records + dev_records  # DEV_TEST stays untouched throughout
    rng = random.Random(CFG.seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    fold_indices = np.array_split(order, CFG.k_folds)

    for fold in range(CFG.k_folds):
        fold_run_id = f"{RUN_ID}_fold{fold}"
        fold_ckpt_dir = PATHS.checkpoint_dir / fold_run_id
        FOLD_CHECKPOINT_DIRS.append(fold_ckpt_dir)

        val_idx = set(fold_indices[fold].tolist())
        fold_train = [pool[i] for i in range(len(pool)) if i not in val_idx]
        fold_val = [pool[i] for i in range(len(pool)) if i in val_idx]
        FOLD_TRAIN_RECORDS.append(fold_train)
        FOLD_VAL_RECORDS.append(fold_val)

        fold_class_weights = compute_class_weights(fold_train, CFG, DEVICE)
        fold_train_ds = DiacritizationDataset(fold_train, aligner, CHAR2ID, CFG.space_label)
        fold_val_ds = DiacritizationDataset(fold_val, aligner, CHAR2ID, CFG.space_label)
        fold_train_loader = DataLoader(fold_train_ds, batch_size=CFG.batch_size,
                                        shuffle=True, collate_fn=_collate)
        fold_val_loader = DataLoader(fold_val_ds, batch_size=CFG.eval_batch_size,
                                      shuffle=False, collate_fn=_collate)

        fold_model = build_model(CFG, fold_class_weights)
        fold_optimizer = build_layerwise_optimizer(fold_model, CFG)
        fold_total_steps = len(fold_train_loader) * CFG.epochs
        fold_scheduler = get_linear_schedule_with_warmup(
            fold_optimizer, num_warmup_steps=int(CFG.warmup_ratio * fold_total_steps),
            num_training_steps=fold_total_steps)
        fold_ckpt = CheckpointManager(fold_ckpt_dir)

        state = fold_ckpt.load_latest(fold_model, fold_optimizer, fold_scheduler)
        f_epoch, f_step, f_best = state["epoch"], state["global_step"], state["best_dev_score"]
        f_best_loss, f_patience = float("inf"), 0

        print(f"\n=== Fold {fold+1}/{CFG.k_folds} ({fold_run_id}) "
              f"| train={len(fold_train)} val={len(fold_val)} "
              f"| resuming from epoch {f_epoch} ===")

        for epoch in range(f_epoch, CFG.epochs):
            tr_loss, tr_acc, f_step, stopped = run_train_epoch(
                fold_model, fold_train_loader, fold_optimizer, fold_scheduler,
                CFG, fold_ckpt, epoch, f_step)
            if stopped:
                break
            v_loss, v_acc = run_eval(fold_model, fold_val_loader, CFG)
            print(f"  fold {fold} epoch {epoch+1}/{CFG.epochs} | train_acc={tr_acc:.4f} "
                  f"| val_loss={v_loss:.4f} val_acc={v_acc:.4f}")

            is_best_f1 = v_acc > f_best
            if is_best_f1:
                f_best = v_acc
            improved = (v_loss < f_best_loss) if CFG.early_stop_metric == "dev_loss" else is_best_f1
            if v_loss < f_best_loss:
                f_best_loss = v_loss
            f_patience = 0 if improved else f_patience + 1

            fold_ckpt.save(fold_model, fold_optimizer, fold_scheduler, epoch + 1, f_step, f_best,
                            is_best=is_best_f1)
            if f_patience >= CFG.early_stop_patience:
                print(f"  fold {fold}: early stopping.")
                break

        print(f"Fold {fold} done. Best val accuracy: {f_best:.4f}")

    print(f"\nK-fold training complete: {CFG.k_folds} folds checkpointed under "
          f"{PATHS.checkpoint_dir}/{RUN_ID}_fold*/")
else:
    print("CFG.k_folds == 1 -> skipping k-fold section, using Section 8's single model.")


# ## 10. Self-Training on Unlabeled `KAGGLE_TEST` Sentences (optional)

@torch.no_grad()
def pseudo_label_sentences(models: List[nn.Module], sentences: List[str], aligner: CharAligner,
                            char2id: Dict[str, int], conf_threshold: float) -> List[dict]:
    for m in models:
        m.eval()
    pseudo_records = []
    for sent_id, text in enumerate(sentences):
        text = clean_arabic_text(text)
        chars = list(text)
        if not chars:
            continue
        n = len(chars)
        enc = aligner.encode(chars)
        input_ids = torch.tensor([enc["input_ids"]], device=DEVICE)
        attn = torch.ones_like(input_ids)
        toks = torch.tensor([[t if t >= 0 else 0 for t in enc["token_idx_per_char"][:n]]], device=DEVICE)
        char_ids = torch.tensor([[char2id.get(c, char2id.get('<UNK>', 1)) for c in chars]], device=DEVICE)
        wf = torch.tensor([compute_is_word_final(chars)], device=DEVICE)
        mask = torch.ones((1, n), dtype=torch.bool, device=DEVICE)

        decoded = majority_vote_decode(models, input_ids, attn, char_ids, toks, wf, mask)[0]
        # confidence proxy: average the models' auxiliary-emission softmax confidence
        # at the majority-voted label, per position
        conf_sum = torch.zeros(n, device=DEVICE)
        for m in models:
            emissions = m._encode(input_ids, attn, toks, char_ids, wf)[0]  # (n, num_labels)
            probs = torch.softmax(emissions, dim=-1)
            pred_ids = torch.tensor(decoded, device=DEVICE)
            conf_sum += probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)
        conf = (conf_sum / len(models)).cpu().tolist()

        labels, keep = [], True
        for i, c in enumerate(chars):
            if c == SPACE_CHAR:
                labels.append(0)
                continue
            if conf[i] < conf_threshold:
                keep = False
                break
            labels.append(decoded[i] if decoded[i] < CFG.num_classes else 0)  # safety clamp
        if keep:
            pseudo_records.append({"sent_id": f"pseudo_{sent_id}", "chars": chars, "labels": labels})
    return pseudo_records


def continue_finetune(m, ckpt_dir: Path, base_train_records: List[dict], pseudo_records: List[dict],
                       val_loader, start_score: float) -> Tuple[float, bool]:
    '''Continue-trains one model on (base_train_records + pseudo_records) for
    up to CFG.self_training_epochs, model-selecting by val accuracy.
    Returns (best_val_acc_achieved, improved_over_start_score).'''
    augmented = base_train_records + pseudo_records
    ds = DiacritizationDataset(augmented, aligner, CHAR2ID, CFG.space_label)
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True, collate_fn=_collate)

    opt = build_layerwise_optimizer(m, CFG)
    total_steps = len(loader) * CFG.self_training_epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(CFG.warmup_ratio * total_steps), num_training_steps=total_steps)
    st_ckpt = CheckpointManager(ckpt_dir)
    state = st_ckpt.load_latest(m, opt, sched)
    step, best_acc = state["global_step"], start_score

    for epoch in range(state["epoch"], CFG.self_training_epochs):
        tr_loss, tr_acc, step, stopped = run_train_epoch(m, loader, opt, sched, CFG, st_ckpt, epoch, step)
        if stopped:
            break
        v_loss, v_acc = run_eval(m, val_loader, CFG)
        print(f"    epoch {epoch+1}/{CFG.self_training_epochs} | train_acc={tr_acc:.4f} | val_acc={v_acc:.4f}")
        is_best = v_acc > best_acc
        st_ckpt.save(m, opt, sched, epoch + 1, step, max(best_acc, v_acc), is_best=is_best)
        if is_best:
            best_acc = v_acc

    improved = st_ckpt.best_path.exists() and best_acc > start_score
    if improved:
        st_ckpt.load_best(m)
    return best_acc, improved


if CFG.self_training_enabled:
    with open(PATHS.raw_test_txt, "r", encoding="utf-8") as f:
        raw_test_sentences = [l.rstrip("\n") for l in f if l.strip()]

    pseudo_records = pseudo_label_sentences(MODELS_FOR_INFERENCE, raw_test_sentences, aligner,
                                             CHAR2ID, CFG.self_training_conf_threshold)
    print(f"Pseudo-labeled {len(pseudo_records)}/{len(raw_test_sentences)} sentences "
          f"above confidence {CFG.self_training_conf_threshold} "
          f"(using {len(MODELS_FOR_INFERENCE)}-model majority vote)")

    if len(pseudo_records) == 0:
        print("No sentences cleared the confidence threshold -- lower "
              "CFG.self_training_conf_threshold or skip self-training for this run.")
    elif CFG.k_folds > 1 and FOLD_CHECKPOINT_DIRS:
        print(f"\nContinuing fine-tune for up to {CFG.self_training_epochs} epochs on EACH of "
              f"{CFG.k_folds} folds (pseudo-labels shared across folds, base train split per-fold)...")
        for fold in range(CFG.k_folds):
            print(f"\n  -- fold {fold} --")
            fm = MODELS_FOR_INFERENCE[fold]
            fold_val_ds = DiacritizationDataset(FOLD_VAL_RECORDS[fold], aligner, CHAR2ID, CFG.space_label)
            fold_val_loader = DataLoader(fold_val_ds, batch_size=CFG.eval_batch_size,
                                          shuffle=False, collate_fn=_collate)
            st_dir = PATHS.checkpoint_dir / f"{RUN_ID}_fold{fold}_selftrain"
            base_loss, base_acc = run_eval(fm, fold_val_loader, CFG)
            best_acc, improved = continue_finetune(fm, st_dir, FOLD_TRAIN_RECORDS[fold],
                                                     pseudo_records, fold_val_loader, base_acc)
            print(f"  fold {fold}: base_val_acc={base_acc:.4f} -> best_val_acc={best_acc:.4f} "
                  f"({'adopted' if improved else 'kept original'})")

        st_report = evaluator.evaluate(MODELS_FOR_INFERENCE, dev_test_loader)
        print(f"\nSelf-trained fold-ensemble DEV_TEST micro-F1: {st_report['micro_f1']:.4f}  "
              f"(baseline was {DEV_TEST_SCORE:.4f})")
        if st_report["micro_f1"] > DEV_TEST_SCORE:
            print("Self-training improved the fold-ensemble DEV_TEST score -- adopting it.")
            DEV_TEST_REPORT = st_report
            DEV_TEST_SCORE = st_report["micro_f1"]
            DEV_TEST_REPORT.update(word_level_metrics_from_predict_fn(
                lambda chars: _predict_chars(MODELS_FOR_INFERENCE, chars), dev_test_records))
        else:
            print("Self-training did not improve the fold-ensemble DEV_TEST score overall.")
    else:
        train_records_augmented = train_records + pseudo_records
        print(f"Augmented train set: {len(train_records)} + {len(pseudo_records)} pseudo "
              f"= {len(train_records_augmented)} sentences")

        st_model = MODELS_FOR_INFERENCE[0]
        st_dir = PATHS.checkpoint_dir / f"{RUN_ID}_selftrain"
        print(f"\nContinuing fine-tune for up to {CFG.self_training_epochs} epochs "
              f"on the pseudo-augmented set...")
        best_acc, improved = continue_finetune(st_model, st_dir, train_records, pseudo_records,
                                                 dev_loader, DEV_TEST_SCORE)

        st_report = evaluator.evaluate([st_model], dev_test_loader)
        print(f"\nSelf-trained DEV_TEST micro-F1: {st_report['micro_f1']:.4f}  "
              f"(baseline was {DEV_TEST_SCORE:.4f})")
        if st_report["micro_f1"] > DEV_TEST_SCORE:
            print("Self-training improved DEV_TEST -- adopting it as the final model/report.")
            DEV_TEST_REPORT = st_report
            DEV_TEST_SCORE = st_report["micro_f1"]
            MODELS_FOR_INFERENCE = [st_model]
            DEV_TEST_REPORT.update(word_level_metrics_from_predict_fn(
                lambda chars: _predict_chars(MODELS_FOR_INFERENCE, chars), dev_test_records))
        else:
            print("Self-training did not improve DEV_TEST -- keeping the original model/report.")
else:
    print("Self-training disabled (CFG.self_training_enabled = False). Skipping.")


# ## 11. Inference on `KAGGLE_TEST` and Submission File

print("Derived class_id -> diacritic mapping:", CLASS_ID_TO_DIACRITIC)

@torch.no_grad()
def diacritize_sentences(models: List[nn.Module], sentences: List[str], aligner: CharAligner,
                          tokenizer, char2id: Dict[str, int], max_tokens: int) -> List[str]:
    for m in models:
        m.eval()
    outputs = []
    for text in sentences:
        text = clean_arabic_text(text)
        chars = list(text)
        if not chars:
            outputs.append("")
            continue

        ranges = partition_indices_for_length(chars, tokenizer, max_tokens)
        full_preds: List[int] = []
        for start, end in ranges:
            frag = chars[start:end]
            full_preds.extend(_predict_chars(models, frag))

        out_chars = [c if c == SPACE_CHAR else c + CLASS_ID_TO_DIACRITIC.get(p, "")
                     for c, p in zip(chars, full_preds)]
        outputs.append("".join(out_chars))
    return outputs


with open(PATHS.raw_test_txt, "r", encoding="utf-8") as f:
    kaggle_test_sentences = [l.rstrip("\n") for l in f if l.strip()]

diacritized_output = diacritize_sentences(MODELS_FOR_INFERENCE, kaggle_test_sentences, aligner,
                                           tokenizer, CHAR2ID, CFG.max_subword_len)

pred_txt_path = PATHS.work_dir / f"{RUN_ID}_predictions.txt"
with open(pred_txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(diacritized_output))
print(f"Wrote diacritized predictions: {pred_txt_path}  "
      f"(ensembling {len(MODELS_FOR_INFERENCE)} model(s))")



import subprocess
submission_path = PATHS.work_dir / f"{RUN_ID}_submission.csv"
result = subprocess.run(
    ["python", str(PATHS.make_submission_py),
     "--ids", str(PATHS.raw_test_ids),
     "--input", str(PATHS.raw_test_txt),
     "--pred", str(pred_txt_path),
     "--out", str(submission_path)],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
else:
    print(f"submission.csv written to: {submission_path}")


# ## 11.5 Sanity-Check `submission.csv` Before Packaging/Upload

sample_sub_path = PATHS.root / "test_data" / "sample_submission.csv"

sub_df = pd.read_csv(submission_path)
sample_df = pd.read_csv(sample_sub_path) if sample_sub_path.exists() else None

print(f"submission.csv rows      : {len(sub_df)}")
if sample_df is not None:
    print(f"sample_submission rows   : {len(sample_df)}")
    print(f"Row count matches        : {len(sub_df) == len(sample_df)}")
    ids_match = set(sub_df['Id']) == set(sample_df['Id'])
    print(f"Id set matches exactly   : {ids_match}")
    if not ids_match:
        missing = set(sample_df['Id']) - set(sub_df['Id'])
        extra = set(sub_df['Id']) - set(sample_df['Id'])
        print(f"  Missing Ids (first 5): {list(missing)[:5]}")
        print(f"  Extra Ids (first 5)  : {list(extra)[:5]}")
else:
    print("sample_submission.csv not found — skipping row/Id comparison.")

print(f"\nLabel range              : [{sub_df['Label'].min()}, {sub_df['Label'].max()}]")
print(f"Any NaN labels            : {sub_df['Label'].isna().any()}")
assert sub_df['Label'].between(0, CFG.num_classes - 1).all(), \
    f"found a label outside 0-{CFG.num_classes - 1}!"

print("\n--- Spot check: first 3 diacritized predictions vs raw input ---")
for raw, pred in list(zip(kaggle_test_sentences, diacritized_output))[:3]:
    print(f"  raw : {raw}")
    print(f"  pred: {pred}")
    print()


# ## 12. Package Run Artifacts into a Structured Zip

def package_run(run_id: str, cfg: TrainConfig, dev_test_report: Dict[str, Any],
                 submission_csv: Path, class_names: List[str],
                 models: List[nn.Module]) -> Path:
    score = dev_test_report["micro_f1"]
    package_dir = PATHS.export_dir / f"{run_id}-{score:.4f}"
    package_dir.mkdir(parents=True, exist_ok=True)

    with open(package_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    report_to_save = {k: v for k, v in dev_test_report.items() if k != "confusion_matrix"}
    with open(package_dir / "dev_test_evaluation.json", "w") as f:
        json.dump(report_to_save, f, indent=2)

    evaluator.plot_confusion(dev_test_report["confusion_matrix"])
    plt.savefig(package_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    shutil.copy(submission_csv, package_dir / "submission.csv")

    if cfg.export_include_weights:
        weights_dir = package_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        for i, m in enumerate(models):
            torch.save(m.state_dict(), weights_dir / f"model_{i}.pt")
        print(f"Exported {len(models)} model weight file(s) to {weights_dir}")

    card = (f"# Run: {run_id}\n\nBackbone: {cfg.backbone}\n"
            f"DEV_TEST micro-F1 (Kaggle proxy): {score:.4f}\n"
            f"DEV_TEST macro-F1: {dev_test_report['macro_f1']:.4f}\n"
            f"DER: {dev_test_report.get('DER', float('nan')):.4f}  "
            f"DER*: {dev_test_report.get('DER_star', float('nan')):.4f}\n"
            f"WER: {dev_test_report.get('WER', float('nan')):.4f}  "
            f"WER*: {dev_test_report.get('WER_star', float('nan')):.4f}\n"
            f"Models ensembled: {dev_test_report.get('n_models_ensembled', 1)}\n"
            f"Chars evaluated: {dev_test_report['n_chars_evaluated']}\n"
            f"Self-training used: {cfg.self_training_enabled}\n"
            f"Weights included: {cfg.export_include_weights}\n"
            f"use_crf: {cfg.use_crf} | n_pool_layers: {cfg.n_pool_layers} | "
            f"lstm_hidden_dim: {cfg.lstm_hidden_dim} | num_lstm_layers: {cfg.num_lstm_layers} | "
            f"char_emb_dim: {cfg.char_emb_dim}\n")
    (package_dir / "README.md").write_text(card, encoding="utf-8")

    zip_path = PATHS.export_dir / f"{run_id}-{score:.4f}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in package_dir.rglob("*"):
            zf.write(file, file.relative_to(package_dir.parent))
    print(f"Packaged: {zip_path}")
    return zip_path

final_zip = package_run(RUN_ID, CFG, DEV_TEST_REPORT, submission_path, CLASS_NAMES, MODELS_FOR_INFERENCE)


# ## 13. Run Summary

print("="*70)
print(f"RUN COMPLETE: {RUN_ID} ({BACKBONE_NAME})")
print(f"DEV_TEST micro-F1 : {DEV_TEST_SCORE:.4f}")
print(f"Export             : {final_zip}")
print("="*70)
print("\nTo run the next backbone: change ACTIVE_MODEL in Section 2.2 and re-run.")


