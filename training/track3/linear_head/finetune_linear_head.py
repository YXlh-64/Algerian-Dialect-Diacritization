# Shared fine-tuning script for 6 of 6 track3/linear_head experiments (auto-consolidated -- identical except ACTIVE_MODEL; a broken '!pip install kaggle' cell present in some copies was dropped).
# Excluded from this shared script (differ for real reasons, kept as their own file): none
# Defaults to --active-model arabert_v02, the best-scoring run in this group (0.91884 private F1). Override with --active-model <key>.

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

# Guard against a known sympy/torch interaction bug (sympy/sympy#29756): some
# sympy versions don't auto-import the `sympy.printing` submodule, but
# torch/fx/experimental/symbolic_shapes.py (pulled in by `import transformers`
# via torch.distributed.tensor) references sympy.printing.StrPrinter in a type
# annotation that's evaluated eagerly -- so `import transformers` can die with
# "AttributeError: module 'sympy' has no attribute 'printing'" even though the
# actual problem is upstream of transformers entirely. Forcing the submodule
# import first is enough; if that's not sufficient (deeper sympy/torch version
# mismatch), fall back to upgrading sympy and retrying once.
try:
    import sympy
    import sympy.printing  # noqa: F401  -- registers sympy.printing before torch needs it
except ImportError:
    pass

try:
    import transformers
except ImportError:
    os.system("pip install -q transformers")
    import transformers
except AttributeError as e:
    if "sympy" in str(e):
        print(f"Hit the known sympy/torch import issue ({e}); upgrading sympy and retrying...")
        os.system("pip install -q -U sympy")
        import importlib
        import sympy
        importlib.reload(sympy)
        import sympy.printing  # noqa: F401
        import transformers
    else:
        raise

try:
    import sklearn
except ImportError:
    os.system("pip install -q scikit-learn")
    import sklearn

import torch.nn as nn
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
    "marbertv2":      "UBC-NLP/MARBERTv2",     # further-pretrained MARBERT, 512-token
                                                # sequences (vs MARBERT's 128) -- worth
                                                # trying alongside plain MARBERT, not
                                                # instead of it
    "dziribert":      "alger-ia/dziribert",
    "alclam":         "rahbi/alclam-base-v2",  # Algerian-dialect BERT trained from
                                                # scratch on 3.4M dialectal sentences
                                                # (arXiv:2407.13097) -- different
                                                # tokenizer/pretraining than DziriBERT,
                                                # plausibly stronger on this exact dialect
}

# <<< CHANGE THIS to switch which model this Kaggle session trains >>>
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--active-model", type=str, default="arabert_v02",
                      choices=list(MODEL_REGISTRY.keys()),
                      help="Key into MODEL_REGISTRY (default: arabert_v02, the best-scoring run in this group at 0.91884 private F1)")
_args, _ = _parser.parse_known_args()
ACTIVE_MODEL: str = _args.active_model
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
    num_classes: int = 16
    char_emb_dim: int = 64
    n_concat_layers: int = 4
    head_hidden_dim: int = 512
    head_dropout: float = 0.20            # bumped 0.15 -> 0.20: runs at 25 epochs
                                           # showed dev_loss rising past ~epoch 10
                                           # while train_f1 kept climbing -- classic
                                           # overfitting on the small (4.8k) train set
    use_deep_head: bool = True
    max_subword_len: int = 256
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 25
    lr_head: float = 3e-4
    lr_backbone_top: float = 2e-5
    layerwise_decay: float = 0.9
    weight_decay: float = 0.08            # bumped 0.05 -> 0.08, same overfitting evidence
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    rare_class_ids: Tuple[int, ...] = (9, 10, 11, 12, 13, 14, 15)
    rare_class_weight: float = 3.0        # bumped 2.0 -> 3.0: Shadda+Fatha (support 85)
                                           # stayed the most informative weak class
                                           # (F1 ~0.72-0.77) across every backbone
    use_rdrop: bool = True
    rdrop_alpha: float = 1.0
    early_stop_metric: str = "dev_loss"   # "dev_loss" (recommended) or "dev_f1".
                                           # dev_loss started rising several epochs
                                           # before dev_f1 visibly plateaued in the
                                           # last runs -- it's the earlier, more
                                           # reliable overfitting signal.
    early_stop_patience: int = 4           # was 2 -- your own exported runs show this
                                            # cost ~0.4pt: camelbert_mix at patience=2
                                            # scored 0.9105, identical config at
                                            # patience=4 scored 0.9145. camelbert_da
                                            # (0.9102) also used patience=2 and is a
                                            # good candidate to rerun with this value.
    seed: int = SEED
    dev_split_for_test: float = 0.20
    max_train_minutes: int = 480
    checkpoint_every_steps: int = 500
    self_training_enabled: bool = True    # ON: models are now consistently 0.90+,
                                           # pseudo-labels are trustworthy (see
                                           # Section 10 for the reasoning)
    self_training_conf_threshold: float = 0.90
    self_training_epochs: int = 4          # shorter continued fine-tune, not a
                                            # full retrain from scratch
    k_folds: int = 5                       # ON: reduces the ~0.4pt run-to-run
                                            # variance already observed between
                                            # identical-config runs, plus gives a
                                            # free ensemble. Set to 1 for the
                                            # original single-split behavior, or
                                            # lower (e.g. 3) if short on session time
                                            # -- 5 folds x up to 25 epochs (though
                                            # convergence is now ~11-15 in practice)
                                            # plus self-training per fold is a real
                                            # multi-hour run; the whole pipeline is
                                            # checkpointed/resumable across restarts.
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


class DiacritizationDataset(Dataset):
    '''Wraps JSONL records ({'chars','labels',...}) into aligned tensors.'''

    def __init__(self, records: List[dict], aligner: CharAligner, char2id: Dict[str, int]):
        self.records = records
        self.aligner = aligner
        self.char2id = char2id
        self.unk_id = char2id.get("<UNK>", 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        chars, labels = rec["chars"], rec["labels"]
        enc = self.aligner.encode(chars)
        n = len(chars)
        char_ids = [self.char2id.get(c, self.unk_id) for c in chars]
        char_labels = [IGNORE_INDEX if chars[i] == SPACE_CHAR else labels[i] for i in range(n)]
        return {
            "input_ids": enc["input_ids"],
            "token_idx_per_char": enc["token_idx_per_char"][:n],
            "char_ids": char_ids,
            "char_labels": char_labels,
            "sent_id": rec.get("sent_id", idx),
        }


IGNORE_INDEX = -100

def collate_fn(batch, pad_token_id: int):
    max_tok = max(len(b["input_ids"]) for b in batch)
    max_char = max(len(b["char_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_tok), pad_token_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_tok), dtype=torch.long)
    char_ids = torch.zeros((B, max_char), dtype=torch.long)
    char_labels = torch.full((B, max_char), IGNORE_INDEX, dtype=torch.long)
    token_idx_per_char = torch.zeros((B, max_char), dtype=torch.long)

    for i, b in enumerate(batch):
        L = len(b["input_ids"]); input_ids[i, :L] = torch.tensor(b["input_ids"]); attn_mask[i, :L] = 1
        C = len(b["char_ids"])
        char_ids[i, :C] = torch.tensor(b["char_ids"])
        char_labels[i, :C] = torch.tensor(b["char_labels"])
        toks = [t if t >= 0 else 0 for t in b["token_idx_per_char"]]
        token_idx_per_char[i, :C] = torch.tensor(toks)

    return {"input_ids": input_ids, "attention_mask": attn_mask, "char_ids": char_ids,
            "char_labels": char_labels, "token_idx_per_char": token_idx_per_char}



from collections import defaultdict, Counter


def word_level_metrics_from_predict_fn(predict_fn, records: List[dict]) -> Dict[str, Any]:
    '''Word-level DER/WER plus sentence exact-match, per-class DER (char
    error rate per diacritic class), and the most common (true, predicted)
    confusion pairs -- for a richer error analysis than DER/WER alone.'''
    total_chars = char_errors = 0
    total_chars_star = char_errors_star = 0
    total_words = word_errors = 0
    total_words_star = word_errors_star = 0
    n_sent = n_sent_exact = 0
    class_total: Dict[int, int] = defaultdict(int)
    class_errors: Dict[int, int] = defaultdict(int)
    confusion_pairs: Counter = Counter()

    for rec in records:
        chars, labels = rec["chars"], rec["labels"]
        preds = predict_fn(chars)

        n_sent += 1
        sent_ok = True
        words, cur = [], []
        for i, c in enumerate(chars):
            if c == SPACE_CHAR:
                if cur:
                    words.append(cur)
                cur = []
                continue
            t, p = labels[i], preds[i]
            cur.append((p, t))
            class_total[t] += 1
            if p != t:
                class_errors[t] += 1
                confusion_pairs[(t, p)] += 1
                sent_ok = False
        if cur:
            words.append(cur)
        if sent_ok:
            n_sent_exact += 1

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

    per_class_der = {c: class_errors[c] / class_total[c]
                      for c in class_total if class_total[c] > 0}

    return {
        "DER": char_errors / max(total_chars, 1),
        "DER_star": char_errors_star / max(total_chars_star, 1),
        "WER": word_errors / max(total_words, 1),
        "WER_star": word_errors_star / max(total_words_star, 1),
        "sentence_exact_match": n_sent_exact / max(n_sent, 1),
        "per_class_der": per_class_der,
        "top_confusions": confusion_pairs.most_common(15),
        "n_chars": total_chars, "n_words": total_words, "n_sentences": n_sent,
    }


# ## 7. Training Utilities

class FocalLoss(nn.Module):
    '''FIXED: pt is now the real (unweighted) softmax probability of the
    true class, computed before any class weight is applied. The original
    version derived pt = exp(-ce) from an already class-weighted ce, so the
    focal modulation term was distorted for every up-weighted rare class --
    exactly the classes this loss exists to help.'''
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.gamma, self.weight, self.ignore_index = gamma, weight, ignore_index

    def forward(self, logits, target):
        mask = target != self.ignore_index
        if not mask.any():
            return logits.sum() * 0.0

        logp_all = torch.log_softmax(logits, dim=-1)
        safe_target = target.clone()
        safe_target[~mask] = 0
        logp_true = logp_all.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)

        pt = logp_true.exp().clamp(min=1e-8, max=1.0)   # real, UNWEIGHTED prob of true class
        ce = -logp_true

        if self.weight is not None:
            alpha_t = self.weight[safe_target]
            ce = ce * alpha_t                             # weight applied AFTER pt is computed

        focal = ((1 - pt) ** self.gamma) * ce
        return focal[mask].mean()


def build_loss_fn(cfg: TrainConfig, device: str) -> nn.Module:
    weight = torch.ones(cfg.num_classes, device=device)
    for c in cfg.rare_class_ids:
        weight[c] = cfg.rare_class_weight
    if cfg.use_focal_loss:
        return FocalLoss(gamma=cfg.focal_gamma, weight=weight)
    return nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX,
                                label_smoothing=cfg.label_smoothing)


def rdrop_kl(logits1: torch.Tensor, logits2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    '''Bidirectional KL between two dropout-perturbed forward passes of the
    same batch, restricted to non-ignored (non-space, non-pad) positions.'''
    p1, p2 = torch.log_softmax(logits1, -1), torch.log_softmax(logits2, -1)
    kl = (torch.nn.functional.kl_div(p1, p2, log_target=True, reduction="none").sum(-1)
          + torch.nn.functional.kl_div(p2, p1, log_target=True, reduction="none").sum(-1)) / 2
    kl = kl[mask]
    return kl.mean() if mask.any() else torch.tensor(0.0, device=logits1.device)


def build_layerwise_optimizer(model: Track3Diacritizer, cfg: TrainConfig):
    '''Discriminative fine-tuning: deeper (earlier) transformer layers get
    progressively smaller learning rates than the classification head.'''
    named_params = list(model.named_parameters())
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

    FIXED: `best.pt` used to duplicate the FULL state (model + optimizer
    moments + scheduler, ~3x model size) on every improvement, on top of
    `latest.pt` already holding the same full state. With 5 main-training
    folds + 5 self-training folds never cleaned up, that is enough to fill
    /kaggle/working and crash torch.save mid-write with a confusing
    "unexpected pos X vs Y" RuntimeError (that error is PyTorch's way of
    surfacing "ran out of disk space", not a code/data bug). `best.pt` only
    needs the model weights -- load_best() never reads anything else -- so
    it is now saved model-only, and `latest.pt` (which DOES need the
    optimizer/scheduler, but only for resuming mid-training) is deletable
    once a fold's training is finished and its weights are safely loaded
    into MODELS_FOR_INFERENCE.'''

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.run_dir / "latest.pt"
        self.best_path = self.run_dir / "best.pt"

    def save(self, model, optimizer, scheduler, epoch: int, global_step: int,
              best_dev_score: float, is_best: bool = False):
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch, "global_step": global_step, "best_dev_score": best_dev_score,
        }
        torch.save(state, self.ckpt_path)
        if is_best:
            # model-only -- best.pt is for inference (load_best), never for
            # resuming an interrupted training step, so no optimizer/scheduler
            # duplication needed here.
            torch.save({"model": state["model"], "epoch": epoch, "global_step": global_step,
                        "best_dev_score": best_dev_score}, self.best_path)

    def load_latest(self, model, optimizer, scheduler):
        if not self.ckpt_path.exists():
            return {"epoch": 0, "global_step": 0, "best_dev_score": -1.0}
        state = torch.load(self.ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state["scheduler"]:
            scheduler.load_state_dict(state["scheduler"])
        print(f"Resumed from checkpoint: epoch={state['epoch']} step={state['global_step']} "
              f"best_dev_score={state['best_dev_score']:.4f}")
        return state

    def load_best(self, model):
        state = torch.load(self.best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        return state

    def free_latest(self):
        '''Deletes latest.pt (full state incl. optimizer/scheduler) once this
        fold/run's training is finished and its weights are already loaded
        elsewhere (e.g. into MODELS_FOR_INFERENCE). Safe to call repeatedly;
        keeps best.pt (small, model-only) intact for later loading/export.'''
        if self.ckpt_path.exists():
            freed_mb = self.ckpt_path.stat().st_size / 1e6
            self.ckpt_path.unlink()
            print(f"Freed {freed_mb:.0f} MB: removed {self.ckpt_path}")


def report_disk_usage(path: Path = Path("/kaggle/working")) -> None:
    '''Prints free/used disk space for the working directory. Call this
    before/after any stage that writes several large checkpoints (k-fold
    training, self-training) so a slow disk-space leak shows up in the logs
    well before it crashes a torch.save call.'''
    try:
        total, used, free = shutil.disk_usage(path)
        print(f"[disk] {path}: {used/1e9:.1f} GB used, {free/1e9:.1f} GB free "
              f"of {total/1e9:.1f} GB total")
    except FileNotFoundError:
        print(f"[disk] {path} does not exist (not running on Kaggle / local dry-run)")

report_disk_usage()


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

        fold_train_ds = DiacritizationDataset(fold_train, aligner, CHAR2ID)
        fold_val_ds = DiacritizationDataset(fold_val, aligner, CHAR2ID)
        fold_train_loader = DataLoader(fold_train_ds, batch_size=CFG.batch_size,
                                        shuffle=True, collate_fn=_collate)
        fold_val_loader = DataLoader(fold_val_ds, batch_size=CFG.eval_batch_size,
                                      shuffle=False, collate_fn=_collate)

        fold_model = Track3Diacritizer(
            BACKBONE_NAME, len(CHAR2ID), CFG.num_classes, CFG.char_emb_dim,
            n_concat_layers=CFG.n_concat_layers, head_hidden_dim=CFG.head_hidden_dim,
            head_dropout=CFG.head_dropout, use_deep_head=CFG.use_deep_head,
        ).to(DEVICE)
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
            tr_loss, tr_f1, f_step, stopped = run_train_epoch(
                fold_model, fold_train_loader, fold_optimizer, fold_scheduler,
                loss_fn, CFG, fold_ckpt, epoch, f_step)
            if stopped:
                break
            v_loss, v_logits, v_labels = run_eval(fold_model, fold_val_loader, loss_fn, CFG)
            v_f1 = np.mean([micro_f1_from_counts(l, y) for l, y in zip(v_logits, v_labels)])
            print(f"  fold {fold} epoch {epoch+1}/{CFG.epochs} | train_f1={tr_f1:.4f} "
                  f"| val_loss={v_loss:.4f} val_f1={v_f1:.4f}")

            is_best_f1 = v_f1 > f_best
            if is_best_f1:
                f_best = v_f1
            improved = (v_loss < f_best_loss) if CFG.early_stop_metric == "dev_loss" else is_best_f1
            if v_loss < f_best_loss:
                f_best_loss = v_loss
            f_patience = 0 if improved else f_patience + 1

            fold_ckpt.save(fold_model, fold_optimizer, fold_scheduler, epoch + 1, f_step, f_best,
                            is_best=is_best_f1)
            if f_patience >= CFG.early_stop_patience:
                print(f"  fold {fold}: early stopping.")
                break

        print(f"Fold {fold} done. Best val micro-F1: {f_best:.4f}")

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
        enc = aligner.encode(chars)
        input_ids = torch.tensor([enc["input_ids"]], device=DEVICE)
        attn = torch.ones_like(input_ids)
        toks = torch.tensor([[t if t >= 0 else 0 for t in enc["token_idx_per_char"][:len(chars)]]],
                             device=DEVICE)
        char_ids = torch.tensor([[char2id.get(c, char2id.get('<UNK>', 1)) for c in chars]], device=DEVICE)

        probs_sum = None
        for m in models:
            logits = m(input_ids, attn, char_ids, toks)[0]
            probs = torch.softmax(logits, dim=-1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        probs_avg = probs_sum / len(models)
        conf, pred = probs_avg.max(-1)

        labels, keep = [], True
        for i, c in enumerate(chars):
            if c == SPACE_CHAR:
                labels.append(0)
                continue
            if conf[i].item() < conf_threshold:
                keep = False
                break
            labels.append(pred[i].item())
        if keep:
            pseudo_records.append({"sent_id": f"pseudo_{sent_id}", "chars": chars, "labels": labels})
    return pseudo_records


def continue_finetune(m, ckpt_dir: Path, base_train_records: List[dict], pseudo_records: List[dict],
                       val_loader, start_score: float,
                       original_best_path: Optional[Path] = None) -> Tuple[float, bool]:
    '''Continue-trains one model on (base_train_records + pseudo_records) for
    up to CFG.self_training_epochs, model-selecting by val_f1 against
    val_loader. Returns (best_val_f1_achieved, improved_over_start_score).

    FIXED: `m` is trained in place every epoch below, so if self-training
    never beats start_score, `m` was previously left holding the LAST
    epoch's weights, not the pre-self-training ones -- "kept original" was
    only true of the printed score, not of the actual model in memory. Pass
    `original_best_path` (the pre-self-training best.pt for this exact
    model) so it can be reloaded when self-training does not help.'''
    augmented = base_train_records + pseudo_records
    ds = DiacritizationDataset(augmented, aligner, CHAR2ID)
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True, collate_fn=_collate)

    opt = build_layerwise_optimizer(m, CFG)
    total_steps = len(loader) * CFG.self_training_epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(CFG.warmup_ratio * total_steps), num_training_steps=total_steps)
    st_ckpt = CheckpointManager(ckpt_dir)
    state = st_ckpt.load_latest(m, opt, sched)
    step, best_f1 = state["global_step"], start_score

    for epoch in range(state["epoch"], CFG.self_training_epochs):
        tr_loss, tr_f1, step, stopped = run_train_epoch(m, loader, opt, sched, loss_fn, CFG,
                                                          st_ckpt, epoch, step)
        if stopped:
            break
        v_loss, v_logits, v_labels = run_eval(m, val_loader, loss_fn, CFG)
        v_f1 = np.mean([micro_f1_from_counts(l, y) for l, y in zip(v_logits, v_labels)])
        print(f"    epoch {epoch+1}/{CFG.self_training_epochs} | train_f1={tr_f1:.4f} | val_f1={v_f1:.4f}")
        is_best = v_f1 > best_f1
        st_ckpt.save(m, opt, sched, epoch + 1, step, max(best_f1, v_f1), is_best=is_best)
        if is_best:
            best_f1 = v_f1

    improved = st_ckpt.best_path.exists() and best_f1 > start_score
    if improved:
        st_ckpt.load_best(m)
    elif original_best_path is not None and original_best_path.exists():
        # roll `m` back to its pre-self-training weights -- without this,
        # "kept original" was a lie about what's actually in memory.
        _orig_state = torch.load(original_best_path, map_location=DEVICE, weights_only=False)
        m.load_state_dict(_orig_state["model"])

    st_ckpt.free_latest()   # full state (incl. optimizer/scheduler) no longer
                             # needed once this fold/run's self-training is decided
    return best_f1, improved


if CFG.self_training_enabled:
    with open(PATHS.raw_test_txt, "r", encoding="utf-8") as f:
        raw_test_sentences = [l.rstrip("\n") for l in f if l.strip()]

    pseudo_records = pseudo_label_sentences(MODELS_FOR_INFERENCE, raw_test_sentences, aligner,
                                             CHAR2ID, CFG.self_training_conf_threshold)
    print(f"Pseudo-labeled {len(pseudo_records)}/{len(raw_test_sentences)} sentences "
          f"above confidence {CFG.self_training_conf_threshold} "
          f"(using {len(MODELS_FOR_INFERENCE)}-model confidence)")

    if len(pseudo_records) == 0:
        print("No sentences cleared the confidence threshold -- lower "
              "CFG.self_training_conf_threshold or skip self-training for this run.")
    elif CFG.k_folds > 1 and FOLD_CHECKPOINT_DIRS:
        print(f"\nContinuing fine-tune for up to {CFG.self_training_epochs} epochs on EACH of "
              f"{CFG.k_folds} folds (pseudo-labels shared across folds, base train split per-fold)...")
        report_disk_usage()
        fold_improved_flags = []
        for fold in range(CFG.k_folds):
            print(f"\n  -- fold {fold} --")
            fm = MODELS_FOR_INFERENCE[fold]
            fold_val_ds = DiacritizationDataset(FOLD_VAL_RECORDS[fold], aligner, CHAR2ID)
            fold_val_loader = DataLoader(fold_val_ds, batch_size=CFG.eval_batch_size,
                                          shuffle=False, collate_fn=_collate)
            st_dir = PATHS.checkpoint_dir / f"{RUN_ID}_fold{fold}_selftrain"
            # start_score uses a fresh eval on this fold's own val split (fair baseline per fold)
            base_loss, base_logits, base_labels = run_eval(fm, fold_val_loader, loss_fn, CFG)
            base_f1 = np.mean([micro_f1_from_counts(l, y) for l, y in zip(base_logits, base_labels)])
            best_f1, improved = continue_finetune(
                fm, st_dir, FOLD_TRAIN_RECORDS[fold], pseudo_records, fold_val_loader, base_f1,
                original_best_path=Path(FOLD_CHECKPOINT_DIRS[fold]) / "best.pt")
            fold_improved_flags.append(improved)
            print(f"  fold {fold}: base_val_f1={base_f1:.4f} -> best_val_f1={best_f1:.4f} "
                  f"({'adopted' if improved else 'kept original'})")

        report_disk_usage()
        st_report = evaluator.evaluate(
            MODELS_FOR_INFERENCE, dev_test_loader,
            predict_fn=lambda chars: _predict_chars(MODELS_FOR_INFERENCE, chars),
            word_metric_records=dev_test_records,
        )
        print(f"\nSelf-trained fold-ensemble DEV_TEST micro-F1: {st_report['micro_f1']:.4f}  "
              f"(baseline was {DEV_TEST_SCORE:.4f})")
        if st_report["micro_f1"] > DEV_TEST_SCORE:
            print("Self-training improved the fold-ensemble DEV_TEST score -- adopting it.")
            DEV_TEST_REPORT = st_report
            DEV_TEST_SCORE = st_report["micro_f1"]
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
        best_f1, improved = continue_finetune(st_model, st_dir, train_records, pseudo_records,
                                                dev_loader, DEV_TEST_SCORE,
                                                original_best_path=ckpt.best_path)

        st_report = evaluator.evaluate(
            [st_model], dev_test_loader,
            predict_fn=lambda chars: _predict_chars([st_model], chars),
            word_metric_records=dev_test_records,
        )
        print(f"\nSelf-trained DEV_TEST micro-F1: {st_report['micro_f1']:.4f}  "
              f"(baseline was {DEV_TEST_SCORE:.4f})")
        if st_report["micro_f1"] > DEV_TEST_SCORE:
            print("Self-training improved DEV_TEST -- adopting it as the final model/report.")
            DEV_TEST_REPORT = st_report
            DEV_TEST_SCORE = st_report["micro_f1"]
            MODELS_FOR_INFERENCE = [st_model]
        else:
            print("Self-training did not improve DEV_TEST -- keeping the original model/report.")
else:
    print("Self-training disabled (CFG.self_training_enabled = False). Skipping.")


print()
print_eval_report(DEV_TEST_REPORT, CLASS_NAMES, name="FINAL (post self-training decision)")


# ## 11. Inference on `KAGGLE_TEST` and Submission File

CLASS_ID_TO_DIACRITIC = {i: ("" if mark in ("(none)", "") else mark)
                          for i, mark in enumerate(CLASS_MARKS)}
print("Derived class_id -> diacritic mapping:", CLASS_ID_TO_DIACRITIC)

@torch.no_grad()
def diacritize_sentences(models: List[nn.Module], sentences: List[str], aligner: CharAligner,
                          char2id: Dict[str, int]) -> List[str]:
    for m in models:
        m.eval()
    outputs = []
    for text in sentences:
        text = clean_arabic_text(text)
        chars = list(text)
        if not chars:
            outputs.append("")
            continue
        enc = aligner.encode(chars)
        input_ids = torch.tensor([enc["input_ids"]], device=DEVICE)
        attn = torch.ones_like(input_ids)
        toks = torch.tensor([[t if t >= 0 else 0 for t in enc["token_idx_per_char"][:len(chars)]]],
                             device=DEVICE)
        char_ids = torch.tensor([[char2id.get(c, char2id.get('<UNK>', 1)) for c in chars]], device=DEVICE)

        probs_sum = None
        for m in models:
            logits = m(input_ids, attn, char_ids, toks)[0]
            probs = torch.softmax(logits, dim=-1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        preds = (probs_sum / len(models)).argmax(-1).tolist()

        out_chars = []
        for c, p in zip(chars, preds):
            out_chars.append(c if c == SPACE_CHAR else c + CLASS_ID_TO_DIACRITIC.get(p, ""))
        outputs.append("".join(out_chars))
    return outputs


with open(PATHS.raw_test_txt, "r", encoding="utf-8") as f:
    kaggle_test_sentences = [l.rstrip("\n") for l in f if l.strip()]

diacritized_output = diacritize_sentences(MODELS_FOR_INFERENCE, kaggle_test_sentences, aligner, CHAR2ID)

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
            f"DEV_TEST macro-F1 (all 16 classes): {dev_test_report.get('macro_f1_all16', float('nan')):.4f}\n"
            f"DEV_TEST macro-F1 ({dev_test_report.get('n_present_classes', '?')} classes present): "
            f"{dev_test_report.get('macro_f1_present_classes', float('nan')):.4f}\n"
            f"DER: {dev_test_report.get('DER', float('nan')):.4f}  "
            f"DER*: {dev_test_report.get('DER_star', float('nan')):.4f}\n"
            f"WER: {dev_test_report.get('WER', float('nan')):.4f}  "
            f"WER*: {dev_test_report.get('WER_star', float('nan')):.4f}\n"
            f"Sentence exact-match: {dev_test_report.get('sentence_exact_match', float('nan')):.4f}\n"
            f"Models ensembled: {dev_test_report.get('n_models_ensembled', 1)}\n"
            f"Chars evaluated: {dev_test_report['n_chars_evaluated']}\n"
            f"Self-training used: {cfg.self_training_enabled}\n"
            f"Weights included: {cfg.export_include_weights}\n"
            f"use_deep_head: {cfg.use_deep_head} | n_concat_layers: {cfg.n_concat_layers} | "
            f"char_emb_dim: {cfg.char_emb_dim} | head_hidden_dim: {cfg.head_hidden_dim}\n")
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
