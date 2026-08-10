# %% [markdown]
# # Track 1 — Focused P2 BiLSTM-CNN-CRF Ensemble
#
# **Goal:** maximize the Track-1 score with one focused, reproducible P2 architecture:
#
# **parallel temporal character CNNs → 3-layer BiLSTM → linear-chain CRF**, trained with five
# independent seeds and blended into one structured ensemble.
#
# The competition metric is macro-F1 across all 16 labels. This notebook therefore optimizes and
# selects models with the same 16-class macro-F1 rather than accuracy. Spaces are context tokens,
# but are excluded from model-selection metrics because they are absent from the submission.
#
# The default `competition` profile trains five P2 seeds, tunes their weights and structured decoder
# on the official dev split, refits each seed on train+dev for its selected epoch count, and writes
# `/kaggle/working/submission.csv`. On Kaggle's **T4 x2** accelerator, one persistent worker is
# assigned to each GPU so two seeds train concurrently; it falls back to sequential execution when
# only one GPU is available. It does not spend runtime benchmarking weaker families.
#
# **No external training data, pretrained transformer, hand-labeled test data, or test-label
# leakage is used.** Soft word priors are learned only from labeled training records.

# %% [markdown]
# ## Why this design
#
# The design is based on primary literature:
#
# - Belinkov & Glass (EMNLP 2015), [Arabic Diacritization with Recurrent Neural
#   Networks](https://aclanthology.org/D15-1274/): character-level bidirectional LSTMs benefit from
#   past and future context; deeper BiLSTMs improved Arabic diacritization.
# - Huang, Xu & Yu (2015), [Bidirectional LSTM-CRF Models for Sequence
#   Tagging](https://arxiv.org/abs/1508.01991): a BiLSTM models input context while a CRF jointly
#   decodes the output sequence.
# - Ma & Hovy (ACL 2016), [End-to-end Sequence Labeling via Bi-directional
#   LSTM-CNNs-CRF](https://aclanthology.org/P16-1101/): CNN representations, bidirectional LSTMs,
#   and CRF decoding form an effective end-to-end tagger.
# - Elmallah et al. (LREC-COLING 2024), [Arabic Diacritization Using Morphologically Informed
#   Character-Level Model](https://aclanthology.org/2024.lrec-main.128/): character RNNs remain
#   competitive for dialectal Arabic, with dropout and early stopping important for generalization.
# - Mohamed & Mubarak (EMNLP 2025), [Advancing Arabic
#   Diacritization](https://aclanthology.org/2025.emnlp-main.846/): a multi-layer character BiLSTM
#   outperformed tested transformer variants. They found CRF did not always help, so this notebook
#   tunes transition strength down to zero when emission-only ensemble decoding scores better.
# - AlKhamissi et al. (WANLP 2020), [Deep
#   Diacritization](https://aclanthology.org/2020.wanlp-1.4/): majority voting improved Arabic
#   diacritization, motivating a diverse architecture ensemble.
# - Cui et al. (CVPR 2019), [Class-Balanced Loss Based on Effective Number of
#   Samples](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html):
#   effective-number weighting is a stable approach for long-tailed labels.
#
# The CNN is adapted to this dataset's character-level formulation: parallel odd-width temporal
# convolutions provide local n-gram features at every character, the BiLSTM supplies sentence
# context, and the CRF learns transitions for exact Viterbi decoding. Multiple seeds provide the
# model diversity that the broader notebook obtained from multiple architecture families.

# %%
from __future__ import annotations

import gc
import itertools
import json
import math
import os
import sys
import argparse
import random
import threading
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from IPython.display import display
except ImportError:
    display = print
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler



REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.track1.bilstm_cnn_crf.bilstm_cnn_crf_model import (
    BiLSTMDiacritizer,
    count_parameters,
)
from evaluation.track1.bilstm_cnn_crf.evaluate_bilstm_cnn_crf import (
    build_sentence_memory,
    build_word_log_priors,
    class_log_prior,
    decode_ensemble,
    metric_summary,
    score_record_predictions,
    tune_ensemble,
    vocalize,
    write_submission,
)


warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_rows", 30)
pd.set_option("display.max_colwidth", 100)


@dataclass
class RunConfig:
    profile: str = "competition"
    seed: int = 2026
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 30
    patience: int = 7
    embedding_dim: int = 128
    boundary_dim: int = 16
    model_dim: int = 256
    cnn_channels: int = 96
    cnn_kernels: tuple[int, ...] = (3, 5, 7)
    hidden_dim: int = 256
    lstm_layers: int = 3
    mlp_dim: int = 256
    dropout: float = 0.30
    learning_rate: float = 2e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    focal_gamma: float = 1.5
    effective_beta: float = 0.999
    max_class_weight: float = 8.0
    crf_aux_weight: float = 0.50
    sampler_max_weight: float = 5.0
    num_workers: int = 2
    max_gpus: int = 2
    parallel_gpu_training: bool = True
    amp: bool = True
    refit_on_full_data: bool = True
    # Disabled because the audit finds a duplicated train/dev sentence with conflicting labels.
    exact_sentence_memory: bool = False
    output_dir: str = "/kaggle/working"


MODEL_REGISTRY = {
    "p2_ensemble": "five-seed character BiLSTM-CNN-CRF ensemble",
}
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--active-model",
    default="p2_ensemble",
    choices=MODEL_REGISTRY,
    help="Track-1 architecture preset selected by run_pipeline.py",
)
_parser.add_argument(
    "--profile",
    default="competition",
    choices=("competition", "smoke"),
    help="Use smoke for a short end-to-end wiring check",
)
_args, _unknown = _parser.parse_known_args()
ACTIVE_MODEL = _args.active_model
PROFILE = _args.profile

CFG = RunConfig(profile=PROFILE)
if PROFILE == "smoke":
    CFG.epochs = 2
    CFG.patience = 2
    CFG.embedding_dim = 48
    CFG.boundary_dim = 8
    CFG.model_dim = 64
    CFG.cnn_channels = 24
    CFG.hidden_dim = 64
    CFG.lstm_layers = 1
    CFG.mlp_dim = 64
    CFG.batch_size = 32
    CFG.eval_batch_size = 64
    CFG.refit_on_full_data = False

MODEL_SPECS = [
    {
        "name": f"p2_bilstm_cnn_crf_seed_{seed}",
        "use_cnn": True,
        "use_crf": True,
        "seed": seed,
    }
    for seed in (3407, 3408, 3409, 3410, 3411)
]

CUDA_DEVICE_COUNT = torch.cuda.device_count()
if CUDA_DEVICE_COUNT:
    TRAINING_DEVICES = [
        torch.device(f"cuda:{index}")
        for index in range(min(CFG.max_gpus, CUDA_DEVICE_COUNT))
    ]
else:
    TRAINING_DEVICES = [torch.device("cpu")]
if not CFG.parallel_gpu_training:
    TRAINING_DEVICES = TRAINING_DEVICES[:1]
DUAL_GPU_ACTIVE = len(TRAINING_DEVICES) > 1
if CFG.output_dir.startswith("/kaggle") and not Path("/kaggle/working").exists():
    CFG.output_dir = str(Path.cwd() / "working/exports/track1/bilstm_cnn_crf/p2_ensemble")
OUTPUT_DIR = Path(CFG.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(
    f"profile={CFG.profile} | devices={[str(device) for device in TRAINING_DEVICES]} | "
    f"parallel_seed_training={DUAL_GPU_ACTIVE}"
)
for device in TRAINING_DEVICES:
    if device.type == "cuda":
        print(f"{device}: {torch.cuda.get_device_name(device.index)}")

# %% [markdown]
# ## 1. Locate and validate the competition data
#
# Attach the competition data to the Kaggle notebook. The path finder also supports this
# repository's local data layout.

# %%
def find_data_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/algerian-dialect-vocalization/Data"),
        Path("/kaggle/input/competitions/algerian-dialect-vocalization"),
        Path("/kaggle/input/algerian-dialect-vocalization/Data"),
        Path("/kaggle/input/algerian-dialect-vocalization"),
        Path("../input/algerian-dialect-vocalization/Data"),
        Path("data"),
        Path("../data"),
        Path("data/kaggle/Data"),
        Path("../data/kaggle/Data"),
    ]
    marker = Path("train_data/train_Algerian-DIAC.jsonl")
    for candidate in candidates:
        if (candidate / marker).exists():
            return candidate.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(
            kaggle_input.glob("**/train_data/train_Algerian-DIAC.jsonl")
        )
        if matches:
            return matches[0].parents[1].resolve()
    raise FileNotFoundError(
        "Could not locate train_data/train_Algerian-DIAC.jsonl. "
        "Attach the Algerian Dialect Vocalization competition data."
    )


DATA_ROOT = find_data_root()
TRAIN_PATH = DATA_ROOT / "train_data/train_Algerian-DIAC.jsonl"
DEV_PATH = DATA_ROOT / "dev_data/dev_Algerian-DIAC.jsonl"
TEST_TEXT_PATH = DATA_ROOT / "test_data/raw_sentences_test.txt"
TEST_IDS_PATH = DATA_ROOT / "test_data/raw_sentences_test_ids.txt"
SAMPLE_SUBMISSION_PATH = DATA_ROOT / "test_data/sample_submission.csv"
VOCAB_PATH = DATA_ROOT / "vocab.json"
print("DATA_ROOT:", DATA_ROOT)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


train_records = read_jsonl(TRAIN_PATH)
dev_records = read_jsonl(DEV_PATH)
vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
test_inputs = TEST_TEXT_PATH.read_text(encoding="utf-8").splitlines()
test_ids = TEST_IDS_PATH.read_text(encoding="utf-8").splitlines()
assert len(test_inputs) == len(test_ids)
test_records = [
    {"sent_id": sent_id, "chars": list(text), "input": text}
    for sent_id, text in zip(test_ids, test_inputs)
]

NUM_LABELS = 16
PAD_ID = vocab["<PAD>"]
UNK_ID = vocab["<UNK>"]
SPACE_ID = vocab[" "]
LABEL_NAMES = [
    "No Diacritic", "Fatha", "Fathatan", "Damma", "Dammatan", "Kasra",
    "Kasratan", "Sukoon", "Shadda", "Shadda+Fatha", "Shadda+Fathatan",
    "Shadda+Damma", "Shadda+Dammatan", "Shadda+Kasra", "Shadda+Kasratan",
    "Shadda+Sukoon",
]


def validate_records(records: list[dict[str, Any]], require_labels: bool) -> None:
    for row_index, record in enumerate(records):
        chars = record["chars"]
        assert "".join(chars) == record["input"], f"input mismatch at row {row_index}"
        assert all(char in vocab for char in chars), f"OOV character at row {row_index}"
        if require_labels:
            labels = record["labels"]
            assert len(chars) == len(labels), f"length mismatch at row {row_index}"
            assert all(0 <= label < NUM_LABELS for label in labels)
            assert all(label == 0 for char, label in zip(chars, labels) if char == " ")


validate_records(train_records, True)
validate_records(dev_records, True)
validate_records(test_records, False)
print(
    f"train={len(train_records):,} | dev={len(dev_records):,} | "
    f"test={len(test_records):,} | vocab={len(vocab)}"
)

# %% [markdown]
# ## 2. Metric-aligned data audit
#
# The official CSV excludes spaces, so all statistics and F1 calculations below use Arabic-letter
# positions only. The long tail is extreme: several legal labels are absent or nearly absent from
# training. A conventional accuracy objective would mostly optimize labels 0, 1, and 7.

# %%
def letter_label_counts(records: list[dict[str, Any]]) -> np.ndarray:
    counts = np.zeros(NUM_LABELS, dtype=np.int64)
    for record in records:
        for char, label in zip(record["chars"], record["labels"]):
            if char != " ":
                counts[label] += 1
    return counts


def length_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    lengths = np.array([len(record["chars"]) for record in records])
    return {
        "sentences": len(lengths), "positions": int(lengths.sum()),
        "min": int(lengths.min()), "median": float(np.median(lengths)),
        "mean": float(lengths.mean()), "p95": float(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
    }


train_counts = letter_label_counts(train_records)
dev_counts = letter_label_counts(dev_records)
distribution = pd.DataFrame({
    "id": np.arange(NUM_LABELS),
    "label": LABEL_NAMES,
    "train_count": train_counts,
    "train_percent": 100 * train_counts / train_counts.sum(),
    "dev_count": dev_counts,
    "dev_percent": 100 * dev_counts / dev_counts.sum(),
})
display(distribution.style.format({"train_percent": "{:.4f}", "dev_percent": "{:.4f}"}))
display(pd.DataFrame(
    [length_summary(train_records), length_summary(dev_records)],
    index=["train", "dev"],
))

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(distribution["id"], distribution["train_count"].clip(lower=0.5))
ax.set_yscale("log")
ax.set_xticks(range(NUM_LABELS))
ax.set_xlabel("Label ID")
ax.set_ylabel("Training letters (log scale)")
ax.set_title("Extreme label imbalance")
plt.show()


def iter_words(record: dict[str, Any], include_labels: bool = True):
    chars = record["chars"]
    labels = record.get("labels")
    start = 0
    for index, char in enumerate(chars + [" "]):
        if char == " ":
            if index > start:
                word = "".join(chars[start:index])
                word_labels = tuple(labels[start:index]) if include_labels else None
                yield word, word_labels, start, index
            start = index + 1


train_word_types = {word for record in train_records for word, _, _, _ in iter_words(record)}
dev_words = [word for record in dev_records for word, _, _, _ in iter_words(record)]
coverage = np.mean([word in train_word_types for word in dev_words])
exact_train_dev_overlap = len(
    {record["input"] for record in train_records}
    & {record["input"] for record in dev_records}
)
print(f"dev word-type coverage from train: {coverage:.2%}")
print(f"exact train/dev sentence overlap: {exact_train_dev_overlap}")

# %% [markdown]
# **Validation policy**
#
# - Classes without dev support receive F1=0 under an explicit 16-label macro average. This makes
#   the dev score conservative but consistent with the stated metric.
# - Train/dev remain untouched for architecture and epoch selection.
# - Only after selection are models refit on train+dev.
# - Soft word priors use train during dev tuning and train+dev for final test inference.
# - Hard sentence memorization is disabled because duplicated labeled sentences can conflict.

# %% [markdown]
# ## 3. Reproducibility, batching, and rare-class sampling

# %%
def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.seed)


def boundary_features(chars: list[str]) -> list[int]:
    """0=space, 1=word-start, 2=word-middle, 3=word-end, 4=single-letter word."""
    features = []
    for index, char in enumerate(chars):
        if char == " ":
            features.append(0)
            continue
        at_start = index == 0 or chars[index - 1] == " "
        at_end = index == len(chars) - 1 or chars[index + 1] == " "
        if at_start and at_end:
            features.append(4)
        elif at_start:
            features.append(1)
        elif at_end:
            features.append(3)
        else:
            features.append(2)
    return features


class DiacritizationDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], vocabulary: dict[str, int]):
        # Pre-encode once. This removes repeated Python tokenization from every epoch and lets both
        # GPU workers spend more time in CUDA kernels.
        self.items = []
        for index, record in enumerate(records):
            chars = record["chars"]
            item = {
                "index": index,
                "sent_id": record["sent_id"],
                "chars": chars,
                "tokens": torch.tensor(
                    [vocabulary.get(char, UNK_ID) for char in chars],
                    dtype=torch.long,
                ),
                "boundaries": torch.tensor(boundary_features(chars), dtype=torch.long),
                "spaces": torch.tensor(
                    [char == " " for char in chars], dtype=torch.bool
                ),
            }
            if "labels" in record:
                item["labels"] = torch.tensor(record["labels"], dtype=torch.long)
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([len(item["tokens"]) for item in items], dtype=torch.long)
    max_length = int(lengths.max())
    batch_size = len(items)
    tokens = torch.full((batch_size, max_length), PAD_ID, dtype=torch.long)
    boundaries = torch.zeros((batch_size, max_length), dtype=torch.long)
    spaces = torch.zeros((batch_size, max_length), dtype=torch.bool)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    has_labels = "labels" in items[0]
    labels = torch.zeros((batch_size, max_length), dtype=torch.long) if has_labels else None

    for row, item in enumerate(items):
        length = len(item["tokens"])
        tokens[row, :length] = item["tokens"]
        boundaries[row, :length] = item["boundaries"]
        spaces[row, :length] = item["spaces"]
        mask[row, :length] = True
        if has_labels:
            labels[row, :length] = item["labels"]

    return {
        "indices": [item["index"] for item in items],
        "sent_ids": [item["sent_id"] for item in items],
        "chars": [item["chars"] for item in items],
        "tokens": tokens,
        "boundaries": boundaries,
        "spaces": spaces,
        "mask": mask,
        "lengths": lengths,
        "labels": labels,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("tokens", "boundaries", "spaces", "mask", "lengths", "labels"):
        if moved.get(key) is not None:
            moved[key] = moved[key].to(device, non_blocking=True)
    return moved


def sentence_sampling_weights(
    records: list[dict[str, Any]], max_weight: float
) -> torch.DoubleTensor:
    counts = letter_label_counts(records)
    reference = float(np.median(counts[counts > 0]))
    class_weights = np.ones(NUM_LABELS, dtype=np.float64)
    for label, count in enumerate(counts):
        if count > 0:
            class_weights[label] = min(max_weight, math.sqrt(reference / count))
    class_weights = np.maximum(class_weights, 1.0)
    weights = []
    for record in records:
        present = {
            label for char, label in zip(record["chars"], record["labels"]) if char != " "
        }
        weights.append(max([class_weights[label] for label in present] or [1.0]))
    return torch.as_tensor(weights, dtype=torch.double)


def make_loader(
    records: list[dict[str, Any]],
    batch_size: int,
    training: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    dataset = DiacritizationDataset(records, vocab)
    sampler = None
    if training:
        generator = torch.Generator()
        generator.manual_seed(seed)
        weights = sentence_sampling_weights(records, CFG.sampler_max_weight)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=generator
        )
    # CUDA plus subprocess workers created from concurrent notebook threads is fragile. The data is
    # tiny and pre-encoded, so worker=0 is faster and safer during dual-GPU execution.
    loader_workers = 0 if DUAL_GPU_ACTIVE else CFG.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
        persistent_workers=loader_workers > 0,
    )

# %% [markdown]
# ## 4. Exact metric and class-balanced focal auxiliary loss
#
# CRF likelihood learns sequence consistency, while an auxiliary class-balanced focal loss keeps
# rare emissions visible to the optimizer. The softmax model uses the focal objective alone.

# %%




def effective_number_weights(
    records: list[dict[str, Any]], beta: float, cap: float
) -> torch.Tensor:
    counts = letter_label_counts(records).astype(np.float64)
    weights = np.zeros(NUM_LABELS, dtype=np.float64)
    present = counts > 0
    weights[present] = (1.0 - beta) / (1.0 - np.power(beta, counts[present]))
    weights[present] /= weights[present].mean()
    weights[present] = np.clip(weights[present], 0.25, cap)
    return torch.tensor(weights, dtype=torch.float32)



# %% [markdown]
# ## 5. Linear-chain CRF
#
# `transitions[from_label, to_label]` is learned jointly with neural emissions. Spaces stay inside
# the sequence and are hard-constrained to label 0.

# %%

# %% [markdown]
# ## 6. Focused P2 BiLSTM-CNN-CRF architecture
#
# Boundary embeddings are deterministic features derived only from spaces. They do not use an
# external segmenter or annotation.

# %%




for spec in MODEL_SPECS:
    preview = BiLSTMDiacritizer(
        len(vocab), NUM_LABELS, spec["use_cnn"], spec["use_crf"], CFG, PAD_ID
    )
    print(f'{spec["name"]}: {count_parameters(preview):,} parameters')
    del preview

# %% [markdown]
# ## 7. Training and inference utilities

# %%
MODEL_INITIALIZATION_LOCK = threading.Lock()


def initialize_model(
    spec: dict[str, Any],
    device: torch.device,
) -> BiLSTMDiacritizer:
    """Deterministically initialize a model without racing over PyTorch's global CPU RNG."""
    with MODEL_INITIALIZATION_LOCK:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec["seed"])
            model = BiLSTMDiacritizer(
                len(vocab), NUM_LABELS, spec["use_cnn"], spec["use_crf"], CFG, PAD_ID
            )
        model = model.to(device)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(spec["seed"])
    return model


def autocast_context(device: torch.device):
    if CFG.amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device: torch.device):
    return torch.cuda.amp.GradScaler(
        enabled=bool(CFG.amp and device.type == "cuda")
    )


def train_epoch(
    model: BiLSTMDiacritizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    class_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    examples = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            loss, _ = model.loss(batch, class_weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(batch["indices"])
        running_loss += float(loss.detach()) * batch_size
        examples += batch_size
    return running_loss / max(examples, 1)


@torch.no_grad()
def predict_records(
    model: BiLSTMDiacritizer,
    records: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    loader = make_loader(records, CFG.eval_batch_size, False, CFG.seed, device)
    outputs: list[Optional[dict[str, Any]]] = [None] * len(records)
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with autocast_context(device):
            emissions = model.emissions(batch)
        paths = model.decode(emissions.float(), batch["mask"])
        log_probabilities = torch.log_softmax(emissions.float(), dim=-1).cpu().numpy()
        for row, record_index in enumerate(raw_batch["indices"]):
            length = int(raw_batch["lengths"][row])
            outputs[record_index] = {
                "sent_id": records[record_index]["sent_id"],
                "chars": records[record_index]["chars"],
                "log_probs": log_probabilities[row, :length].copy(),
                "prediction": np.asarray(paths[row], dtype=np.int64),
            }
    return [output for output in outputs if output is not None]




def transition_snapshot(
    model: BiLSTMDiacritizer,
) -> Optional[dict[str, np.ndarray]]:
    if model.crf is None:
        return None
    return {
        "start": model.crf.start_transitions.detach().float().cpu().numpy().copy(),
        "end": model.crf.end_transitions.detach().float().cpu().numpy().copy(),
        "transitions": model.crf.transitions.detach().float().cpu().numpy().copy(),
    }


def fit_with_validation(
    spec: dict[str, Any],
    training_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model = initialize_model(spec, device)
    class_weights = effective_number_weights(
        training_records, CFG.effective_beta, CFG.max_class_weight
    ).to(device)
    train_loader = make_loader(
        training_records, CFG.batch_size, True, spec["seed"], device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.min_learning_rate
    )
    scaler = make_grad_scaler(device)
    best_score, best_epoch, best_state = -math.inf, 0, None
    stale_epochs, history = 0, []
    started = time.time()

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, class_weights, device
        )
        dev_outputs = predict_records(model, validation_records, device)
        metrics = score_record_predictions(
            validation_records, [output["prediction"] for output in dev_outputs]
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "macro_f1_16": metrics["macro_f1_16"],
            "macro_f1_supported": metrics["macro_f1_supported"],
            "accuracy": metrics["accuracy"],
            "learning_rate": learning_rate,
        })
        print(
            f'{device} | {spec["name"]} | epoch {epoch:02d} | loss={train_loss:.4f} | '
            f'macroF1-16={metrics["macro_f1_16"]:.5f} | '
            f'supported={metrics["macro_f1_supported"]:.5f} | '
            f'acc={metrics["accuracy"]:.5f}'
        )
        if metrics["macro_f1_16"] > best_score + 1e-5:
            best_score = metrics["macro_f1_16"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= CFG.patience:
                print(f'Early stopping {spec["name"]} at epoch {epoch}.')
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    best_outputs = predict_records(model, validation_records, device)
    best_metrics = score_record_predictions(
        validation_records, [output["prediction"] for output in best_outputs]
    )
    transition = transition_snapshot(model)
    checkpoint_path = OUTPUT_DIR / f'{spec["name"]}_selection.pt'
    torch.save({
        "state_dict": best_state,
        "spec": spec,
        "config": asdict(CFG),
        "best_epoch": best_epoch,
        "dev_metrics": {
            key: value for key, value in best_metrics.items()
            if key not in {"per_class_f1", "support"}
        },
    }, checkpoint_path)
    print(
        f'{spec["name"]}: best_epoch={best_epoch}, '
        f'macroF1-16={best_metrics["macro_f1_16"]:.5f}, '
        f'elapsed={(time.time() - started) / 60:.1f} min'
    )
    model.cpu()
    del model
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return {
        "spec": spec,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "history": history,
        "outputs": best_outputs,
        "metrics": best_metrics,
        "transition": transition,
        "checkpoint_path": str(checkpoint_path),
    }


def fit_full_data(
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    epochs: int,
    device: torch.device,
) -> tuple[BiLSTMDiacritizer, list[dict[str, float]]]:
    model = initialize_model(spec, device)
    class_weights = effective_number_weights(
        records, CFG.effective_beta, CFG.max_class_weight
    ).to(device)
    train_loader = make_loader(
        records, CFG.batch_size, True, spec["seed"], device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=CFG.min_learning_rate
    )
    scaler = make_grad_scaler(device)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model, train_loader, optimizer, scaler, class_weights, device
        )
        history.append({"epoch": epoch, "train_loss": loss})
        print(
            f'{device} | {spec["name"]} full | '
            f'epoch {epoch:02d}/{epochs:02d} | loss={loss:.4f}'
        )
        scheduler.step()
    torch.save({
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "spec": spec,
        "config": asdict(CFG),
        "epochs": epochs,
    }, OUTPUT_DIR / f'{spec["name"]}_full.pt')
    return model, history


def run_on_training_devices(
    items: list[Any],
    worker,
    stage_name: str,
) -> list[Any]:
    """Keep one persistent thread per GPU and process that GPU's seeds sequentially."""
    indexed_items = list(enumerate(items))
    assignments = [
        indexed_items[device_index:: len(TRAINING_DEVICES)]
        for device_index in range(len(TRAINING_DEVICES))
    ]

    def run_device_queue(
        device: torch.device,
        queue: list[tuple[int, Any]],
    ) -> list[tuple[int, Any]]:
        completed = []
        for item_index, item in queue:
            print(
                f"\n{'=' * 90}\n"
                f"{stage_name}: item {item_index + 1}/{len(items)} on {device}"
            )
            completed.append((item_index, worker(item, device)))
        return completed

    ordered_results: list[Optional[Any]] = [None] * len(items)
    if len(TRAINING_DEVICES) == 1:
        device_results = run_device_queue(TRAINING_DEVICES[0], assignments[0])
        for item_index, result in device_results:
            ordered_results[item_index] = result
    else:
        with ThreadPoolExecutor(max_workers=len(TRAINING_DEVICES)) as executor:
            futures = [
                executor.submit(run_device_queue, device, queue)
                for device, queue in zip(TRAINING_DEVICES, assignments)
                if queue
            ]
            for future in as_completed(futures):
                for item_index, result in future.result():
                    ordered_results[item_index] = result

    assert all(result is not None for result in ordered_results)
    return list(ordered_results)

# %% [markdown]
# ## 8. Training-only lexical priors and structured ensemble
#
# The training split covers most dev word types. A smoothed distribution over label sequences
# observed for each word is used as a soft emission prior. The BiLSTM can override ambiguous forms.
#
# The transparent ensemble search selects:
#
# 1. simplex weights over the five independently seeded P2 models;
# 2. word-prior strength;
# 3. frequency-logit adjustment for macro-F1;
# 4. CRF transition strength.
#
# Every value is selected only on dev labels.

# %%

















# %% [markdown]
# ## 9. Multi-seed P2 selection on the official dev split
#
# This is the first intensive cell. All five runs use the same P2 architecture and differ only in
# random initialization and sampling order. The rare-class sampler affects only labeled training
# data; validation remains untouched.

# %%
def selection_worker(
    spec: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return fit_with_validation(
        spec,
        train_records,
        dev_records,
        device,
    )


selection_results = run_on_training_devices(
    MODEL_SPECS,
    selection_worker,
    stage_name="dev selection",
)

selection_summary = pd.DataFrame([
    {
        "model": result["spec"]["name"],
        "best_epoch": result["best_epoch"],
        "macro_f1_16": result["metrics"]["macro_f1_16"],
        "macro_f1_supported": result["metrics"]["macro_f1_supported"],
        "accuracy": result["metrics"]["accuracy"],
    }
    for result in selection_results
]).sort_values("macro_f1_16", ascending=False)
display(selection_summary)

fig, ax = plt.subplots(figsize=(10, 5))
for result in selection_results:
    history = pd.DataFrame(result["history"])
    ax.plot(
        history["epoch"], history["macro_f1_16"],
        marker="o", label=result["spec"]["name"],
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Dev macro-F1 (16 classes)")
ax.set_title("Metric-aligned model selection")
ax.legend()
plt.show()

# %%
dev_model_outputs = [result["outputs"] for result in selection_results]
selection_transitions = [result["transition"] for result in selection_results]
ensemble_config, ensemble_dev_predictions, ensemble_search = tune_ensemble(
    dev_model_outputs, selection_transitions, dev_records, train_records
)
ensemble_dev_metrics = score_record_predictions(
    dev_records, ensemble_dev_predictions
)
print("Selected ensemble:")
print({
    **ensemble_config,
    "weights": {
        spec["name"]: float(weight)
        for spec, weight in zip(MODEL_SPECS, ensemble_config["weights"])
    },
})
print(
    f'ensemble macroF1-16={ensemble_dev_metrics["macro_f1_16"]:.5f} | '
    f'supported={ensemble_dev_metrics["macro_f1_supported"]:.5f} | '
    f'accuracy={ensemble_dev_metrics["accuracy"]:.5f}'
)
display(ensemble_search.head(10))
per_class_table = distribution[["id", "label", "dev_count"]].copy()
per_class_table["ensemble_f1"] = ensemble_dev_metrics["per_class_f1"]
display(per_class_table.style.format({"ensemble_f1": "{:.5f}"}))

# %% [markdown]
# ## 10. Refit on train+dev and predict test
#
# Epoch counts and ensemble hyperparameters are frozen. Each seed is retrained from scratch on all
# labeled records, with the same two-GPU queue used during selection. Set
# `refit_on_full_data=False` only to save time.

# %%
full_records = train_records + dev_records


def final_inference_worker(
    result: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    spec = result["spec"]
    if CFG.refit_on_full_data:
        model, history = fit_full_data(
            spec,
            full_records,
            epochs=result["best_epoch"],
            device=device,
        )
    else:
        model = initialize_model(spec, device)
        model.load_state_dict(result["best_state"])
        model.to(device)
        history = []
    outputs = predict_records(model, test_records, device)
    transition = transition_snapshot(model)
    model.cpu()
    del model
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return {
        "spec": spec,
        "history": history,
        "outputs": outputs,
        "transition": transition,
    }


final_jobs = run_on_training_devices(
    selection_results,
    final_inference_worker,
    stage_name="full-data refit and test inference",
)
final_model_outputs = [job["outputs"] for job in final_jobs]
final_transitions = [job["transition"] for job in final_jobs]
full_histories = {
    job["spec"]["name"]: job["history"]
    for job in final_jobs
}

final_predictions = decode_ensemble(
    final_model_outputs,
    test_records,
    final_transitions,
    ensemble_config["weights"],
    build_word_log_priors(full_records),
    ensemble_config["lexical_strength"],
    class_log_prior(full_records),
    ensemble_config["frequency_strength"],
    ensemble_config["transition_strength"],
    build_sentence_memory(full_records),
    exact_sentence_memory=CFG.exact_sentence_memory,
)
assert len(final_predictions) == len(test_records)
for record, prediction in zip(test_records, final_predictions):
    assert len(prediction) == len(record["chars"])
    assert np.all((0 <= prediction) & (prediction < NUM_LABELS))
    assert all(
        predicted == 0
        for char, predicted in zip(record["chars"], prediction)
        if char == " "
    )

# %% [markdown]
# ## 11. Write and validate `submission.csv`
#
# IDs use the original zero-based character position, including spaces in the index; space rows are
# omitted. The assertions compare row order and count with the official sample.

# %%


submission_path = OUTPUT_DIR / "submission.csv"
submission = write_submission(test_records, final_predictions, submission_path)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
assert submission.columns.tolist() == ["Id", "Label"]
assert len(submission) == len(sample_submission)
assert submission["Id"].tolist() == sample_submission["Id"].tolist()
assert submission["Label"].between(0, NUM_LABELS - 1).all()
assert not submission.isna().any().any()
print(f"Wrote {submission_path} with {len(submission):,} rows")
display(submission.head())
display(
    submission["Label"].value_counts().sort_index()
    .rename("test_predictions").to_frame()
)

# %%




vocalized_path = OUTPUT_DIR / "vocalized_predictions.txt"
with vocalized_path.open("w", encoding="utf-8") as handle:
    for record, prediction in zip(test_records, final_predictions):
        handle.write(vocalize(record["chars"], prediction) + "\n")

run_summary = {
    "config": asdict(CFG),
    "models": [
        {
            "spec": result["spec"],
            "best_epoch": result["best_epoch"],
            "dev_macro_f1_16": result["metrics"]["macro_f1_16"],
            "dev_accuracy": result["metrics"]["accuracy"],
        }
        for result in selection_results
    ],
    "ensemble": {
        **ensemble_config,
        "weights": ensemble_config["weights"].tolist(),
        "dev_macro_f1_16": ensemble_dev_metrics["macro_f1_16"],
        "dev_macro_f1_supported": ensemble_dev_metrics["macro_f1_supported"],
        "dev_accuracy": ensemble_dev_metrics["accuracy"],
    },
    "submission": str(submission_path),
}
(OUTPUT_DIR / "run_summary.json").write_text(
    json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Wrote {vocalized_path}")
print(f"Wrote {OUTPUT_DIR / 'run_summary.json'}")

# %% [markdown]
# ## 12. Practical score-improvement order
#
# Run the default notebook once before changing several variables:
#
# 1. **Capacity:** compare `hidden_dim=192/256/320` and `lstm_layers=2/3/4`.
# 2. **Imbalance:** compare `focal_gamma=1.0/1.5/2.0`,
#    `max_class_weight=5/8/12`, and `sampler_max_weight=3/5`.
# 3. **Context:** compare CNN kernels `(3,5)` with `(3,5,7)`.
# 4. **Seeds:** retain the best three to five seeds; more models help only when their errors differ.
# 5. **Lexical prior:** trust the dev-selected strength. Do not tune it from public-leaderboard
#    feedback alone.
# 6. **Final entries:** if two are allowed, keep one conservative single-model submission and one
#    tuned ensemble submission.
#
# High accuracy does not guarantee a high score. Always inspect exact 16-class macro-F1 and
# per-class support/F1.
