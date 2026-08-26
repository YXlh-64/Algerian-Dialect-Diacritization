"""Fine-tune CANINE-S for Algerian dialect diacritization.

The default preset is the best configuration recorded in ``canine.ipynb``:
the factorized two-head model, 512-character sequences, unweighted loss,
cosine decay with warmup, and early stopping on official dev accuracy.

Examples (from the repository root)::

    python run_pipeline.py --track track2 --head-type canine_twohead \
        --model canine_s_twohead --data-dir /path/to/data \
        --skip-install --skip-torch --skip-data-fetch

    python training/track2/canine_twohead/finetune_canine_twohead.py \
        --train-on-all --predict-test --data-dir ./data
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    CanineConfig,
    CanineTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.track2.canine_twohead import CanineTwoHeadForDiacritization
from utils.track2.canine_twohead.data_utils import (
    CLASS_NAMES,
    NUM_LABELS,
    CanineDiacritizationDataset,
    find_test_files,
    load_jsonl,
    normalise_record,
    resolve_dataset_dir,
)


MODEL_REGISTRY = {
    "canine_s_twohead": {
        "checkpoint": "google/canine-s",
        "head_type": "twohead",
        "max_length": 512,
        "batch_size": 16,
        "eval_batch_size": 32,
        "learning_rate": 3e-5,
        "epochs": 30,
        "patience": 5,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "head_dropout": 0.1,
    }
}

ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064A\u0679-\u06D3]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-model", choices=MODEL_REGISTRY, default="canine_s_twohead")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Export directory; defaults to working/exports/track2/canine_s_twohead")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-on-all", action="store_true",
                        help="Train on train+dev after selecting the documented preset.")
    parser.add_argument("--predict-test", action="store_true",
                        help="Write a submission after training; intended with --train-on-all.")
    parser.add_argument("--submission-path", type=str, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


class BestModelKeeper(TrainerCallback):
    """Keep an exact copy of the best weights and avoid fragile reloads.

    CANINE checkpoints can serialize LayerNorm parameters under legacy names;
    restoring the in-memory state avoids mixing best-epoch and last-epoch
    parameters.
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.best: float | None = None
        self.best_epoch: float | None = None
        self.best_state: dict[str, torch.Tensor] | None = None
        self.bad_evals = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        value = (metrics or {}).get("eval_micro_f1")
        model = kwargs.get("model")
        if model is None or value is None or not np.isfinite(value):
            return
        if self.best is None or value > self.best:
            self.best = float(value)
            self.best_epoch = float(state.epoch)
            self.bad_evals = 0
            self.best_state = {
                key: value.detach().to("cpu", copy=True)
                for key, value in model.state_dict().items()
            }
        else:
            self.bad_evals += 1
            if self.bad_evals >= self.patience:
                control.should_training_stop = True

    def restore(self, model) -> None:
        if self.best_state is not None:
            model.load_state_dict({
                key: value.to(model.device)
                for key, value in self.best_state.items()
            })


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    mask = labels != -100
    y_true, y_pred = labels[mask], predictions[mask]
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    return {
        "accuracy": accuracy,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")) if y_true.size else 0.0,
        "DER": 1.0 - accuracy,
    }


def _first_jsonl(directory: Path, split: str) -> Path:
    matches = sorted(directory.joinpath(split).glob("*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No JSONL file found in {directory / split}")
    return matches[0]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_model(preset: dict[str, Any], tokenizer: CanineTokenizer) -> CanineTwoHeadForDiacritization:
    id2label = {index: name for index, name in enumerate(CLASS_NAMES)}
    label2id = {name: index for index, name in enumerate(CLASS_NAMES)}
    config = CanineConfig.from_pretrained(
        preset["checkpoint"],
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )
    config.head_type = preset["head_type"]
    config.head_dropout = preset["head_dropout"]
    model = CanineTwoHeadForDiacritization.from_pretrained(
        preset["checkpoint"], config=config
    )
    # Keep tokenizer in the signature so callers cannot accidentally create a
    # model with a different preprocessing choice without noticing.
    del tokenizer
    return model


def _train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    preset: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, float], int | None]:
    has_eval = eval_dataset is not None
    steps_per_epoch = math.ceil(len(train_dataset) / preset["batch_size"])
    training_kwargs = dict(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=preset["batch_size"],
        per_device_eval_batch_size=preset["eval_batch_size"],
        learning_rate=preset["learning_rate"],
        num_train_epochs=preset["epochs"],
        lr_scheduler_type="cosine",
        warmup_steps=int(preset["warmup_ratio"] * steps_per_epoch * preset["epochs"]),
        weight_decay=preset["weight_decay"],
        save_strategy="no",
        logging_steps=100,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        report_to="none",
        disable_tqdm=False,
        seed=args.seed,
    )
    # Transformers 4.x called this ``evaluation_strategy``; newer releases
    # use ``eval_strategy``.  Keep the experiment runnable in both the local
    # environment and the repository's pinned dependency set.
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_key] = "epoch" if has_eval else "no"
    training_args = TrainingArguments(**training_kwargs)
    keeper = BestModelKeeper(preset["patience"]) if has_eval else None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer, label_pad_token_id=-100
        ),
        compute_metrics=compute_metrics if has_eval else None,
        callbacks=[keeper] if keeper is not None else None,
    )
    started = time.time()
    trainer.train()
    print(f"training completed in {time.time() - started:.0f}s")

    best_epoch = None
    metrics: dict[str, float] = {}
    if keeper is not None:
        keeper.restore(model)
        best_epoch = int(round(keeper.best_epoch)) if keeper.best_epoch is not None else None
        metrics = {
            "dev_accuracy": float(keeper.best or 0.0),
            "dev_micro_f1": float(keeper.best or 0.0),
            "dev_DER": float(1.0 - (keeper.best or 0.0)),
        }
        print(
            f"best dev accuracy={metrics['dev_accuracy']:.4f} "
            f"DER={metrics['dev_DER']:.4f} epoch={best_epoch}"
        )

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, best_epoch


@torch.no_grad()
def _predict_test(model, tokenizer, data_dir: Path, max_length: int, batch_size: int):
    """Return submission rows while preserving the competition's char IDs."""

    text_path, ids_path, sample_path = find_test_files(data_dir)
    raw_sentences = text_path.read_text(encoding="utf-8").split("\n")
    sent_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    while len(raw_sentences) > len(sent_ids) and raw_sentences[-1].strip() == "":
        raw_sentences.pop()
    if len(raw_sentences) != len(sent_ids):
        raise ValueError(f"{len(raw_sentences)} test sentences vs {len(sent_ids)} IDs")

    def clean(text: str) -> list[str]:
        return [char for char in text if char == " " or ARABIC_LETTER_RE.match(char)]

    records = [{"sent_id": sid, "chars": clean(text)} for sid, text in zip(sent_ids, raw_sentences)]
    model.eval()
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        encoded = tokenizer(
            ["".join(record["chars"]) for record in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        predictions = model(**encoded).logits.argmax(dim=-1).cpu().tolist()
        for record, prediction in zip(batch, predictions):
            n = min(len(record["chars"]), len(prediction) - 2)
            for char_index, (char, label) in enumerate(zip(record["chars"][:n], prediction[1 : 1 + n])):
                if char != " ":
                    rows.append({"id": f"{record['sent_id']}_{char_index}", "label": int(label)})

    if sample_path is not None:
        sample = pd.read_csv(sample_path)
        id_col, label_col = sample.columns[:2]
        submission = pd.DataFrame(rows).rename(columns={"id": id_col, "label": label_col})
        if len(submission) != len(sample) or set(submission[id_col]) != set(sample[id_col]):
            raise ValueError("Generated submission IDs do not match sample_submission.csv")
    else:
        submission = pd.DataFrame(rows)
    return submission


def main() -> None:
    args = parse_args()
    preset = dict(MODEL_REGISTRY[args.active_model])
    for key, value in {
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "patience": args.patience,
    }.items():
        if value is not None:
            preset[key] = value

    set_seed(args.seed)
    data_dir = resolve_dataset_dir(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "working/exports/track2/canine_s_twohead"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = load_jsonl(_first_jsonl(data_dir, "train_data"))
    dev_records = load_jsonl(_first_jsonl(data_dir, "dev_data"))
    fit_records = train_records + dev_records if args.train_on_all else train_records
    eval_records = None if args.train_on_all else dev_records

    tokenizer = CanineTokenizer.from_pretrained(preset["checkpoint"])
    train_dataset = CanineDiacritizationDataset(
        fit_records, tokenizer, max_length=preset["max_length"]
    )
    eval_dataset = (
        CanineDiacritizationDataset(eval_records, tokenizer, max_length=preset["max_length"])
        if eval_records is not None
        else None
    )
    model = _build_model(preset, tokenizer)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    metrics, best_epoch = _train(
        model, tokenizer, train_dataset, eval_dataset, preset, output_dir, args
    )

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _write_json(
        output_dir / "training_config.json",
        {
            "track": "track2",
            "head_type": "canine_twohead",
            "model": args.active_model,
            "checkpoint": preset["checkpoint"],
            "max_length": preset["max_length"],
            "epochs": preset["epochs"],
            "best_epoch": best_epoch,
            "learning_rate": preset["learning_rate"],
            "batch_size": preset["batch_size"],
            "seed": args.seed,
            "train_on_all": args.train_on_all,
        },
    )
    if metrics:
        _write_json(output_dir / "dev_metrics.json", metrics)
        print(json.dumps(metrics, indent=2))

    if args.predict_test:
        submission = _predict_test(
            model,
            tokenizer,
            data_dir,
            max_length=preset["max_length"],
            batch_size=preset["eval_batch_size"],
        )
        submission_path = Path(args.submission_path or output_dir / "submission.csv")
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_path, index=False)
        print(f"saved submission: {submission_path} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
