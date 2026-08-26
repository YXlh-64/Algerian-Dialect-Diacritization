from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from transformers import CanineForTokenClassification

from utils.track2.canine_s_model.data_utils import CLASS_NAMES, LABEL_TO_MARKS


def build_label_mapping() -> tuple[dict, dict]:
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
    return id2label, label2id


def load_model(model_name: str = "google/canine-s", model_dir: str | Path | None = None):
    if model_dir is not None:
        return CanineForTokenClassification.from_pretrained(str(model_dir))
    id2label, label2id = build_label_mapping()
    return CanineForTokenClassification.from_pretrained(
        model_name,
        num_labels=16,
        id2label=id2label,
        label2id=label2id,
    )


def save_model_artifacts(model_dir: str | Path, tokenizer, metrics: dict, label_to_marks: Dict[int, str] | None = None):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(model_dir))

    if label_to_marks is None:
        label_to_marks = LABEL_TO_MARKS

    with (model_dir / "label_to_marks.json").open("w", encoding="utf-8") as handle:
        json.dump(label_to_marks, handle, ensure_ascii=False, indent=2)

    with (model_dir / "dev_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def diacritize(chars: Iterable[str], labels: Iterable[int]) -> str:
    return "".join(ch + LABEL_TO_MARKS.get(int(label), "") for ch, label in zip(chars, labels))
