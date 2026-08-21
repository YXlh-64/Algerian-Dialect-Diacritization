import json
from pathlib import Path

import pytest

from utils.track4.Lyes.data import SentenceRecord
from evaluation.track4.Lyes.paper_model_report import (
    load_paper_registry,
    write_prediction_jsonl,
)
from evaluation.track4.Lyes.paper_metrics import load_prediction_jsonl


def test_paper_model_registry_is_complete_and_unique() -> None:
    registry = load_paper_registry(Path("configs/track4/Lyes/paper_models.json"))
    assert len(registry["models"]) == 16
    assert registry["models"][-1]["slug"] == (
        "dual_rope_low_rank_boundary_crf_v10_seed42"
    )


def test_prediction_jsonl_round_trip(tmp_path: Path) -> None:
    records = [
        SentenceRecord(
            sent_id="000001",
            chars=tuple("ب ت"),
            labels=(1, 0, 7),
            input_text="ب ت",
        )
    ]
    predictions = [[1, 0, 7]]
    path = tmp_path / "predictions.jsonl"
    write_prediction_jsonl(path, records, predictions)
    loaded, texts = load_prediction_jsonl(path, records)
    assert loaded == predictions
    assert texts is None


def test_prediction_jsonl_rejects_length_mismatch(tmp_path: Path) -> None:
    records = [
        SentenceRecord(
            sent_id="000001",
            chars=tuple("ب"),
            labels=(1,),
            input_text="ب",
        )
    ]
    with pytest.raises(ValueError, match="length"):
        write_prediction_jsonl(
            tmp_path / "predictions.jsonl", records, [[1, 2]]
        )


def test_registry_loader_rejects_duplicate_slug(tmp_path: Path) -> None:
    registry = json.loads(
        Path("configs/track4/Lyes/paper_models.json").read_text(encoding="utf-8")
    )
    registry["models"][1]["slug"] = registry["models"][0]["slug"]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_paper_registry(path)
