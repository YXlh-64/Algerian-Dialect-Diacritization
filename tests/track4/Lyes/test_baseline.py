from pathlib import Path

import pytest

from experiments.track4.Lyes.baselines import LexiconBaseline, evaluate
from utils.track4.Lyes.data import load_jsonl


DATA_ROOT = Path(__file__).resolve().parents[3] / "Data"


def test_lexicon_baseline_regression() -> None:
    train = load_jsonl(
        DATA_ROOT / "train_data" / "train_Algerian-DIAC.jsonl"
    )
    dev = load_jsonl(DATA_ROOT / "dev_data" / "dev_Algerian-DIAC.jsonl")
    baseline = LexiconBaseline().fit(train)
    result = evaluate(dev, baseline.predict(dev))
    assert result["total"] == 15897
    assert result["correct"] == 13965
    assert result["micro_f1"] == pytest.approx(0.8784676354)
