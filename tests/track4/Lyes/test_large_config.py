from pathlib import Path

import torch

from utils.track4.Lyes.config import load_config
from utils.track4.Lyes.data import BatchCollator, load_jsonl, load_vocab
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer, ModelConfig


ROOT = Path(__file__).resolve().parents[3]


def test_dziriformer_large_11m_configuration_and_forward_pass() -> None:
    config = load_config(ROOT / "configs" / "track4" / "Lyes" /  "dziriformer_large_11m.json")
    vocab = load_vocab(ROOT / config["data"]["vocab"])
    model_config = ModelConfig.from_mapping(
        config["model"], len(vocab), vocab["<PAD>"]
    )
    model = CharDiacritizer(model_config).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 11_000_000 <= parameter_count < 12_000_000

    records = load_jsonl(ROOT / config["data"]["dev"])[:2]
    batch = BatchCollator(vocab)(records)
    with torch.inference_mode():
        logits = model(
            batch["input_ids"], batch["attention_mask"]
        )["logits"]
    assert logits.shape == (2, batch["input_ids"].size(1), 16)
    assert torch.isfinite(logits).all()
