import copy
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from utils.track4.Lyes.config import load_config, validate_config
from utils.track4.Lyes.data import BatchCollator, load_jsonl, load_vocab
from models.track4.Lyes.dual_stream_crf_head import (
    CharDiacritizer,
    ModelConfig,
    RoPEFullSelfAttention,
    RoPEWindowSelfAttention,
    RotaryEmbedding,
)
from training.track4.Lyes.train import _build_scheduler


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "track4" / "Lyes" /  "dziriformer_dual_rope_ce_v6.json"


def _small_model() -> CharDiacritizer:
    return CharDiacritizer(
        ModelConfig(
            vocab_size=43,
            pad_id=0,
            architecture="dual_rope_transformer",
            d_model=32,
            num_layers=3,
            num_heads=4,
            ffn_dim=64,
            dropout=0.0,
            max_length=32,
            attention_window=4,
            conv_kernels=(3,),
            factorized_head=False,
            head_mode="direct",
            dual_local_num_layers=1,
            dual_global_num_layers=1,
            dual_refinement_num_layers=1,
        )
    )


def test_rotary_embedding_preserves_norm_and_relative_geometry() -> None:
    torch.manual_seed(19)
    rotary = RotaryEmbedding(head_dim=8)
    left = torch.randn(1, 1, 1, 8)
    right = torch.randn(1, 1, 1, 8)
    left_at_2 = rotary(left, torch.tensor([[2]]))
    right_at_5 = rotary(right, torch.tensor([[5]]))
    left_at_9 = rotary(left, torch.tensor([[9]]))
    right_at_12 = rotary(right, torch.tensor([[12]]))

    assert torch.allclose(left.norm(), left_at_2.norm(), atol=1.0e-6)
    assert torch.allclose(
        (left_at_2 * right_at_5).sum(),
        (left_at_9 * right_at_12).sum(),
        atol=1.0e-5,
    )


def test_dual_rope_architecture_is_exact_and_has_no_absolute_positions() -> None:
    config = load_config(CONFIG_PATH)
    vocab = load_vocab(ROOT / config["data"]["vocab"])
    model_config = ModelConfig.from_mapping(
        config["model"],
        vocab_size=len(vocab),
        pad_id=vocab["<PAD>"],
        space_id=vocab[" "],
        bos_id=vocab["<BOS>"],
        eos_id=vocab["<EOS>"],
    )
    model = CharDiacritizer(model_config)

    assert model.position_embedding is None
    assert model.conv_frontend is None
    assert model.word_context_encoder is None
    assert model.hint_embedding is None
    assert model.label_head is not None
    assert model.base_head is None
    assert model.shadda_head is None
    assert model.dual_rope_encoder is not None
    encoder = model.dual_rope_encoder
    assert len(encoder.local_blocks) == 6
    assert len(encoder.global_blocks) == 4
    assert len(encoder.refinement_blocks) == 2
    assert all(
        isinstance(block.attention, RoPEWindowSelfAttention)
        for block in encoder.local_blocks
    )
    assert all(
        isinstance(block.attention, RoPEFullSelfAttention)
        for block in (
            list(encoder.global_blocks) + list(encoder.refinement_blocks)
        )
    )
    assert not any(
        "position_embedding" in name for name, _ in model.named_parameters()
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 9_800_000 <= parameter_count < 10_100_000


def test_dual_rope_forward_loss_backward_and_neutral_gate() -> None:
    torch.manual_seed(23)
    model = _small_model()
    input_ids = torch.tensor(
        [[2, 14, 5, 4, 7, 3, 0], [2, 7, 8, 9, 10, 11, 3]]
    )
    attention_mask = input_ids.ne(0)
    targets = torch.tensor(
        [
            [-100, 1, 9, -100, 7, -100, -100],
            [-100, 0, 9, 3, 13, 15, -100],
        ]
    )
    outputs = model(input_ids, attention_mask)

    assert outputs["logits"].shape == (2, 7, 16)
    assert outputs["fusion_gate"].shape == (2, 7, 32)
    assert torch.isfinite(outputs["logits"]).all()
    assert torch.allclose(
        outputs["fusion_gate"][attention_mask],
        torch.full_like(outputs["fusion_gate"][attention_mask], 0.5),
    )
    assert outputs["fusion_gate"][~attention_mask].eq(0.0).all()

    loss = model.compute_loss(outputs, targets, shadda_loss_weight=99.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.dual_rope_encoder is not None
    assert model.dual_rope_encoder.fusion_gate.weight.grad is not None
    assert model.label_head is not None
    assert model.label_head.weight.grad is not None


def test_dual_rope_checkpoint_state_round_trip_is_exact() -> None:
    torch.manual_seed(29)
    original = _small_model().eval()
    restored = _small_model().eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    input_ids = torch.tensor([[2, 14, 5, 4, 7, 3]])
    attention_mask = input_ids.ne(0)
    with torch.inference_mode():
        original_logits = original(input_ids, attention_mask)["logits"]
        restored_logits = restored(input_ids, attention_mask)["logits"]
    assert torch.equal(original_logits, restored_logits)


def test_full_v6_config_runs_real_data_cpu_forward() -> None:
    config = load_config(CONFIG_PATH)
    vocab = load_vocab(ROOT / config["data"]["vocab"])
    records = load_jsonl(ROOT / config["data"]["dev"])[:2]
    batch = BatchCollator(vocab)(records)
    model = CharDiacritizer(
        ModelConfig.from_mapping(
            config["model"],
            vocab_size=len(vocab),
            pad_id=vocab["<PAD>"],
            space_id=vocab[" "],
            bos_id=vocab["<BOS>"],
            eos_id=vocab["<EOS>"],
        )
    ).eval()
    with torch.inference_mode():
        outputs = model(batch["input_ids"], batch["attention_mask"])
    assert outputs["logits"].shape == (
        2,
        batch["input_ids"].size(1),
        16,
    )
    assert torch.isfinite(outputs["logits"]).all()


def test_dual_rope_config_rejects_non_direct_head() -> None:
    config = load_config(CONFIG_PATH)
    invalid = copy.deepcopy(config)
    invalid["model"]["factorized_head"] = True
    invalid["model"]["head_mode"] = "factorized"
    with pytest.raises(ValueError, match="direct or CRF"):
        validate_config(invalid)
    invalid_model_config = ModelConfig(
        vocab_size=43,
        pad_id=0,
        architecture="dual_rope_transformer",
        d_model=32,
        num_layers=3,
        num_heads=4,
        ffn_dim=64,
        head_mode="factorized",
        dual_local_num_layers=1,
        dual_global_num_layers=1,
        dual_refinement_num_layers=1,
    )
    with pytest.raises(ValueError, match="direct or CRF"):
        CharDiacritizer(invalid_model_config)


def test_v6_uses_deterministic_one_cycle_schedule() -> None:
    config = load_config(CONFIG_PATH)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamW(
        [parameter],
        lr=float(config["training"]["learning_rate"]),
    )
    scheduler = _build_scheduler(
        optimizer, config["training"], total_updates=20
    )
    assert isinstance(scheduler, OneCycleLR)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert max(rates) <= config["training"]["learning_rate"]
    assert rates[5] > rates[0]
    assert rates[-1] < rates[5]
