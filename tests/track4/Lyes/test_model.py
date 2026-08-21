import pytest
import torch

from models.track4.Lyes.dual_stream_crf_head import (
    CharDiacritizer,
    FullSelfAttention,
    ModelConfig,
    WindowSelfAttention,
    build_guided_label_hints,
    build_word_features,
)


@pytest.mark.parametrize(
    "architecture", ["plain_transformer", "conv_local_transformer"]
)
@pytest.mark.parametrize("factorized", [False, True])
def test_model_forward_loss_and_backward(
    architecture: str, factorized: bool
) -> None:
    config = ModelConfig(
        vocab_size=43,
        pad_id=0,
        architecture=architecture,
        d_model=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        max_length=32,
        attention_window=4,
        conv_kernels=(3, 5),
        factorized_head=factorized,
    )
    model = CharDiacritizer(config)
    input_ids = torch.tensor(
        [[2, 5, 4, 6, 3, 0, 0], [2, 7, 8, 9, 10, 11, 3]]
    )
    attention_mask = input_ids.ne(0)
    targets = torch.tensor(
        [
            [-100, 1, -100, 7, -100, -100, -100],
            [-100, 0, 9, 3, 13, 15, -100],
        ]
    )
    outputs = model(input_ids, attention_mask)
    assert outputs["logits"].shape == (2, 7, 16)
    assert torch.isfinite(outputs["logits"]).all()
    loss = model.compute_loss(outputs, targets, shadda_loss_weight=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_length_above_configured_limit_is_rejected() -> None:
    config = ModelConfig(
        vocab_size=43,
        pad_id=0,
        d_model=16,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        max_length=4,
        attention_window=4,
        conv_kernels=(3,),
    )
    model = CharDiacritizer(config)
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.ones((1, 5), dtype=torch.long), torch.ones((1, 5), dtype=torch.bool))


def test_gated_joint_head_is_a_normalized_learned_mixture() -> None:
    config = ModelConfig(
        vocab_size=43,
        pad_id=0,
        architecture="conv_local_transformer",
        d_model=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        max_length=32,
        attention_window=4,
        conv_kernels=(3,),
        head_mode="gated_joint",
    )
    model = CharDiacritizer(config)
    input_ids = torch.tensor([[2, 14, 5, 4, 7, 3]])
    attention_mask = input_ids.ne(0)
    targets = torch.tensor([[-100, 1, 9, -100, 7, -100]])
    outputs = model(input_ids, attention_mask)

    assert outputs["joint_logits"].shape == (1, 6, 16)
    assert outputs["head_gate"].shape == (1, 6, 1)
    assert torch.allclose(
        outputs["logits"].exp().sum(dim=-1),
        torch.ones((1, 6)),
        atol=1.0e-6,
    )
    loss = model.compute_loss(outputs, targets, shadda_loss_weight=99.0)
    loss.backward()
    assert model.head_gate is not None
    assert model.head_gate.weight.grad is not None
    assert model.label_head is not None
    assert model.label_head.weight.grad is not None
    assert model.base_head is not None
    assert model.base_head.weight.grad is not None


def test_direct_head_is_isolated_from_factorized_heads() -> None:
    config = ModelConfig(
        vocab_size=43,
        pad_id=0,
        architecture="conv_local_transformer",
        d_model=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        max_length=32,
        attention_window=4,
        conv_kernels=(3,),
        factorized_head=False,
        head_mode="direct",
    )
    model = CharDiacritizer(config)
    assert model.label_head is not None
    assert model.base_head is None
    assert model.shadda_head is None
    assert model.head_gate is None

    input_ids = torch.tensor([[2, 14, 5, 4, 7, 3]])
    attention_mask = input_ids.ne(0)
    targets = torch.tensor([[-100, 1, 9, -100, 7, -100]])
    outputs = model(input_ids, attention_mask)
    loss = model.compute_loss(outputs, targets, shadda_loss_weight=99.0)
    loss.backward()

    assert model.label_head.weight.grad is not None
    assert torch.isfinite(model.label_head.weight.grad).all()


def test_guided_hints_use_blank_zero_and_label_plus_one() -> None:
    torch.manual_seed(17)
    targets = torch.tensor(
        [[-100, 0, 7, 15, -100], [-100, 2, 9, 1, -100]]
    )
    hints = build_guided_label_hints(targets, mask_steps=10)
    assert hints.shape == targets.shape
    assert hints[:, 0].eq(0).all()
    assert hints[:, -1].eq(0).all()
    for hint, target in zip(hints.reshape(-1), targets.reshape(-1)):
        assert int(hint) in (0, int(target) + 1)


def test_hierarchical_word_features_preserve_word_boundaries() -> None:
    input_ids = torch.tensor([[2, 14, 5, 4, 7, 8, 9, 3, 0]])
    attention_mask = input_ids.ne(0)
    content, word_ids, forward, reverse = build_word_features(
        input_ids,
        attention_mask,
        space_id=4,
        bos_id=2,
        eos_id=3,
        max_word_length=32,
    )
    assert content.tolist() == [
        [False, True, True, False, True, True, True, False, False]
    ]
    assert word_ids.tolist() == [[-1, 0, 0, -1, 1, 1, 1, -1, -1]]
    assert forward.tolist() == [[0, 0, 1, 0, 0, 1, 2, 0, 0]]
    assert reverse.tolist() == [[0, 1, 0, 0, 2, 1, 0, 0, 0]]


def test_hierarchical_model_forward_and_backward() -> None:
    config = ModelConfig(
        vocab_size=43,
        pad_id=0,
        architecture="hierarchical_transformer",
        d_model=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        max_length=32,
        attention_window=4,
        conv_kernels=(3,),
        word_num_layers=1,
        word_ffn_dim=64,
    )
    model = CharDiacritizer(config)
    input_ids = torch.tensor([[2, 14, 5, 4, 7, 8, 3]])
    attention_mask = input_ids.ne(0)
    targets = torch.tensor([[-100, 1, 9, -100, 7, 0, -100]])
    outputs = model(input_ids, attention_mask)
    assert outputs["logits"].shape == (1, 7, 16)
    loss = model.compute_loss(outputs, targets, shadda_loss_weight=1.0)
    loss.backward()
    assert model.word_context_encoder is not None
    assert model.word_context_encoder.pool_score.weight.grad is not None


def test_mixed_attention_inserts_periodic_global_blocks() -> None:
    config = ModelConfig(
        vocab_size=43,
        pad_id=0,
        architecture="conv_local_transformer",
        d_model=32,
        num_layers=6,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        max_length=32,
        attention_window=4,
        conv_kernels=(3,),
        global_attention_every=3,
    )
    model = CharDiacritizer(config)
    attention_types = [type(block.attention) for block in model.blocks]
    assert attention_types == [
        WindowSelfAttention,
        WindowSelfAttention,
        FullSelfAttention,
        WindowSelfAttention,
        WindowSelfAttention,
        FullSelfAttention,
    ]
