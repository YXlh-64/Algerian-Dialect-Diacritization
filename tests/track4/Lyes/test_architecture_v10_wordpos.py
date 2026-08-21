from pathlib import Path

import torch

from utils.track4.Lyes.config import load_config
from utils.track4.Lyes.labels import IGNORE_INDEX
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer, ModelConfig, build_word_features


CONFIG_PATH = Path("configs/track4/Lyes/architecture_v10/wordpos_crf_v10.json")


def test_word_positions_reset_at_spaces_and_track_both_directions() -> None:
    input_ids = torch.tensor([[2, 10, 11, 4, 12, 13, 14, 3, 0]])
    mask = input_ids.ne(0)
    content, word_ids, forward, reverse = build_word_features(
        input_ids,
        mask,
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


def test_wordpos_crf_is_an_isolated_input_feature_ablation() -> None:
    resolved = load_config(CONFIG_PATH)
    config = ModelConfig.from_mapping(
        resolved["model"],
        vocab_size=43,
        pad_id=0,
        space_id=4,
        bos_id=2,
        eos_id=3,
    )
    model = CharDiacritizer(config)
    assert config.word_position_features is True
    assert config.resolved_head_mode == "crf"
    assert model.position_embedding is None
    assert model.word_context_encoder is None
    assert model.word_position_embedding is not None
    assert model.reverse_word_position_embedding is not None
    assert model.word_initial_embedding is not None
    assert model.word_final_embedding is not None
    assert sum(parameter.numel() for parameter in model.parameters()) == 9_907_504


def test_wordpos_crf_forward_backward_and_decode() -> None:
    resolved = load_config(CONFIG_PATH)
    config = ModelConfig.from_mapping(
        resolved["model"],
        vocab_size=43,
        pad_id=0,
        space_id=4,
        bos_id=2,
        eos_id=3,
    )
    model = CharDiacritizer(config)
    input_ids = torch.tensor(
        [[2, 10, 11, 4, 12, 13, 3], [2, 14, 4, 15, 16, 17, 3]]
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    targets = torch.tensor(
        [
            [IGNORE_INDEX, 1, 7, IGNORE_INDEX, 0, 9, IGNORE_INDEX],
            [IGNORE_INDEX, 3, IGNORE_INDEX, 5, 8, 0, IGNORE_INDEX],
        ]
    )
    outputs = model(input_ids, attention_mask)
    loss = model.compute_loss(outputs, targets, shadda_loss_weight=1.0)
    loss.backward()
    predictions = model.decode_outputs(outputs)
    assert torch.isfinite(loss)
    assert predictions.shape == input_ids.shape
    assert model.word_position_embedding.weight.grad is not None
    assert model.word_initial_embedding.weight.grad is not None
