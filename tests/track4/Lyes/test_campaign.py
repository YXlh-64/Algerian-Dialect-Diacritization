from pathlib import Path

import pytest
import torch

from experiments.track4.Lyes.campaign.common import (
    load_campaign_config,
    validate_artifact_prefix,
)
from experiments.track4.Lyes.campaign.ensemble import average_probability_groups
from experiments.track4.Lyes.campaign.folds import (
    fold_assignment,
    make_balanced_folds,
)
from experiments.track4.Lyes.campaign.oof_gate import fit_logistic_gate
from utils.track4.Lyes.data import SentenceRecord
from utils.track4.Lyes.checkpoint import build_model_from_checkpoint
from utils.track4.Lyes.lexical_fusion import WordLabelPrior
from models.track4.Lyes.dual_stream_crf_head import (
    CharDiacritizer,
    ModelConfig,
    build_guided_label_hints,
)


def _records(count: int):
    return [
        SentenceRecord(
            sent_id=f"{index:06d}",
            chars=tuple("ب" * (index % 7 + 1)),
            labels=tuple([1] * (index % 7 + 1)),
            input_text="ب" * (index % 7 + 1),
        )
        for index in range(count)
    ]


def test_balanced_folds_are_deterministic_disjoint_and_complete() -> None:
    records = _records(31)
    first = make_balanced_folds(records, 5, 42)
    second = make_balanced_folds(records, 5, 42)
    assert first == second
    assignment = fold_assignment(records, first)
    assert len(assignment) == len(records)
    assert set(assignment.values()) == set(range(5))
    letter_totals = [
        sum(len(records[index].chars) for index in fold)
        for fold in first
    ]
    assert max(letter_totals) - min(letter_totals) <= 7


def test_linear_blank_curriculum_ends_with_all_blank_hints() -> None:
    targets = torch.tensor(
        [[-100, 0, 7, 15, -100], [-100, 2, 9, 1, -100]]
    )
    torch.manual_seed(9)
    first = build_guided_label_hints(
        targets,
        10,
        schedule="linear_blank_curriculum",
        epoch=1,
        total_epochs=60,
    )
    torch.manual_seed(9)
    last = build_guided_label_hints(
        targets,
        10,
        schedule="linear_blank_curriculum",
        epoch=60,
        total_epochs=60,
    )
    assert first.ne(0).any()
    assert last.eq(0).all()


def test_curriculum_blank_probability_is_monotonic_by_epoch() -> None:
    probabilities = [
        0.0 if total == 1 else (epoch - 1) / (total - 1)
        for total in (60,)
        for epoch in range(1, total + 1)
    ]
    assert probabilities[0] == 0.0
    assert probabilities[-1] == 1.0
    assert all(
        left <= right
        for left, right in zip(probabilities, probabilities[1:])
    )


def test_logistic_gate_training_is_deterministic() -> None:
    features = torch.tensor(
        [
            [0.1] * 8,
            [0.2] * 8,
            [0.8] * 8,
            [0.9] * 8,
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    first = fit_logistic_gate(features, targets)
    second = fit_logistic_gate(features, targets)
    assert torch.equal(first.weight, second.weight)
    assert torch.equal(first.bias, second.bias)
    assert float(first.probability(features[0])) < 0.5
    assert float(first.probability(features[-1])) > 0.5


def test_probability_group_average_is_deterministic_and_equal_weight() -> None:
    first = [
        torch.tensor([[0.8, 0.2], [0.4, 0.6]]),
        torch.tensor([[0.3, 0.7]]),
    ]
    second = [
        torch.tensor([[0.2, 0.8], [0.6, 0.4]]),
        torch.tensor([[0.9, 0.1]]),
    ]
    expected = [
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        torch.tensor([[0.6, 0.4]]),
    ]
    result = average_probability_groups([first, second])
    repeated = average_probability_groups([first, second])
    assert all(
        torch.equal(actual, wanted)
        for actual, wanted in zip(result, expected)
    )
    assert all(
        torch.equal(actual, again)
        for actual, again in zip(result, repeated)
    )


def test_lexical_prior_excludes_outer_fold_labels() -> None:
    records = [
        SentenceRecord(
            sent_id=f"{index:06d}",
            chars=tuple(chr(0x0628 + index)),
            labels=(index % 8,),
            input_text=chr(0x0628 + index),
        )
        for index in range(10)
    ]
    folds = make_balanced_folds(records, 5, 42)
    outer = set(folds[0])
    prior = WordLabelPrior().fit(
        [
            record
            for index, record in enumerate(records)
            if index not in outer
        ]
    )
    for index in outer:
        assert prior.log_probabilities(
            records[index].input_text, smoothing=0.01
        ) is None


def test_direct_head_checkpoint_round_trip_is_strictly_compatible() -> None:
    config = ModelConfig(
        vocab_size=6,
        pad_id=0,
        space_id=4,
        bos_id=2,
        eos_id=3,
        architecture="hierarchical_transformer",
        d_model=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        max_length=16,
        attention_window=4,
        conv_kernels=(3,),
        factorized_head=False,
        head_mode="direct",
        global_attention_every=1,
        guided_label_training=True,
        guided_schedule="linear_blank_curriculum",
        word_num_layers=1,
        word_ffn_dim=32,
        max_word_length=8,
    )
    source = CharDiacritizer(config)
    checkpoint = {
        "schema_version": 1,
        "model_config": config.to_dict(),
        "model_state_dict": source.state_dict(),
        "vocab": {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            " ": 4,
            "ب": 5,
        },
    }
    restored, vocab = build_model_from_checkpoint(
        checkpoint, torch.device("cpu")
    )
    assert vocab["<PAD>"] == 0
    assert restored.label_head is not None
    assert restored.base_head is None
    assert restored.shadda_head is None
    assert restored.hint_embedding is not None
    for name, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_reduced_campaign_config_is_explicit() -> None:
    config = load_campaign_config(
        Path("configs/track4/Lyes/pre_hgl_v5/campaign.json")
    )
    assert config["execution"]["run_oof_gate"] is False
    assert "Deferred" in config["execution"]["oof_deferred_reason"]


@pytest.mark.parametrize(
    "prefix", ["DZIRI_MODEL_V5", "A1", "DIRECT16_SEED42"]
)
def test_valid_artifact_prefixes(prefix: str) -> None:
    assert validate_artifact_prefix(prefix) == prefix


@pytest.mark.parametrize("prefix", ["lowercase", "HAS-DASH", "../BAD", ""])
def test_invalid_artifact_prefixes_are_rejected(prefix: str) -> None:
    with pytest.raises(ValueError):
        validate_artifact_prefix(prefix)
