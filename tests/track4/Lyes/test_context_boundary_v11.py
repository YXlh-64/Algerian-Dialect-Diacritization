import itertools
from pathlib import Path

import torch

from utils.track4.Lyes.checkpoint import build_model_from_checkpoint
from utils.track4.Lyes.config import load_config
from experiments.track4.Lyes.context_boundary_v11 import load_evaluation_config
from utils.track4.Lyes.labels import IGNORE_INDEX
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer, LinearChainCRF, ModelConfig


MODEL_CONFIG = Path("configs/track4/Lyes/context_boundary_v11/model.json")
EVALUATION_CONFIG = Path("configs/track4/Lyes/context_boundary_v11/evaluation.json")


def _model() -> CharDiacritizer:
    resolved = load_config(MODEL_CONFIG)
    config = ModelConfig.from_mapping(
        resolved["model"],
        vocab_size=43,
        pad_id=0,
        space_id=4,
        bos_id=2,
        eos_id=3,
    )
    return CharDiacritizer(config)


def _enumerated_context_scores(
    crf: LinearChainCRF,
    emissions: torch.Tensor,
    gates: torch.Tensor,
) -> list[tuple[tuple[int, ...], torch.Tensor]]:
    assert crf.boundary_left is not None
    assert crf.boundary_right is not None
    residual = crf.boundary_left @ crf.boundary_right
    scores = []
    for path in itertools.product(
        range(crf.num_labels), repeat=emissions.size(0)
    ):
        score = crf.start_transitions[path[0]] + emissions[0, path[0]]
        for index in range(1, len(path)):
            transition = crf.transitions + gates[index] * residual
            score = (
                score
                + transition[path[index - 1], path[index]]
                + emissions[index, path[index]]
            )
        score = score + crf.end_transitions[path[-1]]
        scores.append((path, score))
    return scores


def test_context_crf_partition_gold_viterbi_and_marginals_are_exact() -> None:
    crf = LinearChainCRF(
        num_labels=2, boundary_rank=1, context_conditioned=True
    )
    assert crf.boundary_left is not None
    assert crf.boundary_right is not None
    with torch.no_grad():
        crf.start_transitions.copy_(torch.tensor([0.2, -0.1]))
        crf.end_transitions.copy_(torch.tensor([-0.3, 0.4]))
        crf.transitions.copy_(torch.tensor([[0.5, -0.2], [0.1, 0.3]]))
        crf.boundary_left.copy_(torch.tensor([[0.7], [-0.4]]))
        crf.boundary_right.copy_(torch.tensor([[0.6, -0.5]]))
    emissions = torch.tensor(
        [[[0.3, -0.4], [9.0, 9.0], [-0.2, 0.7], [0.1, 0.2]]]
    )
    mask = torch.tensor([[True, False, True, True]])
    gates = torch.tensor([[0.1, 0.0, 0.4, 0.9]])
    targets = torch.tensor([[1, IGNORE_INDEX, 0, 1]])
    packed_emissions = emissions[0, [0, 2, 3]]
    packed_gates = gates[0, [0, 2, 3]]
    enumerated = _enumerated_context_scores(
        crf, packed_emissions, packed_gates
    )
    expected_partition = torch.logsumexp(
        torch.stack([score for _, score in enumerated]), dim=0
    )
    expected_path = max(
        enumerated, key=lambda item: float(item[1].detach())
    )[0]
    expected_gold = dict(enumerated)[(1, 0, 1)]

    assert torch.allclose(
        crf.log_partition(emissions, mask, transition_gate=gates)[0],
        expected_partition,
        atol=1.0e-6,
    )
    assert torch.allclose(
        crf.gold_score(
            emissions, targets, mask, transition_gate=gates
        )[0],
        expected_gold,
        atol=1.0e-6,
    )
    decoded = crf.decode(emissions, mask, transition_gate=gates)
    assert tuple(decoded[0, [0, 2, 3]].tolist()) == expected_path
    marginals = crf.log_marginals(
        emissions, mask, transition_gate=gates
    )
    assert torch.allclose(
        marginals[mask].exp().sum(dim=-1),
        torch.ones(3),
        atol=1.0e-6,
    )


def test_zero_residual_is_exactly_equivalent_to_standard_crf() -> None:
    standard = LinearChainCRF(num_labels=3)
    contextual = LinearChainCRF(
        num_labels=3, boundary_rank=2, context_conditioned=True
    )
    with torch.no_grad():
        standard.start_transitions.normal_()
        standard.end_transitions.normal_()
        standard.transitions.normal_()
        contextual.start_transitions.copy_(standard.start_transitions)
        contextual.end_transitions.copy_(standard.end_transitions)
        contextual.transitions.copy_(standard.transitions)
    emissions = torch.randn(2, 5, 3)
    mask = torch.tensor(
        [[True, False, True, True, False], [False, True, True, False, True]]
    )
    gates = torch.rand(2, 5)

    assert torch.equal(
        contextual._boundary_transition_matrix(), contextual.transitions
    )
    assert torch.allclose(
        standard.log_partition(emissions, mask),
        contextual.log_partition(
            emissions, mask, transition_gate=gates
        ),
        atol=1.0e-6,
    )
    assert torch.equal(
        standard.decode(emissions, mask),
        contextual.decode(emissions, mask, transition_gate=gates),
    )


def test_full_context_boundary_model_isolated_change_and_gradients() -> None:
    model = _model()
    assert model.config.resolved_head_mode == "context_low_rank_boundary_crf"
    assert model.config.crf_boundary_rank == 2
    assert model.crf is not None
    assert model.crf.context_conditioned
    assert model.crf_context_gate is not None
    assert model.crf.boundary_left is not None
    assert model.crf.boundary_right is not None
    assert sum(parameter.numel() for parameter in model.parameters()) == 9_890_418

    input_ids = torch.tensor([[2, 10, 11, 4, 12, 13, 3]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    targets = torch.tensor(
        [[IGNORE_INDEX, 1, 9, IGNORE_INDEX, 7, 0, IGNORE_INDEX]]
    )
    outputs = model(input_ids, attention_mask)
    assert outputs["crf_boundary_indicator"].tolist() == [
        [False, False, False, False, True, False, False]
    ]
    assert torch.equal(
        outputs["crf_transition_gate"][outputs["crf_mask"]],
        torch.full((4,), 0.5),
    )
    assert not outputs["crf_transition_gate"][~outputs["crf_mask"]].any()
    loss = model.compute_loss(outputs, targets, shadda_loss_weight=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.crf.boundary_right.grad is not None
    assert torch.count_nonzero(model.crf.boundary_right.grad) > 0

    with torch.no_grad():
        model.crf.boundary_right.fill_(0.01)
    model.zero_grad(set_to_none=True)
    outputs = model(input_ids, attention_mask)
    model.compute_loss(outputs, targets, shadda_loss_weight=1.0).backward()
    assert model.crf_context_gate.weight.grad is not None
    assert torch.count_nonzero(model.crf_context_gate.weight.grad) > 0


def test_old_crf_checkpoint_without_rank_field_remains_compatible() -> None:
    config = ModelConfig(
        vocab_size=8,
        pad_id=0,
        space_id=4,
        bos_id=2,
        eos_id=3,
        architecture="plain_transformer",
        d_model=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        max_length=16,
        head_mode="crf",
    )
    model = CharDiacritizer(config)
    raw_config = config.to_dict()
    del raw_config["crf_boundary_rank"]
    checkpoint = {
        "schema_version": 1,
        "model_config": raw_config,
        "model_state_dict": model.state_dict(),
        "vocab": {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            " ": 4,
            "ا": 5,
            "ب": 6,
            "ت": 7,
        },
    }
    restored, _ = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
    assert restored.config.crf_boundary_rank == 2
    assert restored.config.resolved_head_mode == "crf"


def test_context_boundary_evaluation_config_is_strict() -> None:
    config = load_evaluation_config(EVALUATION_CONFIG)
    assert config["expected_head_mode"] == "context_low_rank_boundary_crf"
    assert config["controls"]["minimum_neural_gain"] == 15


def test_context_boundary_is_an_isolated_v7_decoder_change() -> None:
    control = load_config(Path("configs/track4/Lyes/model.json"))
    candidate = load_config(MODEL_CONFIG)
    control_model = dict(control["model"])
    candidate_model = dict(candidate["model"])
    assert control_model.pop("head_mode") == "crf"
    assert (
        candidate_model.pop("head_mode")
        == "context_low_rank_boundary_crf"
    )
    assert control_model == candidate_model

    control_training = dict(control["training"])
    candidate_training = dict(candidate["training"])
    control_training.pop("num_workers")
    candidate_training.pop("num_workers")
    assert control_training == candidate_training
    assert control["seed"] == candidate["seed"] == 42
    assert control["data"] == candidate["data"]
