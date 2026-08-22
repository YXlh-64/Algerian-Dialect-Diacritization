import itertools
from pathlib import Path

import pytest
import torch

from utils.track4.Lyes.config import load_config
from utils.track4.Lyes.data import BatchCollator, load_jsonl, load_vocab
from experiments.track4.Lyes.dual_rope_boundary_crf_v8 import load_v8_config
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer, LinearChainCRF, ModelConfig


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    ROOT / "configs" / "track4" / "Lyes" /  "dziriformer_dual_rope_boundary_crf_v8.json"
)
CAMPAIGN_PATH = ROOT / "configs" / "track4" / "Lyes" /  "dual_rope_v8" / "campaign.json"


def _enumerated_scores(
    crf: LinearChainCRF,
    emissions: torch.Tensor,
    boundaries: torch.Tensor,
) -> list[tuple[tuple[int, ...], torch.Tensor]]:
    if crf.boundary_transitions is None:
        raise AssertionError("test requires boundary transitions")
    results = []
    for path in itertools.product(
        range(crf.num_labels), repeat=emissions.size(0)
    ):
        score = crf.start_transitions[path[0]] + emissions[0, path[0]]
        for index in range(1, len(path)):
            transitions = (
                crf.boundary_transitions
                if bool(boundaries[index])
                else crf.transitions
            )
            score = (
                score
                + transitions[path[index - 1], path[index]]
                + emissions[index, path[index]]
            )
        score = score + crf.end_transitions[path[-1]]
        results.append((path, score))
    return results


def test_boundary_crf_matches_brute_force_for_partition_gold_and_viterbi() -> None:
    crf = LinearChainCRF(num_labels=2, boundary_conditioned=True)
    assert crf.boundary_transitions is not None
    with torch.no_grad():
        crf.start_transitions.copy_(torch.tensor([0.2, -0.1]))
        crf.end_transitions.copy_(torch.tensor([-0.3, 0.4]))
        crf.transitions.copy_(torch.tensor([[0.6, -0.7], [0.2, 0.1]]))
        crf.boundary_transitions.copy_(
            torch.tensor([[-0.8, 0.9], [0.7, -0.4]])
        )
    emissions = torch.tensor(
        [[[0.3, -0.4], [9.0, 9.0], [-0.2, 0.7]]]
    )
    targets = torch.tensor([[0, -100, 1]])
    mask = torch.tensor([[True, False, True]])
    boundary_mask = torch.tensor([[False, False, True]])
    packed_emissions = emissions[0, [0, 2]]
    packed_boundaries = boundary_mask[0, [0, 2]]
    enumerated = _enumerated_scores(
        crf, packed_emissions, packed_boundaries
    )
    expected_partition = torch.logsumexp(
        torch.stack([score for _, score in enumerated]), dim=0
    )
    expected_path = max(
        enumerated, key=lambda item: float(item[1].detach())
    )[0]
    expected_gold = dict(enumerated)[(0, 1)]

    assert torch.allclose(
        crf.log_partition(emissions, mask, boundary_mask)[0],
        expected_partition,
        atol=1.0e-6,
    )
    assert torch.allclose(
        crf.gold_score(emissions, targets, mask, boundary_mask)[0],
        expected_gold,
        atol=1.0e-6,
    )
    decoded = crf.decode(emissions, mask, boundary_mask)
    assert tuple(decoded[0, [0, 2]].tolist()) == expected_path
    assert int(decoded[0, 1]) == 0


def test_boundary_crf_marginals_match_enumerated_distribution() -> None:
    crf = LinearChainCRF(num_labels=2, boundary_conditioned=True)
    assert crf.boundary_transitions is not None
    with torch.no_grad():
        crf.transitions.copy_(torch.tensor([[0.5, -0.5], [0.1, 0.2]]))
        crf.boundary_transitions.copy_(
            torch.tensor([[-0.6, 0.8], [0.9, -0.3]])
        )
    emissions = torch.tensor(
        [[[0.1, 0.4], [8.0, 8.0], [0.7, -0.2]]],
        requires_grad=True,
    )
    targets = torch.tensor([[1, -100, 0]])
    mask = torch.tensor([[True, False, True]])
    boundary_mask = torch.tensor([[False, False, True]])
    enumerated = _enumerated_scores(
        crf, emissions[0, [0, 2]], boundary_mask[0, [0, 2]]
    )
    score_tensor = torch.stack([score for _, score in enumerated])
    weights = torch.softmax(score_tensor, dim=0)
    expected = torch.zeros(2, 2)
    for (path, _), weight in zip(enumerated, weights):
        expected[0, path[0]] += weight.detach()
        expected[1, path[1]] += weight.detach()

    marginals = crf.log_marginals(
        emissions, mask, boundary_mask
    )[0, [0, 2]].exp()
    assert torch.allclose(marginals, expected, atol=1.0e-6)
    loss = crf.negative_log_likelihood(
        emissions, targets, mask, boundary_mask
    )
    loss.backward()
    assert emissions.grad is not None
    assert crf.transitions.grad is not None
    assert crf.boundary_transitions.grad is not None


def test_boundary_mask_contract_is_strict() -> None:
    emissions = torch.zeros(1, 2, 2)
    mask = torch.tensor([[True, True]])
    boundaries = torch.tensor([[False, True]])
    standard = LinearChainCRF(num_labels=2)
    conditioned = LinearChainCRF(num_labels=2, boundary_conditioned=True)
    with pytest.raises(ValueError, match="does not accept"):
        standard.log_partition(emissions, mask, boundaries)
    with pytest.raises(ValueError, match="requires a boundary"):
        conditioned.log_partition(emissions, mask)
    with pytest.raises(ValueError, match="only mark scored"):
        conditioned.log_partition(
            emissions,
            torch.tensor([[True, False]]),
            boundaries,
        )


def test_full_boundary_crf_config_forward_backward_and_boundary_mask() -> None:
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
    assert model.crf is not None
    assert model.crf.boundary_transitions is not None
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == 9_890_352
    )

    records = load_jsonl(ROOT / config["data"]["dev"])[:2]
    batch = BatchCollator(vocab)(records)
    outputs = model(batch["input_ids"], batch["attention_mask"])
    expected_boundaries = (
        outputs["crf_mask"]
        & torch.nn.functional.pad(
            batch["input_ids"][:, :-1].eq(vocab[" "]),
            (1, 0),
            value=False,
        )
    )
    assert torch.equal(outputs["crf_boundary_mask"], expected_boundaries)
    assert not outputs["crf_boundary_mask"][:, 1].any()

    loss = model.compute_loss(
        outputs, batch["targets"], shadda_loss_weight=1.0
    )
    loss.backward()
    predictions = model.decode_outputs(outputs)
    probabilities = model.probabilities(outputs)
    assert predictions.shape == batch["targets"].shape
    assert torch.isfinite(loss)
    assert model.crf.transitions.grad is not None
    assert model.crf.boundary_transitions.grad is not None
    assert torch.allclose(
        probabilities[outputs["crf_mask"]].sum(dim=-1),
        torch.ones_like(
            probabilities[outputs["crf_mask"]].sum(dim=-1)
        ),
        atol=1.0e-4,
    )


def test_v8_campaign_config_is_strict() -> None:
    config = load_v8_config(CAMPAIGN_PATH)
    assert config["acceptance"] == {
        "boundary_neural_must_exceed_correct": 14816,
        "boundary_v2_must_exceed_correct": 14977,
        "boundary_ensemble_v2_must_exceed_correct": 14977,
        "crossfit_gate_must_exceed_correct": 14978,
    }
    assert config["crossfit_gate"] == {
        "fold_count": 5,
        "fold_seed": 842,
        "lexical_smoothing": 0.01,
        "decision_threshold": 0.5,
    }
