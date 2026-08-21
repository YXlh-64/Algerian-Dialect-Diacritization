import itertools
from pathlib import Path

import torch

from utils.track4.Lyes.config import load_config
from utils.track4.Lyes.data import BatchCollator, load_jsonl, load_vocab
from models.track4.Lyes.dual_stream_crf_head import (
    CharDiacritizer,
    LinearChainCRF,
    ModelConfig,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "track4" / "Lyes" / "model.json"


def _enumerated_scores(
    crf: LinearChainCRF,
    emissions: torch.Tensor,
) -> list[tuple[tuple[int, ...], torch.Tensor]]:
    results = []
    for path in itertools.product(
        range(crf.num_labels), repeat=emissions.size(0)
    ):
        score = crf.start_transitions[path[0]] + emissions[0, path[0]]
        for index in range(1, len(path)):
            score = (
                score
                + crf.transitions[path[index - 1], path[index]]
                + emissions[index, path[index]]
            )
        score = score + crf.end_transitions[path[-1]]
        results.append((path, score))
    return results


def test_crf_partition_and_viterbi_match_brute_force_with_interior_mask() -> None:
    crf = LinearChainCRF(num_labels=2)
    with torch.no_grad():
        crf.start_transitions.copy_(torch.tensor([0.2, -0.1]))
        crf.end_transitions.copy_(torch.tensor([-0.3, 0.4]))
        crf.transitions.copy_(torch.tensor([[0.5, -0.2], [0.1, 0.3]]))
    emissions = torch.tensor(
        [[[0.3, -0.4], [9.0, 9.0], [-0.2, 0.7]]]
    )
    mask = torch.tensor([[True, False, True]])
    packed = emissions[0, [0, 2]]
    enumerated = _enumerated_scores(crf, packed)
    expected_partition = torch.logsumexp(
        torch.stack([score for _, score in enumerated]), dim=0
    )
    expected_path = max(
        enumerated, key=lambda item: float(item[1].detach())
    )[0]

    assert torch.allclose(
        crf.log_partition(emissions, mask)[0],
        expected_partition,
        atol=1.0e-6,
    )
    decoded = crf.decode(emissions, mask)
    assert tuple(decoded[0, [0, 2]].tolist()) == expected_path
    assert int(decoded[0, 1]) == 0


def test_crf_gold_score_and_marginals_are_exact() -> None:
    crf = LinearChainCRF(num_labels=2)
    emissions = torch.tensor(
        [[[0.1, 0.4], [8.0, 8.0], [0.7, -0.2]]],
        requires_grad=True,
    )
    targets = torch.tensor([[1, -100, 0]])
    mask = torch.tensor([[True, False, True]])
    enumerated = _enumerated_scores(crf, emissions[0, [0, 2]])
    gold = dict(enumerated)[(1, 0)]
    assert torch.allclose(
        crf.gold_score(emissions, targets, mask)[0], gold
    )

    log_marginals = crf.log_marginals(emissions, mask)
    assert torch.allclose(
        log_marginals[mask].exp().sum(dim=-1),
        torch.ones(2),
        atol=1.0e-6,
    )
    loss = crf.negative_log_likelihood(emissions, targets, mask)
    loss.backward()
    assert emissions.grad is not None
    assert torch.isfinite(emissions.grad).all()


def test_full_dual_rope_crf_config_forward_backward_and_decode() -> None:
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
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == 9_890_096
    )

    records = load_jsonl(ROOT / config["data"]["dev"])[:2]
    batch = BatchCollator(vocab)(records)
    outputs = model(batch["input_ids"], batch["attention_mask"])
    loss = model.compute_loss(
        outputs, batch["targets"], shadda_loss_weight=1.0
    )
    loss.backward()
    predictions = model.decode_outputs(outputs)
    assert predictions.shape == batch["targets"].shape
    assert torch.isfinite(loss)
    assert model.crf.transitions.grad is not None
    assert outputs["crf_mask"][:, 1:-1].sum() == batch[
        "targets"
    ].ne(-100).sum()
