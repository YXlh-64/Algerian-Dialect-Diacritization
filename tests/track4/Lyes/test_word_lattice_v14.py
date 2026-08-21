import itertools
from pathlib import Path

import pytest
import torch

from utils.track4.Lyes.data import SentenceRecord
from experiments.track4.Lyes.filtered_word_lattice_v14 import (
    evaluate_oracle,
    load_campaign_config,
)
from models.track4.Lyes.dual_stream_crf_head import CharDiacritizer, LinearChainCRF, ModelConfig
from models.track4.Lyes.word_lattice import (
    WordCandidate,
    WordCandidateTransformer,
    WordLattice,
    base_word_scores,
    build_word_lattice,
    candidate_hash,
    lattice_marginals,
    lattice_viterbi,
)


ROOT = Path(__file__).resolve().parents[3]


def test_k_best_segments_matches_brute_force() -> None:
    crf = LinearChainCRF(3)
    with torch.no_grad():
        crf.transitions.copy_(
            torch.tensor(
                [[0.1, -0.2, 0.3], [0.0, 0.4, -0.1], [-0.3, 0.2, 0.5]]
            )
        )
    emissions = torch.tensor(
        [[0.2, 0.1, -0.4], [0.0, 0.7, 0.3], [0.4, -0.2, 0.1]]
    )
    paths, scores = crf.k_best_segments(emissions, 5)

    expected = []
    for path in itertools.product(range(3), repeat=3):
        score = sum(float(emissions[i, label]) for i, label in enumerate(path))
        score += sum(
                float(crf.transitions[path[i - 1], path[i]].detach())
            for i in range(1, len(path))
        )
        expected.append((score, path))
    expected.sort(key=lambda item: (-item[0], item[1]))
    assert paths.tolist() == [list(path) for _, path in expected[:5]]
    assert scores.tolist() == pytest.approx(
        [score for score, _ in expected[:5]], abs=1e-6
    )


def test_lattice_generation_keeps_baseline_first_and_unique() -> None:
    record = SentenceRecord(
        sent_id="000001",
        chars=("ا", "ب", " ", "ت"),
        labels=(1, 7, 0, 3),
        input_text="اب ت",
    )
    crf = LinearChainCRF(16)
    emissions = torch.zeros(4, 16)
    emissions[0, 1] = 3.0
    emissions[1, 7] = 3.0
    emissions[3, 3] = 3.0
    baseline = [1, 7, 0, 3]
    lattice = build_word_lattice(record, emissions, baseline, crf, 4)
    assert lattice.spans == ((0, 2), (3, 4))
    assert lattice.candidates[0][0].labels == (1, 7)
    assert lattice.candidates[1][0].labels == (3,)
    assert all(
        len({candidate.labels for candidate in group}) == len(group)
        for group in lattice.candidates
    )
    assert all(len(group) == 4 for group in lattice.candidates)


def _tiny_lattice() -> WordLattice:
    first = (
        WordCandidate((0,), 0.3, candidate_hash((0,))),
        WordCandidate((1,), 0.1, candidate_hash((1,))),
    )
    second = (
        WordCandidate((2,), 0.2, candidate_hash((2,))),
        WordCandidate((3,), 0.0, candidate_hash((3,))),
    )
    return WordLattice(
        sent_id="000002",
        char_count=3,
        spans=((0, 1), (2, 3)),
        candidates=(first, second),
        baseline_labels=(0, 0, 2),
    )


def test_lattice_viterbi_and_marginals_match_brute_force() -> None:
    lattice = _tiny_lattice()
    transitions = torch.zeros(16, 16)
    transitions[1, 3] = 0.8
    starts = torch.zeros(16)
    starts[1] = 0.2
    ends = torch.zeros(16)
    ends[3] = 0.1
    scores = base_word_scores(lattice)

    combinations = []
    for first, second in itertools.product(range(2), repeat=2):
        first_label = lattice.candidates[0][first].labels[-1]
        second_label = lattice.candidates[1][second].labels[0]
        score = (
            scores[0][first]
            + starts[first_label]
            + transitions[first_label, second_label]
            + scores[1][second]
            + ends[second_label]
        )
        combinations.append((float(score), first, second))
    combinations.sort(key=lambda row: (-row[0], row[1], row[2]))
    best = combinations[0]
    expected = [
        lattice.candidates[0][best[1]].labels[0],
        0,
        lattice.candidates[1][best[2]].labels[0],
    ]
    assert lattice_viterbi(lattice, scores, transitions, starts, ends) == expected

    marginals = lattice_marginals(lattice, scores, transitions, starts, ends)
    assert torch.allclose(marginals.sum(dim=-1), torch.ones(3), atol=1e-6)
    normalizer = torch.logsumexp(
        torch.tensor([row[0] for row in combinations]), dim=0
    )
    expected_first = torch.zeros(16)
    expected_second = torch.zeros(16)
    for score, first, second in combinations:
        probability = torch.exp(torch.tensor(score) - normalizer)
        expected_first[lattice.candidates[0][first].labels[0]] += probability
        expected_second[lattice.candidates[1][second].labels[0]] += probability
    assert torch.allclose(marginals[0], expected_first, atol=1e-6)
    assert marginals[1, 0] == 1.0
    assert torch.allclose(marginals[2], expected_second, atol=1e-6)


def test_encode_is_forward_compatible() -> None:
    torch.manual_seed(5)
    model = CharDiacritizer(
        ModelConfig(
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
            dropout=0.0,
            max_length=12,
            attention_window=4,
            factorized_head=False,
            head_mode="crf",
        )
    ).eval()
    input_ids = torch.tensor([[2, 5, 6, 3]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    hidden, fusion_gate = model.encode(input_ids, attention_mask)
    outputs = model(input_ids, attention_mask)
    assert fusion_gate is None
    assert torch.equal(outputs["logits"], model.label_head(hidden))


def test_candidate_transformer_backward_and_mask_validation() -> None:
    scorer = WordCandidateTransformer(context_dim=16, d_model=16, num_heads=2, ffn_dim=32)
    context = torch.randn(3, 4, 16)
    labels = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0], [6, 7, 8, 9]])
    mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False], [True, True, True, True]]
    )
    residuals = scorer(context, labels, mask)
    assert residuals.shape == (3,)
    residuals.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in scorer.parameters()
    )
    assert context.grad is None


def test_v14_config_locks_candidate_counts_and_moderate_gate() -> None:
    config = load_campaign_config(
        ROOT / "configs" / "track4" / "Lyes" /  "filtered_word_lattice_v14" / "campaign.json"
    )
    assert config["candidate_counts"] == [4, 8]
    assert config["oracle_gate"] == {
        "minimum_recoverable_letters": 20,
        "minimum_recoverable_exact_words": 10,
    }


def test_oracle_metrics_count_recoverable_letters_and_words() -> None:
    train = [
        SentenceRecord(
            sent_id="000010",
            chars=("ا", "ب"),
            labels=(1, 7),
            input_text="اب",
        )
    ]
    evaluation = [
        SentenceRecord(
            sent_id="000011",
            chars=("ا", "ب"),
            labels=(1, 7),
            input_text="اب",
        )
    ]
    baseline = (0, 7)
    group = (
        WordCandidate(baseline, 1.0, candidate_hash(baseline)),
        WordCandidate((1, 7), 0.9, candidate_hash((1, 7))),
    )
    lattice = WordLattice(
        sent_id="000011",
        char_count=2,
        spans=((0, 2),),
        candidates=(group,),
        baseline_labels=baseline,
    )
    metrics = evaluate_oracle(train, evaluation, [lattice], 4)
    assert metrics["recoverable_correct_letters"] == 1
    assert metrics["recoverable_exact_words"] == 1
    assert metrics["recoverable_exact_sentences"] == 1
