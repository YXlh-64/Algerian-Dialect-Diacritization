from pathlib import Path

from experiments.track4.Lyes.architecture_v10_evaluate import (
    _gate_results,
    load_evaluation_config,
)


def test_architecture_v10_evaluation_contract_is_strict() -> None:
    config = load_evaluation_config(
        Path("configs/track4/Lyes/architecture_v10/evaluation.json")
    )
    assert set(config["experiments"]) == {
        "wordpos_crf",
        "factorized_crf",
        "low_rank_boundary_crf",
    }
    assert config["controls"]["production_v2_correct"] == 14977


def test_architecture_gate_requires_every_diagnostic() -> None:
    controls = load_evaluation_config(
        Path("configs/track4/Lyes/architecture_v10/evaluation.json")
    )["controls"]
    paper = {
        "correct_letters": 14831,
        "word_correct": 3040,
        "shadda": {"accuracy": 0.983},
        "tanween": {"accuracy": 0.9998},
    }
    diagnostics = {"oov_correct": 2746}
    assert _gate_results(controls, paper, diagnostics)["all_passed"] is True
    diagnostics["oov_correct"] = 2745
    assert _gate_results(controls, paper, diagnostics)["all_passed"] is False
