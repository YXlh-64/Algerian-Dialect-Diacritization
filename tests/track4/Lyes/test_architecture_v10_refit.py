from experiments.track4.Lyes.architecture_v10_refit import choose_candidate


def test_refit_selection_ignores_rejected_architectures() -> None:
    selected = choose_candidate(
        [
            {
                "architecture_accepted": False,
                "experiment": "wordpos_crf",
                "system_name": "Rejected",
                "neural_correct": 16000,
                "parameter_count": 1,
            }
        ]
    )
    assert selected["slug"] == "dual_rope_crf_v7_control"


def test_refit_selection_uses_correct_then_parameter_count() -> None:
    selected = choose_candidate(
        [
            {
                "architecture_accepted": True,
                "experiment": "factorized_crf",
                "system_name": "Factorized",
                "neural_correct": 14840,
                "parameter_count": 9888554,
            },
            {
                "architecture_accepted": True,
                "experiment": "low_rank_boundary_crf",
                "system_name": "Low rank",
                "neural_correct": 14840,
                "parameter_count": 9890160,
            },
        ]
    )
    assert selected["slug"] == "factorized_crf"
