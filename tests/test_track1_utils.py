"""Regression tests for Track-1 shared record, evaluation, and loss helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from evaluation.track1.bilstm_cnn_crf.evaluate_bilstm_cnn_crf import (
    decode_ensemble,
    metric_summary,
    score_record_predictions,
    vocalize,
)
from utils.track1.data import iter_words, letter_label_counts, validate_records

try:
    import torch
    import torch.nn.functional as F

    from models.track1.bilstm_cnn_crf.bilstm_cnn_crf_model import (
        class_balanced_focal_loss,
    )
    from training.track1.bilstm_cnn_crf.data import DataSettings
    from training.track1.bilstm_cnn_crf.engine import TrainingContext, run_training
except ImportError:  # Allows data/evaluation tests in environments without torch.
    torch = None
    F = None
    class_balanced_focal_loss = None
    DataSettings = None
    TrainingContext = None
    run_training = None


class RecordUtilityTests(unittest.TestCase):
    def test_word_spans_and_labels(self) -> None:
        record = {
            "chars": list("اب ج"),
            "labels": [1, 3, 0, 5],
            "input": "اب ج",
        }
        self.assertEqual(
            list(iter_words(record)),
            [("اب", (1, 3), 0, 2), ("ج", (5,), 3, 4)],
        )
        self.assertEqual(
            list(iter_words(record, include_labels=False)),
            [("اب", None, 0, 2), ("ج", None, 3, 4)],
        )

    def test_letter_counts_exclude_spaces(self) -> None:
        records = [
            {"chars": list("اب ج"), "labels": [1, 3, 0, 1]},
        ]
        counts = letter_label_counts(records)
        self.assertEqual(int(counts[0]), 0)
        self.assertEqual(int(counts[1]), 2)
        self.assertEqual(int(counts[3]), 1)

    def test_validation_rejects_nonzero_space_label(self) -> None:
        records = [
            {"input": "ا ب", "chars": list("ا ب"), "labels": [1, 7, 3]},
        ]
        vocabulary = {"ا": 0, " ": 1, "ب": 2}
        with self.assertRaisesRegex(ValueError, "nonzero space label"):
            validate_records(records, vocabulary, require_labels=True)


class EvaluationUtilityTests(unittest.TestCase):
    def test_metric_and_emission_only_decode(self) -> None:
        records = [
            {"sent_id": "x", "input": "اب", "chars": ["ا", "ب"], "labels": [1, 3]},
        ]
        log_probs = np.array(
            [
                [0.0, 3.0] + [-5.0] * 14,
                [0.0, -2.0, -2.0, 4.0] + [-5.0] * 12,
            ]
        )
        predictions = decode_ensemble(
            [[{"log_probs": log_probs}]],
            records,
            [None],
            np.array([1.0]),
            {},
            0.0,
            np.zeros(16),
            0.0,
            0.0,
        )
        self.assertEqual(predictions[0].tolist(), [1, 3])
        self.assertEqual(
            score_record_predictions(records, predictions)["accuracy"], 1.0
        )
        self.assertEqual(metric_summary([0, 1], [0, 1])["accuracy"], 1.0)
        self.assertEqual(vocalize(["ا"], [1]), "اَ")


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FocalLossTests(unittest.TestCase):
    def test_matches_original_formula_on_regular_logits(self) -> None:
        emissions = torch.tensor(
            [[[0.2, 1.3, -0.7], [2.0, -0.5, 0.1]]], dtype=torch.float32
        )
        labels = torch.tensor([[1, 0]])
        mask = torch.tensor([[True, True]])
        weights = torch.tensor([1.0, 2.0, 0.5])
        gamma = 1.5

        selected = emissions[mask]
        selected_labels = labels[mask]
        old_cross_entropy = F.cross_entropy(
            selected, selected_labels, weight=weights, reduction="none"
        )
        old_probability = (
            torch.softmax(selected, dim=-1)
            .gather(1, selected_labels.unsqueeze(1))
            .squeeze(1)
        )
        expected = ((1.0 - old_probability).pow(gamma) * old_cross_entropy).mean()
        actual = class_balanced_focal_loss(emissions, labels, mask, weights, gamma)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_remains_finite_for_large_logits(self) -> None:
        emissions = torch.tensor([[[10000.0, -10000.0, 0.0]]])
        labels = torch.tensor([[0]])
        mask = torch.tensor([[True]])
        loss = class_balanced_focal_loss(
            emissions, labels, mask, torch.ones(3), gamma=1.5
        )
        self.assertTrue(torch.isfinite(loss))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainingLoopTests(unittest.TestCase):
    def test_early_stopping_records_triggering_epoch(self) -> None:
        model = torch.nn.Linear(1, 1)
        config = SimpleNamespace(
            effective_beta=0.999,
            max_class_weight=8.0,
            batch_size=1,
            learning_rate=0.01,
            weight_decay=0.0,
            min_learning_rate=0.001,
            amp=False,
            patience=2,
        )
        settings = DataSettings(vocabulary={"<PAD>": 0}, pad_id=0, unk_id=0)
        context = TrainingContext(config=config, data=settings, output_dir=Path("."))
        validation = [{"chars": ["ا"], "labels": [0]}]
        metric_sequence = [
            {"macro_f1_16": 0.5, "macro_f1_supported": 0.5, "accuracy": 0.5},
            {"macro_f1_16": 0.4, "macro_f1_supported": 0.4, "accuracy": 0.4},
            {"macro_f1_16": 0.3, "macro_f1_supported": 0.3, "accuracy": 0.3},
        ]
        with (
            patch(
                "training.track1.bilstm_cnn_crf.engine.effective_number_weights",
                return_value=torch.ones(16),
            ),
            patch("training.track1.bilstm_cnn_crf.engine.make_loader", return_value=[]),
            patch(
                "training.track1.bilstm_cnn_crf.engine.train_epoch", return_value=1.0
            ),
            patch(
                "training.track1.bilstm_cnn_crf.engine.predict_records",
                return_value=[{"prediction": np.array([0])}],
            ),
            patch(
                "training.track1.bilstm_cnn_crf.engine.score_record_predictions",
                side_effect=metric_sequence,
            ),
        ):
            result = run_training(
                model,
                [{"chars": ["ا"], "labels": [0]}],
                epochs=5,
                device=torch.device("cpu"),
                seed=7,
                spec_name="test",
                context=context,
                validation_records=validation,
                use_early_stopping=True,
            )
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["history"][-1]["epoch"], 3)


if __name__ == "__main__":
    unittest.main()
