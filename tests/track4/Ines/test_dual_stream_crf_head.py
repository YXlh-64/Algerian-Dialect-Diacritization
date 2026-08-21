import unittest

import torch

from models.track4.Ines.dual_stream_crf_head_model import (
    CRF,
    Track4DualStreamCRF,
    gather_letters,
)


class DualStreamCRFHeadTests(unittest.TestCase):
    def test_gather_letters_preserves_letter_order(self):
        emissions = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
        tags = torch.tensor([[1, 0, 3, 0]])
        letter_mask = torch.tensor([[True, False, True, False]])

        compact_emissions, compact_tags, compact_mask, indices, lengths = (
            gather_letters(emissions, tags, letter_mask)
        )

        torch.testing.assert_close(
            compact_emissions[0], emissions[0, torch.tensor([0, 2])]
        )
        self.assertEqual(compact_tags.tolist(), [[1, 3]])
        self.assertEqual(compact_mask.tolist(), [[True, True]])
        self.assertEqual(indices.tolist(), [[0, 2]])
        self.assertEqual(lengths.tolist(), [2])

    def test_crf_empty_batch_has_zero_differentiable_loss(self):
        crf = CRF(num_tags=3)
        emissions = torch.randn(2, 1, 3, requires_grad=True)
        tags = torch.zeros(2, 1, dtype=torch.long)
        mask = torch.zeros(2, 1, dtype=torch.bool)

        loss = crf(emissions, tags, mask)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(emissions.grad, torch.zeros_like(emissions))

    def test_model_loss_and_decode_exclude_spaces_and_padding(self):
        torch.manual_seed(42)
        model = Track4DualStreamCRF(
            vocab_size=8,
            num_labels=4,
            dim=8,
            n_heads=2,
            local_layers=1,
            global_layers=1,
            final_layers=1,
            local_window=1,
            dropout=0.0,
            max_seq_len=8,
            unscored_label_id=0,
        )
        char_ids = torch.tensor([[1, 2, 3, 7]])
        labels = torch.tensor([[1, 0, 2, 0]])
        pad_mask = torch.tensor([[False, False, False, True]])
        is_space = torch.tensor([[False, True, False, False]])

        loss = model.loss(char_ids, labels, pad_mask, is_space)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.crf.transitions.grad)

        model.eval()
        predictions = model.predict(char_ids, pad_mask, is_space)
        self.assertEqual(predictions.shape, char_ids.shape)
        self.assertEqual(predictions[0, 1].item(), 0)
        self.assertEqual(predictions[0, 3].item(), 0)


if __name__ == "__main__":
    unittest.main()
