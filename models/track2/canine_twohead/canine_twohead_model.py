"""CANINE-S with a factorized shadda/vowel classification head.

The 16 labels used by the competition have a useful product structure:

    label = 8 * shadda + vowel

Predicting the two factors separately lets the model share evidence between
plain and shadda-bearing versions of the same vowel.  The combined
log-probabilities are returned as the 16-class logits expected by
``Trainer`` and by the official character-level metric.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CanineModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.canine.modeling_canine import CaninePreTrainedModel


class CanineTwoHeadForDiacritization(CaninePreTrainedModel):
    """Pretrained CANINE-S encoder plus shadda and vowel heads."""

    def __init__(self, config):
        super().__init__(config)
        self.canine = CanineModel(config)
        self.dropout = nn.Dropout(getattr(config, "head_dropout", 0.1))
        self.shadda_head = nn.Linear(config.hidden_size, 2)
        self.vowel_head = nn.Linear(config.hidden_size, 8)

        # This is deliberately a plain attribute.  It is set by the training
        # script after loading and is not part of the best-performing config.
        self.class_weights = None
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.canine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden = self.dropout(outputs.last_hidden_state)

        shadda_log_probs = F.log_softmax(self.shadda_head(hidden).float(), dim=-1)
        vowel_log_probs = F.log_softmax(self.vowel_head(hidden).float(), dim=-1)

        # (B, T, 2, 8) -> (B, T, 16), with index 8*shadda + vowel.
        log_probs = (shadda_log_probs.unsqueeze(-1) + vowel_log_probs.unsqueeze(-2)).flatten(-2)

        loss = None
        if labels is not None:
            loss = F.nll_loss(
                log_probs.reshape(-1, self.config.num_labels),
                labels.reshape(-1),
                weight=self.class_weights,
                ignore_index=-100,
            )

        return TokenClassifierOutput(loss=loss, logits=log_probs)
