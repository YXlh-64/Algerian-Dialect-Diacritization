import torch
import torch.nn as nn

from models.track4.SmailRoumaissa.transformer import Backbone
from models.track4.SmailRoumaissa.heads import PerWordCRFHead


class TransformerCNNCRFTagger(nn.Module):
    """Backbone + decomposed emission head + per-word linear-chain CRF."""

    def __init__(self, vocab_size: int, pad_id: int, **backbone_kwargs):
        super().__init__()
        self.backbone = Backbone(vocab_size, pad_id, **backbone_kwargs)
        self.crf_head = PerWordCRFHead(self.backbone.dim)

    def loss(self, input_ids, attn_mask, is_letter, labels):
        h = self.backbone(input_ids, attn_mask)
        tags = labels.clone()
        tags[tags == -100] = 0
        return self.crf_head.loss(h, is_letter, tags)

    @torch.no_grad()
    def decode(self, input_ids, attn_mask, is_letter):
        h = self.backbone(input_ids, attn_mask)
        return self.crf_head.decode(h, is_letter)

    @torch.no_grad()
    def marginal_log_probs(self, input_ids, attn_mask, base_temperature: float = 1.0,
                            shadda_temperature: float = 1.0):
        h = self.backbone(input_ids, attn_mask)
        return self.crf_head.marginal_log_probs(h, base_temperature, shadda_temperature)

    @torch.no_grad()
    def decode_from_log_probs(self, log_probs, is_letter):
        return self.crf_head.decode_from_emissions(log_probs, is_letter)


def build_model(vocab_size: int, pad_id: int, **backbone_kwargs):
    """Builds the (only) supported architecture: Transformer + CNN + CRF,
    now with T5-style relative position attention and gated CNN fusion."""
    return TransformerCNNCRFTagger(vocab_size, pad_id, **backbone_kwargs)
