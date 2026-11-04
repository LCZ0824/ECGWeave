"""ECGWeave classifier with attention-conditioned variance."""

import torch
from torch import nn

from .model import JointMultiresClassifier, ModelConfig
from .stats_reader import StatsInteractionReader


class PostFFNOnly(nn.Module):
    """Retain the post-exchange FFN without additional lead attention."""

    def __init__(self, post_ffn):
        super().__init__()
        self.post_ffn = post_ffn

    def forward(self, value):
        return self.post_ffn(value)


def build_classifier(config=None):
    """Construct the backbone, statistical reader, and class-specific heads."""
    config = ModelConfig() if config is None else config
    model = JointMultiresClassifier(config, arm="wide_joint")
    original_weight = model.classifier_weight
    with torch.random.fork_rng(devices=[]):
        reader = StatsInteractionReader(config)
        loaded = reader.load_state_dict(model.reader.state_dict(), strict=False)
        if (
            set(loaded.missing_keys)
            != {
                "variance_norm.weight",
                "variance_norm.bias",
                "variance_projection.weight",
            }
            or loaded.unexpected_keys
        ):
            raise ValueError("Shared statistical-reader parameters do not match")
        model.classifier_weight = nn.Parameter(
            original_weight.detach()[:, : config.readout_dim].clone()
        )
    model.reader = reader.to(original_weight)
    model.fine.lead_attention = nn.Identity()
    for exchange in model.exchanges:
        exchange.lead_attention = PostFFNOnly(exchange.lead_attention.post_ffn)
    return model
