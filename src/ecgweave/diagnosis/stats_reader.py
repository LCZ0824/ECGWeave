"""First- and second-moment class-query readout."""

import torch
from torch import nn

from . import base_model as base
from .natural_reader import JointInteractionReader, split_heads


def coverage_moments(q, key, value, types, scale_logits, durations):
    """Aggregate first and second moments with the same attention in FP32."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        source = value.float()
        moments = base.coverage_attention(
            q.float(),
            key.float(),
            torch.cat((source, source.square()), dim=-1),
            types,
            scale_logits.float(),
            durations,
        )
        mean, second = moments.chunk(2, dim=-1)
        variance = (second - mean.square()).clamp_min(0)
    return mean, variance


class StatsInteractionReader(JointInteractionReader):
    """Add attention-conditioned variance as a learned query residual."""

    def __init__(self, config):
        super().__init__(config)
        self.feature_dim = config.readout_dim
        with torch.random.fork_rng(devices=[]):
            self.variance_norm = nn.LayerNorm(config.readout_dim)
            self.variance_projection = nn.Linear(
                config.readout_dim, config.readout_dim, bias=False
            )
            # Zero-initialize the projection without an additional gate.
            nn.init.zeros_(self.variance_projection.weight)

    def forward(self, h, u, raw):
        c = self.config
        batch = h.shape[0]
        key, value = self.memory(h, u, raw)
        query = self.query[None].expand(batch, -1, -1)
        durations = (c.patch_size / c.sample_rate, 4 / c.sample_rate)
        for index, layer in enumerate(self.rounds):
            q = split_heads(layer.query(query), c.readout_heads)
            if index == 1:
                context, variance = coverage_moments(
                    q, key, value, self.memory_types, self.scale_logits, durations
                )
            else:
                context = base.coverage_attention(
                    q, key, value, self.memory_types, self.scale_logits, durations
                )
            context = context.transpose(1, 2).reshape(batch, c.n_classes, c.readout_dim)
            query = query + layer.output(context.to(query.dtype))
            query = query + layer.ffn(query)
        statistics = variance.transpose(1, 2).reshape(batch, c.n_classes, c.readout_dim)
        query = query + self.variance_projection(
            self.variance_norm(statistics.log1p().to(query.dtype))
        )
        return self.norm(query)
