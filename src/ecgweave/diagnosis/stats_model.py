"""当前论文stats_prior20分类器。"""

import torch
from torch import nn

from .model import JointMultiresClassifier, ModelConfig
from .stats_reader import StatsInteractionReader


class PostFFNOnly(nn.Module):
    """保留交换后的FFN及其参数名称。"""

    def __init__(self, post_ffn):
        super().__init__()
        self.post_ffn = post_ffn

    def forward(self, value):
        return self.post_ffn(value)


def build_classifier(config=None):
    """完整构造共同骨干后替换读取器，与冻结训练实现的初始化一致。"""
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
            raise ValueError("统计读取器的共同参数不匹配")
        model.classifier_weight = nn.Parameter(
            original_weight.detach()[:, : config.readout_dim].clone()
        )
    model.reader = reader.to(original_weight)
    model.fine.lead_attention = nn.Identity()
    for exchange in model.exchanges:
        exchange.lead_attention = PostFFNOnly(exchange.lead_attention.post_ffn)
    return model
