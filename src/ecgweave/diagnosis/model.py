# 公开版适配说明：包内相对导入；不改计算。
"""四个性能候选：宽细流与交换、逐次FFN、统一内容读出及类别解码。"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from . import base_model as base
from . import pro_model as pro
from .natural_reader import ContentMemoryReader, JointInteractionReader
from .spec import SUPPORTED_ARMS


@dataclass(frozen=True)
class ModelConfig(base.ModelConfig):
    fine_dim: int = 128
    exchange_dim: int = 128


class FineFFN(nn.Module):
    """交换后的逐位置重编码；单一零输出初始化，残差系数为1。"""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.input = nn.Linear(width, 2 * width)
        self.output = nn.Linear(2 * width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, u):
        return u + self.output(F.gelu(self.input(self.norm(u))))


class TemporalFineFFN(FineFFN):
    """沿每条导联的真实时间轴混合隐藏特征，保持采样网格与导联边界。"""

    def __init__(self, width, dilation):
        super().__init__(width)
        self.depthwise = nn.Conv1d(
            2 * width,
            2 * width,
            9,
            padding=4 * dilation,
            dilation=dilation,
            groups=2 * width,
        )

    def forward(self, u):
        batch, leads, time, width = u.shape
        hidden = self.input(self.norm(u))
        hidden = hidden.reshape(batch * leads, time, 2 * width).transpose(1, 2)
        hidden = (
            self.depthwise(hidden)
            .transpose(1, 2)
            .reshape(batch, leads, time, 2 * width)
        )
        return u + self.output(F.gelu(hidden))


class ExchangeLeadAttention(base.SameTimeLeadAttention):
    def __init__(self, config, temporal_dilation=None):
        super().__init__(config.fine_dim, config.fine_heads, config.residual_init)
        self.post_ffn = (
            FineFFN(config.fine_dim)
            if temporal_dilation is None
            else TemporalFineFFN(config.fine_dim, temporal_dilation)
        )

    def forward(self, u):
        return self.post_ffn(super().forward(u))


class JointMultiresClassifier(pro.JointMultiresClassifier):
    def __init__(self, config, arm="wide_joint"):
        if arm not in SUPPORTED_ARMS:
            raise ValueError(f"未知实验臂：{arm}")
        super().__init__(config, "gpt_pro")
        self.arm = arm
        # 共同骨干先完整初始化，再增加分支，避免候选改变其他共有初值。
        for index, exchange in enumerate(self.exchanges):
            attention = ExchangeLeadAttention(
                config, 2**index if arm == "temporal_joint" else None
            )
            loaded = attention.load_state_dict(
                exchange.lead_attention.state_dict(), strict=False
            )
            if loaded.unexpected_keys or any(
                not name.startswith("post_ffn.") for name in loaded.missing_keys
            ):
                raise ValueError("交换后FFN未保留共有导联注意力")
            exchange.lead_attention = attention
        if arm in ("wide_clean", "label_decoder"):
            self.reader = ContentMemoryReader(
                config, label_self_attention=arm == "label_decoder"
            )
            self.classifier_weight = nn.Parameter(
                torch.randn(config.n_classes, self.reader.feature_dim) * 0.02
            )
        else:
            reader = JointInteractionReader(config)
            reader.load_state_dict(self.reader.state_dict(), strict=True)
            self.reader = reader

    def added_parameter_groups(self):
        groups = {
            f"fine_ffn.{index}": list(exchange.lead_attention.post_ffn.parameters())
            for index, exchange in enumerate(self.exchanges)
        }
        if isinstance(self.reader, ContentMemoryReader):
            groups["reader.memory"] = [
                parameter
                for module in (
                    self.reader.coarse_content,
                    self.reader.fine_norm,
                    self.reader.fine_content,
                    self.reader.key,
                    self.reader.value,
                )
                for parameter in module.parameters()
            ]
            for index, layer in enumerate(self.reader.self_attention):
                groups[f"reader.self_attention.{index}"] = list(layer.parameters())
        return groups

    def parameter_count_by_role(self):
        counts = {}
        for name, parameter in self.named_parameters():
            role = name.split(".", 1)[0]
            counts[role] = counts.get(role, 0) + parameter.numel()
        return counts
