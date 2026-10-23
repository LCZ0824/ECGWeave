# 公开版适配说明：包内相对导入；不改计算。
"""统一精度的联合读取，以及只从内容记忆解码的两轮类别读取。"""

import torch
from torch import nn
from torch.nn import functional as F

from . import base_model as base
from . import pro_model as pro


def joint_coverage_read(q, k, v, raw_values, types, scale_logits, durations):
    """一次寻址同时汇聚学习特征与原波，继承调用处的普通AMP策略。"""
    physical = raw_values.to(v.dtype)[:, None].expand(-1, v.shape[1], -1, -1)
    context = base.coverage_attention(
        q, k, torch.cat((v, physical), -1), types, scale_logits, durations
    )
    return context[..., : v.shape[-1]], context[..., v.shape[-1] :]


def split_heads(value, heads):
    return value.reshape(value.shape[0], -1, heads, value.shape[-1] // heads).transpose(
        1, 2
    )


class JointInteractionReader(pro.InteractionReader):
    """保留原交互内容；第二轮合并Value，不再另算FP32原波注意力。"""

    def memory(self, h, u, raw):
        coarse_keys = self.coarse_key(h)
        coarse_keys = (
            coarse_keys
            + self.lead_embed.to(coarse_keys.dtype)
            + self.coarse_time.to(coarse_keys.dtype)
        )
        fine_keys = self.fine_key(u) + self.raw_key(raw)
        fine_keys = (
            fine_keys
            + self.lead_embed.to(fine_keys.dtype)
            + self.fine_time.to(fine_keys.dtype)
        )
        keys = torch.cat((coarse_keys.flatten(1, 2), fine_keys.flatten(1, 2)), 1)
        inputs = torch.cat((F.layer_norm(u, (self.config.fine_dim,)), raw), -1)
        fine_values = (
            self.fine_value(u) + self.raw_value(raw) + self.interaction(inputs)
        )
        values = torch.cat(
            (self.coarse_value(h).flatten(1, 2), fine_values.flatten(1, 2)), 1
        )
        return split_heads(keys, self.config.readout_heads), split_heads(
            values, self.config.readout_heads
        )

    def forward(self, h, u, raw):
        c = self.config
        batch = h.shape[0]
        k, v = self.memory(h, u, raw)
        durations = (c.patch_size / c.sample_rate, 4 / c.sample_rate)
        query = self.query[None].expand(batch, -1, -1)
        for index, layer in enumerate(self.rounds):
            q = split_heads(layer.query(query), c.readout_heads)
            if index == 1:
                raw_values = torch.cat(
                    (
                        raw.new_zeros(batch, c.n_leads * c.n_patches, 4),
                        raw.flatten(1, 2),
                    ),
                    1,
                )
                context, physical = joint_coverage_read(
                    q, k, v, raw_values, self.memory_types, self.scale_logits, durations
                )
            else:
                context = base.coverage_attention(
                    q, k, v, self.memory_types, self.scale_logits, durations
                )
            query = query + layer.output(
                context.transpose(1, 2).reshape(batch, c.n_classes, c.readout_dim)
            )
            query = query + layer.ffn(query)
        return torch.cat(
            (
                self.norm(query),
                physical.transpose(1, 2).reshape(batch, c.n_classes, -1),
            ),
            -1,
        )


class LabelSelfAttention(nn.Module):
    """病例条件的类别状态交流；无标签输入、无类别互斥softmax。"""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query):
        batch, length, width = query.shape
        qkv = (
            self.qkv(self.norm(query))
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        context = F.scaled_dot_product_attention(*qkv.unbind(0))
        return query + self.output(
            context.transpose(1, 2).reshape(batch, length, width)
        )


class ContentMemoryReader(nn.Module):
    """原波与细流只有一个内容入口；最终分类只读取类别状态。"""

    def __init__(self, config, label_self_attention=False):
        super().__init__()
        self.config = config
        width = config.readout_dim
        self.feature_dim = width
        self.query = nn.Parameter(torch.randn(config.n_classes, width) * 0.02)
        self.coarse_content = nn.Sequential(
            nn.LayerNorm(config.embed_dim), nn.Linear(config.embed_dim, width)
        )
        self.fine_norm = nn.LayerNorm(config.fine_dim)
        # 只归一化学习特征，raw4保留全数据固定的mV单位。
        self.fine_content = nn.Sequential(
            nn.Linear(config.fine_dim + 4, width), nn.GELU(), nn.Linear(width, width)
        )
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.lead_embed = nn.Parameter(torch.randn(1, config.n_leads, 1, width) * 0.02)
        for name, steps, stride in (
            ("coarse_time", config.n_patches, config.patch_size),
            ("fine_time", config.n_fine, 4),
        ):
            centers = (
                torch.arange(steps, dtype=torch.float32) * stride + (stride - 1) / 2
            ) / 20
            self.register_buffer(name, base._position_encoding(centers, width))
        self.rounds = nn.ModuleList(base.ReadoutRound(width) for _ in range(2))
        self.scale_logits = nn.Parameter(torch.zeros(config.n_classes, 2))
        self.norm = nn.LayerNorm(width)
        self.register_buffer(
            "memory_types",
            torch.cat(
                (
                    torch.zeros(config.n_leads * config.n_patches, dtype=torch.long),
                    torch.ones(config.n_leads * config.n_fine, dtype=torch.long),
                )
            ),
        )
        # 分支初始化不推进共有部分的随机数流，clean与decoder的共有初值一致。
        with torch.random.fork_rng(devices=[]):
            self.self_attention = nn.ModuleList(
                LabelSelfAttention(width, config.readout_heads)
                for _ in range(2 if label_self_attention else 0)
            )

    def memory(self, h, u, raw):
        coarse = self.coarse_content(h)
        fine = self.fine_content(torch.cat((self.fine_norm(u), raw), -1))
        content = torch.cat((coarse.flatten(1, 2), fine.flatten(1, 2)), 1)
        keys = self.key(content)
        address = torch.cat(
            (
                (self.lead_embed + self.coarse_time)
                .expand(1, self.config.n_leads, -1, -1)
                .flatten(1, 2),
                (self.lead_embed + self.fine_time)
                .expand(1, self.config.n_leads, -1, -1)
                .flatten(1, 2),
            ),
            1,
        )
        keys = keys + address.to(keys.dtype)
        return split_heads(keys, self.config.readout_heads), split_heads(
            self.value(content), self.config.readout_heads
        )

    def forward(self, h, u, raw):
        c = self.config
        batch = h.shape[0]
        k, v = self.memory(h, u, raw)
        query = self.query[None].expand(batch, -1, -1)
        durations = (c.patch_size / c.sample_rate, 4 / c.sample_rate)
        for index, layer in enumerate(self.rounds):
            q = split_heads(layer.query(query), c.readout_heads)
            context = base.coverage_attention(
                q, k, v, self.memory_types, self.scale_logits, durations
            )
            query = query + layer.output(
                context.transpose(1, 2).reshape(batch, c.n_classes, c.readout_dim)
            )
            if self.self_attention:
                # 首次cross-attention之后才交流，输入已含本例ECG证据。
                query = self.self_attention[index](query)
            query = query + layer.ffn(query)
        return self.norm(query)
