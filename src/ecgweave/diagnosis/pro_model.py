# 公开版适配说明：包内相对导入；不改计算。
"""P的汇聚前交互与谨慎吸收G的局部双端参考；基础模型独立快照只读。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from . import base_model as base


ModelConfig = base.ModelConfig


def local_time_basis(config):
    """时间三角基：非负、L2归一，跨导联同步，最大非零距离小于1秒。"""
    step = 0.5
    seconds = (
        torch.arange(config.n_patches) * config.patch_size + (config.patch_size - 1) / 2
    ) / config.sample_rate
    knots = (
        torch.arange(math.ceil(config.seq_len / config.sample_rate / step) + 1) * step
    )
    phi = (1 - (seconds[:, None] - knots[None]).abs() / step).clamp_min(0)
    return F.normalize(phi, dim=-1).repeat(config.n_leads, 1)


def augment_local_reference(q, k, v, reference, coefficient, gate, phi):
    """精确追加 g_i g_j K_ij r_i^T M r_j；不构造逐对注意力mask。"""
    dimension = q.shape[-1]
    r = reference.detach().to(q.dtype) * gate.to(q.dtype)
    matrix = coefficient.tanh().to(q.dtype)
    time = phi.to(q.dtype)
    qr = torch.einsum("bnp,hpq->bhnq", r, matrix)
    extra_q = (qr[..., None] * time[None, None, :, None]).flatten(-2) * math.sqrt(
        dimension
    )
    extra_k = (r[..., None] * time[None, :, None]).flatten(-2)
    extra_k = extra_k[:, None].expand(-1, q.shape[1], -1, -1)
    padded = math.ceil((dimension + extra_q.shape[-1]) / 8) * 8
    return (
        F.pad(torch.cat((q, extra_q), -1), (0, padded - dimension - extra_q.shape[-1])),
        F.pad(torch.cat((k, extra_k), -1), (0, padded - dimension - extra_k.shape[-1])),
        F.pad(v, (0, padded - v.shape[-1])),
        dimension**-0.5,
    )


class LocalGatedRelations(base.BoundedRelations):
    """门控学习任务中的参考使用方式，不解释为定界正确性概率。"""

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("time_basis", local_time_basis(config))
        self.gates = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(config.embed_dim),
                nn.Linear(config.embed_dim, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            for _ in range(config.depth - self.first_layer + 1)
        )
        for gate in self.gates:
            nn.init.zeros_(gate[3].weight)
            nn.init.constant_(gate[3].bias, math.log(9))

    def forward(self, block, x, r, layer):
        batch, length, width = x.shape
        dimension = width // block.heads
        index = layer - self.first_layer
        qkv = (
            block.qkv(block.norm1(x))
            .reshape(batch, length, 3, block.heads, dimension)
            .permute(2, 0, 3, 1, 4)
        )
        gate = self.gates[index](x).sigmoid()
        q, k, v, scale = augment_local_reference(
            *qkv.unbind(0), r, self.coefficients[index], gate, self.time_basis
        )
        context = F.scaled_dot_product_attention(q, k, v, scale=scale)[..., :dimension]
        x = x + block.proj(context.transpose(1, 2).reshape(batch, length, width))
        return x + block.mlp(block.norm2(x))


class InteractionReader(base.JointLabelReader):
    def __init__(self, config):
        super().__init__(config)
        self.interaction = nn.Sequential(
            nn.Linear(config.fine_dim + 4, 32),
            nn.GELU(),
            nn.Linear(32, config.readout_dim, bias=False),
        )
        nn.init.zeros_(self.interaction[2].weight)

    def forward(self, h, u, raw):
        c = self.config
        batch = h.shape[0]
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
        # s0=1mV，仅固定单位；不做每记录或每窗口幅值标准化。
        inputs = torch.cat((F.layer_norm(u, (c.fine_dim,)), raw), -1)
        fine_values = (
            self.fine_value(u) + self.raw_value(raw) + self.interaction(inputs)
        )
        values = torch.cat(
            (self.coarse_value(h).flatten(1, 2), fine_values.flatten(1, 2)), 1
        )
        k = keys.reshape(
            batch, -1, c.readout_heads, c.readout_dim // c.readout_heads
        ).transpose(1, 2)
        v = values.reshape(
            batch, -1, c.readout_heads, c.readout_dim // c.readout_heads
        ).transpose(1, 2)
        durations = (c.patch_size / c.sample_rate, 4 / c.sample_rate)
        query = self.query[None].expand(batch, -1, -1)
        for index, layer in enumerate(self.rounds):
            q = (
                layer.query(query)
                .reshape(batch, c.n_classes, c.readout_heads, -1)
                .transpose(1, 2)
            )
            context = base.coverage_attention(
                q, k, v, self.memory_types, self.scale_logits, durations
            )
            query = query + layer.output(
                context.transpose(1, 2).reshape(batch, c.n_classes, c.readout_dim)
            )
            query = query + layer.ffn(query)
            if index == 1:
                with torch.autocast(device_type=h.device.type, enabled=False):
                    physical_values = torch.cat(
                        (
                            raw.new_zeros(batch, c.n_leads * c.n_patches, 4),
                            raw.flatten(1, 2),
                        ),
                        1,
                    )[:, None].expand(-1, c.readout_heads, -1, -1)
                    physical = base.coverage_attention(
                        q.float(),
                        k.float(),
                        physical_values.float(),
                        self.memory_types,
                        self.scale_logits.float(),
                        durations,
                    )
        with torch.autocast(device_type=h.device.type, enabled=False):
            return torch.cat(
                (
                    self.norm(query.float()),
                    physical.transpose(1, 2).reshape(batch, c.n_classes, -1),
                ),
                -1,
            )


class JointMultiresClassifier(base.JointMultiresClassifier):
    def __init__(self, config, arm="gpt_pro"):
        super().__init__(config)
        # 在完整基础模型构造后添加新模块，保留同seed的所有共有初值。
        self.arm = arm
        if arm == "gpt_pro":
            reader = InteractionReader(config)
            loaded = reader.load_state_dict(self.reader.state_dict(), strict=False)
            if (
                set(loaded.missing_keys)
                != {
                    "interaction.0.weight",
                    "interaction.0.bias",
                    "interaction.2.weight",
                }
                or loaded.unexpected_keys
            ):
                raise ValueError("P候选共有reader初值未完整保留")
            self.reader = reader
        elif arm == "gemini_local":
            relations = LocalGatedRelations(config)
            with torch.no_grad():
                relations.coefficients.copy_(self.relations.coefficients)
            self.relations = relations
        else:
            raise ValueError(f"未知实验臂：{arm}")
