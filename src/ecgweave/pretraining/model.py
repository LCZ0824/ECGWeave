# 公开版适配说明：包内相对导入；不改计算。
"""冻结 Step1 MAE 的有效支持适配；参数键兼容，不改变既有模型文件。"""

import torch
from torch.nn import functional as F

from . import core

ModelConfig = core.ModelConfig
LossConfig = core.LossConfig


def expand_valid(valid, batch, leads, patches, device):
    """接受时间 [B,K] 或导联 [B,L,K] 的完整 patch 有效性。"""
    if valid is None:
        return torch.ones(batch, leads, patches, dtype=torch.bool, device=device)
    if valid.dtype != torch.bool:
        raise ValueError("patch_valid 必须为 bool")
    if valid.shape == (batch, patches):
        valid = valid[:, None].expand(-1, leads, -1)
    if valid.shape != (batch, leads, patches):
        raise ValueError("patch_valid 形状不匹配")
    return valid.to(device)


def make_valid_mask(patch_valid, mask_ratio, mode, generator=None, n_leads=12):
    """在真实支持内沿用冻结遮挡配方；不为极短记录调低遮挡率。"""
    if patch_valid.ndim not in (2, 3):
        raise ValueError("patch_valid 需要 [B,K] 或 [B,L,K]")
    b, k = patch_valid.shape[0], patch_valid.shape[-1]
    valid = expand_valid(patch_valid, b, n_leads, k, patch_valid.device)
    # 同步十二导联支持是当前数据合同，避免导联缺失破坏组同步语义。
    if not torch.equal(valid, valid[:, :1].expand_as(valid)):
        raise ValueError("当前配方要求全部导联共享有效时间支持")
    counts = valid[:, 0].sum(-1)
    for count in counts.tolist():
        keep = round(count * (1 - mask_ratio))
        if not 0 < keep < count:
            raise ValueError("有效支持无法同时提供可见 patch 与被遮目标")
    if bool(valid.all()):
        return core.make_mask(b, n_leads, k, mask_ratio, mode, generator, valid.device)
    result = torch.ones_like(valid)
    for i, count in enumerate(counts.tolist()):
        selected = core.make_mask(
            1, n_leads, count, mask_ratio, mode, generator, valid.device
        )
        result[i, :, valid[i, 0]] = selected[0]
    return result


def attention_block(block, x, key_valid):
    """复用冻结参数；SDPA bool True 表示允许作为 key。"""
    if bool(key_valid.all()):
        return block(x)
    b, length, width = x.shape
    qkv = (
        block.qkv(block.norm1(x))
        .reshape(b, length, 3, block.heads, width // block.heads)
        .permute(2, 0, 3, 1, 4)
    )
    attended = (
        F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], attn_mask=key_valid[:, None, None, :]
        )
        .transpose(1, 2)
        .reshape(b, length, width)
    )
    x = x + block.proj(attended)
    return x + block.mlp(block.norm2(x))


def valid_target(patches, valid, config):
    """完整有效 patch 的局部矩；padding 不参与任何目标统计。"""
    if bool(valid.all()):
        return core.reconstruction_target(patches, config)
    patches = torch.where(valid[..., None], patches.float(), 0.0)
    if config.mode == "raw":
        return patches
    if config.mode == "patch_softnorm":
        return core.reconstruction_target(patches, config)
    weights = valid.to(patches.dtype)
    count = weights.sum(-1, keepdim=True).clamp_min(1) * patches.shape[-1]
    mean = patches.sum((-2, -1), keepdim=True) / count[..., None]
    centered = torch.where(valid[..., None], patches - mean, 0.0)
    b, leads, k, _ = patches.shape
    moments = torch.stack((centered.mean(-1), centered.square().mean(-1)), dim=2)
    # 池化三个量的同一分母相消，得到有效 patch 数加权矩。
    pooled = F.avg_pool1d(
        moments.reshape(b * leads, 2, k),
        config.local_window,
        stride=1,
        padding=config.local_window // 2,
        count_include_pad=True,
    ).reshape(b, leads, 2, k)
    denom = (
        F.avg_pool1d(
            weights.reshape(b * leads, 1, k),
            config.local_window,
            stride=1,
            padding=config.local_window // 2,
            count_include_pad=True,
        )
        .reshape(b, leads, k)
        .clamp_min(1 / config.local_window)
    )
    local_mean = (pooled[:, :, 0] / denom)[..., None]
    local_var = (pooled[:, :, 1] / denom - local_mean[..., 0].square()).clamp_min(0)[
        ..., None
    ]
    target = (centered - local_mean) / torch.sqrt(local_var + config.soft_floor_mv**2)
    return torch.where(valid[..., None], target, 0.0)


class ValidECGMAE(core.ECGMAE):
    """固定 10 秒布局、可变可见数；state_dict 与冻结 ECGMAE 相同。"""

    def forward(self, x, mask, patch_valid=None):
        c = self.config
        b = x.shape[0]
        valid = expand_valid(patch_valid, b, c.n_leads, c.n_patches, x.device)
        if mask.dtype != torch.bool or mask.shape != valid.shape:
            raise ValueError("mask 需要匹配输入的 bool [B,L,K]")
        if not valid.any(-1).all():
            raise ValueError("每导联必须存在有效 patch")
        visible = (valid & ~mask).flatten(1)
        counts = visible.sum(-1)
        if (counts == 0).any():
            raise ValueError("每条记录须有可见 token")
        safe = torch.where(valid[..., None], self.patchify(x), 0.0)
        tokens = self._embed(self.unpatchify(safe))
        # padding gather 槽独立 key mask；scatter 只写真实可见位置。
        width = int(counts.max())
        indices = torch.zeros(b, width, dtype=torch.long, device=x.device)
        key_valid = torch.arange(width, device=x.device)[None] < counts[:, None]
        for row in range(b):
            indices[row, : counts[row]] = visible[row].nonzero().flatten()
        tokens = tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))
        for block in self.encoder_blocks:
            tokens = attention_block(block, tokens, key_valid)
        latent = self.decoder_embed(self.encoder_norm(tokens))
        restored = (
            self.mask_token.to(latent.dtype)
            .expand(b, c.n_leads * c.n_patches, -1)
            .clone()
        )
        for row in range(b):
            restored[row, indices[row, : counts[row]]] = latent[row, : counts[row]]
        restored = restored.reshape(b, c.n_leads, c.n_patches, c.decoder_embed_dim)
        restored = (
            restored
            + self.decoder_lead_embed.to(restored.dtype)
            + self.decoder_time_pos.to(restored.dtype)
        )
        restored = restored.reshape(b * c.n_leads, c.n_patches, c.decoder_embed_dim)
        for block in self.decoder_blocks:
            restored = attention_block(
                block, restored, valid.reshape(b * c.n_leads, c.n_patches)
            )
        pred = self.decoder_pred(self.decoder_norm(restored))
        return pred.reshape(b, c.n_leads, c.n_patches, c.patch_size)

    def loss(self, pred, x, mask, config=None, patch_valid=None):
        patches = self.patchify(x)
        valid = expand_valid(patch_valid, *patches.shape[:3], x.device)
        if (
            pred.shape != patches.shape
            or mask.shape != valid.shape
            or mask.dtype != torch.bool
        ):
            raise ValueError("预测/目标/遮挡形状或类型不匹配")
        targets = valid & mask
        counts = targets.sum((1, 2))
        if (counts == 0).any():
            raise ValueError("每条记录必须有有效被遮目标")
        target = valid_target(patches, valid, config or LossConfig("raw"))
        error = (pred.float() - target).square().mean(-1)
        return (torch.where(targets, error, 0.0).sum((1, 2)) / counts).mean()
