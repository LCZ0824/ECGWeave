"""Masked autoencoding with valid-sample support."""

import torch
from torch.nn import functional as F

from . import core

ModelConfig = core.ModelConfig
LossConfig = core.LossConfig


def expand_valid(valid, batch, leads, patches, device):
    """Expand temporal [B, K] or per-lead [B, L, K] validity masks."""
    if valid is None:
        return torch.ones(batch, leads, patches, dtype=torch.bool, device=device)
    if valid.dtype != torch.bool:
        raise ValueError("patch_valid must be bool")
    if valid.shape == (batch, patches):
        valid = valid[:, None].expand(-1, leads, -1)
    if valid.shape != (batch, leads, patches):
        raise ValueError("patch_valid shape does not match")
    return valid.to(device)


def make_valid_mask(patch_valid, mask_ratio, mode, generator=None, n_leads=12):
    """Apply the masking recipe within valid sample support."""
    if patch_valid.ndim not in (2, 3):
        raise ValueError("patch_valid must have shape [B, K] or [B, L, K]")
    b, k = patch_valid.shape[0], patch_valid.shape[-1]
    valid = expand_valid(patch_valid, b, n_leads, k, patch_valid.device)
    # Require shared time support for synchronized lead masking.
    if not torch.equal(valid, valid[:, :1].expand_as(valid)):
        raise ValueError("All leads must share valid time support")
    counts = valid[:, 0].sum(-1)
    for count in counts.tolist():
        keep = round(count * (1 - mask_ratio))
        if not 0 < keep < count:
            raise ValueError(
                "Valid support cannot provide both visible and masked target patches"
            )
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
    """Apply attention using True key-mask entries as valid positions."""
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
    """Compute local target moments from fully valid patches."""
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
    # Shared pooling denominators cancel in support-weighted moments.
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
    """Masked autoencoder supporting variable valid-token counts."""

    def forward(self, x, mask, patch_valid=None):
        c = self.config
        b = x.shape[0]
        valid = expand_valid(patch_valid, b, c.n_leads, c.n_patches, x.device)
        if mask.dtype != torch.bool or mask.shape != valid.shape:
            raise ValueError("mask must be bool [B, L, K] matching the input")
        if not valid.any(-1).all():
            raise ValueError("Each lead must have a valid patch")
        visible = (valid & ~mask).flatten(1)
        counts = visible.sum(-1)
        if (counts == 0).any():
            raise ValueError("Each record must have visible tokens")
        safe = torch.where(valid[..., None], self.patchify(x), 0.0)
        tokens = self._embed(self.unpatchify(safe))
        # Mask padded gather slots and scatter only valid visible positions.
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
            raise ValueError(
                "Prediction, target, or mask shape or dtype does not match"
            )
        targets = valid & mask
        counts = targets.sum((1, 2))
        if (counts == 0).any():
            raise ValueError("Each record must have valid masked targets")
        target = valid_target(patches, valid, config or LossConfig("raw"))
        error = (pred.float() - target).square().mean(-1)
        return (torch.where(targets, error, 0.0).sum((1, 2)) / counts).mean()
