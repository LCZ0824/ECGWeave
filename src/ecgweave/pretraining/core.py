"""Multilead masked autoencoding with a shared per-lead decoder."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Lead order: I, II, III, aVR, aVL, aVF, V1-V6.
# The six limb leads have two degrees of freedom.
LIMB_LEAD_COUNT = 6


@dataclass(frozen=True)
class ModelConfig:
    n_leads: int = 12
    seq_len: int = 5000
    patch_size: int = 20
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    decoder_embed_dim: int = 128
    decoder_depth: int = 2
    decoder_num_heads: int = 4
    mlp_ratio: float = 4.0

    def __post_init__(self):
        if self.n_leads < 1 or self.patch_size < 1 or self.seq_len < 1:
            raise ValueError(
                "Lead count, sequence length, and patch size must be positive"
            )
        if self.seq_len % self.patch_size:
            raise ValueError("seq_len must be divisible by patch_size")
        if self.depth < 1 or self.decoder_depth < 1 or self.mlp_ratio <= 0:
            raise ValueError(
                "Encoder and decoder depths and MLP ratios must be positive"
            )
        for width, heads in (
            (self.embed_dim, self.num_heads),
            (self.decoder_embed_dim, self.decoder_num_heads),
        ):
            if width < 2 or width % 2 or heads < 1 or width % heads:
                raise ValueError(
                    "Dimensions must be positive, even, and divisible by their head counts"
                )

    @property
    def n_patches(self) -> int:
        return self.seq_len // self.patch_size


MASK_MODES = ("independent", "mixed", "limbsync", "limbsync_mixed")
LOSS_MODES = ("raw", "patch_softnorm", "local_softnorm")


@dataclass(frozen=True)
class LossConfig:
    """Reconstruction target normalization settings.

    soft_floor_mv shares the waveform amplitude unit. local_window is an odd
    number of patches; five 40-ms patches span 200 ms."""

    mode: str = "raw"
    soft_floor_mv: float = 0.02
    local_window: int = 5

    def __post_init__(self):
        if self.mode not in LOSS_MODES:
            raise ValueError(f"Unknown loss space: {self.mode}")
        if not self.soft_floor_mv > 0:
            raise ValueError("soft_floor_mv must be positive")
        if self.local_window < 1 or self.local_window % 2 == 0:
            raise ValueError("local_window must be a positive odd number of patches")


def reconstruction_target(patches: Tensor, config: LossConfig) -> Tensor:
    """Build raw, patch-normalized, or locally normalized reconstruction targets."""
    if config.mode == "raw":
        return patches
    patches = patches.float()
    floor_sq = config.soft_floor_mv**2
    if config.mode == "patch_softnorm":
        mean = patches.mean(-1, keepdim=True)
        var = patches.var(-1, unbiased=False, keepdim=True)
        return (patches - mean) / torch.sqrt(var + floor_sq)
    # Subtract the lead DC offset to reduce moment cancellation error.
    centered = patches - patches.mean(dim=(-2, -1), keepdim=True)
    batch, leads, n_patches, _ = centered.shape
    moments = torch.stack((centered.mean(-1), centered.square().mean(-1)), dim=2)
    # Equal-size patches allow equally weighted window moments.
    pooled = F.avg_pool1d(
        moments.reshape(batch * leads, 2, n_patches),
        kernel_size=config.local_window,
        stride=1,
        padding=config.local_window // 2,
        count_include_pad=False,
    ).reshape(batch, leads, 2, n_patches)
    local_mean = pooled[:, :, 0].unsqueeze(-1)
    local_var = (pooled[:, :, 1] - pooled[:, :, 0].square()).clamp_min(0).unsqueeze(-1)
    return (centered - local_mean) / torch.sqrt(local_var + floor_sq)


def masked_reconstruction_loss(
    pred: Tensor, patches: Tensor, mask: Tensor, config: LossConfig
) -> Tensor:
    """Mean samplewise squared error over masked patches."""
    if pred.shape != patches.shape or mask.shape != patches.shape[:-1]:
        raise ValueError("Prediction, waveform, and mask shapes do not match")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("Loss requires a bool mask with at least one masked patch")
    target = reconstruction_target(patches.float(), config)
    per_patch = (pred.float() - target).square().mean(-1)
    return per_patch[mask].mean()


def make_mask(
    batch: int,
    n_leads: int,
    n_patches: int,
    mask_ratio: float,
    mode: str,
    generator: torch.Generator | None,
    device: torch.device | str,
) -> Tensor:
    """Return a [B, L, K] mask; True denotes a hidden patch.

    Each lead retains round(K * (1 - mask_ratio)) positions. Modes share the
    same random draws; limbsync synchronizes the six limb leads."""
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown masking mode: {mode}")
    if mode.startswith("limbsync") and n_leads != 12:
        raise ValueError(
            "Limb-synchronized masking requires the standard 12-lead order"
        )
    if not 0 <= mask_ratio < 1:
        raise ValueError("mask_ratio must be in [0, 1)")
    n_keep = round(n_patches * (1 - mask_ratio))
    if min(batch, n_leads, n_patches, n_keep) < 1:
        raise ValueError(
            "Batch, lead, patch, and visible-patch counts must be positive"
        )
    rng_device = generator.device if generator is not None else device
    independent_scores = torch.rand(
        batch, n_leads, n_patches, generator=generator, device=rng_device
    )
    shared_scores = torch.rand(
        batch, 1, n_patches, generator=generator, device=rng_device
    )
    use_shared = torch.rand(batch, 1, 1, generator=generator, device=rng_device) < 0.5
    scores = independent_scores
    if mode.startswith("limbsync"):
        # Mask limb leads together to prevent reconstruction from linear identities.
        limb = independent_scores[:, :1].expand(-1, LIMB_LEAD_COUNT, -1)
        scores = torch.cat((limb, independent_scores[:, LIMB_LEAD_COUNT:]), dim=1)
    if mode in ("mixed", "limbsync_mixed"):
        scores = torch.where(use_shared, shared_scores, scores)
    visible_indices = scores.argsort(dim=-1)[..., :n_keep]
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(-1, visible_indices, False)
    return mask.to(device=device)


def _sincos(n_positions: int, width: int) -> Tensor:
    positions = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(width // 2, dtype=torch.float32)
        / (width // 2)
    )
    angles = positions * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)[None, None]


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block using scaled dot-product attention."""

    def __init__(self, width: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        qkv = (
            self.qkv(self.norm1(x))
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.proj(attended)
        return x + self.mlp(self.norm2(x))


class ECGMAE(nn.Module):
    """Encode visible multilead patches and reconstruct each lead with a shared decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config
        self.patch_embed = nn.Linear(c.patch_size, c.embed_dim)
        self.lead_embed = nn.Parameter(torch.zeros(1, c.n_leads, 1, c.embed_dim))
        self.register_buffer("time_pos", _sincos(c.n_patches, c.embed_dim))
        self.encoder_blocks = nn.ModuleList(
            TransformerBlock(c.embed_dim, c.num_heads, c.mlp_ratio)
            for _ in range(c.depth)
        )
        self.encoder_norm = nn.LayerNorm(c.embed_dim, eps=1e-6)
        self.decoder_embed = nn.Linear(c.embed_dim, c.decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, c.decoder_embed_dim))
        self.decoder_lead_embed = nn.Parameter(
            torch.zeros(1, c.n_leads, 1, c.decoder_embed_dim)
        )
        self.register_buffer(
            "decoder_time_pos", _sincos(c.n_patches, c.decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            TransformerBlock(c.decoder_embed_dim, c.decoder_num_heads, c.mlp_ratio)
            for _ in range(c.decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(c.decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(c.decoder_embed_dim, c.patch_size)
        self.apply(self._init_weights)
        nn.init.normal_(self.lead_embed, std=0.02)
        nn.init.normal_(self.decoder_lead_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def patchify(self, x: Tensor) -> Tensor:
        """Reshape [B, L, T] waveforms into [B, L, K, P] patches."""
        c = self.config
        if x.ndim != 3 or x.shape[1:] != (c.n_leads, c.seq_len):
            raise ValueError(
                f"Expected [B,{c.n_leads},{c.seq_len}] input; received {tuple(x.shape)}"
            )
        return x.reshape(x.shape[0], c.n_leads, c.n_patches, c.patch_size)

    def unpatchify(self, patches: Tensor) -> Tensor:
        c = self.config
        if patches.ndim != 4 or patches.shape[1:] != (
            c.n_leads,
            c.n_patches,
            c.patch_size,
        ):
            raise ValueError("Patches must have shape [B, L, K, P] matching config")
        return patches.reshape(patches.shape[0], c.n_leads, c.seq_len)

    def _embed(self, x: Tensor) -> Tensor:
        patches = self.patch_embed(self.patchify(x))
        # Cast positional embeddings to preserve the autocast dtype.
        patches = (
            patches
            + self.lead_embed.to(patches.dtype)
            + self.time_pos.to(patches.dtype)
        )
        return patches.flatten(1, 2)

    def _visible_indices(self, mask: Tensor, batch: int) -> Tensor:
        if mask.dtype != torch.bool or mask.shape != (
            batch,
            self.config.n_leads,
            self.config.n_patches,
        ):
            raise ValueError("mask must be bool [B, L, K] matching the input")
        visible = ~mask.flatten(1)
        counts = visible.sum(1)
        if not torch.all(counts == counts[0]) or counts[0] == 0:
            raise ValueError("Records must have equal, nonzero visible-token counts")
        positions = torch.arange(visible.shape[1], device=mask.device).expand(batch, -1)
        return positions[visible].reshape(batch, -1)

    def _encode_visible(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self._embed(x)
        indices = self._visible_indices(mask, x.shape[0])
        tokens = tokens.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )
        for block in self.encoder_blocks:
            tokens = block(tokens)
        return self.encoder_norm(tokens), indices

    def encode_visible(self, x: Tensor, mask: Tensor) -> Tensor:
        """Return [B, V, D] visible tokens in lead-major, then time-major order."""
        return self._encode_visible(x, mask)[0]

    def encode_dense(
        self, x: Tensor, layers: tuple[int, ...] | None = None
    ) -> Tensor | dict[int, Tensor]:
        """Encode all waveform patches into [B, L, K, D].

        Optional layers use one-based indices and return normalized features."""
        c = self.config
        if layers is not None and any(layer < 1 or layer > c.depth for layer in layers):
            raise ValueError(f"Layer index must be in [1,{c.depth}]")
        tokens = self._embed(x)
        selected = {}
        for index, block in enumerate(self.encoder_blocks, 1):
            tokens = block(tokens)
            if layers is not None and index in layers:
                selected[index] = self.encoder_norm(tokens).reshape(
                    x.shape[0], c.n_leads, c.n_patches, c.embed_dim
                )
        if layers is not None:
            return {layer: selected[layer] for layer in layers}
        return self.encoder_norm(tokens).reshape(
            x.shape[0], c.n_leads, c.n_patches, c.embed_dim
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        c = self.config
        batch = x.shape[0]
        latent, indices = self._encode_visible(x, mask)
        latent = self.decoder_embed(latent)
        restored = self.mask_token.to(latent.dtype).expand(
            batch, c.n_leads * c.n_patches, -1
        )
        restored = restored.scatter(
            1, indices.unsqueeze(-1).expand(-1, -1, c.decoder_embed_dim), latent
        )
        restored = restored.reshape(batch, c.n_leads, c.n_patches, c.decoder_embed_dim)
        restored = (
            restored
            + self.decoder_lead_embed.to(restored.dtype)
            + self.decoder_time_pos.to(restored.dtype)
        )
        # Decode within each lead; cross-lead information comes from the encoder.
        restored = restored.reshape(batch * c.n_leads, c.n_patches, c.decoder_embed_dim)
        for block in self.decoder_blocks:
            restored = block(restored)
        prediction = self.decoder_pred(self.decoder_norm(restored))
        return prediction.reshape(batch, c.n_leads, c.n_patches, c.patch_size)

    def loss(
        self, pred: Tensor, x: Tensor, mask: Tensor, config: LossConfig | None = None
    ) -> Tensor:
        """Compute masked reconstruction loss in the selected target space."""
        return masked_reconstruction_loss(
            pred, self.patchify(x), mask, config or LossConfig("raw")
        )
