"""Coarse and fine waveform encoding with soft-reference attention."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_leads: int = 12
    seq_len: int = 5000
    patch_size: int = 20
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    fine_dim: int = 64
    fine_heads: int = 4
    exchange_dim: int = 64
    exchange_heads: int = 4
    readout_dim: int = 128
    readout_heads: int = 4
    n_classes: int = 71
    exchange_layers: tuple[int, int, int] = (4, 8, 12)
    relation_start_layer: int = 5
    beta_max: float = 1.0
    residual_init: float = 0.1
    sample_rate: float = 500.0

    def __post_init__(self):
        object.__setattr__(self, "exchange_layers", tuple(self.exchange_layers))
        if min(self.n_leads, self.seq_len, self.patch_size, self.n_classes) < 1:
            raise ValueError("Lead, sample, patch, and class counts must be positive")
        if self.seq_len % self.patch_size or self.patch_size % 4:
            raise ValueError(
                "Sequence length must be divisible by patch size, and patch size by four"
            )
        for width, heads in (
            (self.embed_dim, self.num_heads),
            (self.fine_dim, self.fine_heads),
            (self.exchange_dim, self.exchange_heads),
            (self.readout_dim, self.readout_heads),
        ):
            if width < 2 or width % 2 or heads < 1 or width % heads:
                raise ValueError(
                    "Feature widths must be positive, even, and divisible by their head counts"
                )
        if self.depth < 3 or self.mlp_ratio <= 0 or self.sample_rate <= 0:
            raise ValueError(
                "Encoder depth must be at least three; MLP ratio and sample rate must be positive"
            )
        if (
            len(self.exchange_layers) != 3
            or tuple(sorted(set(self.exchange_layers))) != self.exchange_layers
            or self.exchange_layers[0] < 1
            or self.exchange_layers[-1] != self.depth
        ):
            raise ValueError(
                "Exchange layers must increase strictly and end at the final encoder layer"
            )
        if not 1 <= self.relation_start_layer <= self.depth:
            raise ValueError("Reference start layer must be within the encoder")
        if self.beta_max != 1.0 or self.residual_init != 0.1:
            raise ValueError(
                "This configuration requires beta_max=1 and residual_init=0.1"
            )

    @property
    def n_patches(self) -> int:
        return self.seq_len // self.patch_size

    @property
    def n_fine(self) -> int:
        return self.seq_len // 4


def _position_encoding(positions: Tensor, width: int) -> Tensor:
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(width // 2, dtype=torch.float32)
        / (width // 2)
    )
    angles = positions[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), -1)[None, None]


def _sincos(n_positions: int, width: int) -> Tensor:
    """Fixed float32 sinusoidal time encoding."""
    return _position_encoding(torch.arange(n_positions, dtype=torch.float32), width)


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


class MAEEncoder(nn.Module):
    """Encoder-only masked-autoencoder backbone."""

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
        self.apply(self._init_weights)
        nn.init.normal_(self.lead_embed, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _embed(self, x: Tensor) -> Tensor:
        c = self.config
        patches = self.patch_embed(
            x.reshape(x.shape[0], c.n_leads, c.n_patches, c.patch_size)
        )
        patches = (
            patches
            + self.lead_embed.to(patches.dtype)
            + self.time_pos.to(patches.dtype)
        )
        return patches.flatten(1, 2)


def pack_four(x: Tensor) -> Tensor:
    """Reshape consecutive groups of four samples without changing mV units."""
    if x.ndim != 3 or x.shape[-1] % 4:
        raise ValueError(
            "Four-sample packing requires [B, L, T] with T divisible by four"
        )
    return x.reshape(x.shape[0], x.shape[1], x.shape[2] // 4, 4)


def supported_reference(reference: Tensor, support: Tensor) -> Tensor:
    """Detach references and suppress relation bias where support is invalid."""
    r = reference.detach().float()
    mass = r.sum(-1, keepdim=True)
    valid = (
        support.detach()
        & torch.isfinite(r).all(-1)
        & (r >= 0).all(-1)
        & torch.isfinite(mass[..., 0])
        & (mass[..., 0] > 0)
    )
    return torch.where(valid[..., None], r / mass.clamp_min(1e-12), 0)


def augment_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    posterior: Tensor,
    coefficient: Tensor,
    beta_max: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Append reference-relation coordinates to Q/K for scaled attention."""
    dimension = q.shape[-1]
    r = posterior.detach().to(dtype=q.dtype)
    matrix = coefficient.tanh().to(dtype=q.dtype) * beta_max
    extra_q = torch.einsum("bnp,hpq->bhnq", r, matrix) * math.sqrt(dimension)
    extra_k = r[:, None].expand(-1, q.shape[1], -1, -1)
    padded = math.ceil((dimension + 4) / 8) * 8
    return (
        F.pad(torch.cat((q, extra_q), -1), (0, padded - dimension - 4)),
        F.pad(torch.cat((k, extra_k), -1), (0, padded - dimension - 4)),
        F.pad(v, (0, padded - v.shape[-1])),
        dimension**-0.5,
    )


class BoundedRelations(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.first_layer = config.relation_start_layer
        self.beta_max = config.beta_max
        self.coefficients = nn.Parameter(
            torch.randn(config.depth - self.first_layer + 1, config.num_heads, 4, 4)
            * 0.02
        )

    def forward(
        self, block: TransformerBlock, x: Tensor, r: Tensor, layer: int
    ) -> Tensor:
        batch, length, width = x.shape
        dimension = width // block.heads
        qkv = (
            block.qkv(block.norm1(x))
            .reshape(batch, length, 3, block.heads, dimension)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v, scale = augment_reference(
            *qkv.unbind(0),
            r,
            self.coefficients[layer - self.first_layer],
            self.beta_max,
        )
        context = F.scaled_dot_product_attention(q, k, v, scale=scale)[..., :dimension]
        x = x + block.proj(context.transpose(1, 2).reshape(batch, length, width))
        return x + block.mlp(block.norm2(x))


def local_indices(
    query_steps: int, source_steps: int, direction: str, ratio: int = 5
) -> tuple[Tensor, Tensor]:
    """Aligned same-lead neighborhoods: 35 fine cells or five coarse cells."""
    if direction == "coarse_to_fine":
        index = (
            torch.arange(query_steps)[:, None] * ratio
            + ratio // 2
            + torch.arange(-17, 18)[None]
        )
    elif direction == "fine_to_coarse":
        index = torch.arange(query_steps)[:, None] // ratio + torch.arange(-2, 3)[None]
    else:
        raise ValueError(f"Unknown local exchange direction: {direction}")
    valid = (index >= 0) & (index < source_steps)
    return index.clamp(0, source_steps - 1), valid


class LocalCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        source_dim: int,
        attention_dim: int,
        heads: int,
        query_steps: int,
        source_steps: int,
        direction: str,
        ratio: int = 5,
    ):
        super().__init__()
        self.heads = heads
        self.width = attention_dim
        self.query_norm = nn.LayerNorm(query_dim)
        self.source_norm = nn.LayerNorm(source_dim)
        self.q = nn.Linear(query_dim, attention_dim)
        self.k = nn.Linear(source_dim, attention_dim)
        self.v = nn.Linear(source_dim, attention_dim)
        self.output = nn.Linear(attention_dim, query_dim)
        index, valid = local_indices(query_steps, source_steps, direction, ratio)
        self.register_buffer("index", index)
        self.register_buffer("valid", valid)

    def forward(self, query: Tensor, source: Tensor) -> Tensor:
        batch, leads, steps, _ = query.shape
        dimension = self.width // self.heads
        q = self.q(self.query_norm(query)).reshape(
            batch, leads, steps, self.heads, dimension
        )
        k = self.k(self.source_norm(source))[:, :, self.index].reshape(
            batch, leads, steps, -1, self.heads, dimension
        )
        v = self.v(source)[:, :, self.index].reshape_as(k)
        scores = (q.unsqueeze(3) * k).sum(-1).float() / math.sqrt(dimension)
        scores = scores.masked_fill(~self.valid[None, None, :, :, None], -torch.inf)
        weights = scores.softmax(3).to(v.dtype)
        context = (
            (weights[..., None] * v).sum(3).reshape(batch, leads, steps, self.width)
        )
        return self.output(context)


class FineResidual(nn.Module):
    def __init__(self, width: int, dilation: int, residual: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.convolution = nn.Conv1d(
            width, width, 5, padding=2 * dilation, dilation=dilation, groups=width
        )
        self.channel = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.strength = nn.Parameter(torch.tensor(residual))

    def forward(self, u: Tensor) -> Tensor:
        batch, leads, time, width = u.shape
        feature = self.norm(u).reshape(batch * leads, time, width).transpose(1, 2)
        feature = self.convolution(feature).transpose(1, 2).reshape_as(u)
        return u + self.strength * self.channel(feature)


class SameTimeLeadAttention(nn.Module):
    """Attend across leads independently at each time position."""

    def __init__(self, width: int, heads: int, residual: float = 0.1):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.strength = nn.Parameter(torch.tensor(residual))

    def forward(self, u: Tensor) -> Tensor:
        batch, leads, time, width = u.shape
        x = self.norm(u).transpose(1, 2).reshape(batch * time, leads, width)
        qkv = (
            self.qkv(x)
            .reshape(batch * time, leads, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        context = F.scaled_dot_product_attention(*qkv.unbind(0))
        context = context.transpose(1, 2).reshape(batch, time, leads, width)
        return u + self.strength * self.output(context.transpose(1, 2))


class FineWaveEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input = nn.Linear(4, config.fine_dim)
        self.lead_embed = nn.Parameter(
            torch.randn(1, config.n_leads, 1, config.fine_dim) * 0.02
        )
        # Use 40 ms as the coordinate unit and retain original fine-cell centers.
        centers = (torch.arange(config.n_fine, dtype=torch.float32) * 4 + 1.5) / 20
        self.register_buffer("time_pos", _position_encoding(centers, config.fine_dim))
        self.local = nn.ModuleList(
            FineResidual(config.fine_dim, dilation, config.residual_init)
            for dilation in (1, 2, 4)
        )
        self.fusion = nn.Linear(config.fine_dim * 4, config.fine_dim)
        self.lead_attention = SameTimeLeadAttention(
            config.fine_dim, config.fine_heads, config.residual_init
        )

    def forward(self, signals: Tensor) -> tuple[Tensor, Tensor]:
        raw = pack_four(signals)
        u = self.input(raw)
        u = u + self.lead_embed.to(u.dtype) + self.time_pos.to(u.dtype)
        levels = [u]
        for layer in self.local:
            u = layer(u)
            levels.append(u)
        u = self.fusion(torch.cat(levels, -1))
        return self.lead_attention(u), raw


class BidirectionalExchange(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.to_coarse = LocalCrossAttention(
            config.embed_dim,
            config.fine_dim,
            config.exchange_dim,
            config.exchange_heads,
            config.n_patches,
            config.n_fine,
            "coarse_to_fine",
            config.patch_size // 4,
        )
        self.to_fine = LocalCrossAttention(
            config.fine_dim,
            config.embed_dim,
            config.exchange_dim,
            config.exchange_heads,
            config.n_fine,
            config.n_patches,
            "fine_to_coarse",
            config.patch_size // 4,
        )
        self.strength = nn.Parameter(torch.full((2,), config.residual_init))
        self.lead_attention = SameTimeLeadAttention(
            config.fine_dim, config.fine_heads, config.residual_init
        )

    def forward(self, h: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        h = h + self.strength[0] * self.to_coarse(h, u)
        u = u + self.strength[1] * self.to_fine(u, h)
        return h, self.lead_attention(u)


def coverage_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    types: Tensor,
    scale_logits: Tensor,
    durations: tuple[float, float] | Tensor,
) -> Tensor:
    """Add scale and sampling-duration biases without a dense attention mask."""
    dimension, value_dimension = q.shape[-1], v.shape[-1]
    duration = torch.as_tensor(durations, dtype=q.dtype, device=q.device)
    bias = scale_logits.to(q.dtype).log_softmax(-1) + duration.log()
    bias = bias - bias.mean(-1, keepdim=True)
    extra_q = bias[None, None].expand(q.shape[0], q.shape[1], -1, -1)
    extra_q = extra_q * math.sqrt(dimension)
    extra_k = F.one_hot(types, 2).to(k.dtype)[None, None]
    extra_k = extra_k.expand(k.shape[0], k.shape[1], -1, -1)
    padded = math.ceil(max(dimension + 2, value_dimension) / 8) * 8
    aq = F.pad(torch.cat((q, extra_q), -1), (0, padded - dimension - 2))
    ak = F.pad(torch.cat((k, extra_k), -1), (0, padded - dimension - 2))
    av = F.pad(v, (0, padded - value_dimension))
    return F.scaled_dot_product_attention(aq, ak, av, scale=dimension**-0.5)[
        ..., :value_dimension
    ]


class ReadoutRound(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.output = nn.Linear(width, width)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )


class JointLabelReader(nn.Module):
    """Two-round class-query attention over features and raw voltages."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.query = nn.Parameter(
            torch.randn(config.n_classes, config.readout_dim) * 0.02
        )
        self.coarse_key = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.readout_dim),
        )
        self.fine_key = nn.Sequential(
            nn.LayerNorm(config.fine_dim),
            nn.Linear(config.fine_dim, config.readout_dim),
        )
        self.raw_key = nn.Linear(4, config.readout_dim, bias=False)
        self.coarse_value = nn.Linear(config.embed_dim, config.readout_dim)
        self.fine_value = nn.Linear(config.fine_dim, config.readout_dim)
        self.raw_value = nn.Linear(4, config.readout_dim, bias=False)
        self.lead_embed = nn.Parameter(
            torch.randn(1, config.n_leads, 1, config.readout_dim) * 0.02
        )
        for name, steps, stride in (
            ("coarse_time", config.n_patches, config.patch_size),
            ("fine_time", config.n_fine, 4),
        ):
            centers = (
                torch.arange(steps, dtype=torch.float32) * stride + (stride - 1) / 2
            ) / 20
            self.register_buffer(name, _position_encoding(centers, config.readout_dim))
        self.rounds = nn.ModuleList(ReadoutRound(config.readout_dim) for _ in range(2))
        self.scale_logits = nn.Parameter(torch.zeros(config.n_classes, 2))
        self.norm = nn.LayerNorm(config.readout_dim)
        self.feature_dim = config.readout_dim + 4 * config.readout_heads
        self.register_buffer(
            "memory_types",
            torch.cat(
                (
                    torch.zeros(config.n_leads * config.n_patches, dtype=torch.long),
                    torch.ones(config.n_leads * config.n_fine, dtype=torch.long),
                )
            ),
        )

    def forward(self, h: Tensor, u: Tensor, raw: Tensor) -> Tensor:
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
        values = torch.cat(
            (
                self.coarse_value(h).flatten(1, 2),
                (self.fine_value(u) + self.raw_value(raw)).flatten(1, 2),
            ),
            1,
        )
        k = keys.reshape(batch, -1, c.readout_heads, c.readout_dim // c.readout_heads)
        k = k.transpose(1, 2)
        v = values.reshape(batch, -1, c.readout_heads, c.readout_dim // c.readout_heads)
        v = v.transpose(1, 2)
        durations = (c.patch_size / c.sample_rate, 4 / c.sample_rate)
        query = self.query[None].expand(batch, -1, -1)
        for index, layer in enumerate(self.rounds):
            q = layer.query(query).reshape(batch, c.n_classes, c.readout_heads, -1)
            q = q.transpose(1, 2)
            context = coverage_attention(
                q, k, v, self.memory_types, self.scale_logits, durations
            )
            query = query + layer.output(
                context.transpose(1, 2).reshape(batch, c.n_classes, c.readout_dim)
            )
            query = query + layer.ffn(query)
            if index == 1:
                # Read raw-voltage values with the same attention weights.
                # Use FP32 to preserve voltage differences lost by BF16 quantization.
                with torch.autocast(device_type=h.device.type, enabled=False):
                    physical_values = torch.cat(
                        (
                            raw.new_zeros(batch, c.n_leads * c.n_patches, 4),
                            raw.flatten(1, 2),
                        ),
                        1,
                    )[:, None].expand(-1, c.readout_heads, -1, -1)
                    physical = coverage_attention(
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


class JointMultiresClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = MAEEncoder(config)
        self.fine = FineWaveEncoder(config)
        self.exchanges = nn.ModuleList(BidirectionalExchange(config) for _ in range(3))
        self.relations = BoundedRelations(config)
        self.reader = JointLabelReader(config)
        self.classifier_weight = nn.Parameter(
            torch.randn(config.n_classes, self.reader.feature_dim) * 0.02
        )
        self.classifier_bias = nn.Parameter(torch.zeros(config.n_classes))

    def load_mae_state(self, state_dict: Mapping[str, Tensor]) -> dict:
        """Validate and load encoder weights, reporting unused decoder parameters."""
        expected = self.encoder.state_dict()
        missing = sorted(set(expected) - state_dict.keys())
        ignored = sorted(
            key
            for key in state_dict
            if key.startswith("decoder_") or key == "mask_token"
        )
        unexpected = sorted(set(state_dict) - set(expected) - set(ignored))
        if missing or unexpected:
            raise ValueError(
                f"MAE state keys do not match: missing={missing}, unexpected={unexpected}"
            )
        for name, target in expected.items():
            source = state_dict[name]
            if (
                not isinstance(source, Tensor)
                or source.shape != target.shape
                or source.dtype != target.dtype
            ):
                raise ValueError(f"MAE tensor shape or dtype does not match: {name}")
        encoder_state = {name: state_dict[name] for name in expected}
        self.encoder.load_state_dict(encoder_state, strict=True)
        identical = all(
            torch.equal(value.detach().cpu(), state_dict[name].detach().cpu())
            for name, value in self.encoder.state_dict().items()
        )
        if not identical:
            raise RuntimeError("Loaded MAE tensors do not match the supplied state")
        return {
            "all_tensors_equal": identical,
            "loaded_tensor_count": len(expected),
            "loaded_keys": sorted(expected),
            "ignored_decoder_keys": ignored,
        }

    @torch.no_grad()
    def initialize_label_bias(self, prior: Tensor):
        prior = torch.as_tensor(
            prior, dtype=torch.float32, device=self.classifier_bias.device
        )
        if (
            prior.shape != (self.config.n_classes,)
            or not torch.isfinite(prior).all()
            or ((prior < 0) | (prior > 1)).any()
        ):
            raise ValueError(
                "Class priors must be finite training prevalences in [0, 1]"
            )
        epsilon = torch.finfo(prior.dtype).eps
        self.classifier_bias.copy_(prior.clamp(epsilon, 1 - epsilon).logit())

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        c = self.config
        x, reference, support = (
            batch["signals"],
            batch["reference"],
            batch["reference_support"],
        )
        if (
            x.ndim != 3
            or x.shape[1:] != (c.n_leads, c.seq_len)
            or x.dtype != torch.float32
        ):
            raise ValueError(
                f"Waveforms must be float32 [B,{c.n_leads},{c.seq_len}] mV"
            )
        expected_reference = (x.shape[0], c.n_leads, c.n_patches, 4)
        if reference.shape != expected_reference or reference.dtype != torch.float32:
            raise ValueError("References must be float32 [B, L, K, 4]")
        if support.shape != expected_reference[:-1] or support.dtype != torch.bool:
            raise ValueError("Reference support must be bool [B, L, K]")
        r = supported_reference(reference, support).flatten(1, 2)
        u, raw = self.fine(x)
        h = self.encoder._embed(x)
        stage = 0
        for layer, block in enumerate(self.encoder.encoder_blocks, 1):
            h = (
                block(h)
                if layer < c.relation_start_layer
                else self.relations(block, h, r, layer)
            )
            if layer == c.exchange_layers[stage]:
                h, u = self.exchanges[stage](
                    h.reshape(x.shape[0], c.n_leads, c.n_patches, c.embed_dim), u
                )
                h = h.flatten(1, 2)
                stage = min(stage + 1, 2)
        h = self.encoder.encoder_norm(h).reshape(
            x.shape[0], c.n_leads, c.n_patches, c.embed_dim
        )
        features = self.reader(h, u, raw)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return (features * self.classifier_weight[None]).sum(
                -1
            ) + self.classifier_bias
