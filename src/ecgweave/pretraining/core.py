"""同步十二导联 MAE：可见 token 联合编码、共享的逐导联重建解码器。"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# 导联固定顺序为 data_prepare.LEADS：I, II, III, aVR, aVL, aVF, V1–V6。
# 前 6 个肢体导联只有 I、II 两个自由度（Einthoven/Goldberger 线性恒等式）。
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
            raise ValueError("导联数、序列长度与 patch 长度必须为正")
        if self.seq_len % self.patch_size:
            raise ValueError("seq_len 必须整除 patch_size")
        if self.depth < 1 or self.decoder_depth < 1 or self.mlp_ratio <= 0:
            raise ValueError("编码器/解码器层数与 MLP 比例必须为正")
        for width, heads in (
            (self.embed_dim, self.num_heads),
            (self.decoder_embed_dim, self.decoder_num_heads),
        ):
            if width < 2 or width % 2 or heads < 1 or width % heads:
                raise ValueError("维度必须为正偶数，并能被对应注意力头数整除")

    @property
    def n_patches(self) -> int:
        return self.seq_len // self.patch_size


MASK_MODES = ("independent", "mixed", "limbsync", "limbsync_mixed")
LOSS_MODES = ("raw", "patch_softnorm", "local_softnorm")


@dataclass(frozen=True)
class LossConfig:
    """重建损失空间。归一化统计量取自目标信号，只定义损失，不进入编码器输入。

    soft_floor_mv 与模型输入同单位（幅度尺度 1 mV 时即 mV），防止近平坦段的噪声
    被无限放大；local_window 为奇数个 patch，40 ms patch 下 5 即 200 ms，约一个
    P 波或 QRS 加过渡，使归一化尺度与波形尺度一致。
    """

    mode: str = "raw"
    soft_floor_mv: float = 0.02
    local_window: int = 5

    def __post_init__(self):
        if self.mode not in LOSS_MODES:
            raise ValueError(f"未知损失空间：{self.mode}")
        if not self.soft_floor_mv > 0:
            raise ValueError("软底限必须为正")
        if self.local_window < 1 or self.local_window % 2 == 0:
            raise ValueError("局部窗必须是正奇数个 patch")


def reconstruction_target(patches: Tensor, config: LossConfig) -> Tensor:
    """patches [B,L,K,P] → 损失空间中的同形状目标。

    raw：原始幅值。patch_softnorm：(x−μ_p)/√(σ_p²+ε²)，逐 patch 统计。
    local_softnorm：(x−μ_w)/√(σ_w²+ε²)，μ_w、σ_w 取同导联以该 patch 为中心的
    local_window 个 patch 的全部样点，记录边缘按可用 patch 截断。
    """
    if config.mode == "raw":
        return patches
    patches = patches.float()
    floor_sq = config.soft_floor_mv**2
    if config.mode == "patch_softnorm":
        mean = patches.mean(-1, keepdim=True)
        var = patches.var(-1, unbiased=False, keepdim=True)
        return (patches - mean) / torch.sqrt(var + floor_sq)
    # 先去掉整条导联的直流：目标对它恒不变，只为减小 E[x²]−E[x]² 的相消误差
    centered = patches - patches.mean(dim=(-2, -1), keepdim=True)
    batch, leads, n_patches, _ = centered.shape
    moments = torch.stack((centered.mean(-1), centered.square().mean(-1)), dim=2)
    # 每个 patch 样点数相同，窗内 patch 矩的等权平均即窗内全部样点的矩
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
    """被遮 patch 上、所选损失空间中的逐样点平方误差均值；可见 patch 不计。"""
    if pred.shape != patches.shape or mask.shape != patches.shape[:-1]:
        raise ValueError("预测、原波形与遮挡形状不一致")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("loss 要求 bool mask 且至少有一个被遮 patch")
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
    """返回 [B,L,K] bool；True 为遮挡。所有模式消耗完全相同的随机数。

    每条导联保留 round(K*(1-mask_ratio)) 个位置。
    - mixed：每条记录以 0.5 概率让全部导联共享一个时间遮挡，否则逐导联独立。
    - limbsync：6 个肢体导联复用 I 导联的独立遮挡分数（同一时间遮挡），
      胸导联逐导联独立；不额外抽随机数，因此 I 与胸导联和 independent 逐位相同。
    - limbsync_mixed：mixed 的非共享记录改用 limbsync。
    generator 可以独立于训练全局随机流，且可位于 CPU 或 CUDA。
    """
    if mode not in MASK_MODES:
        raise ValueError(f"未知遮挡模式：{mode}")
    if mode.startswith("limbsync") and n_leads != 12:
        raise ValueError("肢体组同步遮挡只适用于标准十二导联顺序")
    if not 0 <= mask_ratio < 1:
        raise ValueError("mask_ratio 必须位于 [0,1)")
    n_keep = round(n_patches * (1 - mask_ratio))
    if min(batch, n_leads, n_patches, n_keep) < 1:
        raise ValueError("batch、导联、patch 和可见 patch 数必须为正")
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
        # 肢体导联秩为 2：同一时刻只要另有两个肢体导联可见即可线性复原，故整组同遮
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
    """无 dropout 的标准 pre-LN Transformer，使用 PyTorch SDPA。"""

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
    """输入 [B,L,T] mV 波形；输出 [B,L,K,P] 原物理量 patch 预测。

    不在模型中归一化波形或重建目标。时间位置固定，导联位置可训练。
    编码器仅接收可见 token；共享解码器以 B*L 为批维分别处理每条导联。
    """

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
        """仅重排时间轴，不混合导联；[B,L,T] → [B,L,K,P]。"""
        c = self.config
        if x.ndim != 3 or x.shape[1:] != (c.n_leads, c.seq_len):
            raise ValueError(
                f"需要 [B,{c.n_leads},{c.seq_len}] 输入，实为 {tuple(x.shape)}"
            )
        return x.reshape(x.shape[0], c.n_leads, c.n_patches, c.patch_size)

    def unpatchify(self, patches: Tensor) -> Tensor:
        c = self.config
        if patches.ndim != 4 or patches.shape[1:] != (
            c.n_leads,
            c.n_patches,
            c.patch_size,
        ):
            raise ValueError("patch 形状必须为 [B,L,K,P] 且与 config 一致")
        return patches.reshape(patches.shape[0], c.n_leads, c.seq_len)

    def _embed(self, x: Tensor) -> Tensor:
        patches = self.patch_embed(self.patchify(x))
        # 显式转 dtype 保持 autocast 路径，不由 fp32 位置编码提升回 fp32。
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
            raise ValueError("mask 必须为与输入匹配的 [B,L,K] bool 张量")
        visible = ~mask.flatten(1)
        counts = visible.sum(1)
        if not torch.all(counts == counts[0]) or counts[0] == 0:
            raise ValueError("每条记录须有相同且非零的可见 token 数")
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
        """返回 [B,V,D]；可见位置按 lead-major/time-major 顺序排列。"""
        return self._encode_visible(x, mask)[0]

    def encode_dense(
        self, x: Tensor, layers: tuple[int, ...] | None = None
    ) -> Tensor | dict[int, Tensor]:
        """完整波形稠密编码，输出 [B,L,K,D]，与预训练编码器完全共用参数。

        layers 使用 1-based 编码器层号；指定时返回同序字典，所选层统一
        使用 encoder_norm。不给 layers 时只返回末层，不进行池化。
        """
        c = self.config
        if layers is not None and any(layer < 1 or layer > c.depth for layer in layers):
            raise ValueError(f"层号必须位于 [1,{c.depth}]")
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
        # 解码器跨时间关注自身导联；跨导联信息仅来自联合编码器的 latent。
        restored = restored.reshape(batch * c.n_leads, c.n_patches, c.decoder_embed_dim)
        for block in self.decoder_blocks:
            restored = block(restored)
        prediction = self.decoder_pred(self.decoder_norm(restored))
        return prediction.reshape(batch, c.n_leads, c.n_patches, c.patch_size)

    def loss(
        self, pred: Tensor, x: Tensor, mask: Tensor, config: LossConfig | None = None
    ) -> Tensor:
        """被遮 patch 的重建损失；默认原 mV MSE（先导配方），包括低幅与常量片段。"""
        return masked_reconstruction_loss(
            pred, self.patchify(x), mask, config or LossConfig("raw")
        )
