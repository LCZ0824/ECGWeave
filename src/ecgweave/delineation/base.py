# 公开版适配说明：抽取Segmenter12及其常量/辅助函数；standardize来自同快照ecguni/augment.py。
"""基础十二导联定界头；从冻结实现按职责抽取。"""

import torch
from torch import nn
from torch.nn import functional as F
from ..pretraining.core import ECGMAE

NUM_CLASSES = 4
IGNORE = -100
FCN_CHANNELS = 128
FCN_DROPOUT = 0.1
STEM_CHANNELS = 32
STEM_KERNEL = 17
FUSE_CHANNELS = 64
FUSE_KERNEL = 9
DECODER_ATTRIBUTES = (
    "decoder_embed",
    "mask_token",
    "decoder_lead_embed",
    "decoder_time_pos",
    "decoder_blocks",
    "decoder_norm",
    "decoder_pred",
)


def standardize(x: torch.Tensor) -> torch.Tensor:
    """官方 Standardize(axis=(-1,-2))：整窗零均值、总体标准差归一，std=0 时置 0。"""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    safe = torch.where(std > 0, std, torch.ones_like(std))
    return torch.where(std > 0, (x - mean) / safe, torch.zeros_like(x))


def linear_upsample_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """与 F.interpolate(mode="linear", align_corners=False) 等价的固定矩阵 [n_in, n_out]。

    线性插值对输入是线性的，插值单位基即得矩阵；用矩阵乘法代替插值，使 CUDA 反向可确定。
    """
    basis = torch.eye(n_in).unsqueeze(0)
    return F.interpolate(basis, size=n_out, mode="linear", align_corners=False)[0]


def standardize_leads(x: torch.Tensor) -> torch.Tensor:
    """[B,L,T] 逐导联官方 Standardize（常数导联置 0），作为 Stem 输入。"""
    return standardize(x.reshape(-1, x.shape[-1])).reshape(x.shape)


class Segmenter12(nn.Module):
    """编码器只看原始 mV；Stem 看逐导联标准化波形；读出逐导联共享，跨导联信息只来自编码器。"""

    def __init__(self, mae: ECGMAE):
        super().__init__()
        for name in DECODER_ATTRIBUTES:
            delattr(mae, name)
        self.encoder = mae
        config = mae.config
        self.patch_size = config.patch_size
        self.readout = nn.Sequential(
            nn.Conv1d(config.embed_dim, FCN_CHANNELS, 3, padding=1, bias=False),
            nn.BatchNorm1d(FCN_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(FCN_DROPOUT)
        self.cls_seg = nn.Conv1d(FCN_CHANNELS, NUM_CLASSES * config.patch_size, 1)
        self.stem = nn.Sequential(
            nn.Conv1d(
                1, STEM_CHANNELS, STEM_KERNEL, padding=STEM_KERNEL // 2, bias=False
            ),
            nn.BatchNorm1d(STEM_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                STEM_CHANNELS,
                STEM_CHANNELS,
                STEM_KERNEL,
                padding=STEM_KERNEL - 1,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm1d(STEM_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(
                FCN_CHANNELS + STEM_CHANNELS + NUM_CLASSES,
                FUSE_CHANNELS,
                FUSE_KERNEL,
                padding=FUSE_KERNEL // 2,
                bias=False,
            ),
            nn.BatchNorm1d(FUSE_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Conv1d(FUSE_CHANNELS, NUM_CLASSES, 1)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)
        self.register_buffer(
            "upsample",
            linear_upsample_matrix(config.n_patches, config.seq_len),
            persistent=False,
        )

    def _token_heads(self, x: torch.Tensor):
        batch, leads, _ = x.shape
        tokens = self.encoder.encode_dense(x)
        n_tokens = tokens.shape[2]
        hidden = self.readout(
            tokens.reshape(batch * leads, n_tokens, -1).transpose(1, 2)
        )
        logits = self.cls_seg(self.dropout(hidden))
        # 输出通道 p*classes+c 对应 token 内第 p 点的类别 c（与单导联 subtoken 头相同排列）
        coarse = logits.reshape(batch * leads, self.patch_size, NUM_CLASSES, n_tokens)
        coarse = coarse.permute(0, 2, 3, 1).reshape(
            batch * leads, NUM_CLASSES, n_tokens * self.patch_size
        )
        return hidden, coarse

    @staticmethod
    def _to_output(logits: torch.Tensor, batch: int, leads: int) -> torch.Tensor:
        return logits.reshape(batch, leads, NUM_CLASSES, -1).permute(0, 2, 1, 3)

    def coarse_logits(self, x: torch.Tensor) -> torch.Tensor:
        """仅 subtoken 读出的 logits [B,4,L,T]（零初始化残差下即初始输出）。"""
        return self._to_output(self._token_heads(x)[1], x.shape[0], x.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,L,T] 原始 mV → logits [B,4,L,T]。"""
        batch, leads, length = x.shape
        hidden, coarse = self._token_heads(x)
        upsampled = hidden @ self.upsample.to(hidden.dtype)
        stem = self.stem(standardize_leads(x).reshape(batch * leads, 1, length))
        fused = self.fuse(
            torch.cat((upsampled, stem.to(upsampled.dtype), coarse), dim=1)
        )
        return self._to_output(coarse + self.refine(fused), batch, leads)
