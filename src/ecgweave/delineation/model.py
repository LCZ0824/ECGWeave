# 公开版适配说明：包内导入；工厂直接构造同结构，不隐式读取S*。
"""保持已有Encoder与主预测路径的多深度/局部几何读出。"""

import torch
from torch import nn
from torch.nn import functional as F

from . import base as FT

ARCHITECTURES = ("base", "multidepth", "derivative", "gls_local", "multidepth_gls")


def derivative_channels(x, *, seams=None):
    """原波形及02式两阶差分；拼接±2点先去掉差分伪影再归一。"""
    x = x.float()
    guard = None
    if seams is not None:
        if seams.shape != x.shape or seams.dtype != torch.bool:
            raise ValueError("拼接位置须为与逐导联波形相同形状的bool")
        guard = F.max_pool1d(seams.float(), 5, 1, 2).bool()
    channels, current = [x], x
    for _ in range(2):
        padded = F.pad(current, (1, 0), mode="replicate")
        current = current - padded[..., :-1]
        clean = current.masked_fill(guard, 0) if guard is not None else current
        scale = clean.std(dim=-1, keepdim=True).clamp_min(1e-6)
        channels.append(clean / scale)
    return torch.cat(channels, dim=1)


class LocalGeometryBlock(nn.Module):
    """02原始网格深度卷积＋LN/MLP残差；500Hz下匹配时间范围。"""

    def __init__(self, channels=32):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 13, padding=6, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, 4 * channels)
        self.reduce = nn.Linear(4 * channels, channels)
        self.scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x):
        residual = self.depthwise(x).transpose(1, 2)
        residual = self.reduce(F.gelu(self.expand(self.norm(residual))))
        return x + (residual * self.scale).transpose(1, 2)


class IntegratedSegmenter12(FT.Segmenter12):
    """联合Encoder后逐导联共享局部读出，输出仍为[B,4,L,T]。"""

    def __init__(self, base, architecture):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"未知结构：{architecture}")
        nn.Module.__init__(self)
        self.architecture = architecture
        self.patch_size = base.patch_size
        for name, module in base.named_children():
            self.add_module(name, module)
        self.register_buffer("upsample", base.upsample.clone(), persistent=False)
        self.multidepth = architecture in ("multidepth", "multidepth_gls")
        self.geometry = architecture in ("gls_local", "multidepth_gls")
        # 新分支不消费既有初始化/训练主随机流，所有旧模块已严格加载。
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(230930)
            self.depth_projections = nn.ModuleDict()
            if self.multidepth:
                width = self.encoder.config.embed_dim
                for layer in (3, 6, 9):
                    projection = nn.Linear(width, width, bias=False)
                    nn.init.zeros_(projection.weight)
                    self.depth_projections[str(layer)] = projection
            if architecture == "derivative":
                old = self.stem[0]
                self.derivative_projection = nn.Conv1d(
                    2,
                    old.out_channels,
                    old.kernel_size,
                    stride=old.stride,
                    padding=old.padding,
                    dilation=old.dilation,
                    bias=False,
                ).to(device=old.weight.device, dtype=old.weight.dtype)
                nn.init.zeros_(self.derivative_projection.weight)
            if self.geometry:
                self.local_geometry = nn.Sequential(
                    nn.Conv1d(3, 32, 13, padding=6, bias=False),
                    nn.GroupNorm(1, 32),
                    LocalGeometryBlock(),
                    LocalGeometryBlock(),
                )
                self.local_output = nn.Conv1d(32, 32, 1, bias=False)
                nn.init.zeros_(self.local_output.weight)

    def dense_features(self, x, *, feature_mask=None):
        if self.multidepth:
            layers = self.encoder.encode_dense(x, layers=(3, 6, 9, 12))
            tokens = layers[12]
            if feature_mask is not None:
                tokens = tokens * feature_mask
            for layer in (3, 6, 9):
                feature = layers[layer]
                if feature_mask is not None:
                    feature = feature * feature_mask
                tokens = tokens + self.depth_projections[str(layer)](feature)
            return tokens
        tokens = self.encoder.encode_dense(x)
        return tokens if feature_mask is None else tokens * feature_mask

    def _token_heads(self, x, *, feature_mask=None):
        batch, leads, _ = x.shape
        tokens = self.dense_features(x, feature_mask=feature_mask)
        n_tokens = tokens.shape[2]
        hidden = self.readout(
            tokens.reshape(batch * leads, n_tokens, -1).transpose(1, 2)
        )
        logits = self.cls_seg(self.dropout(hidden))
        coarse = logits.reshape(batch * leads, self.patch_size, 4, n_tokens)
        coarse = coarse.permute(0, 2, 3, 1).reshape(
            batch * leads, 4, n_tokens * self.patch_size
        )
        return hidden, coarse

    def forward(self, x, *, feature_mask=None, seams=None):
        batch, leads, length = x.shape
        hidden, coarse = self._token_heads(x, feature_mask=feature_mask)
        upsampled = hidden @ self.upsample.to(hidden.dtype)
        local_x = FT.standardize_leads(x).reshape(batch * leads, 1, length)
        local_seams = None if seams is None else seams.reshape(batch * leads, 1, length)
        if self.architecture == "derivative":
            derivative = derivative_channels(local_x, seams=local_seams)[:, 1:]
            projected = self.stem[0](local_x) + self.derivative_projection(derivative)
            stem = self.stem[1:](projected)
        else:
            stem = self.stem(local_x)
            if self.geometry:
                geometry = self.local_geometry(
                    derivative_channels(local_x, seams=local_seams)
                )
                stem = stem + self.local_output(geometry).to(stem.dtype)
        fused = self.fuse(
            torch.cat((upsampled, stem.to(upsampled.dtype), coarse), dim=1)
        )
        return self._to_output(coarse + self.refine(fused), batch, leads)


def build_segmenter(config=None, architecture="gls_local"):
    """创建结构；检查点由调用者显式加载，不读取本地实验目录。"""
    from ..pretraining.core import ECGMAE, ModelConfig

    return IntegratedSegmenter12(
        FT.Segmenter12(ECGMAE(config or ModelConfig())), architecture
    )
