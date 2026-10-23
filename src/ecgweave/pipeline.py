"""原始mV波形、固定软参考与最终分类器的公开接口。"""

import torch

from .diagnosis.model import JointMultiresClassifier, ModelConfig
from .delineation.model import build_segmenter


LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
STATE_ORDER = ("BG", "P", "QRS", "T")


def validate_signals(signals):
    """公开输入为500Hz、10秒、float32原始mV；不隐式缩放或重采样。"""
    if signals.ndim != 3 or signals.shape[1:] != (12, 5000) or len(signals) < 1:
        raise ValueError("signals须为非空[B,12,5000]，导联顺序见LEAD_ORDER")
    if signals.dtype != torch.float32 or not torch.isfinite(signals).all():
        raise ValueError("signals须为有限float32原始mV")


def load_classifier(checkpoint, device="cpu"):
    """仅加载可信本地检查点；兼容原训练字典，不读取F0或患者缓存。"""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**saved["model_config"])
    model = JointMultiresClassifier(config, arm=saved["config"]["arm"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()


def load_delineator(checkpoint, device="cpu"):
    """严格加载最终gls_local定界器的纯state_dict，不读取S*或F0。"""
    model = build_segmenter()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).requires_grad_(False).eval()


def pool_reference(probabilities, signals, sample_support=None, patch_size=20):
    """完整20点支持取概率均值；常数导联及无效格置零，输出[B,L,K,4]。"""
    batch, leads, length = signals.shape
    if length % patch_size or probabilities.shape != (batch, 4, leads, length):
        raise ValueError("四态概率须对齐[B,4,L,T]，T可被patch_size整除")
    support = torch.ones_like(signals, dtype=torch.bool)
    if sample_support is not None:
        if sample_support.shape != signals.shape or sample_support.dtype != torch.bool:
            raise ValueError("sample_support须为与signals同形的bool")
        support = sample_support
    support = support & (signals.amax(-1) > signals.amin(-1))[..., None]
    shape = (batch, leads, length // patch_size, patch_size)
    valid = support.reshape(shape).all(-1)
    probability = probabilities.float().permute(0, 2, 3, 1)
    reference = probability.reshape(*shape, 4).mean(-2)
    valid = valid & torch.isfinite(reference).all(-1) & (reference >= 0).all(-1)
    mass = reference.sum(-1, keepdim=True)
    valid = valid & (mass[..., 0] > 0)
    reference = reference / mass.clamp_min(1e-12)
    return reference.masked_fill(~valid[..., None], 0), valid


@torch.inference_mode()
def waveform_batch(signals, delineator, sample_support=None):
    """以固定定界器一次前向生成对齐软参考；不读取记录级标签。"""
    validate_signals(signals)
    if delineator.training or any(p.requires_grad for p in delineator.parameters()):
        raise ValueError("生成参考前须将定界器置eval并requires_grad_(False)")
    probabilities = delineator(signals).float().softmax(1)
    reference, support = pool_reference(probabilities, signals, sample_support)
    return {"signals": signals, "reference": reference, "reference_support": support}
