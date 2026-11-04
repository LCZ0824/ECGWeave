"""Waveform preprocessing, soft references, and checkpoint loading."""

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch

from .diagnosis.model import JointMultiresClassifier, ModelConfig
from .diagnosis.stats_model import build_classifier
from .delineation.model import build_segmenter


LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
STATE_ORDER = ("BG", "P", "QRS", "T")


def validate_signals(signals):
    """Require finite float32 [B, 12, 5000] signals at 500 Hz in mV."""
    if signals.ndim != 3 or signals.shape[1:] != (12, 5000) or len(signals) < 1:
        raise ValueError("signals must be nonempty [B, 12, 5000]; see LEAD_ORDER")
    if signals.dtype != torch.float32 or not torch.isfinite(signals).all():
        raise ValueError("signals must be finite float32 waveforms in mV")


def load_classifier(checkpoint, device="cpu"):
    """Load a trusted local classifier checkpoint."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**saved["model_config"])
    settings = saved["config"]
    if settings.get("structure") == "performance_composite":
        if settings["reader_kind"] != "stats" or not settings["lead_attention_off"]:
            raise ValueError("This entry point supports the stats_prior20 structure")
        model = build_classifier(config)
    else:
        model = JointMultiresClassifier(config, arm=settings["arm"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()


def load_delineator(checkpoint, device="cpu"):
    """Strictly load a delineator state_dict and freeze its parameters."""
    model = build_segmenter()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).requires_grad_(False).eval()


def pool_reference(probabilities, signals, sample_support=None, patch_size=20):
    """Average valid 20-sample probability windows into [B, L, K, 4] references."""
    batch, leads, length = signals.shape
    if length % patch_size or probabilities.shape != (batch, 4, leads, length):
        raise ValueError(
            "Probabilities must align with [B, 4, L, T], with T divisible by patch_size"
        )
    support = torch.ones_like(signals, dtype=torch.bool)
    if sample_support is not None:
        if sample_support.shape != signals.shape or sample_support.dtype != torch.bool:
            raise ValueError(
                "sample_support must be bool with the same shape as signals"
            )
        support = sample_support
    support = support & (signals.amax(-1) > signals.amin(-1))[..., None]
    shape = (batch, leads, length // patch_size, patch_size)
    valid = support.reshape(shape).all(-1)
    probability = probabilities.float().permute(0, 2, 3, 1)
    reference = probability.reshape(*shape, 4).mean(-2)
    valid = valid & torch.isfinite(reference).all(-1) & (reference >= 0).all(-1)
    mass = reference.sum(-1, keepdim=True)
    valid = valid & (mass[..., 0] > 0)
    return reference.masked_fill(~valid[..., None], 0), valid


def filter500(signals):
    """Filter the delineator input while preserving raw classifier waveforms."""
    high = butter(5, 0.67, btype="highpass", fs=500, output="sos")
    low = butter(5, 40, btype="lowpass", fs=500, output="sos")
    return sosfiltfilt(low, sosfiltfilt(high, signals)).astype(np.float32)


@torch.inference_mode()
def waveform_batch(signals, delineator, sample_support=None):
    """Generate soft references with a frozen delineator."""
    validate_signals(signals)
    if delineator.training or any(p.requires_grad for p in delineator.parameters()):
        raise ValueError(
            "The delineator must be in eval mode with requires_grad_(False)"
        )
    filtered = torch.from_numpy(filter500(signals.detach().cpu().numpy())).to(
        signals.device
    )
    probabilities = delineator(filtered).float().softmax(1)
    reference, support = pool_reference(probabilities, signals, sample_support)
    # Quantize posteriors to float16, then restore FP32 and normalize.
    reference = reference.half().float()
    reference = reference / reference.sum(-1, keepdim=True).clamp_min(1e-12)
    return {"signals": signals, "reference": reference, "reference_support": support}
