"""Pseudo-label loss for a fixed teacher."""

import torch
from torch.nn import functional as F
from . import base as FT


def pseudo_ce(
    logits, targets, confidence, valid_mask, *, threshold, denominator, weights=None
):
    """Four-state CE; the valid denominator includes rejected pseudo-label positions."""
    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError("logits must have shape [B, 4, L, T]")
    shape = logits.shape[:1] + logits.shape[2:]
    if any(t.shape != shape for t in (targets, confidence, valid_mask)):
        raise ValueError("targets, confidence, and support must have shape [B, L, T]")
    if targets.dtype != torch.long or valid_mask.dtype != torch.bool:
        raise ValueError("targets must be long and valid must be bool")
    if denominator not in ("valid", "accepted") or not 0 <= threshold <= 1:
        raise ValueError("Invalid denominator or confidence threshold")
    eligible = valid_mask & (targets != FT.IGNORE)
    if ((targets[eligible] < 0) | (targets[eligible] >= 4)).any():
        raise ValueError("Target class index is out of range")
    accepted = eligible & (confidence.detach() >= threshold)
    weights = torch.ones_like(confidence) if weights is None else weights.detach()
    if (
        weights.shape != shape
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError("Invalid sample-weight shape or values")
    support = eligible if denominator == "valid" else accepted
    normalizer = (weights * support).sum()
    safe_targets = torch.where(eligible, targets, 0)
    one_hot = F.one_hot(safe_targets, 4).movedim(-1, 1)
    point_ce = -(one_hot * logits.float().log_softmax(1)).sum(1)
    loss = (point_ce * accepted * weights).sum() / normalizer.clamp_min(1)
    valid_counts = [int((eligible & (targets == c)).sum()) for c in range(4)]
    accepted_counts = [int((accepted & (targets == c)).sum()) for c in range(4)]
    return loss, {
        "valid_points": sum(valid_counts),
        "accepted_points": sum(accepted_counts),
        "class_valid": valid_counts,
        "class_accepted": accepted_counts,
        "class_acceptance": [
            a / v if v else 0.0
            for a, v in zip(accepted_counts, valid_counts, strict=True)
        ],
        "denominator_weight": float(normalizer),
        "accepted_weight": float((weights * accepted).sum()),
    }
