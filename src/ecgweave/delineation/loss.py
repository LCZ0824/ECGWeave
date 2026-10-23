# 公开版适配说明：抽取原pseudo_ce；包内常量导入。
"""固定教师伪标签损失；正式配置按全部有效点作分母。"""

import torch
from torch.nn import functional as F
from . import base as FT


def pseudo_ce(
    logits, targets, confidence, valid_mask, *, threshold, denominator, weights=None
):
    """四类CE；valid分母包括未过阈值点；加权时分母为对应支持的权重和。"""
    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError("logits形状须为[B,4,L,T]")
    shape = logits.shape[:1] + logits.shape[2:]
    if any(t.shape != shape for t in (targets, confidence, valid_mask)):
        raise ValueError("伪标签/置信度/支持形状须为[B,L,T]")
    if targets.dtype != torch.long or valid_mask.dtype != torch.bool:
        raise ValueError("targets须long且valid须bool")
    if denominator not in ("valid", "accepted") or not 0 <= threshold <= 1:
        raise ValueError("分母或阈值无效")
    eligible = valid_mask & (targets != FT.IGNORE)
    if ((targets[eligible] < 0) | (targets[eligible] >= 4)).any():
        raise ValueError("硬标签类别越界")
    accepted = eligible & (confidence.detach() >= threshold)
    weights = torch.ones_like(confidence) if weights is None else weights.detach()
    if (
        weights.shape != shape
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError("样点权重形状/值无效")
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
