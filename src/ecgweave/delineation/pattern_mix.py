"""同步导联的原始供体双向 Pattern 搬运；不读取 U 真值。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

MIN_WIDTH = 500
MAX_WIDTH = 2500
THRESHOLD = 0.8


def best_match(query, donors, starts, *, query_valid=None, donor_valid=None):
    """在共同有效的 0..3 标签上计算四类宏 IoU，双空类记 0。

    一个候选同时覆盖全部导联。并列按供体行、起点升序决定；完全无
    共同有效点时返回 None，而非把零支持误当背景匹配。
    """
    query, donors = np.asarray(query), np.asarray(donors)
    if query.ndim != 2 or donors.ndim != 3 or donors.shape[1] != query.shape[0]:
        raise ValueError("匹配要求 query[L,W] 与 donors[N,L,T]")
    width = query.shape[-1]
    starts = np.unique(np.asarray(starts, dtype=np.int64))
    if starts.ndim != 1 or width < 1:
        raise ValueError("窗口宽度须为正，候选起点须为一维")
    if np.any(starts < 0) or np.any(starts + width > donors.shape[-1]):
        raise ValueError("候选窗口超出供体时间范围")
    q_valid = (query >= 0) & (query < 4)
    d_valid = (donors >= 0) & (donors < 4)
    if query_valid is not None:
        query_valid = np.asarray(query_valid, dtype=bool)
        if query_valid.shape != query.shape:
            raise ValueError("query_valid 须与 query 同形")
        q_valid &= query_valid
    if donor_valid is not None:
        donor_valid = np.asarray(donor_valid, dtype=bool)
        if donor_valid.shape != donors.shape:
            raise ValueError("donor_valid 须与 donors 同形")
        d_valid &= donor_valid
    if not len(starts) or not len(donors):
        return None
    candidates = np.stack(
        [donors[..., start : start + width] for start in starts], axis=1
    )
    common = q_valid[None, None] & np.stack(
        [d_valid[..., start : start + width] for start in starts], axis=1
    )
    supported = common.any(axis=(-2, -1))
    if not supported.any():
        return None
    score = np.zeros(candidates.shape[:2], dtype=np.float64)
    for label in range(4):
        q, d = query == label, candidates == label
        intersection = ((q & d) & common).sum(axis=(-2, -1))
        union = ((q | d) & common).sum(axis=(-2, -1))
        score += np.divide(
            intersection, union, out=np.zeros_like(score), where=union > 0
        )
    score /= 4
    score[~supported] = -np.inf
    row, column = np.unravel_index(score.argmax(), score.shape)
    return {
        "row": int(row),
        "start": int(starts[column]),
        "iou": float(score[row, column]),
    }


def _counts(labels):
    labels = np.asarray(labels)
    known = (labels >= 0) & (labels < 4)
    return np.bincount(labels[known], minlength=4)


def _mark_seams(seams, row, start, width, length):
    for position in (start, start + width):
        if 0 < position < length:
            seams[row, :, position] = True


def mix_batches(
    x_l,
    y_l,
    weak_u,
    strong_u,
    probabilities,
    *,
    valid_u,
    seed,
    threshold=THRESHOLD,
    seams_l=None,
    seams_u=None,
):
    """同批双向搬运同步导联窗口，两个方向均只读取原始输入。

    L2U 已知人工标签映射为 one-hot/conf=1；未知位置 q/conf=0、
    valid=False、target=-100。U2L 的窗口置信度仅在原供体有效点计算，
    FP64 累计且严格大于 FP32 阈值；原目的人工 ignore 永远保留。
    seams 只标内域切口，导数模型自行决定扩张范围。
    """
    if x_l.ndim != 3:
        raise ValueError("混合要求非空 [B,L,T] 波形，T至少500")
    length = x_l.shape[-1]
    if (
        length < MIN_WIDTH
        or x_l.shape[0] == 0
        or x_l.shape[1] == 0
        or weak_u.ndim != 3
        or weak_u.shape[0] == 0
        or weak_u.shape[1:] != x_l.shape[1:]
        or strong_u.shape != weak_u.shape
        or y_l.shape != x_l.shape
        or probabilities.shape != (len(weak_u), 4, *x_l.shape[1:])
        or valid_u.shape != weak_u.shape
    ):
        raise ValueError("混合要求非空 [B,L,T] 与 [B,4,L,T]，T至少500")
    if (
        y_l.dtype != torch.long
        or valid_u.dtype != torch.bool
        or any(t.dtype != torch.float32 for t in (x_l, weak_u, strong_u, probabilities))
    ):
        raise ValueError("L 标签为 long，U 有效性为 bool，波形与概率为 FP32")
    if (
        len({t.device for t in (x_l, y_l, weak_u, strong_u, probabilities, valid_u)})
        != 1
    ):
        raise ValueError("所有输入须位于同一设备")

    seams_l = torch.zeros_like(y_l, dtype=torch.bool) if seams_l is None else seams_l
    seams_u = torch.zeros_like(valid_u) if seams_u is None else seams_u
    probabilities = probabilities.detach()
    confidence, pseudo = probabilities.max(1)
    l_known = (y_l >= 0) & (y_l < 4)
    l_np = y_l.detach().cpu().numpy()
    l_valid_np = l_known.cpu().numpy()
    u_np = pseudo.cpu().numpy()
    u_valid_np = valid_u.cpu().numpy()
    c_np = confidence.cpu().numpy()
    threshold_fp32 = np.float32(threshold)
    rng = np.random.default_rng(int(seed))
    width = int(rng.integers(MIN_WIDTH, min(MAX_WIDTH, length) + 1))
    stride = width // 2
    starts = list(range(0, length - width + 1, stride))
    # 先抽 L 位置，再抽 U 位置；不消费 NumPy/PyTorch 全局随机流。
    l_starts = rng.integers(0, length - width + 1, size=len(x_l))
    u_starts = rng.integers(0, length - width + 1, size=len(weak_u))
    output = {
        "x_l": x_l.clone(),
        "y_l": y_l.clone(),
        "strong_u": strong_u.clone(),
        "target_u": pseudo.masked_fill(~valid_u, -100),
        "confidence_u": confidence.masked_fill(~valid_u, 0),
        "accepted_u": (confidence >= float(threshold_fp32)) & valid_u,
        "probabilities_u": probabilities.masked_fill(~valid_u[:, None], 0),
        "valid_u": valid_u.clone(),
        "seams_l": seams_l.clone(),
        "seams_u": seams_u.clone(),
    }
    copied_true = np.zeros(4, dtype=np.int64)
    removed_true = np.zeros(4, dtype=np.int64)
    copied_pseudo = np.zeros(4, dtype=np.int64)
    plans = []
    accepted_l, accepted_u = 0, 0
    for direction, queries, donors, query_valid, donor_valid, query_starts in (
        ("l2u", u_np, l_np, u_valid_np, l_valid_np, u_starts),
        ("u2l", l_np, u_np, l_valid_np, u_valid_np, l_starts),
    ):
        for row, start_value in enumerate(query_starts):
            start = int(start_value)
            dst = slice(start, start + width)
            chosen = best_match(
                queries[row, :, dst],
                donors,
                starts,
                query_valid=query_valid[row, :, dst],
                donor_valid=donor_valid,
            )
            donor = source = score = mean_confidence = None
            common_points = 0
            accepted = False
            if chosen is not None:
                donor, source, score = chosen["row"], chosen["start"], chosen["iou"]
                src = slice(source, source + width)
                common_points = int(
                    (query_valid[row, :, dst] & donor_valid[donor, :, src]).sum()
                )
                if direction == "u2l":
                    mean_confidence = float(
                        c_np[donor, :, src][donor_valid[donor, :, src]].mean(
                            dtype=np.float64
                        )
                    )
                    accepted = mean_confidence > threshold_fp32
                else:
                    accepted = True
            plans.append(
                {
                    "direction": direction,
                    "query_row": row,
                    "query_start": start,
                    "query_window": [start, start + width],
                    "donor_row": donor,
                    "donor_start": source,
                    "donor_window": None
                    if source is None
                    else [source, source + width],
                    "width": width,
                    "iou": score,
                    "score": score,
                    "mean_confidence": mean_confidence,
                    "mean_conf": mean_confidence,
                    "common_support_points": common_points,
                    "accepted": bool(accepted),
                }
            )
            if not accepted:
                continue
            if direction == "l2u":
                source_labels = y_l[donor, :, src]
                source_known = l_known[donor, :, src]
                output["strong_u"][row, :, dst] = x_l[donor, :, src]
                output["seams_u"][row, :, dst] = seams_l[donor, :, src]
                output["target_u"][row, :, dst] = source_labels.masked_fill(
                    ~source_known, -100
                )
                output["valid_u"][row, :, dst] = source_known
                output["confidence_u"][row, :, dst] = source_known.to(confidence.dtype)
                output["accepted_u"][row, :, dst] = source_known
                output["probabilities_u"][row, :, :, dst] = (
                    F.one_hot(source_labels.masked_fill(~source_known, 0), 4)
                    .permute(2, 0, 1)
                    .to(probabilities.dtype)
                    * source_known[None]
                )
                copied_true += _counts(l_np[donor, :, src])
                accepted_u += 1
                _mark_seams(output["seams_u"], row, start, width, length)
            else:
                keep = l_known[row, :, dst] & valid_u[donor, :, src]
                output["x_l"][row, :, dst] = weak_u[donor, :, src]
                output["seams_l"][row, :, dst] = seams_u[donor, :, src]
                output["y_l"][row, :, dst] = pseudo[donor, :, src].masked_fill(
                    ~keep, -100
                )
                removed_true += _counts(l_np[row, :, dst])
                copied_pseudo += _counts(u_np[donor, :, src][keep.cpu().numpy()])
                accepted_l += 1
                _mark_seams(output["seams_l"], row, start, width, length)

    original_true = _counts(l_np)
    human_labels = original_true - removed_true + copied_true
    output["diagnostics"] = {
        "seed": int(seed),
        "width": width,
        "stride": stride,
        "candidate_starts": starts,
        "plans": plans,
        "threshold": float(threshold),
        "threshold_fp32": float(threshold_fp32),
        "u2l_mean_accumulator": "float64",
        "u2l_mean_support": "所选原 U 供体片段的有效导联×时间点",
        "empty_union_iou": 0,
        "common_support_iou": "query/donor 均有效且标签在0..3；合并导联×时间计四类宏IoU",
        "no_common_support": "不匹配、不搬运、不产生切口",
        "ignore_label": -100,
        "ignore_contract": "L2U未知人工标签置q/conf=0且valid=False；U2L保留原目的ignore，供体无效亦ignore",
        "l2u_acceptance_fraction": accepted_u / len(weak_u),
        "u2l_acceptance_fraction": accepted_l / len(x_l),
        "l2u_copied_true_l_by_class": copied_true.tolist(),
        "u2l_removed_true_l_by_class": removed_true.tolist(),
        "u2l_copied_pseudo_by_class": copied_pseudo.tolist(),
        "human_label_points_by_class": human_labels.tolist(),
        "human_label_points_total": int(human_labels.sum()),
        "human_label_fraction_l": int((original_true - removed_true).sum())
        / y_l.numel(),
        "human_label_fraction_u": int(copied_true.sum()) / pseudo.numel(),
        "cut_scope": "原供体填充边界随波形搬运，追加内域start/start+width；不扩张邻域",
        "time_scope": "500Hz增强后的采样网格；不一概对应原患者未增强生理时间",
    }
    return output
