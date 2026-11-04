"""完整自然分布BCE与固定系数的成对/先验对齐宏辅助。"""

from contextlib import nullcontext
import math

import numpy as np
import torch
from torch.nn import functional as F


def mixed_batch(uniform_rows, sampler, prevalence):
    """32条为16均匀+2组4正4负；10条尾批为2均匀+1组4正4负。"""
    uniform_rows = np.asarray(uniform_rows, dtype=np.int64)
    layout = {32: (16, 2, 4), 10: (2, 1, 4), 2: (0, 1, 1)}
    if len(uniform_rows) not in layout:
        raise ValueError("仅接受32条完整批、10条尾批或2条CPU预检批")
    start, classes, side = layout[len(uniform_rows)]
    pieces, conditions = [uniform_rows[:start]], []
    for task, rows in sampler.draw(classes=classes, per_side=side):
        pieces.append(rows)
        conditions.append(
            {
                "task": task,
                "start": start,
                "per_side": side,
                "prevalence": float(prevalence[task]),
            }
        )
        start += 2 * side
    return np.concatenate(pieces), conditions


def loss_components(logits, labels, conditions, objective, records, start=0):
    """返回本微批对完整更新的BCE和辅助贡献；start是全局记录偏移。

    records与conditions均指完整更新。BCE除完整记录数，辅助除完整
    条件组数，因此对多个微批的结果直接相加即可还原完整风险。
    """
    if logits.shape != labels.shape or logits.shape[1] != 71 or not conditions:
        raise ValueError("损失要求完整71列同形标签和非空条件组")
    weights = logits.new_ones(len(logits))
    # 均匀记录微批也保留可求导的零，供分项autograd校准使用。
    auxiliary = logits.sum() * 0
    end = start + len(logits)
    for condition in conditions:
        first, side = condition["start"], condition["per_side"]
        last = first + 2 * side
        if first < end and last > start:
            if not start <= first < last <= end:
                raise ValueError("完整正负条件组禁止跨越微批边界")
            offset, task, prevalence = (
                first - start,
                condition["task"],
                condition["prevalence"],
            )
            weights[offset : offset + side] = 2 * prevalence
            weights[offset + side : offset + 2 * side] = 2 * (1 - prevalence)
            positive = logits[offset : offset + side, task]
            negative = logits[offset + side : offset + 2 * side, task]
            if objective["kind"] == "pair":
                value = F.softplus(negative[None, :] - positive[:, None]).mean()
            elif objective["kind"] == "prior":
                bias = math.log(prevalence) - math.log1p(-prevalence)
                value = 0.5 * (
                    F.softplus(-(positive - bias)).mean()
                    + F.softplus(negative - bias).mean()
                )
            else:
                raise ValueError("辅助目标必须为pair或prior")
            auxiliary = auxiliary + value / len(conditions)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1)
    return (bce * weights).sum() / records, auxiliary


def update(model, optimizer, batch, conditions, config, device, scale=None):
    """单优化器微批梯度累加；scale仅兼容旧驱动接口，不读取或更新。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    records = len(batch["labels"])
    coefficient = config["objective"]["aux_coefficient"]
    total, classification, auxiliary = 0.0, 0.0, 0.0
    for start in range(0, records, config["microbatch"]):
        end = min(start + config["microbatch"], records)
        piece = {key: value[start:end] for key, value in batch.items()}
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if torch.device(device).type == "cuda"
            else nullcontext()
        )
        with context:
            logits = model(piece)
        bce, aux = loss_components(
            logits.float(),
            piece["labels"],
            conditions,
            config["objective"],
            records,
            start,
        )
        loss = bce + coefficient * aux
        if not torch.isfinite(loss):
            raise FloatingPointError("性能组合损失非有限")
        loss.backward()
        total += float(loss.detach())
        classification += float(bce.detach())
        auxiliary += float(aux.detach())
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["clip_norm"], error_if_nonfinite=True
    )
    optimizer.step()
    return {
        "loss": total,
        "bce": classification,
        "auxiliary": auxiliary,
        "aux_coefficient": coefficient,
        "gradient_norm": float(norm),
    }
