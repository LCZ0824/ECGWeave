# 公开版适配说明：包内相对导入；不改计算。
"""全71类均匀条件排序，保持记录级经验宏风险及精确呈现预算。"""

import copy

import numpy as np
import torch
from torch.nn import functional as F


def rank_budget(records, passes, classes=8, per_side=4):
    total = records * passes
    if min(records, passes, classes, per_side) < 1 or total % (2 * per_side):
        raise ValueError("记录预算必须容纳整数个完整正负类别批")
    complete, tail = divmod(total // (2 * per_side), classes)
    return [classes] * complete + ([tail] if tail else [])


class ConditionSampler:
    def __init__(self, labels, train_rows, seed=0):
        self.rng = np.random.RandomState(seed)
        rows = np.asarray(train_rows, dtype=np.int64)
        target = np.asarray(labels)[rows]
        if (
            target.shape != (len(rows), 71)
            or len(np.unique(rows)) != len(rows)
            or not np.isin(target, [0, 1]).all()
        ):
            raise ValueError("采样器要求完整71列、唯一TRAIN行和二元标签")
        self.positive = [rows[target[:, c] == 1] for c in range(71)]
        self.negative = [rows[target[:, c] == 0] for c in range(71)]
        if any(not len(pool) for pool in self.positive + self.negative):
            raise ValueError("TRAIN每类均须存在正负记录")

    def draw(self, classes=8, per_side=4):
        tasks = self.rng.choice(71, classes, replace=False)
        return [
            (
                int(c),
                np.concatenate(
                    [
                        self.rng.choice(self.positive[c], per_side, replace=True),
                        self.rng.choice(self.negative[c], per_side, replace=True),
                    ]
                ),
            )
            for c in tasks
        ]

    def state_dict(self):
        return copy.deepcopy(self.rng.get_state())

    def load_state_dict(self, state):
        self.rng.set_state(copy.deepcopy(state))


def class_objective(logits, labels, task, scale, prevalence, per_side=4):
    """返回平方hinge与逆采样概率校正后的完整71 BCE。

    条件批正负各半，BCE按真实TRAIN主类阳性率还原总体记录权重，
    以免将重采样后的标签共现分布误当作原总体分布。
    """
    if (
        logits.shape != (2 * per_side, 71)
        or labels.shape != logits.shape
        or scale.shape != (71,)
        or not 0 <= task < 71
        or not 0 < prevalence < 1
    ):
        raise ValueError("条件排序要求同形完整71列、合法类别与阳性率")
    scores = logits[:, task]
    margin = (scores[:per_side, None] - scores[None, per_side:]) / scale[task]
    rank = torch.relu(1 - margin).square().mean()
    per_record = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    ).mean(1)
    bce = (
        prevalence * per_record[:per_side].mean()
        + (1 - prevalence) * per_record[per_side:].mean()
    )
    return rank, bce


def best_point(history):
    return max(history, key=lambda point: (point["macro_auroc"], -point["index"]))
