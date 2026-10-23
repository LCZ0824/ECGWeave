# 公开版适配说明：包内相对导入；不改计算。
"""检验宏类别条件风险、记录预算和同分选择，不依赖患者数据。"""

import numpy as np
import pytest
import torch

from ecgweave.diagnosis.objective import (
    ConditionSampler,
    best_point,
    class_objective,
    rank_budget,
)


def test_budget_preserves_exact_record_count_and_partial_class_tail():
    schedule = rank_budget(17418, 20, classes=8, per_side=4)
    assert len(schedule) == 5444
    assert schedule[-1] == 1
    assert sum(schedule) * 8 == 348360
    with pytest.raises(ValueError):
        rank_budget(3, 1, classes=8, per_side=4)


def test_conditional_sampling_has_label_identity_and_exact_resume():
    labels = np.tile(np.array([[0], [1], [0], [1]]), (1, 71))
    sampler = ConditionSampler(labels, np.arange(4), seed=5)
    first = sampler.draw(8, 4)
    assert len(set(c for c, _ in first)) == 8
    for c, rows in first:
        assert (labels[rows[:4], c] == 1).all()
        assert (labels[rows[4:], c] == 0).all()
    state = sampler.state_dict()
    expected = sampler.draw(8, 4)
    sampler.load_state_dict(state)
    actual = sampler.draw(8, 4)
    for (a, x), (b, y) in zip(expected, actual, strict=True):
        assert a == b
        np.testing.assert_array_equal(x, y)


def test_rank_depends_on_order_and_gradient_pushes_positive_up():
    logits = torch.zeros(8, 71, requires_grad=True)
    labels = torch.zeros_like(logits)
    labels[:4, 3] = 1
    rank, bce = class_objective(logits, labels, 3, torch.ones(71), 0.25, 4)
    assert rank.item() == 1.0
    assert bce.item() == pytest.approx(np.log(2), abs=1e-7)
    rank.backward()
    assert (logits.grad[:4, 3] < 0).all()
    assert (logits.grad[4:, 3] > 0).all()
    assert logits.grad[:, 4:].count_nonzero() == 0
    logits = logits.detach()
    logits[:4, 3] = 2
    assert class_objective(logits, labels, 3, torch.ones(71), 0.25, 4)[0] == 0


def test_importance_weighted_bce_matches_population_exactly():
    # 条件批各取两个阳/阴；总体阳性率1/4，各条件内部取值恒定。
    logits = torch.cat([torch.full((2, 71), 2.0), torch.full((2, 71), -1.0)])
    labels = torch.cat([torch.ones(2, 71), torch.zeros(2, 71)])
    _, corrected = class_objective(logits, labels, 0, torch.ones(71), 0.25, 2)
    population_logits = torch.cat([logits[:1], logits[2:3].expand(3, -1)])
    population_labels = torch.cat([labels[:1], labels[2:3].expand(3, -1)])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        population_logits, population_labels
    )
    torch.testing.assert_close(corrected, expected)


def test_selection_uses_auc_and_earliest_tie_not_ap():
    history = [
        {"index": 0, "macro_auroc": 0.94, "macro_ap": 0.6},
        {"index": 1, "macro_auroc": 0.95, "macro_ap": 0.4},
        {"index": 2, "macro_auroc": 0.95, "macro_ap": 0.8},
    ]
    assert best_point(history) is history[1]
