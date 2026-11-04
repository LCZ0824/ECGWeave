"""Mixed sampling, weighted losses, and optimizer update tests."""

import copy
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ecgweave.diagnosis.objective import ConditionSampler


def implementation():
    from ecgweave.diagnosis import prior_objective

    return prior_objective


def configuration(kind):
    return {
        "seed": 0,
        "microbatch": 16,
        "clip_norm": 5.0,
        "objective": {"kind": kind, "aux_coefficient": 0.03},
    }


def conditions_for(records):
    starts = [16, 24] if records == 32 else [2 if records == 10 else 0]
    return [
        {
            "task": 3 + index,
            "start": start,
            "per_side": 1 if records == 2 else 4,
            "prevalence": 0.03 + 0.27 * index,
        }
        for index, start in enumerate(starts)
    ]


def reference(logits, labels, conditions, kind):
    weights = torch.ones(len(logits), device=logits.device)
    auxiliary = []
    for condition in conditions:
        start, side, task, prevalence = (
            condition["start"],
            condition["per_side"],
            condition["task"],
            condition["prevalence"],
        )
        weights[start : start + side] = 2 * prevalence
        weights[start + side : start + 2 * side] = 2 * (1 - prevalence)
        positive = logits[start : start + side, task]
        negative = logits[start + side : start + 2 * side, task]
        if kind == "pair":
            auxiliary.append(F.softplus(negative[None, :] - positive[:, None]).mean())
        else:
            bias = math.log(prevalence / (1 - prevalence))
            auxiliary.append(
                (
                    F.softplus(-(positive - bias)).mean()
                    + F.softplus(negative - bias).mean()
                )
                / 2
            )
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1)
    return (bce * weights).mean(), torch.stack(auxiliary).mean()


@pytest.mark.parametrize(
    "records,classes,prefix,side", [(32, 2, 16, 4), (10, 1, 2, 4), (2, 1, 0, 1)]
)
def test_mixed_sampling_preserves_uniform_records_and_complete_conditions(
    records, classes, prefix, side
):
    module = implementation()
    labels = ((np.arange(100)[:, None] + np.arange(71)) % 5 == 0).astype(np.float32)
    sampler = ConditionSampler(labels, np.arange(100), seed=19)
    uniform = np.arange(records)
    rows, conditions = module.mixed_batch(uniform, sampler, labels.mean(0))
    np.testing.assert_array_equal(rows[:prefix], uniform[:prefix])
    assert len(rows) == records and len(conditions) == classes
    assert len({item["task"] for item in conditions}) == classes
    for index, item in enumerate(conditions):
        assert item["start"] == prefix + index * 2 * side
        assert item["per_side"] == side
        start, task = item["start"], item["task"]
        assert (labels[rows[start : start + side], task] == 1).all()
        assert (labels[rows[start + side : start + 2 * side], task] == 0).all()
        assert item["prevalence"] == pytest.approx(float(labels[:, task].mean()))


@pytest.mark.parametrize("kind", ["pair", "prior"])
@pytest.mark.parametrize("records", [32, 10, 2])
def test_components_match_dense_math_and_all_71_bce_columns(kind, records):
    module = implementation()
    logits = torch.linspace(-4, 3, records * 71).reshape(records, 71)
    labels = (torch.arange(records * 71).reshape(records, 71) % 3 == 0).float()
    conditions = conditions_for(records)
    expected = reference(logits, labels, conditions, kind)
    actual = module.loss_components(
        logits, labels, conditions, configuration(kind)["objective"], records
    )
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target)
    changed = labels.clone()
    changed[:, 30:] = 1 - changed[:, 30:]
    other_bce, other_aux = module.loss_components(
        logits, changed, conditions, configuration(kind)["objective"], records
    )
    assert not torch.isclose(actual[0], other_bce)
    torch.testing.assert_close(actual[1], other_aux)


def test_pair_gradient_has_correct_direction_bound_and_bias_invariance():
    module = implementation()
    logits = torch.zeros(10, 71, requires_grad=True)
    conditions = conditions_for(10)
    _, auxiliary = module.loss_components(
        logits, torch.zeros_like(logits), conditions, {"kind": "pair"}, 10
    )
    gradient = torch.autograd.grad(auxiliary, logits)[0]
    assert (gradient[2:6, 3] < 0).all()
    assert (gradient[6:10, 3] > 0).all()
    assert float(gradient.abs().max()) <= 1 / 4
    torch.testing.assert_close(gradient.sum(0), torch.zeros(71))
    _, shifted = module.loss_components(
        logits + torch.arange(71),
        torch.zeros_like(logits),
        conditions,
        {"kind": "pair"},
        10,
    )
    torch.testing.assert_close(auxiliary, shifted)


def test_prior_auxiliary_and_natural_bce_share_the_posterior_optimum():
    module = implementation()
    prevalence = 0.2
    # Positive and negative pools have morphology ratios 3:1 and 1:3.
    scores = torch.tensor(
        [
            math.log(prevalence / (1 - prevalence)) + math.log(3),
            math.log(prevalence / (1 - prevalence)) - math.log(3),
        ],
        requires_grad=True,
    )
    indices = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1])
    logits = torch.zeros(8, 71)
    logits[:, 3] = scores[indices]
    labels = torch.zeros_like(logits)
    labels[:4, 3] = 1
    condition = {"task": 3, "start": 0, "per_side": 4, "prevalence": prevalence}
    bce, auxiliary = module.loss_components(
        logits, labels, [condition], {"kind": "prior"}, 8
    )
    torch.testing.assert_close(
        torch.autograd.grad(auxiliary, scores, retain_graph=True)[0],
        torch.zeros_like(scores),
        atol=2e-8,
        rtol=0,
    )
    torch.testing.assert_close(
        torch.autograd.grad(bce, scores)[0], torch.zeros_like(scores), atol=2e-9, rtol=0
    )


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 71)

    def forward(self, batch):
        return self.head(batch["signals"])


@pytest.mark.parametrize("kind", ["pair", "prior"])
@pytest.mark.parametrize("records", [32, 10, 2])
def test_microbatch_gradient_and_step_match_full_batch(kind, records):
    module = implementation()
    torch.manual_seed(421)
    model = LinearModel()
    expected = copy.deepcopy(model)
    model.eval()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 4)
    batch = {
        "signals": torch.randn(records, 3),
        "labels": torch.randint(0, 2, (records, 71)).float(),
    }
    conditions, config = conditions_for(records), configuration(kind)
    bce, auxiliary = reference(expected(batch), batch["labels"], conditions, kind)
    (bce + config["objective"]["aux_coefficient"] * auxiliary).backward()
    norm = torch.nn.utils.clip_grad_norm_(expected.parameters(), 5)
    torch.optim.SGD(expected.parameters(), lr=0.03).step()
    scale = torch.ones(71)
    result = module.update(
        model,
        torch.optim.SGD(model.parameters(), lr=0.03),
        batch,
        conditions,
        config,
        "cpu",
        scale,
    )
    assert model.training
    assert result["bce"] == pytest.approx(float(bce.detach()), rel=1e-6)
    assert result["auxiliary"] == pytest.approx(float(auxiliary.detach()), rel=1e-6)
    assert result["gradient_norm"] == pytest.approx(float(norm), rel=1e-5)
    assert result["loss"] == pytest.approx(
        float((bce + 0.03 * auxiliary).detach()), rel=1e-6
    )
    torch.testing.assert_close(scale, torch.ones(71))
    for actual, target in zip(model.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, target.grad, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(actual, target, atol=1e-7, rtol=1e-5)


def test_conditions_cannot_cross_microbatch_boundary():
    module = implementation()
    model = LinearModel()
    initial = copy.deepcopy(model.state_dict())
    batch = {"signals": torch.randn(24, 3), "labels": torch.zeros(24, 71)}
    condition = {"task": 3, "start": 12, "per_side": 4, "prevalence": 0.1}
    with pytest.raises(ValueError, match="microbatch|cross"):
        module.update(
            model,
            torch.optim.SGD(model.parameters(), lr=1),
            batch,
            [condition],
            configuration("pair"),
            "cpu",
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name])
