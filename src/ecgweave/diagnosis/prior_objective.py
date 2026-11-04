"""Prevalence-weighted BCE and fixed-prior auxiliary objectives."""

from contextlib import nullcontext
import math

import numpy as np
import torch
from torch.nn import functional as F


def mixed_batch(uniform_rows, sampler, prevalence):
    """Use 16 uniform records and two four-positive/four-negative groups per batch."""
    uniform_rows = np.asarray(uniform_rows, dtype=np.int64)
    layout = {32: (16, 2, 4), 10: (2, 1, 4), 2: (0, 1, 1)}
    if len(uniform_rows) not in layout:
        raise ValueError(
            "Expected a 32-record full batch, 10-record final batch, or two-record smoke batch"
        )
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
    """Return microbatch contributions to the full-update BCE and auxiliary loss.

    records and conditions describe the full update; start is the record offset.
    Summing the returned contributions recovers the full-update objective."""
    if logits.shape != labels.shape or logits.shape[1] != 71 or not conditions:
        raise ValueError(
            "Loss requires matching 71-column labels and nonempty conditional groups"
        )
    weights = logits.new_ones(len(logits))
    # Keep a differentiable zero for uniform-only microbatches.
    auxiliary = logits.sum() * 0
    end = start + len(logits)
    for condition in conditions:
        first, side = condition["start"], condition["per_side"]
        last = first + 2 * side
        if first < end and last > start:
            if not start <= first < last <= end:
                raise ValueError(
                    "Conditional groups must not cross microbatch boundaries"
                )
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
                raise ValueError("Auxiliary objective must be pair or prior")
            auxiliary = auxiliary + value / len(conditions)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1)
    return (bce * weights).sum() / records, auxiliary


def update(model, optimizer, batch, conditions, config, device, scale=None):
    """Accumulate microbatch gradients with one optimizer; scale is unused."""
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
            raise FloatingPointError("Combined loss is not finite")
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
