# 公开版适配说明：包内相对导入；不改计算。
"""联合导联 Pattern 的来源、支持域及确定性合同。"""

from __future__ import annotations


import numpy as np
import pytest
import torch
import torch.nn.functional as F


def _pattern():
    from ecgweave.delineation import pattern_mix

    return pattern_mix


def test_best_match_computes_macro_iou_on_common_support():
    query = np.array([[0, 0, 1, 1], [2, 2, 3, 3]])
    donors = np.array([[[0, 1, 1, 1], [2, 3, 3, 3]]])
    module = _pattern()
    assert module.best_match(query, donors, [0])["iou"] == pytest.approx(7 / 12)

    query_valid = np.ones_like(query, dtype=bool)
    donor_valid = np.ones_like(donors, dtype=bool)
    query_valid[1, 1] = False
    donor_valid[0, 0, 1] = False
    match = module.best_match(
        query, donors, [0], query_valid=query_valid, donor_valid=donor_valid
    )
    assert match == {"row": 0, "start": 0, "iou": 1.0}


def test_best_match_uses_zero_for_empty_classes_and_stable_row_start_ties():
    query = np.zeros((12, 4), dtype=np.int64)
    donors = np.zeros((2, 12, 8), dtype=np.int64)
    match = _pattern().best_match(query, donors, [4, 0])
    assert match == {"row": 0, "start": 0, "iou": 0.25}


def test_best_match_excludes_unknown_labels_and_unsupported_candidates():
    query = np.array([[0, 1, -100, 3]])
    donors = np.array([[[0, 1, 2, 3]], [[0, 1, -100, 3]]])
    donor_valid = np.ones_like(donors, dtype=bool)
    donor_valid[0] = False
    module = _pattern()
    match = module.best_match(query, donors, [0], donor_valid=donor_valid)
    assert match == {"row": 1, "start": 0, "iou": 0.75}
    assert (
        module.best_match(
            query, donors, [0], query_valid=np.zeros_like(query, dtype=bool)
        )
        is None
    )
    assert module.best_match(query, np.full_like(donors, -100), [0]) is None
    assert module.best_match(query, donors, []) is None


def _inputs(length=1000, labeled=2, unlabeled=3, confidence=0.81):
    time = torch.arange(length)
    lead = torch.arange(12)[:, None]
    l_row = torch.arange(labeled)[:, None, None]
    u_row = torch.arange(unlabeled)[:, None, None]
    x_l = (100_000 + 100_000 * l_row + 1_000 * lead + time).float()
    weak_u = (900_000 + 100_000 * u_row + 1_000 * lead + time).float()
    strong_u = -weak_u.clone()
    y_l = (time // 23 + lead + l_row).remainder(4).long()
    pseudo = (time // 19 + 2 * lead + u_row).remainder(4).long()
    y_l[:, :, 30:65] = -100
    y_l[0, 0] = -100
    valid_u = torch.ones_like(pseudo, dtype=torch.bool)
    valid_u[:, :, 20:55] = False
    valid_u[:, 1] = False
    one_hot = F.one_hot(pseudo, 4).permute(0, 3, 1, 2).float()
    probabilities = (
        one_hot * confidence + (1 - one_hot) * ((1 - confidence) / 3)
    ).requires_grad_()
    return {
        "x_l": x_l,
        "y_l": y_l,
        "weak_u": weak_u,
        "strong_u": strong_u,
        "probabilities": probabilities,
        "valid_u": valid_u,
    }


def _counts(labels):
    known = (labels >= 0) & (labels < 4)
    return torch.bincount(labels[known], minlength=4).cpu().numpy()


def test_original_padding_seams_follow_original_donors():
    inputs = _inputs(length=1000)
    seams_l = torch.zeros_like(inputs["y_l"], dtype=torch.bool)
    seams_u = torch.zeros_like(inputs["valid_u"])
    seams_l[..., 100:900:37] = True
    seams_u[..., 50:950:43] = True
    output = _pattern().mix_batches(**inputs, seed=73, seams_l=seams_l, seams_u=seams_u)
    expected = {"l": seams_l.clone(), "u": seams_u.clone()}
    for plan in output["diagnostics"]["plans"]:
        if not plan["accepted"]:
            continue
        row, start, width = plan["query_row"], plan["query_start"], plan["width"]
        donor, source = plan["donor_row"], plan["donor_start"]
        target, original = (
            ("u", seams_l) if plan["direction"] == "l2u" else ("l", seams_u)
        )
        expected[target][row, :, start : start + width] = original[
            donor, :, source : source + width
        ]
        for cut in (start, start + width):
            if 0 < cut < 1000:
                expected[target][row, :, cut] = True
    assert torch.equal(output["seams_l"], expected["l"])
    assert torch.equal(output["seams_u"], expected["u"])


def _reference_match(query, donors, starts, query_valid, donor_valid):
    """独立按候选逐项核验，不调用模块搜索函数。"""
    best = None
    for row in range(len(donors)):
        for start in sorted(starts):
            source = donors[row, :, start : start + query.shape[-1]]
            common = query_valid & donor_valid[row, :, start : start + query.shape[-1]]
            common &= (query >= 0) & (query < 4) & (source >= 0) & (source < 4)
            if not common.any():
                continue
            score = 0.0
            for label in range(4):
                query_class, source_class = query == label, source == label
                union = ((query_class | source_class) & common).sum()
                if union:
                    score += ((query_class & source_class) & common).sum() / union
            score /= 4
            if best is None or score > best[2]:
                best = (row, start, score)
    return best


def test_mix_reconstructs_original_donors_for_different_batch_sizes():
    inputs = _inputs()
    originals = {name: tensor.detach().clone() for name, tensor in inputs.items()}
    module = _pattern()
    output = module.mix_batches(**inputs, seed=73)
    expected = {name: originals[name].clone() for name in ("x_l", "y_l", "strong_u")}
    confidence, pseudo = originals["probabilities"].max(1)
    expected["valid_u"] = originals["valid_u"].clone()
    expected["target_u"] = pseudo.masked_fill(~originals["valid_u"], -100)
    expected["confidence_u"] = confidence.masked_fill(~originals["valid_u"], 0)
    expected["accepted_u"] = (confidence >= 0.8) & originals["valid_u"]
    expected["probabilities_u"] = originals["probabilities"].masked_fill(
        ~originals["valid_u"][:, None], 0
    )
    expected["seams_l"] = torch.zeros_like(originals["y_l"], dtype=torch.bool)
    expected["seams_u"] = torch.zeros_like(originals["valid_u"])
    copied_true, removed_true, copied_pseudo = [
        np.zeros(4, dtype=np.int64) for _ in range(3)
    ]

    rng = np.random.default_rng(73)
    width = int(rng.integers(500, 1001))
    l_starts = rng.integers(0, 1000 - width + 1, size=2)
    u_starts = rng.integers(0, 1000 - width + 1, size=3)
    diagnostics = output["diagnostics"]
    candidates = list(range(0, 1000 - width + 1, width // 2))
    assert diagnostics["width"] == width
    assert diagnostics["stride"] == width // 2
    assert diagnostics["candidate_starts"] == candidates
    assert [(p["direction"], p["query_row"]) for p in diagnostics["plans"]] == [
        ("l2u", 0),
        ("l2u", 1),
        ("l2u", 2),
        ("u2l", 0),
        ("u2l", 1),
    ]
    l_known = (originals["y_l"] >= 0) & (originals["y_l"] < 4)
    for plan in diagnostics["plans"]:
        row, start = plan["query_row"], plan["query_start"]
        assert start == int((u_starts if plan["direction"] == "l2u" else l_starts)[row])
        dst = slice(start, start + width)
        queries, donors, query_valid, donor_valid = (
            (pseudo, originals["y_l"], originals["valid_u"], l_known)
            if plan["direction"] == "l2u"
            else (originals["y_l"], pseudo, l_known, originals["valid_u"])
        )
        match = _reference_match(
            queries[row, :, dst].numpy(),
            donors.numpy(),
            candidates,
            query_valid[row, :, dst].numpy(),
            donor_valid.numpy(),
        )
        assert (plan["donor_row"], plan["donor_start"]) == match[:2]
        assert plan["iou"] == pytest.approx(match[2])
        assert plan["score"] == plan["iou"]
        assert plan["query_window"] == [start, start + width]
        donor, source = plan["donor_row"], plan["donor_start"]
        src = slice(source, source + width)
        assert plan["donor_window"] == [source, source + width]
        assert plan["accepted"]
        if plan["direction"] == "l2u":
            assert plan["mean_confidence"] is None
            source_labels = originals["y_l"][donor, :, src]
            source_known = l_known[donor, :, src]
            expected["strong_u"][row, :, dst] = originals["x_l"][donor, :, src]
            expected["target_u"][row, :, dst] = source_labels.masked_fill(
                ~source_known, -100
            )
            expected["valid_u"][row, :, dst] = source_known
            expected["confidence_u"][row, :, dst] = source_known.float()
            expected["accepted_u"][row, :, dst] = source_known
            probabilities = F.one_hot(source_labels.masked_fill(~source_known, 0), 4)
            expected["probabilities_u"][row, :, :, dst] = (
                probabilities.permute(2, 0, 1).float() * source_known[None]
            )
            copied_true += _counts(source_labels)
            seam_name = "seams_u"
        else:
            donor_valid = originals["valid_u"][donor, :, src]
            mean = float(
                confidence[donor, :, src][donor_valid].numpy().mean(dtype=np.float64)
            )
            assert plan["mean_confidence"] == mean
            assert plan["mean_conf"] == mean
            keep = l_known[row, :, dst] & donor_valid
            copied_labels = pseudo[donor, :, src].masked_fill(~keep, -100)
            expected["x_l"][row, :, dst] = originals["weak_u"][donor, :, src]
            expected["y_l"][row, :, dst] = copied_labels
            removed_true += _counts(originals["y_l"][row, :, dst])
            copied_pseudo += _counts(copied_labels)
            seam_name = "seams_l"
        for cut in (start, start + width):
            if 0 < cut < 1000:
                expected[seam_name][row, :, cut] = True

    for name, tensor in expected.items():
        assert torch.equal(output[name], tensor), name
        assert output[name].device == inputs["x_l"].device
    for name, tensor in inputs.items():
        assert torch.equal(tensor.detach(), originals[name]), name
    assert inputs["probabilities"].requires_grad
    assert not output["probabilities_u"].requires_grad
    assert output["probabilities_u"].grad_fn is None
    human = _counts(originals["y_l"]) - removed_true + copied_true
    assert diagnostics["l2u_copied_true_l_by_class"] == copied_true.tolist()
    assert diagnostics["u2l_removed_true_l_by_class"] == removed_true.tolist()
    assert diagnostics["u2l_copied_pseudo_by_class"] == copied_pseudo.tolist()
    assert diagnostics["human_label_points_by_class"] == human.tolist()
    assert diagnostics["human_label_points_total"] == int(human.sum())
    assert diagnostics["u2l_acceptance_fraction"] == 1.0


@pytest.mark.parametrize("confidence, accepted", [(0.8, False), (0.81, True)])
def test_u2l_strict_threshold_uses_original_teacher_and_keeps_destination_ignore(
    confidence, accepted
):
    inputs = _inputs(length=500, labeled=1, unlabeled=1, confidence=confidence)
    inputs["y_l"].fill_(1)
    inputs["y_l"][:, :, 10:20] = -100
    inputs["y_l"][:, 4] = -100
    inputs["valid_u"].fill_(True)
    with torch.no_grad():
        inputs["probabilities"].fill_((1 - confidence) / 3)
        inputs["probabilities"][:, 2] = confidence
    originals = {name: tensor.detach().clone() for name, tensor in inputs.items()}
    output = _pattern().mix_batches(**inputs, seed=19)
    l2u, u2l = output["diagnostics"]["plans"]
    assert l2u["accepted"]
    assert u2l["accepted"] is accepted
    assert u2l["iou"] == 0
    assert u2l["mean_confidence"] == float(np.float32(confidence))
    assert output["diagnostics"]["u2l_acceptance_fraction"] == float(accepted)
    assert not output["seams_l"].any()
    assert not output["seams_u"].any()
    assert torch.equal(output["strong_u"], originals["x_l"])
    known = originals["y_l"] != -100
    assert torch.equal(output["target_u"], originals["y_l"])
    assert torch.equal(output["valid_u"], known)
    assert torch.equal(output["confidence_u"], known.float())
    assert torch.equal(output["accepted_u"], known)
    expected_q = torch.zeros_like(originals["probabilities"])
    expected_q[:, 1] = known.float()
    assert torch.equal(output["probabilities_u"], expected_q)
    expected_y = originals["y_l"].clone()
    if accepted:
        expected_y[known] = 2
    assert torch.equal(output["y_l"], expected_y)
    assert torch.equal(
        output["x_l"], originals["weak_u"] if accepted else originals["x_l"]
    )
    for name, tensor in inputs.items():
        assert torch.equal(tensor.detach(), originals[name]), name


def test_u2l_averages_only_valid_donor_points_and_masks_invalid_sources():
    inputs = _inputs(length=500, labeled=1, unlabeled=1)
    inputs["y_l"].fill_(1)
    inputs["y_l"][:, :, :13] = -100
    inputs["valid_u"][:, :, 100:] = False
    inputs["valid_u"][:, :, :100] = True
    with torch.no_grad():
        inputs["probabilities"][:, :, :, 100:] = 0.25
    original_q = inputs["probabilities"].detach().clone()
    source_pseudo = original_q.argmax(1)
    output = _pattern().mix_batches(**inputs, seed=11)
    u2l = output["diagnostics"]["plans"][1]
    assert u2l["accepted"]
    assert u2l["mean_confidence"] == float(np.float32(0.81))
    expected_y = source_pseudo.masked_fill(~inputs["valid_u"], -100)
    expected_y[:, :, :13] = -100
    assert torch.equal(output["y_l"], expected_y)
    assert torch.equal(output["x_l"], inputs["weak_u"])
    assert torch.equal(inputs["probabilities"].detach(), original_q)


def test_seams_mark_only_actual_interior_cuts_at_same_time_on_all_leads():
    output = _pattern().mix_batches(**_inputs(length=600), seed=7)
    assert 500 <= output["diagnostics"]["width"] <= 600
    for name, direction in (("seams_l", "u2l"), ("seams_u", "l2u")):
        actual = output[name]
        assert actual.dtype == torch.bool
        assert torch.equal(actual, actual[:, :1].expand_as(actual))
        expected = torch.zeros_like(actual)
        for plan in output["diagnostics"]["plans"]:
            if plan["direction"] == direction and plan["accepted"]:
                for cut in (plan["query_start"], plan["query_start"] + plan["width"]):
                    if 0 < cut < 600:
                        expected[plan["query_row"], :, cut] = True
        assert torch.equal(actual, expected)
        assert actual.any()


def test_seed_replay_preserves_numpy_and_torch_global_random_streams():
    inputs = _inputs(length=600)
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    module = _pattern()
    first = module.mix_batches(**inputs, seed=9123)
    second = module.mix_batches(**inputs, seed=9123)
    for name in first:
        if name == "diagnostics":
            assert first[name] == second[name]
        else:
            assert torch.equal(first[name], second[name]), name
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    assert np.array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]
    assert torch.equal(torch_state, torch.get_rng_state())


@pytest.mark.parametrize("empty_support", ["unlabeled", "labeled"])
def test_mix_does_not_match_without_any_common_support(empty_support):
    inputs = _inputs(length=500)
    if empty_support == "unlabeled":
        inputs["valid_u"].fill_(False)
    else:
        inputs["y_l"].fill_(-100)
    output = _pattern().mix_batches(**inputs, seed=1)
    assert torch.equal(output["x_l"], inputs["x_l"])
    assert torch.equal(output["y_l"], inputs["y_l"])
    assert torch.equal(output["strong_u"], inputs["strong_u"])
    for plan in output["diagnostics"]["plans"]:
        assert not plan["accepted"]
        assert plan["donor_row"] is None
        assert plan["donor_start"] is None
        assert plan["donor_window"] is None
        assert plan["score"] is None
        assert plan["mean_conf"] is None
    assert not output["seams_l"].any()
    assert not output["seams_u"].any()
    assert output["diagnostics"]["u2l_acceptance_fraction"] == 0
    assert output["diagnostics"]["human_label_points_total"] == int(
        _counts(inputs["y_l"]).sum()
    )


def test_legacy_threshold_argument_controls_u2l_acceptance():
    inputs = _inputs(length=500, labeled=1, unlabeled=1, confidence=0.81)
    output = _pattern().mix_batches(**inputs, seed=3, threshold=0.85)
    assert not output["diagnostics"]["plans"][1]["accepted"]
    assert torch.equal(output["x_l"], inputs["x_l"])


def test_mix_rejects_waveform_shorter_than_500():
    with pytest.raises(ValueError, match="至少500"):
        _pattern().mix_batches(**_inputs(length=499), seed=1)
