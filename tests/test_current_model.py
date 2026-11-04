"""当前论文模型的结构、目标与前端；仅使用合成数据。"""

from dataclasses import asdict
import math

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch import nn
from torch.nn import functional as F

from ecgweave.diagnosis.model import ModelConfig
from ecgweave.diagnosis.stats_model import build_classifier
from ecgweave.diagnosis.prior_objective import loss_components
from ecgweave.pipeline import filter500, load_classifier, waveform_batch


def tiny_config():
    return ModelConfig(
        n_leads=2,
        seq_len=40,
        embed_dim=12,
        depth=3,
        num_heads=3,
        fine_dim=16,
        exchange_dim=16,
        readout_dim=16,
        n_classes=5,
        exchange_layers=(1, 2, 3),
        relation_start_layer=2,
    )


def test_current_parameter_identity():
    model = build_classifier()
    assert type(model.reader).__name__ == "StatsInteractionReader"
    assert sum(p.numel() for p in model.parameters()) == 6639486
    assert model.classifier_weight.shape == (71, 128)
    assert isinstance(model.fine.lead_attention, nn.Identity)
    assert all(
        type(e.lead_attention).__name__ == "PostFFNOnly" for e in model.exchanges
    )


def test_current_checkpoint_roundtrip_hidden_and_backward(tmp_path):
    config = tiny_config()
    model = build_classifier(config).eval()
    checkpoint = tmp_path / "synthetic.pt"
    torch.save(
        {
            "model_config": asdict(config),
            "config": {
                "arm": "wide_joint",
                "candidate": "stats_prior20",
                "structure": "performance_composite",
                "reader_kind": "stats",
                "lead_attention_off": True,
            },
            "model": model.state_dict(),
        },
        checkpoint,
    )
    restored = load_classifier(checkpoint)
    batch = {
        "signals": torch.randn(2, 2, 40),
        "reference": torch.full((2, 2, 2, 4), 0.25, requires_grad=True),
        "reference_support": torch.ones(2, 2, 2, dtype=torch.bool),
    }
    hidden = []
    hook = restored.reader.register_forward_hook(lambda m, a, out: hidden.append(out))
    logits = restored(batch)
    hook.remove()
    torch.testing.assert_close(logits, model(batch), rtol=0, atol=0)
    assert hidden[0].shape == (2, 5, 16)
    rebuilt = (hidden[0].float() * restored.classifier_weight[None]).sum(-1)
    rebuilt = rebuilt + restored.classifier_bias
    torch.testing.assert_close(logits, rebuilt, rtol=0, atol=0)
    logits.square().mean().backward()
    assert batch["reference"].grad is None
    assert restored.reader.variance_projection.weight.grad is not None
    assert torch.isfinite(restored.reader.variance_projection.weight.grad).all()


def test_prior_loss_uses_training_prevalence_and_microbatch_sum():
    logits = torch.randn(32, 71, requires_grad=True)
    labels = torch.randint(0, 2, (32, 71)).float()
    groups = [
        {"task": 3, "start": 16, "per_side": 4, "prevalence": 0.12},
        {"task": 9, "start": 24, "per_side": 4, "prevalence": 0.37},
    ]
    objective = {"kind": "prior"}
    bce, aux = loss_components(logits, labels, groups, objective, 32)
    weights = torch.ones(32)
    expected_aux = logits.sum() * 0
    for item in groups:
        start, rho, task = item["start"], item["prevalence"], item["task"]
        weights[start : start + 4] = 2 * rho
        weights[start + 4 : start + 8] = 2 * (1 - rho)
        bias = math.log(rho / (1 - rho))
        expected_aux = expected_aux + 0.25 * (
            F.softplus(-(logits[start : start + 4, task] - bias)).mean()
            + F.softplus(logits[start + 4 : start + 8, task] - bias).mean()
        )
    expected_bce = (
        F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1)
        * weights
    ).mean()
    torch.testing.assert_close(bce, expected_bce)
    torch.testing.assert_close(aux, expected_aux)
    pieces = [
        loss_components(
            logits[i : i + 16], labels[i : i + 16], groups, objective, 32, i
        )
        for i in (0, 16)
    ]
    torch.testing.assert_close(bce, sum(p[0] for p in pieces))
    torch.testing.assert_close(aux, sum(p[1] for p in pieces))
    full_grad = torch.autograd.grad(
        bce + 0.029492154911704947 * aux, logits, retain_graph=True
    )[0]
    piece_grad = torch.autograd.grad(
        sum(p[0] + 0.029492154911704947 * p[1] for p in pieces), logits
    )[0]
    torch.testing.assert_close(full_grad, piece_grad)


def test_native_filter_matches_frozen_recipe():
    signals = np.random.default_rng(0).normal(size=(1, 12, 5000)).astype(np.float32)
    high = butter(5, 0.67, btype="highpass", fs=500, output="sos")
    low = butter(5, 40, btype="lowpass", fs=500, output="sos")
    expected = sosfiltfilt(low, sosfiltfilt(high, signals)).astype(np.float32)
    np.testing.assert_array_equal(filter500(signals), expected)


def test_only_delineator_receives_filtered_signal():
    class Frontend(nn.Module):
        def forward(self, values):
            self.seen = values.clone()
            return torch.zeros(len(values), 4, 12, 5000)

    signals = torch.randn(1, 12, 5000)
    original = signals.clone()
    frontend = Frontend().eval()
    batch = waveform_batch(signals, frontend)
    torch.testing.assert_close(
        frontend.seen, torch.from_numpy(filter500(signals.numpy()))
    )
    torch.testing.assert_close(signals, original, rtol=0, atol=0)
    assert batch["signals"] is signals
    assert batch["reference"].shape == (1, 12, 250, 4)
    assert batch["reference_support"].all()


def test_cli_rejects_bf16_on_cpu_before_loading_weights():
    import pytest
    from ecgweave.predict import main

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--input",
                "synthetic.npz",
                "--checkpoint",
                "synthetic.pt",
                "--output",
                "output.npz",
                "--precision",
                "bf16",
            ]
        )


def test_waveform_reference_matches_frozen_half_precision_storage():
    class Frontend(nn.Module):
        def forward(self, values):
            state = torch.arange(4).reshape(1, 4, 1, 1).float()
            return (values[:, None] * (state + 1) + state).contiguous()

    signals = torch.randn(1, 12, 5000)
    frontend = Frontend().eval()
    logits = frontend(torch.from_numpy(filter500(signals.numpy())))
    expected = logits.float().softmax(1).reshape(1, 4, 12, 250, 20).mean(-1)
    expected = expected.permute(0, 2, 3, 1).half().float()
    expected = expected / expected.sum(-1, keepdim=True).clamp_min(1e-12)
    actual = waveform_batch(signals, frontend)
    torch.testing.assert_close(actual["reference"], expected, rtol=0, atol=0)
