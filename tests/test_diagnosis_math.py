# 公开版适配说明：替换导入；主管集成测试另由公开接口测试覆盖。
"""新结构的数学、信息流与可独立加载合同；不访问患者数据。"""

import math

import pytest
import torch
from torch.nn import functional as F

from ecgweave.diagnosis import base_model as base
from ecgweave.diagnosis.model import (
    FineFFN,
    JointMultiresClassifier,
    ModelConfig,
    TemporalFineFFN,
)
from ecgweave.diagnosis.natural_reader import (
    ContentMemoryReader,
    JointInteractionReader,
    joint_coverage_read,
)
from ecgweave.diagnosis.spec import SUPPORTED_ARMS as GPUS


torch.set_num_threads(2)


def small_config():
    return ModelConfig(
        n_leads=2,
        seq_len=40,
        patch_size=20,
        embed_dim=12,
        depth=3,
        num_heads=3,
        fine_dim=16,
        fine_heads=4,
        exchange_dim=16,
        exchange_heads=4,
        readout_dim=16,
        readout_heads=4,
        n_classes=5,
        exchange_layers=(1, 2, 3),
        relation_start_layer=2,
    )


def batch_for(config):
    return {
        "signals": torch.randn(2, config.n_leads, config.seq_len),
        "reference": torch.rand(2, config.n_leads, config.n_patches, 4),
        "reference_support": torch.ones(
            2, config.n_leads, config.n_patches, dtype=torch.bool
        ),
    }


def test_joint_read_matches_explicit_attention_and_gradients():
    torch.manual_seed(4)
    q = torch.randn(2, 2, 3, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 7, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 7, 6, dtype=torch.float64, requires_grad=True)
    raw = torch.randn(2, 7, 4, dtype=torch.float64, requires_grad=True)
    scale = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    types = torch.tensor([0, 0, 1, 1, 1, 1, 1])
    durations = (0.04, 0.008)
    values = torch.cat((v, raw[:, None].expand(-1, 2, -1, -1)), -1)
    bias = scale.log_softmax(-1) + torch.tensor(durations, dtype=torch.float64).log()
    score = q @ k.transpose(-2, -1) / math.sqrt(8)
    weights = (score + bias[:, types][None, None]).softmax(-1)
    expected = weights @ values
    feature, physical = joint_coverage_read(q, k, v, raw, types, scale, durations)
    actual = torch.cat((feature, physical), -1)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), (q, k, v, raw, scale))
    expected_grads = torch.autograd.grad(
        (expected * upstream).sum(), (q, k, v, raw, scale)
    )
    for observed, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(observed, reference, atol=1e-8, rtol=1e-8)


def test_joint_reader_inherits_amp_and_uses_one_read_per_round(monkeypatch):
    config = small_config()
    reader = JointInteractionReader(config)
    h = torch.randn(2, 2, 2, 12)
    u = torch.randn(2, 2, 10, 16)
    raw = torch.randn(2, 2, 10, 4)
    untouched = raw.clone()
    calls = []
    implementation = base.coverage_attention

    def capture(q, k, v, *args):
        calls.append((q.dtype, k.dtype, v.dtype, v.shape[-1]))
        return implementation(q, k, v, *args)

    monkeypatch.setattr(base, "coverage_attention", capture)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = reader(h, u, raw)
    assert actual.shape == (2, 5, 32)
    assert calls == [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, 4),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, 8),
    ]
    assert torch.equal(raw, untouched)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("arm", GPUS)
def test_shared_backbone_and_three_identity_ffns(arm):
    config = small_config()
    torch.manual_seed(9)
    anchor = JointMultiresClassifier(config, "wide_joint")
    torch.manual_seed(9)
    model = JointMultiresClassifier(config, arm)
    anchor_state = anchor.state_dict()
    for name, tensor in model.state_dict().items():
        if (
            name.startswith(("encoder.", "fine.", "relations.", "exchanges."))
            and "post_ffn" not in name
        ):
            assert torch.equal(tensor, anchor_state[name]), name
    u = torch.randn(2, 2, 10, config.fine_dim)
    for index, exchange in enumerate(model.exchanges):
        ffn = exchange.lead_attention.post_ffn
        assert torch.equal(ffn(u), u)
        if arm == "temporal_joint":
            assert isinstance(ffn, TemporalFineFFN)
            assert ffn.depthwise.dilation == (2**index,)
        else:
            assert isinstance(ffn, FineFFN)
    assert sum(model.parameter_count_by_role().values()) == sum(
        p.numel() for p in model.parameters()
    )


@pytest.mark.parametrize("factory", [lambda: FineFFN(8), lambda: TemporalFineFFN(8, 2)])
def test_ffn_is_identity_then_both_projections_learn(factory):
    torch.manual_seed(2)
    ffn = factory()
    x = torch.randn(2, 2, 10, 8)
    target = torch.randn_like(x)
    initial = {name: p.detach().clone() for name, p in ffn.named_parameters()}
    optimizer = torch.optim.AdamW(ffn.parameters(), lr=0.01, weight_decay=0)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(ffn(x), target).backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in ffn.parameters()
        )
        optimizer.step()
    for key in ("input.weight", "output.weight"):
        assert not torch.equal(dict(ffn.named_parameters())[key], initial[key])
    if isinstance(ffn, TemporalFineFFN):
        assert not torch.equal(ffn.depthwise.weight, initial["depthwise.weight"])


def test_temporal_ffn_mixes_time_without_crossing_lead_boundaries():
    ffn = TemporalFineFFN(8, 2)
    with torch.no_grad():
        ffn.input.weight.zero_()
        ffn.input.bias.zero_()
        ffn.input.weight[0, 0] = 1
        ffn.depthwise.weight.zero_()
        ffn.depthwise.bias.zero_()
        ffn.depthwise.weight[0, 0, 5] = 1
        ffn.output.weight.zero_()
        ffn.output.weight[0, 0] = 1
    x = torch.randn(1, 2, 20, 8, requires_grad=True)
    derivative = torch.autograd.grad(ffn(x)[0, 0, 8, 0], x)[0]
    assert derivative[0, 0, 10].abs().sum() > 0
    assert derivative[0, 1].count_nonzero() == 0
    assert derivative[0, 0, 9].count_nonzero() == 0


@pytest.mark.parametrize("self_attention", [False, True])
def test_content_memory_has_one_raw_entry_and_no_classifier_bypass(self_attention):
    config = small_config()
    reader = ContentMemoryReader(config, label_self_attention=self_attention)
    assert reader.feature_dim == config.readout_dim
    assert not hasattr(reader, "raw_key") and not hasattr(reader, "raw_value")
    h = torch.randn(2, 2, 2, 12)
    u = torch.randn(2, 2, 10, 16)
    raw = torch.randn(2, 2, 10, 4)
    before = raw.clone()
    actual = reader(h, u, raw)
    assert actual.shape == (2, 5, 16)
    assert (reader(h, u, raw + 1) - actual).abs().max() > 1e-6
    assert torch.equal(raw, before)


def test_label_self_attention_is_conditioned_on_ecg_and_never_on_targets():
    config = small_config()
    model = JointMultiresClassifier(config, "label_decoder").eval()
    batch = batch_for(config)
    seen = []
    handle = model.reader.self_attention[0].register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    with torch.no_grad():
        first = model(batch)
        with_labels = model({**batch, "labels": torch.full((2, 5), float("nan"))})
        model({**batch, "signals": batch["signals"] + 0.7})
    handle.remove()
    assert torch.equal(first, with_labels)
    assert not torch.equal(seen[0], seen[2])
