"""Attention-moment readout and gradient tests."""

import math

import torch
from torch.nn import functional as F

from ecgweave.diagnosis import base_model as base
from ecgweave.diagnosis.natural_reader import JointInteractionReader, split_heads


def implementations():
    from ecgweave.diagnosis import stats_reader

    return stats_reader, None


def tiny_config():
    return base.ModelConfig(
        n_leads=2,
        seq_len=40,
        patch_size=20,
        embed_dim=16,
        depth=3,
        num_heads=2,
        fine_dim=8,
        fine_heads=2,
        exchange_dim=8,
        exchange_heads=2,
        readout_dim=16,
        readout_heads=2,
        n_classes=3,
        exchange_layers=(1, 2, 3),
        relation_start_layer=1,
    )


def inputs(config, requires_grad=False):
    shapes = (
        (2, config.n_leads, config.n_patches, config.embed_dim),
        (2, config.n_leads, config.n_fine, config.fine_dim),
        (2, config.n_leads, config.n_fine, 4),
    )
    return tuple(torch.randn(shape, requires_grad=requires_grad) for shape in shapes)


def dense_weights(query, key, types, scale_logits, durations):
    logits = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    bias = scale_logits.log_softmax(-1) + query.new_tensor(durations).log()
    return (logits + bias[:, types][None, None]).softmax(-1)


def dense_stats_reference(reader, h, u, raw):
    c = reader.config
    key, value = reader.memory(h, u, raw)
    query = reader.query[None].expand(h.shape[0], -1, -1)
    durations = (c.patch_size / c.sample_rate, 4 / c.sample_rate)
    for index, layer in enumerate(reader.rounds):
        q = split_heads(layer.query(query), c.readout_heads)
        attention = dense_weights(
            q, key, reader.memory_types, reader.scale_logits, durations
        )
        context = attention @ value
        if index == 1:
            variance = (attention @ value.square() - context.square()).clamp_min(0)
        query = query + layer.output(
            context.transpose(1, 2).reshape(h.shape[0], c.n_classes, c.readout_dim)
        )
        query = query + layer.ffn(query)
    statistics = variance.transpose(1, 2).reshape(
        h.shape[0], c.n_classes, c.readout_dim
    )
    query = query + reader.variance_projection(reader.variance_norm(statistics.log1p()))
    return reader.norm(query)


def test_stats_initialization_preserves_original_parameters_and_rng():
    module, _ = implementations()
    config = tiny_config()
    torch.manual_seed(31)
    original = JointInteractionReader(config)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(31)
    reader = module.StatsInteractionReader(config)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert reader.feature_dim == config.readout_dim
    assert torch.count_nonzero(reader.variance_projection.weight) == 0
    assert reader.variance_projection.bias is None
    for name, value in original.state_dict().items():
        assert torch.equal(reader.state_dict()[name], value), name


def test_zero_stats_projection_recovers_original_content_readout():
    module, _ = implementations()
    torch.manual_seed(85)
    config = tiny_config()
    original = JointInteractionReader(config)
    reader = module.StatsInteractionReader(config)
    loaded = reader.load_state_dict(original.state_dict(), strict=False)
    assert set(loaded.missing_keys) == {
        "variance_norm.weight",
        "variance_norm.bias",
        "variance_projection.weight",
    }
    assert not loaded.unexpected_keys
    values = inputs(config)
    torch.testing.assert_close(
        reader(*values),
        original(*values)[..., : config.readout_dim],
        rtol=2e-5,
        atol=2e-6,
    )


def test_stats_output_matches_dense_formula_with_active_variance_path():
    module, _ = implementations()
    torch.manual_seed(392)
    config = tiny_config()
    reader = module.StatsInteractionReader(config)
    with torch.no_grad():
        reader.variance_projection.weight.normal_(std=0.03)
        reader.scale_logits.copy_(torch.tensor([[0.7, -0.2], [-0.8, 0.3], [0.1, 0.6]]))
    values = inputs(config)
    torch.testing.assert_close(
        reader(*values), dense_stats_reference(reader, *values), rtol=3e-5, atol=3e-6
    )


def test_coverage_moments_squares_half_values_in_fp32_and_clamps_variance():
    module, _ = implementations()
    q = torch.zeros(1, 1, 1, 4, dtype=torch.float16)
    key = torch.zeros(1, 1, 2, 4, dtype=torch.float16)
    value = torch.tensor([[[[0.0, 1000.0], [1000.0, 1000.0]]]], dtype=torch.float16)
    mean, variance = module.coverage_moments(
        q, key, value, torch.tensor([0, 1]), torch.zeros(1, 2), (1.0, 1.0)
    )
    assert mean.dtype == variance.dtype == torch.float32
    assert torch.isfinite(variance).all() and (variance >= 0).all()
    torch.testing.assert_close(mean, torch.tensor([[[[500.0, 1000.0]]]]))
    torch.testing.assert_close(variance, torch.tensor([[[[250000.0, 0.0]]]]))


def test_stats_first_backward_learns_projection_and_shared_memory():
    module, _ = implementations()
    torch.manual_seed(129)
    config = tiny_config()
    reader = module.StatsInteractionReader(config)
    values = inputs(config, requires_grad=True)
    features = reader(*values)
    assert features.shape == (2, config.n_classes, config.readout_dim)
    classifier = torch.randn(config.n_classes, config.readout_dim) * 0.02
    logits = (features * classifier[None]).sum(-1)
    F.binary_cross_entropy_with_logits(
        logits, torch.randint(0, 2, logits.shape).float()
    ).backward()
    for parameter in reader.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for parameter in (
        reader.variance_projection.weight,
        reader.raw_value.weight,
        reader.coarse_value.weight,
    ):
        assert parameter.grad.abs().sum() > 0
    for value in values:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
