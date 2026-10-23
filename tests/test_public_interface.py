"""公开封装的跨阶段合同；只用合成数据。"""

from dataclasses import asdict
import importlib.util

import pytest
import torch


torch.set_num_threads(2)


def api():
    assert importlib.util.find_spec("ecgweave"), "缺少可安装的ECGWeave公开包"
    from ecgweave import pipeline

    return pipeline


def test_reference_pooling_retains_only_complete_nonconstant_support():
    pipeline = api()
    probabilities = torch.zeros(1, 4, 2, 40)
    probabilities[:, 1, :, :20] = 1
    probabilities[:, 2, :, 20:] = 1
    signals = torch.arange(40).float().expand(1, 2, 40).clone()
    signals[:, 1] = 2
    support = torch.ones_like(signals, dtype=torch.bool)
    support[:, 0, 39] = False
    reference, valid = pipeline.pool_reference(probabilities, signals, support)
    assert valid.tolist() == [[[True, False], [False, False]]]
    torch.testing.assert_close(reference[0, 0, 0], torch.tensor([0.0, 1, 0, 0]))
    assert torch.count_nonzero(reference[~valid]) == 0


def test_checkpoint_loader_preserves_logits_and_feature_reconstruction(tmp_path):
    pipeline = api()
    from ecgweave.diagnosis.model import JointMultiresClassifier, ModelConfig

    config = ModelConfig(
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
    model = JointMultiresClassifier(config).eval()
    path = tmp_path / "synthetic.pt"
    torch.save(
        {
            "model_config": asdict(config),
            "config": {"arm": "wide_joint"},
            "model": model.state_dict(),
        },
        path,
    )
    restored = pipeline.load_classifier(path)
    batch = {
        "signals": torch.randn(2, 2, 40),
        "reference": torch.rand(2, 2, 2, 4),
        "reference_support": torch.ones(2, 2, 2, dtype=torch.bool),
    }
    features = []
    handle = restored.reader.register_forward_hook(
        lambda module, args, value: features.append(value)
    )
    with torch.inference_mode():
        observed = restored(batch)
        torch.testing.assert_close(observed, model(batch), rtol=0, atol=0)
        reconstructed = (features[0] * restored.classifier_weight[None]).sum(
            -1
        ) + restored.classifier_bias
        torch.testing.assert_close(observed, reconstructed, rtol=0, atol=0)
    handle.remove()
    restored.train()
    batch["reference"].requires_grad_()
    restored(batch).square().mean().backward()
    assert batch["reference"].grad is None
    assert restored.classifier_weight.grad is not None


def test_pretraining_masked_padding_has_no_loss_or_input_effect():
    api()
    from ecgweave.pretraining.core import LossConfig, ModelConfig
    from ecgweave.pretraining.model import ValidECGMAE, make_valid_mask

    config = ModelConfig(
        seq_len=80,
        embed_dim=12,
        depth=2,
        num_heads=3,
        decoder_embed_dim=8,
        decoder_depth=1,
        decoder_num_heads=2,
    )
    model = ValidECGMAE(config)
    signal = torch.randn(2, 12, 80)
    valid = torch.tensor([[True, True, True, False]] * 2)
    mask = make_valid_mask(valid, 0.75, "limbsync_mixed", n_leads=12)
    prediction = model(signal, mask, valid)
    loss = model.loss(prediction, signal, mask, LossConfig("local_softnorm"), valid)
    changed = signal.clone()
    changed[..., 60:] = 10000
    torch.testing.assert_close(model(changed, mask, valid), prediction, rtol=0, atol=0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.patch_embed.weight.grad is not None


def test_pseudo_loss_uses_all_valid_points_and_masks_rejected_gradients():
    api()
    from ecgweave.delineation.loss import pseudo_ce

    logits = torch.zeros(1, 4, 1, 4, requires_grad=True)
    targets = torch.zeros(1, 1, 4, dtype=torch.long)
    confidence = torch.tensor([[[1.0, 0.5, 0.5, 0.5]]])
    valid = torch.ones(1, 1, 4, dtype=torch.bool)
    loss, stats = pseudo_ce(
        logits, targets, confidence, valid, threshold=0.95, denominator="valid"
    )
    torch.testing.assert_close(loss, torch.tensor(4.0).log() / 4)
    assert stats["valid_points"] == 4 and stats["accepted_points"] == 1
    loss.backward()
    assert torch.count_nonzero(logits.grad[..., 1:]) == 0


def test_delineator_strict_roundtrip_and_backward():
    pipeline = api()
    from ecgweave.delineation.model import build_segmenter
    from ecgweave.pretraining.core import ModelConfig

    config = ModelConfig(
        n_leads=2,
        seq_len=80,
        embed_dim=12,
        depth=3,
        num_heads=3,
        decoder_embed_dim=8,
        decoder_depth=1,
        decoder_num_heads=2,
    )
    model = build_segmenter(config)
    signal = torch.randn(2, 2, 80)
    logits = model(signal)
    assert logits.shape == (2, 4, 2, 80)
    logits.square().mean().backward()
    assert model.encoder.patch_embed.weight.grad is not None
    assert model.local_output.weight.grad is not None
    restored = build_segmenter(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(signal), model(signal), rtol=0, atol=0)
        reference, valid = pipeline.pool_reference(logits.softmax(1), signal)
    assert reference.shape == (2, 2, 4, 4)
    assert valid.all()


@pytest.mark.parametrize("bad", ["length", "dtype", "finite"])
def test_public_input_rejects_mismatched_signal_contract(bad):
    pipeline = api()
    signal = torch.randn(1, 12, 5000)
    if bad == "length":
        signal = signal[..., :-1]
    elif bad == "dtype":
        signal = signal.double()
    else:
        signal[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        pipeline.validate_signals(signal)
