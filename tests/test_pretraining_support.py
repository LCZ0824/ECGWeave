# 公开版适配说明：包内相对导入；不改计算。
"""CPU 行为测试：固定张量中的真实支持与冻结配方兼容。"""

import pytest
import torch
from ecgweave.pretraining import model as m

torch.set_num_threads(2)


def model():
    torch.manual_seed(42)
    c = m.ModelConfig(
        embed_dim=12,
        depth=1,
        num_heads=3,
        decoder_embed_dim=12,
        decoder_depth=1,
        decoder_num_heads=3,
        mlp_ratio=1,
    )
    return m.ValidECGMAE(c).eval()


def setup(lengths):
    x = torch.randn(len(lengths), 12, 5000) * 0.2
    valid = torch.arange(250)[None] < torch.tensor(lengths)[:, None]
    mask = m.make_valid_mask(
        valid, 0.75, "limbsync_mixed", torch.Generator().manual_seed(9)
    )
    return x, valid, mask


def test_all_valid_frozen_equivalence():
    net = model()
    old = m.core.ECGMAE(net.config).eval()
    old.load_state_dict(net.state_dict(), strict=True)
    x, valid, mask = setup([250, 250])
    frozen_mask = m.core.make_mask(
        2, 12, 250, 0.75, "limbsync_mixed", torch.Generator().manual_seed(9), "cpu"
    )
    assert torch.equal(mask, frozen_mask)
    pred = net(x, mask, valid)
    expected = old(x, mask)
    torch.testing.assert_close(pred, expected, atol=1e-6, rtol=1e-5)
    cfg = m.LossConfig("local_softnorm")
    torch.testing.assert_close(
        net.loss(pred, x, mask, cfg, valid), old.loss(expected, x, mask, cfg)
    )


def test_padding_values_and_gradients():
    net = model()
    x, valid, mask = setup([7, 13])
    points = valid.repeat_interleave(20, -1)[:, None].expand_as(x)
    x.requires_grad_()
    altered = torch.where(points, x.detach(), torch.full_like(x, 987.0))
    pred = net(x, mask, valid)
    pred2 = net(altered, mask, valid)
    valid3 = valid[:, None].expand(-1, 12, -1)
    torch.testing.assert_close(pred[valid3], pred2[valid3])
    cfg = m.LossConfig("local_softnorm")
    loss = net.loss(pred, x, mask, cfg, valid)
    torch.testing.assert_close(loss, net.loss(pred2, altered, mask, cfg, valid))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(x.grad[~points]) == 0


def test_mixed_batch_matches_each_record():
    net = model()
    x, valid, mask = setup([3, 17])
    pred = net(x, mask, valid)
    losses = []
    for row in range(2):
        single = net(x[row : row + 1], mask[row : row + 1], valid[row : row + 1])
        v = valid[row : row + 1, None].expand(-1, 12, -1)
        torch.testing.assert_close(
            pred[row : row + 1][v], single[v], atol=2e-6, rtol=1e-5
        )
        losses.append(
            net.loss(
                single,
                x[row : row + 1],
                mask[row : row + 1],
                m.LossConfig("local_softnorm"),
                valid[row : row + 1],
            )
        )
    torch.testing.assert_close(
        net.loss(pred, x, mask, m.LossConfig("local_softnorm"), valid),
        torch.stack(losses).mean(),
    )


def test_tiny_valid_and_impossible_masks():
    net = model()
    x, valid, mask = setup([3])
    assert torch.isfinite(
        net.loss(net(x, mask, valid), x, mask, m.LossConfig("local_softnorm"), valid)
    )
    one = torch.arange(250)[None] < 1
    with pytest.raises(ValueError, match="可见|目标"):
        m.make_valid_mask(one, 0.75, "limbsync_mixed", torch.Generator())
    with pytest.raises(ValueError, match="目标"):
        net.loss(
            torch.zeros(1, 12, 250, 20),
            x,
            torch.zeros_like(mask),
            m.LossConfig("local_softnorm"),
            valid,
        )


def test_local_target_matches_independent_window_statistics():
    torch.manual_seed(5)
    patches = torch.randn(2, 12, 9, 20)
    valid = torch.tensor(
        [[1, 1, 0, 1, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    valid3 = valid[:, None].expand(-1, 12, -1)
    cfg = m.LossConfig("local_softnorm")
    actual = m.valid_target(patches, valid3, cfg)
    for b in range(2):
        for k in valid[b].nonzero().flatten().tolist():
            lo, hi = max(0, k - 2), min(9, k + 3)
            support = patches[b, :, lo:hi][:, valid[b, lo:hi]].flatten(1)
            expected = (patches[b, :, k] - support.mean(-1, keepdim=True)) / torch.sqrt(
                support.var(-1, unbiased=False, keepdim=True) + cfg.soft_floor_mv**2
            )
            torch.testing.assert_close(actual[b, :, k], expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", m.core.MASK_MODES)
def test_mask_modes_preserve_validity_and_restore_rng(mode):
    valid = torch.arange(250)[None] < torch.tensor([3, 29])[:, None]
    rng = torch.Generator().manual_seed(10)
    state = rng.get_state()
    mask = m.make_valid_mask(valid, 0.75, mode, rng)
    rng.set_state(state)
    assert torch.equal(mask, m.make_valid_mask(valid, 0.75, mode, rng))
    expanded = valid[:, None].expand(-1, 12, -1)
    assert mask[~expanded].all()
    assert ((~mask).sum(-1) == torch.tensor([1, 7])[:, None]).all()
    if mode.startswith("limbsync"):
        assert torch.equal(mask[:, :6], mask[:, :1].expand(-1, 6, -1))
