import numpy as np

from tablehop.rl.grpo import group_advantages


def test_group_normalisation():
    r = np.array([1.0, 0.0, 0.0, 1.0, 0.2, 0.4, 0.6, 0.8])
    a = group_advantages(r, 4)
    for g in a.reshape(2, 4):
        assert abs(g.mean()) < 1e-9 and abs(g.std() - 1) < 1e-3


def test_identical_rewards_zero_advantage():
    a = group_advantages(np.array([0.5] * 8 + [1.0, 0.0] * 4), 8)
    assert np.allclose(a[:8], 0.0)
    assert np.abs(a[8:]).max() > 0.9
