"""Accelerator-only numerical checks of segment discount factors and agentic advantages."""

import numpy as np
from easydel.trainers._online_rl.math import cross_segment_discount_factors
from easydel.trainers.agentic_moshpit.utils import (
    compute_advantages_agentic,
    compute_advantages_step,
    compute_discounted_returns,
    normalize_rewards_batch,
    normalize_rewards_group,
)
from jax import numpy as jnp


def test_returns_discount_only_response_boundaries():
    mask = np.array([1, 1, 0, 0, 1, 1, 0])
    rewards = np.array([99, 2, 99, 99, 99, 4, 99], dtype=np.float32)
    actual = compute_discounted_returns(rewards, mask, gamma=0.5)
    np.testing.assert_allclose(actual, [4, 4, 0, 0, 4, 4, 0])
    per_step = compute_discounted_returns(np.array([2, 4]), mask, gamma=0.5)
    np.testing.assert_allclose(per_step, [4, 4])


def test_agentic_advantages_vary_by_segment_but_not_within_response():
    rewards = [np.array([0.0, 4.0]), np.array([0.0, 0.0])]
    mask = np.array([[1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0]])
    actual = compute_advantages_agentic(rewards, mask, group_size=2, gamma=0.5)
    expected = (np.array([2.0, 4.0, 0.0, 0.0]) - 1.5) / (np.std([2.0, 4.0, 0.0, 0.0]) + 1e-4)
    np.testing.assert_allclose(actual[0], [expected[0], expected[0], 0, expected[1], expected[1], 0])
    np.testing.assert_allclose(actual[1], [expected[2], expected[2], 0, expected[3], expected[3], 0])
    assert actual[0, 0] != actual[0, 3]
    assert compute_advantages_step(rewards, 2, gamma=0.5).shape == (2,)


def test_reward_normalizers_preserve_public_group_and_batch_behavior():
    rewards = jnp.asarray([1.0, 3.0, 10.0, 10.0])
    np.testing.assert_allclose(normalize_rewards_group(rewards, 2), [-1 / 1.0001, 1 / 1.0001, 0, 0], atol=1e-6)
    values = np.asarray([1.0, 3.0, 10.0, 10.0])
    np.testing.assert_allclose(
        normalize_rewards_batch(rewards), (values - values.mean()) / (values.std() + 1e-4), atol=1e-6
    )


def test_shared_cross_segment_factor_has_independent_expected_values():
    factors = cross_segment_discount_factors(jnp.asarray([0, 2, 3]), gamma=0.5, lam=0.8)
    # TPU float32 power uses an approximate transcendental lowering (the
    # measured relative error for 0.4**2 is 9.54e-7), not a matmul whose
    # precision can be pinned. Retain the independent decimal reference.
    np.testing.assert_allclose(factors, [1, 0.16, 0.064], rtol=2e-6, atol=1e-8)
