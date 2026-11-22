"""Accelerator-only regressions for public compaction numerical primitives."""

import jax
import numpy as np
import pytest
from easydel.trainers.compaction_reinforcement_learning_trainer._fn import (
    apply_cross_segment_gae_correction,
    compaction_ppo_policy_loss,
    cross_segment_advantage_factors,
)
from jax import numpy as jnp


def test_compaction_factors_keep_trajectory_rows_and_segment_order():
    # Rows are deliberately interleaved and not sorted by segment ID. The two
    # rows in trajectory 10's final segment must both count for its first row.
    trajectories = jnp.asarray([10, 20, 10, 10, 20])
    segments = jnp.asarray([0, 3, 2, 2, 1])
    mask = jnp.asarray([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]])
    lambdas = jnp.asarray([0.5, 0.8, 0.5, 0.4, 0.6])
    expected = np.asarray([0.25**4, 1.0, 1.0, 1.0, 0.3**2])
    actual = cross_segment_advantage_factors(trajectories, segments, mask, gamma=0.5, gae_lambdas=lambdas)
    # TPU's float32 pow uses an approximate transcendental lowering (~1.6e-6
    # relative error for 0.3**2); there is no dot precision setting to pin here.
    # Keep the independent reference and a tight relative roundoff allowance.
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.0)
    corrected = apply_cross_segment_gae_correction(
        jnp.ones((5, 3)), trajectories, segments, mask, gamma=0.5, gae_lambdas=lambdas
    )
    np.testing.assert_allclose(corrected, np.broadcast_to(expected[:, None], (5, 3)), rtol=2e-6, atol=0.0)


@pytest.mark.parametrize("normalizer, denominator", [(None, 4.0), (8.0, 8.0), (0.0, 1.0)])
def test_public_compaction_ppo_loss_clipping_masking_and_gradients(normalizer, denominator):
    behavior = jnp.full((2, 3), -5.0)
    current = behavior + jnp.log(jnp.asarray([[2.0, 0.5, 100.0], [1.1, 0.9, 100.0]]))
    advantages = jnp.asarray([[2.0, -2.0, 1.0], [3.0, -3.0, 1.0]])
    mask = jnp.asarray([[1, 1, 0], [1, 1, 0]])

    def loss(logps, adv):
        return compaction_ppo_policy_loss(logps, behavior, adv, mask, cliprange=0.2, normalizer=normalizer)

    actual, (logps_gradient, advantage_gradient) = jax.value_and_grad(loss, argnums=(0, 1))(current, advantages)
    # Independent hand calculation: the first row clips both signs; the
    # second row retains both unclipped terms. Masked tokens contribute zero.
    expected = (-2.4 + 1.6 - 3.3 + 2.7) / denominator
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        logps_gradient,
        np.asarray([[0.0, 0.0, 0.0], [-3.3, 2.7, 0.0]]) / denominator,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(advantage_gradient, np.zeros((2, 3)))


def test_public_compaction_ppo_empty_action_mask_has_finite_zero_loss():
    logps = jnp.asarray([[-2.0, -3.0]])
    actual = compaction_ppo_policy_loss(logps, logps, jnp.ones_like(logps), jnp.zeros_like(logps), cliprange=0.2)
    np.testing.assert_array_equal(actual, 0.0)
