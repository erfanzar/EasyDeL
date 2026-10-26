"""Accelerator-only numerics for the public compaction PPO objective."""

import jax
import numpy as np
import pytest
from easydel.trainers.compaction_reinforcement_learning_trainer._fn import compaction_ppo_policy_loss
from jax import numpy as jnp


@pytest.mark.parametrize("normalizer,denominator", [(None, 4.0), (10.0, 10.0)])
def test_compaction_compatibility_wrapper_loss_and_gradients(normalizer, denominator):
    # Clip the positive and negative first-row terms. Keep the second row's
    # interior terms, while excluding extreme masked ratios from all reductions.
    behavior = jnp.full((2, 3), -4.0, dtype=jnp.float32)
    current = behavior + jnp.log(jnp.asarray([[1.5, 0.5, 20.0], [1.1, 0.9, 20.0]], dtype=jnp.float32))
    advantages = jnp.asarray([[2.0, -3.0, 100.0], [4.0, -5.0, 100.0]], dtype=jnp.float32)
    mask = jnp.asarray([[1, 1, 0], [1, 1, 0]], dtype=jnp.int32)

    def loss(logps, old_logps, adv):
        return compaction_ppo_policy_loss(logps, old_logps, adv, mask, cliprange=0.2, normalizer=normalizer)

    actual, gradients = jax.value_and_grad(loss, argnums=(0, 1, 2))(current, behavior, advantages)
    expected = (-2.4 + 2.4 - 4.4 + 4.5) / denominator
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    expected_gradient = np.asarray([[0.0, 0.0, 0.0], [-4.4, 4.5, 0.0]]) / denominator
    np.testing.assert_allclose(gradients[0], expected_gradient, rtol=2e-5, atol=2e-6)
    # The public surrogate differentiates the log-ratio in either argument;
    # the trainer supplies the rollout probabilities as fixed data.
    np.testing.assert_allclose(gradients[1], -expected_gradient, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(gradients[2], np.zeros((2, 3)))
