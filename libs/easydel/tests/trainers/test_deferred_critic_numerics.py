"""Accelerator-only regression for zero-baseline PPO reward transport. Not CPU-safe."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.trainers._online_rl.trainer import OnlineActorCriticTrainer
from jax import numpy as jnp


@pytest.mark.parametrize("whiten_advantages", [False, True])
def test_deferred_critic_transport_preserves_rewards_and_current_critic_targets(whiten_advantages):
    trainer = object.__new__(OnlineActorCriticTrainer)
    trainer.arguments = SimpleNamespace(
        gamma=0.93,
        lam=0.81,
        policy_gae_mode="fixed",
        critic_gae_mode="match_policy",
        whiten_advantages=whiten_advantages,
    )
    trainer.critic_state = SimpleNamespace(graphstate=None, graphother=None)
    trainer.compute_critic_values = lambda *args: jnp.asarray([[9, 2, 3, 4, 8], [9, 5, 6, 7, 8]], dtype=jnp.float32)
    rewards = jnp.asarray([[0.5, -0.2, 1.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)
    mask = jnp.asarray([[1, 1, 1], [1, 0, 0]], dtype=jnp.float32)
    zeros = jnp.zeros_like(rewards)
    advantages, returns = trainer._compute_gae(rewards, zeros, mask)
    batch = trainer._adapt_ppo_rollout_batch(
        {
            "input_ids": jnp.asarray([[1, 2, 3, 4, 5, 6], [1, 2, 3, 0, 0, 0]]),
            "attention_mask": jnp.asarray([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
            "completion_mask": mask,
            "old_logps": -jnp.ones_like(rewards),
            "old_values": zeros,
            "advantages": advantages,
            "returns": returns,
        }
    )
    np.testing.assert_allclose(batch["rewards"], [[0, 0, 0.5, -0.2, 1], [0, 0, 2, 0, 0]], atol=1e-6)
    np.testing.assert_array_equal(batch["action_mask"], [[0, 0, 1, 1, 1], [0, 0, 1, 0, 0]])
    np.testing.assert_array_equal(batch["done_mask"], [[0, 0, 0, 0, 1], [0, 0, 1, 0, 0]])
    np.testing.assert_array_equal(batch["old_values"], [[9, 2, 3, 4, 8], [9, 5, 6, 7, 8]])
    # Independent recurrence for critic returns on the first row's actions.
    delta_last = 1.0 - 8.0
    delta_middle = -0.2 + 0.93 * 8.0 - 4.0
    delta_first = 0.5 + 0.93 * 4.0 - 3.0
    last_advantage = delta_last
    middle_advantage = delta_middle + 0.93 * 0.81 * last_advantage
    first_advantage = delta_first + 0.93 * 0.81 * middle_advantage
    np.testing.assert_allclose(
        batch["critic_returns"][0, 2:],
        [3 + first_advantage, 4 + middle_advantage, 8 + last_advantage],
        atol=1e-5,
    )
