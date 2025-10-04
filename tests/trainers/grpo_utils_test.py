import jax
from jax import numpy as jnp

from easydel.trainers.training_utils import compute_group_advantages, compute_length_reward


def test_compute_group_advantages_zscore_has_zero_mean():
    rewards = jnp.array([1.0, 3.0, 2.0, 4.0], dtype=jnp.float32)
    advantages, means, stds, collapsed = compute_group_advantages(
        rewards,
        num_generations=2,
        epsilon=1e-6,
        enforce_mixed=False,
    )
    grouped_adv = advantages.reshape(-1, 2)
    mean_per_group = jnp.mean(grouped_adv, axis=1)
    assert jnp.allclose(mean_per_group, 0.0, atol=1e-5)
    assert jnp.all(stds > 0)
    assert not jnp.any(collapsed)


def test_compute_group_advantages_jitter_for_collapsed_group():
    rewards = jnp.ones((4,), dtype=jnp.float32)
    advantages, _, _, collapsed = compute_group_advantages(
        rewards,
        num_generations=2,
        epsilon=1e-6,
        enforce_mixed=True,
        jitter=0.5,
    )
    assert bool(collapsed[0])
    first_group = advantages.reshape(-1, 2)[0]
    assert first_group[0] > first_group[1]


def test_compute_length_reward_linear_matches_dapo_piecewise():
    lengths = jnp.array([10, 18, 22], dtype=jnp.float32)
    shaped = compute_length_reward(
        lengths,
        max_completion_length=20,
        cache_tokens=5,
        mode="linear",
        scale=1.0,
    )
    expected = jnp.array([0.0, (15.0 - 18.0) / 5.0, -1.0], dtype=jnp.float32)
    assert jnp.allclose(shaped, expected, atol=1e-6)


def test_compute_length_reward_punitive_penalises_overflow():
    lengths = jnp.array([14, 17, 25], dtype=jnp.float32)
    shaped = compute_length_reward(
        lengths,
        max_completion_length=20,
        cache_tokens=5,
        mode="punitive",
        scale=1.0,
    )
    expected = jnp.array([0.0, -2.0 / 5.0, -(25.0 - 15.0) / 5.0], dtype=jnp.float32)
    assert jnp.allclose(shaped, expected, atol=1e-6)
