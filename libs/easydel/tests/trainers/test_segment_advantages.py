"""Accelerator-only numerical checks of segment discount factors and agentic advantages."""

import numpy as np
from easydel.trainers._online_rl.math import cross_segment_discount_factors
from easydel.trainers.agentic_moshpit.utils import (
    compute_advantages_agentic,
    compute_advantages_gigpo,
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


def test_agentic_advantages_normalize_per_step_not_pooled_across_timesteps():
    # Rollout 0: step rewards [1.0, 4.0] -> G_0 = 3.0, G_1 = 4.0
    # Rollout 1: step rewards [0.0, 4.0] -> G_0 = 2.0, G_1 = 4.0
    # Step 0 differs across the group (3.0 vs 2.0 -> +/- 0.5 / (0.5 + 1e-4)),
    # while step 1 is tied (4.0 vs 4.0 -> std == 0 -> 0.0 advantage).
    rewards = [np.array([1.0, 4.0]), np.array([0.0, 4.0])]
    mask = np.array([[1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0]])
    actual = compute_advantages_agentic(rewards, mask, group_size=2, gamma=0.5)
    step0_pos = 0.5 / (0.5 + 1e-4)
    np.testing.assert_allclose(actual[0], [step0_pos, step0_pos, 0.0, 0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(actual[1], [-step0_pos, -step0_pos, 0.0, 0.0, 0.0, 0.0], atol=1e-5)
    assert actual[0, 0] != actual[0, 3]
    assert compute_advantages_step(rewards, 2, gamma=0.5).shape == (2,)


def test_agentic_advantages_identical_five_step_successes_receive_zero_advantage():
    # Regression for RL-H1: pooling across timesteps previously gave turn 0 -1.375!
    rewards = [
        np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    ]
    mask = np.array(
        [
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        ],
        dtype=np.int32,
    )
    actual = compute_advantages_agentic(rewards, mask, group_size=2, gamma=0.95)
    np.testing.assert_allclose(actual, np.zeros_like(actual, dtype=np.float32), atol=1e-7)


def test_agentic_advantages_explicit_turn_ids_and_empty_masks():
    # 0-token turn 0 + adjacent turns without 0-gap: turn_ids keeps attribution exact.
    rewards = [np.array([0.0, 4.0]), np.array([2.0, 0.0])]
    # Row 0: turn 0 had 0 tokens; tokens [0, 1] belong to turn 1.
    # Row 1: turns 0 and 1 are adjacent without an observation gap: [0, 0, 1, 1].
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
    turn_ids = np.array([[1, 1, -1, -1], [0, 0, 1, 1]], dtype=np.int32)
    actual = compute_advantages_agentic(rewards, mask, group_size=2, gamma=0.5, turn_ids=turn_ids)
    # Row 0 returns: [2.0, 4.0]; Row 1 returns: [2.0, 0.0].
    # Step 0 returns across group: [2.0, 2.0] -> std 0 -> 0.0.
    # Step 1 returns across group: [4.0, 0.0] -> mean 2.0, std 2.0 -> +2/(2+1e-4) vs -2/(2+1e-4).
    step1_mag = 2.0 / (2.0 + 1e-4)
    np.testing.assert_allclose(actual[0], [step1_mag, step1_mag, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(actual[1], [0.0, 0.0, -step1_mag, -step1_mag], atol=1e-5)

    # Empty / 0-length masks must not crash on mask[:-1].
    empty_seq = compute_advantages_agentic(rewards, np.zeros((2, 0), dtype=np.int32), group_size=2, gamma=0.5)
    assert empty_seq.shape == (2, 0)
    empty_batch = compute_advantages_agentic([], np.zeros((0, 4), dtype=np.int32), group_size=2, gamma=0.5)
    assert empty_batch.shape == (0, 4)


def test_step_and_gigpo_advantages_use_per_turn_returns_and_honor_scale_rewards():
    # Equal G_0 (1.0 vs 1.0) but different G_1 (0.0 vs 2.0) with gamma=0.5:
    # Old compute_advantages_step broadcast G_0 only and returned all zeros!
    step_rewards = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 2.0], dtype=np.float32)]
    mask = np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=np.int32)
    turn_ids = np.array([[0, 0, -1, 1, 1], [0, 0, -1, 1, 1]], dtype=np.int32)

    adv_2d = compute_advantages_step(step_rewards, group_size=2, gamma=0.5, response_masks=mask, turn_ids=turn_ids)
    step1_mag = 1.0 / (1.0 + 1e-4)
    np.testing.assert_allclose(adv_2d[0], [0.0, 0.0, 0.0, -step1_mag, -step1_mag], atol=1e-5)
    np.testing.assert_allclose(adv_2d[1], [0.0, 0.0, 0.0, step1_mag, step1_mag], atol=1e-5)

    adv_1d = compute_advantages_step(step_rewards, group_size=2, gamma=0.5)
    assert adv_1d[1] > adv_1d[0]

    # GiGPO honors scale_rewards="none" on step_adv as well as episode_adv.
    ep_rewards = jnp.asarray([10.0, 10.0], dtype=jnp.float32)  # zero episode adv so only step_adv contributes
    scaled_step_rewards = [np.array([0.0, 0.0], dtype=np.float32), np.array([0.0, 10.0], dtype=np.float32)]
    gigpo_none, _ = compute_advantages_gigpo(
        ep_rewards,
        scaled_step_rewards,
        group_size=2,
        episode_weight=1.0,
        step_weight=1.0,
        gamma=0.5,
        scale_rewards="none",
        response_masks=mask,
        turn_ids=turn_ids,
    )
    # Unscaled turn 1 centered returns: -5.0 vs +5.0 (NOT divided by std=5.0!).
    np.testing.assert_allclose(np.asarray(gigpo_none[0]), [-2.5, -2.5, 0.0, -5.0, -5.0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(gigpo_none[1]), [2.5, 2.5, 0.0, 5.0, 5.0], atol=1e-5)


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
