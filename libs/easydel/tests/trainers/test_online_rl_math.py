# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import numpy as np
from easydel.trainers._online_rl.math import (
    cross_segment_advantage_correction,
    following_segment_token_counts,
    global_token_denominator,
    global_token_mean,
    length_adaptive_lambda,
    skip_observation_gae,
    target_aligned_action_mask,
    target_aligned_token_ids,
)
from jax import numpy as jnp


def _numpy_skip_observation_gae(
    rewards,
    values,
    action_mask,
    transition_mask,
    *,
    gamma,
    lam,
    bootstrap_values,
):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    action_mask = np.asarray(action_mask, dtype=bool)
    transition_mask = np.asarray(transition_mask, dtype=bool)
    output = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    for row in range(rewards.shape[0]):
        next_value = float(bootstrap_values[row])
        next_advantage = 0.0
        for position in range(rewards.shape[1] - 1, -1, -1):
            if not action_mask[row, position]:
                continue
            continues = float(transition_mask[row, position])
            delta = float(rewards[row, position]) + gamma * next_value * continues - float(values[row, position])
            advantage = delta + gamma * float(lam[row]) * next_advantage * continues
            output[row, position] = advantage
            returns[row, position] = advantage + float(values[row, position])
            next_value = float(values[row, position])
            next_advantage = advantage
    return output, returns


def _numpy_following_segment_counts(segment_ids, action_mask):
    segment_ids = np.asarray(segment_ids)
    action_mask = np.asarray(action_mask, dtype=bool)
    result = np.zeros_like(segment_ids, dtype=np.int32)
    for row in range(segment_ids.shape[0]):
        action_positions = np.flatnonzero(action_mask[row])
        for position in action_positions:
            current_segment = segment_ids[row, position]
            result[row, position] = sum(
                bool(action_mask[row, later]) and segment_ids[row, later] != current_segment
                for later in range(position + 1, segment_ids.shape[1])
            )
    return result


def test_target_alignment_uses_next_token_roles_and_valid_source_context():
    input_ids = jnp.asarray([[10, 11, 12, 13], [0, 20, 21, 22]])
    action_mask = jnp.asarray([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=jnp.int32)
    attention_mask = jnp.asarray([[1, 1, 1, 1], [0, 1, 1, 1]], dtype=jnp.int32)

    np.testing.assert_array_equal(np.asarray(target_aligned_token_ids(input_ids)), [[11, 12, 13], [20, 21, 22]])
    np.testing.assert_array_equal(
        np.asarray(target_aligned_action_mask(action_mask, attention_mask)),
        [[1, 0, 1], [0, 1, 0]],
    )


def test_length_adaptive_lambda_matches_published_formula_and_handles_empty_rows():
    lengths = jnp.asarray([0, 1, 2, 10], dtype=jnp.int32)
    actual = np.asarray(length_adaptive_lambda(lengths, alpha=1.5))
    expected = np.asarray([0.0, 1.0 - 1.0 / 1.5, 1.0 - 1.0 / 3.0, 1.0 - 1.0 / 15.0])

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_skip_observation_gae_matches_independent_action_chain_reference():
    rewards = np.asarray(
        [
            [0.2, 50.0, 0.0, -0.1, 25.0, 1.0, 0.0],
            [0.0, 0.4, 10.0, 0.0, -0.2, 5.0, 0.8],
        ],
        dtype=np.float32,
    )
    values = np.asarray(
        [
            [0.4, 99.0, 88.0, 0.3, 77.0, 0.1, 66.0],
            [55.0, 0.5, 44.0, 33.0, 0.2, 22.0, 0.6],
        ],
        dtype=np.float32,
    )
    action_mask = np.asarray(
        [
            [1, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
        ],
        dtype=bool,
    )
    transition_mask = np.asarray(
        [
            [1, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    gamma = 0.97
    lam = np.asarray([0.8, 0.65], dtype=np.float32)
    bootstrap = np.asarray([9.0, 0.25], dtype=np.float32)

    expected_advantages, expected_returns = _numpy_skip_observation_gae(
        rewards,
        values,
        action_mask,
        transition_mask,
        gamma=gamma,
        lam=lam,
        bootstrap_values=bootstrap,
    )
    actual_advantages, actual_returns = jax.jit(
        lambda r, v, a, t, b: skip_observation_gae(
            r,
            v,
            a,
            gamma=gamma,
            lam=jnp.asarray(lam),
            transition_mask=t,
            bootstrap_values=b,
        )
    )(rewards, values, action_mask, transition_mask, bootstrap)

    np.testing.assert_allclose(np.asarray(actual_advantages), expected_advantages, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(actual_returns), expected_returns, rtol=1e-5, atol=1e-5)
    assert np.all(np.asarray(actual_advantages)[~action_mask] == 0)
    assert np.all(np.asarray(actual_returns)[~action_mask] == 0)


def test_skip_observation_gae_infers_next_action_across_observation_tokens():
    rewards = jnp.asarray([[0.0, 100.0, 0.0, 1.0]], dtype=jnp.float32)
    values = jnp.asarray([[0.5, 20.0, 30.0, 0.25]], dtype=jnp.float32)
    action_mask = jnp.asarray([[1, 0, 0, 1]], dtype=jnp.int32)

    advantages, _ = skip_observation_gae(rewards, values, action_mask, gamma=1.0, lam=1.0)

    # Last action: 1 - .25 = .75. First action bridges directly to it:
    # (.25 - .5) + .75 = .5. Observation rewards/values are ignored.
    np.testing.assert_allclose(np.asarray(advantages), [[0.5, 0.0, 0.0, 0.75]], atol=1e-6)


def test_cross_segment_correction_counts_only_later_optimized_tokens():
    advantages = np.asarray(
        [
            [2.0, 90.0, 3.0, 80.0, 4.0, 5.0, 70.0],
            [1.0, 60.0, 2.0, 3.0, 50.0, 4.0, 40.0],
        ],
        dtype=np.float32,
    )
    action_mask = np.asarray(
        [
            [1, 0, 1, 0, 1, 1, 0],
            [1, 0, 1, 1, 0, 1, 0],
        ],
        dtype=bool,
    )
    segment_ids = np.asarray(
        [
            [0, 0, 0, 1, 1, 2, 2],
            [4, 4, 5, 5, 5, 8, 8],
        ],
        dtype=np.int32,
    )
    gamma = 0.9
    lam = np.asarray([0.8, 0.7], dtype=np.float32)

    expected_counts = _numpy_following_segment_counts(segment_ids, action_mask)
    expected = np.zeros_like(advantages)
    for row in range(advantages.shape[0]):
        expected[row] = np.where(
            action_mask[row],
            advantages[row] * (gamma * lam[row]) ** expected_counts[row],
            0.0,
        )

    actual_counts = following_segment_token_counts(segment_ids, action_mask)
    actual = jax.jit(lambda a, s, m: cross_segment_advantage_correction(a, s, m, gamma=gamma, lam=jnp.asarray(lam)))(
        advantages, segment_ids, action_mask
    )

    np.testing.assert_array_equal(np.asarray(actual_counts), expected_counts)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


def test_global_token_reduction_uses_one_batch_denominator():
    values = jnp.asarray([[2.0, 10.0, 6.0], [8.0, 20.0, 30.0]])
    mask = jnp.asarray([[1, 0, 1], [1, 0, 0]])

    assert float(global_token_denominator(mask)) == 3.0
    assert np.isclose(float(global_token_mean(values, mask)), (2.0 + 6.0 + 8.0) / 3.0)
    assert float(global_token_denominator(jnp.zeros_like(mask))) == 1.0
    assert float(global_token_mean(values, jnp.zeros_like(mask))) == 0.0
