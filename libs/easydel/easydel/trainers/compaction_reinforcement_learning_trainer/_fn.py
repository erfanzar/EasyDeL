# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numerical primitives for Compaction Reinforcement Learning."""

from __future__ import annotations

import jax
from jax import numpy as jnp


def cross_segment_advantage_factors(
    trajectory_ids: jax.Array,
    segment_ids: jax.Array,
    action_mask: jax.Array,
    *,
    gamma: float,
    gae_lambdas: jax.Array | float,
) -> jax.Array:
    """Compute ``(gamma * lambda) ** N_after`` for each segment row.

    ``N_after`` counts optimized tokens in strictly later segments belonging
    to the same parent trajectory. Prompt, observation, and padding tokens do
    not contribute because their ``action_mask`` entries are zero.
    """
    trajectory_ids = jnp.asarray(trajectory_ids).reshape(-1)
    segment_ids = jnp.asarray(segment_ids).reshape(-1)
    if trajectory_ids.shape != segment_ids.shape:
        raise ValueError("`trajectory_ids` and `segment_ids` must have the same shape.")
    if action_mask.ndim != 2 or action_mask.shape[0] != trajectory_ids.shape[0]:
        raise ValueError("`action_mask` must have shape [number_of_segment_rows, target_length].")
    lambdas = jnp.asarray(gae_lambdas, dtype=jnp.float32)
    if lambdas.ndim == 0:
        lambdas = jnp.broadcast_to(lambdas, trajectory_ids.shape)
    if lambdas.shape != trajectory_ids.shape:
        raise ValueError("`gae_lambdas` must be scalar or have one value per segment row.")

    token_counts = jnp.sum(action_mask.astype(jnp.float32), axis=1)
    same_trajectory = trajectory_ids[:, None] == trajectory_ids[None, :]
    is_later_segment = segment_ids[None, :] > segment_ids[:, None]
    tokens_after = jnp.sum(
        same_trajectory.astype(jnp.float32) * is_later_segment.astype(jnp.float32) * token_counts[None, :],
        axis=1,
    )
    return jnp.power(float(gamma) * lambdas, tokens_after)


def apply_cross_segment_gae_correction(
    local_advantages: jax.Array,
    trajectory_ids: jax.Array,
    segment_ids: jax.Array,
    action_mask: jax.Array,
    *,
    gamma: float,
    gae_lambdas: jax.Array | float,
) -> jax.Array:
    """Apply CompactionRL's published correction to local segment GAE."""
    factors = cross_segment_advantage_factors(
        trajectory_ids,
        segment_ids,
        action_mask,
        gamma=gamma,
        gae_lambdas=gae_lambdas,
    )
    return local_advantages * factors[:, None]


def compaction_ppo_policy_loss(
    current_logps: jax.Array,
    behavior_logps: jax.Array,
    advantages: jax.Array,
    action_mask: jax.Array,
    *,
    cliprange: float,
    normalizer: jax.Array | float | None = None,
) -> jax.Array:
    """Compute the token-global clipped PPO actor loss used by CompactionRL."""
    action_mask = action_mask.astype(jnp.float32)
    advantages = jax.lax.stop_gradient(advantages)
    ratio = jnp.exp(current_logps - behavior_logps)
    unclipped = -advantages * ratio
    clipped = -advantages * jnp.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
    if normalizer is None:
        normalizer = jnp.maximum(jnp.sum(action_mask), 1.0)
    else:
        normalizer = jnp.maximum(jnp.asarray(normalizer, dtype=jnp.float32), 1.0)
    return jnp.sum(jnp.maximum(unclipped, clipped) * action_mask) / normalizer


__all__ = (
    "apply_cross_segment_gae_correction",
    "compaction_ppo_policy_loss",
    "cross_segment_advantage_factors",
)
