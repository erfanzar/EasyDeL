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

"""Reward normalization and advantage computation utilities for agentic MoshPit.

This module provides functions for computing advantages from multi-turn
agent trajectories, supporting multiple reward modes:

- Episode-level (GRPO): Standard group-relative advantages on terminal rewards.
- Step-level (step_reinforce): Discounted per-step returns normalized within groups.
- GiGPO: Combined episode + step rewards with separate normalization.
- Agentic reinforce: Segment-aware discounted returns respecting response boundaries.

All functions operate on NumPy/JAX arrays and are designed to be called
from the trainer's ``_preprocess_batch_input`` method.
"""

from __future__ import annotations

import jax
import numpy as np
from jax import numpy as jnp


def compute_discounted_returns(
    step_rewards: np.ndarray,
    response_mask: np.ndarray,
    gamma: float = 0.95,
) -> np.ndarray:
    """Compute discounted returns for each step in a trajectory.

    Works backwards through the trajectory, accumulating discounted
    future rewards. Only counts rewards at response boundaries
    (where response_mask transitions from 1 to 0).

    Args:
        step_rewards: Per-step rewards, shape ``[num_steps]``.
        response_mask: Binary mask indicating response token positions,
            shape ``[seq_len]``.
        gamma: Discount factor for future rewards.

    Returns:
        Discounted returns per step, shape ``[num_steps]``.
    """
    num_steps = len(step_rewards)
    returns = np.zeros(num_steps, dtype=np.float32)
    running_return = 0.0
    for t in reversed(range(num_steps)):
        running_return = step_rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


def compute_segment_discounted_returns(
    step_rewards: np.ndarray,
    segment_ids: np.ndarray,
    gamma: float = 0.95,
) -> np.ndarray:
    """Compute discounted returns respecting segment (response turn) boundaries.

    Each segment represents one agent response turn. Returns are
    computed within segments and discounted across segments.

    This implements the "agentic_reinforce" advantage estimator from ROLL,
    where each response turn's return considers the discounted value of
    all future turns.

    Args:
        step_rewards: Per-segment rewards, shape ``[num_segments]``.
        segment_ids: Segment index for each token, shape ``[seq_len]``.
            Tokens in segment ``i`` have ``segment_ids[token] == i``.
        gamma: Discount factor across segments.

    Returns:
        Per-segment discounted returns, shape ``[num_segments]``.
    """
    num_segments = len(step_rewards)
    returns = np.zeros(num_segments, dtype=np.float32)
    running_return = 0.0
    for t in reversed(range(num_segments)):
        running_return = step_rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


def normalize_rewards_group(
    rewards: jax.Array,
    group_size: int,
) -> jax.Array:
    """Normalize rewards within groups (GRPO-style).

    Subtracts group mean and divides by group std for each group
    of ``group_size`` trajectories.

    Args:
        rewards: Flat reward array, shape ``[batch_size]``.
        group_size: Number of trajectories per group.

    Returns:
        Normalized rewards, shape ``[batch_size]``.
    """
    grouped = rewards.reshape(-1, group_size)
    mean = jnp.nanmean(grouped, axis=-1, keepdims=True)
    std = jnp.nanstd(grouped, axis=-1, keepdims=True)
    is_zero = jnp.isclose(std, 0.0)
    normalized = (grouped - mean) / (std + 1e-4)
    normalized = jnp.where(is_zero, 0.0, normalized)
    return jnp.nan_to_num(normalized.reshape(-1))


def normalize_rewards_batch(rewards: jax.Array) -> jax.Array:
    """Normalize rewards across the entire batch.

    Args:
        rewards: Reward array, shape ``[batch_size]``.

    Returns:
        Normalized rewards, shape ``[batch_size]``.
    """
    mean = jnp.nanmean(rewards)
    std = jnp.nanstd(rewards)
    is_zero = jnp.isclose(std, 0.0)
    normalized = (rewards - mean) / (std + 1e-4)
    return jnp.where(is_zero, 0.0, jnp.nan_to_num(normalized))


def compute_advantages_episode(
    rewards: jax.Array,
    group_size: int,
    scale_rewards: str = "group",
) -> tuple[jax.Array, jax.Array]:
    """Compute episode-level advantages (standard GRPO).

    Each trajectory's advantage is its reward minus the group mean,
    optionally divided by group or batch standard deviation.

    Args:
        rewards: Per-trajectory rewards, shape ``[batch_size]``.
        group_size: Number of trajectories per prompt group.
        scale_rewards: Scaling mode: "group", "batch", or "none".

    Returns:
        Tuple of (advantages, std_rewards) arrays.
    """
    grouped = rewards.reshape(-1, group_size)
    mean_grouped = jnp.nanmean(grouped, axis=-1)
    advantages = rewards - mean_grouped.repeat(group_size, axis=0)

    if scale_rewards == "group":
        std_rewards = jnp.nanstd(grouped, axis=-1)
        std_rewards = std_rewards.repeat(group_size, axis=0)
    elif scale_rewards == "batch":
        std_rewards = jnp.nanstd(rewards)
        std_rewards = jnp.broadcast_to(std_rewards, advantages.shape)
    else:
        std_rewards = jnp.ones_like(advantages)

    if scale_rewards != "none":
        advantages = jnp.where(
            jnp.isclose(std_rewards, 0.0),
            0.0,
            advantages / (std_rewards + 1e-4),
        )
    advantages = jnp.nan_to_num(advantages)

    return advantages, std_rewards


def compute_advantages_step(
    step_rewards_list: list[np.ndarray],
    group_size: int,
    gamma: float = 0.95,
) -> np.ndarray:
    """Compute step-level discounted advantages.

    For each trajectory, computes discounted returns from per-step
    rewards, then normalizes across the group.

    Args:
        step_rewards_list: List of per-trajectory step reward arrays.
        group_size: Number of trajectories per group.
        gamma: Discount factor.

    Returns:
        Per-trajectory discounted returns, shape ``[num_trajectories]``.
    """
    returns = np.array(
        [compute_discounted_returns(sr, np.ones(len(sr)), gamma)[0] for sr in step_rewards_list],
        dtype=np.float32,
    )
    grouped = returns.reshape(-1, group_size)
    mean = np.nanmean(grouped, axis=-1, keepdims=True)
    std = np.nanstd(grouped, axis=-1, keepdims=True)
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    normalized = (grouped - mean) / (std + 1e-4)
    return normalized.reshape(-1)


def compute_advantages_gigpo(
    episode_rewards: jax.Array,
    step_rewards_list: list[np.ndarray],
    group_size: int,
    episode_weight: float = 1.0,
    step_weight: float = 1.0,
    gamma: float = 0.95,
    scale_rewards: str = "group",
) -> tuple[jax.Array, jax.Array]:
    """Compute GiGPO-style combined episode + step advantages.

    Combines normalized episode-level rewards with normalized
    step-level discounted returns using configurable weights.

    Args:
        episode_rewards: Per-trajectory episode rewards.
        step_rewards_list: List of per-trajectory step reward arrays.
        group_size: Number of trajectories per group.
        episode_weight: Weight for episode-level advantages.
        step_weight: Weight for step-level advantages.
        gamma: Discount factor for step rewards.
        scale_rewards: Scaling mode for episode advantages.

    Returns:
        Tuple of (combined advantages, std_rewards).
    """
    episode_adv, std_rewards = compute_advantages_episode(episode_rewards, group_size, scale_rewards)
    step_adv = jnp.array(
        compute_advantages_step(step_rewards_list, group_size, gamma),
        dtype=episode_adv.dtype,
    )
    combined = episode_weight * episode_adv + step_weight * step_adv
    return combined, std_rewards
