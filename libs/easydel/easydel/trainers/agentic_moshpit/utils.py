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
        step_rewards: Per-step rewards, shape ``[num_steps]``, or token-aligned
            rewards with the same length as ``response_mask``. In token mode,
            only the reward at each response boundary contributes.
        response_mask: Binary mask indicating response token positions,
            shape ``[seq_len]``; for per-step rewards it must contain one
            contiguous response segment per step.
        gamma: Discount factor for future rewards.

    Returns:
        Discounted returns per step, shape ``[num_steps]``.
    """
    rewards = np.asarray(step_rewards, dtype=np.float32)
    mask = np.asarray(response_mask, dtype=bool)
    if rewards.ndim != 1 or mask.ndim != 1:
        raise ValueError("Rewards and response_mask must be one-dimensional.")
    boundaries = mask & ~np.r_[mask[1:], False] if mask.size else mask
    returns = np.zeros(len(rewards), dtype=np.float32)
    running_return = 0.0
    if len(rewards) == len(mask):
        # Token-aligned rewards: discount once per response, not once per
        # observation or generated token, and score only response boundaries.
        for t in reversed(range(len(rewards))):
            if boundaries[t]:
                running_return = rewards[t] + gamma * running_return
            if mask[t]:
                returns[t] = running_return
    else:
        if np.count_nonzero(boundaries) != len(rewards):
            raise ValueError("Per-step rewards must match the number of response segments.")
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
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

    if scale_rewards == "group":
        advantages = normalize_rewards_group(rewards, group_size)
    elif scale_rewards != "none":
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
        [
            compute_discounted_returns(sr, np.tile([1, 0], len(sr)), gamma)[0] if len(sr) else 0.0
            for sr in step_rewards_list
        ],
        dtype=np.float32,
    )
    grouped = returns.reshape(-1, group_size)
    mean = np.nanmean(grouped, axis=-1, keepdims=True)
    std = np.nanstd(grouped, axis=-1, keepdims=True)
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    normalized = (grouped - mean) / (std + 1e-4)
    return normalized.reshape(-1)


def compute_advantages_agentic(
    step_rewards_list: list[np.ndarray],
    response_masks: np.ndarray,
    group_size: int,
    gamma: float = 0.95,
) -> np.ndarray:
    """Return group-normalized credit per response segment, not per episode.

    Args:
        step_rewards_list: One reward sequence per trajectory.
        response_masks: Batched token masks; each contiguous response run is a
            segment. Observations and padding always receive zero advantage.
        group_size: Number of trajectories sharing a prompt.
        gamma: Discount applied only between response segments.

    Returns:
        Float32 advantages with the same shape as ``response_masks``.
    """
    masks = np.asarray(response_masks, dtype=bool)
    if group_size <= 0 or masks.ndim != 2 or len(masks) != len(step_rewards_list) or len(masks) % group_size:
        raise ValueError("Response masks must form complete trajectory groups.")
    values = np.zeros(masks.shape, dtype=np.float32)
    group_returns: list[list[float]] = [[] for _ in range(len(masks) // group_size)]
    for row, (rewards, mask) in enumerate(zip(step_rewards_list, masks, strict=True)):
        returns = compute_discounted_returns(rewards, np.tile([1, 0], len(rewards)), gamma)
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
        if len(starts) > len(returns):
            raise ValueError("Every response segment needs a step reward.")
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            values[row, start:end] = returns[index]
            group_returns[row // group_size].append(float(returns[index]))
    for group, returns in enumerate(group_returns):
        if not returns:
            continue
        rows = slice(group * group_size, (group + 1) * group_size)
        std = float(np.nanstd(returns))
        values[rows] = np.where(masks[rows], (values[rows] - np.nanmean(returns)) / (std + 1e-4), 0.0)
    return np.nan_to_num(values)


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
