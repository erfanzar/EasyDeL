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


def _compute_per_step_normalized_returns(
    step_rewards_list: list[np.ndarray],
    group_size: int,
    gamma: float = 0.95,
    scale_rewards: str = "group",
    eps: float = 1e-4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute discounted returns ``G_t`` per step and normalize each timestep ``t`` separately.

    Returns ``(raw_returns_per_traj, normalized_returns_per_traj)``, where each
    element is a 1D float32 array of length ``len(step_rewards_list[row])``.
    """
    if group_size <= 0 or len(step_rewards_list) % group_size != 0:
        raise ValueError("step_rewards_list must form complete trajectory groups.")
    if scale_rewards not in ("group", "batch", "none"):
        raise ValueError(f"Invalid scale_rewards: {scale_rewards!r}. Expected 'group', 'batch', or 'none'.")

    num_groups = len(step_rewards_list) // group_size
    raw_returns: list[np.ndarray] = []
    group_step_values: list[dict[int, list[float]]] = [{} for _ in range(num_groups)]
    batch_step_values: dict[int, list[float]] = {}

    for row, step_rewards in enumerate(step_rewards_list):
        sr = np.asarray(step_rewards, dtype=np.float32)
        if sr.size == 0:
            returns = np.zeros(0, dtype=np.float32)
        else:
            returns = compute_discounted_returns(sr, np.tile([1, 0], len(sr)), gamma)
        raw_returns.append(returns)
        group_idx = row // group_size
        for t, val in enumerate(returns):
            fval = float(val)
            group_step_values[group_idx].setdefault(t, []).append(fval)
            batch_step_values.setdefault(t, []).append(fval)

    batch_step_std: dict[int, float] = {}
    if scale_rewards == "batch":
        for t, vals in batch_step_values.items():
            arr = np.asarray(vals, dtype=np.float64)
            batch_step_std[t] = float(np.nanstd(arr)) if arr.size and not np.all(np.isnan(arr)) else 0.0

    group_step_stats: list[dict[int, tuple[float, float]]] = [{} for _ in range(num_groups)]
    for group_idx, step_map in enumerate(group_step_values):
        for t, vals in step_map.items():
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0 or np.all(np.isnan(arr)):
                group_step_stats[group_idx][t] = (0.0, 0.0)
            else:
                mean = float(np.nanmean(arr))
                std = float(np.nanstd(arr)) if scale_rewards == "group" else batch_step_std.get(t, 0.0)
                group_step_stats[group_idx][t] = (mean, std)

    normalized_returns: list[np.ndarray] = []
    for row, returns in enumerate(raw_returns):
        if returns.size == 0:
            normalized_returns.append(np.zeros(0, dtype=np.float32))
            continue
        group_idx = row // group_size
        norm = np.zeros_like(returns, dtype=np.float32)
        for t, val in enumerate(returns):
            if np.isnan(val):
                norm[t] = 0.0
                continue
            mean, std = group_step_stats[group_idx][t]
            centered = float(val) - mean
            if scale_rewards == "none":
                norm[t] = np.float32(centered)
            elif not np.isfinite(std) or std < eps:
                norm[t] = np.float32(0.0)
            else:
                norm[t] = np.float32(centered / (std + eps))
        normalized_returns.append(np.nan_to_num(norm))

    return raw_returns, normalized_returns


def compute_advantages_step(
    step_rewards_list: list[np.ndarray],
    group_size: int,
    gamma: float = 0.95,
    response_masks: np.ndarray | None = None,
    turn_ids: np.ndarray | None = None,
    scale_rewards: str = "group",
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute step-level discounted advantages.

    Computes discounted returns ``G_t`` for every turn ``t`` in each trajectory
    and normalizes each timestep ``t`` across trajectories in the group (rather
    than broadcasting only ``G_0`` to all turns).

    When ``response_masks`` is provided, returns a 2D token-aligned advantage
    array ``[num_trajectories, seq_len]`` where each response token receives
    its own turn's normalized advantage ``A_t``. Otherwise returns a 1D
    per-trajectory summary ``[num_trajectories]`` averaging each trajectory's
    per-turn normalized advantages.

    Args:
        step_rewards_list: List of per-trajectory step reward arrays.
        group_size: Number of trajectories per group.
        gamma: Discount factor.
        response_masks: Optional binary response mask ``[num_trajectories, seq_len]``.
        turn_ids: Optional per-token turn index ``[num_trajectories, seq_len]``
            (``-1`` for non-response tokens, ``0..T-1`` for assistant turns).
        scale_rewards: Scaling mode: ``"group"``, ``"batch"``, or ``"none"``.
        eps: Numerical stability epsilon and zero-std guard threshold.

    Returns:
        Float32 advantages of shape ``[num_trajectories, seq_len]`` when
        ``response_masks`` is provided, or ``[num_trajectories]`` otherwise.
    """
    if response_masks is not None:
        return compute_advantages_agentic(
            step_rewards_list=step_rewards_list,
            response_masks=response_masks,
            group_size=group_size,
            gamma=gamma,
            turn_ids=turn_ids,
            scale_rewards=scale_rewards,
            eps=eps,
        )

    _, normalized_per_traj = _compute_per_step_normalized_returns(
        step_rewards_list=step_rewards_list,
        group_size=group_size,
        gamma=gamma,
        scale_rewards=scale_rewards,
        eps=eps,
    )
    return np.array(
        [float(np.mean(advs)) if advs.size else 0.0 for advs in normalized_per_traj],
        dtype=np.float32,
    )


def compute_advantages_agentic(
    step_rewards_list: list[np.ndarray],
    response_masks: np.ndarray,
    group_size: int,
    gamma: float = 0.95,
    turn_ids: np.ndarray | None = None,
    scale_rewards: str = "group",
    eps: float = 1e-4,
) -> np.ndarray:
    """Return per-step group-normalized credit per response turn, not pooled across timesteps.

    Each step index ``t`` is normalized across trajectories in the same group
    at the same step index ``t`` (with zero-std guarding so identical rollouts
    receive ``0.0`` advantage rather than spurious cross-timestep variance).
    When ``turn_ids`` is supplied, token-to-turn attribution uses those explicit
    turn indices so 0-token turns or omitted empty observations never shift
    ``returns[k]`` onto the wrong turn.

    Args:
        step_rewards_list: One reward sequence per trajectory.
        response_masks: Batched token masks ``[batch_size, seq_len]``.
            Observations and padding always receive zero advantage.
        group_size: Number of trajectories sharing a prompt.
        gamma: Discount applied only between response turns.
        turn_ids: Optional explicit per-token turn indices matching
            ``response_masks`` shape (``>= 0`` for assistant turns, ``-1`` for
            non-response tokens).
        scale_rewards: Scaling mode: ``"group"``, ``"batch"``, or ``"none"``.
        eps: Numerical stability epsilon and zero-std guard threshold.

    Returns:
        Float32 advantages with the same shape as ``response_masks``.
    """
    masks = np.asarray(response_masks, dtype=bool)
    if group_size <= 0 or masks.ndim != 2 or len(masks) != len(step_rewards_list) or len(masks) % group_size:
        raise ValueError("Response masks must form complete trajectory groups.")
    if turn_ids is not None:
        turn_id_arr = np.asarray(turn_ids, dtype=np.int32)
        if turn_id_arr.shape != masks.shape:
            raise ValueError(f"turn_ids shape {turn_id_arr.shape} must match response_masks shape {masks.shape}.")
    else:
        turn_id_arr = None

    if masks.shape[0] == 0 or masks.shape[1] == 0:
        return np.zeros(masks.shape, dtype=np.float32)

    _, normalized_per_traj = _compute_per_step_normalized_returns(
        step_rewards_list=step_rewards_list,
        group_size=group_size,
        gamma=gamma,
        scale_rewards=scale_rewards,
        eps=eps,
    )

    values = np.zeros(masks.shape, dtype=np.float32)
    for row, (step_advs, mask) in enumerate(zip(normalized_per_traj, masks, strict=True)):
        if not np.any(mask):
            continue
        if turn_id_arr is not None:
            row_turns = turn_id_arr[row]
            valid = mask & (row_turns >= 0)
            if not np.any(valid):
                continue
            unique_turns = np.unique(row_turns[valid])
            if int(unique_turns.max()) >= len(step_advs):
                raise ValueError("Every response segment needs a step reward.")
            for t in unique_turns:
                t_int = int(t)
                values[row, valid & (row_turns == t_int)] = step_advs[t_int]
        else:
            starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
            ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
            if len(starts) > len(step_advs):
                raise ValueError("Every response segment needs a step reward.")
            for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                values[row, start:end] = step_advs[index]

    return np.nan_to_num(values).astype(np.float32)


def compute_advantages_gigpo(
    episode_rewards: jax.Array,
    step_rewards_list: list[np.ndarray],
    group_size: int,
    episode_weight: float = 1.0,
    step_weight: float = 1.0,
    gamma: float = 0.95,
    scale_rewards: str = "group",
    response_masks: np.ndarray | None = None,
    turn_ids: np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute GiGPO-style combined episode + step advantages.

    Combines normalized episode-level rewards with normalized
    step-level discounted returns using configurable weights, honoring
    ``scale_rewards`` on both components.

    Args:
        episode_rewards: Per-trajectory episode rewards.
        step_rewards_list: List of per-trajectory step reward arrays.
        group_size: Number of trajectories per group.
        episode_weight: Weight for episode-level advantages.
        step_weight: Weight for step-level advantages.
        gamma: Discount factor for step rewards.
        scale_rewards: Scaling mode for both episode and step advantages
            (``"group"``, ``"batch"``, or ``"none"``).
        response_masks: Optional 2D response masks for token-aligned per-turn
            step advantages.
        turn_ids: Optional explicit per-token turn indices aligned with
            ``response_masks``.

    Returns:
        Tuple of (combined advantages, std_rewards).
    """
    episode_adv, std_rewards = compute_advantages_episode(episode_rewards, group_size, scale_rewards)
    step_adv = jnp.array(
        compute_advantages_step(
            step_rewards_list,
            group_size,
            gamma,
            response_masks=response_masks,
            turn_ids=turn_ids,
            scale_rewards=scale_rewards,
        ),
        dtype=episode_adv.dtype,
    )
    if step_adv.ndim == 2:
        mask_arr = jnp.asarray(response_masks, dtype=bool)
        combined = jnp.where(mask_arr, episode_weight * episode_adv[:, None] + step_weight * step_adv, 0.0)
    else:
        combined = episode_weight * episode_adv + step_weight * step_adv
    return combined, std_rewards
