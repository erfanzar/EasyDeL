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

"""Cycle-safe token-global policy objectives shared by online RL trainers."""

from __future__ import annotations

import jax
from jax import numpy as jnp

_MIN_SAFE_LOG_RATIO: float = -80.0
_MAX_SAFE_LOG_RATIO: float = 30.0


def _normalizer(action_mask: jax.Array, normalizer: jax.Array | float | None) -> jax.Array:
    """Resolve a local or explicitly accumulation-correct token denominator."""
    if normalizer is None:
        return jnp.maximum(jnp.sum(action_mask), 1.0)
    count = jnp.asarray(normalizer, dtype=jnp.float32)
    return jnp.where(count > 0.0, count, 1.0)


def clipped_ppo_policy_loss(
    current_logps: jax.Array,
    behavior_logps: jax.Array,
    advantages: jax.Array,
    action_mask: jax.Array,
    *,
    cliprange: float,
    normalizer: jax.Array | float | None = None,
    return_clip_fraction: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the token-global clipped PPO surrogate and optional diagnostics.

    Args:
        current_logps: Current-policy token log probabilities.
        behavior_logps: Rollout-policy token log probabilities.
        advantages: Fixed token advantages, detached from differentiation.
        action_mask: Mask selecting generated action tokens.
        cliprange: Symmetric importance-ratio clipping width.
        normalizer: Optional global token denominator, already corrected for
            gradient accumulation by the caller. Defaults to local action count.
        return_clip_fraction: Also return the local fraction of action tokens
            whose clipped loss exceeds their unclipped loss.

    Returns:
        Scalar policy loss, or ``(loss, clip_fraction)`` when requested.
    """
    action_mask = action_mask.astype(jnp.float32)
    mask_bool = action_mask > 0.0
    advantages = jax.lax.stop_gradient(jnp.where(mask_bool, advantages.astype(jnp.float32), 0.0))
    log_ratio = (current_logps - behavior_logps).astype(jnp.float32)
    # Mask non-action positions and clamp before exp() so masked or outlier
    # tokens never overflow to inf and poison reductions or VJP with NaN.
    safe_log_ratio = jnp.clip(
        jnp.where(mask_bool, log_ratio, 0.0),
        _MIN_SAFE_LOG_RATIO,
        _MAX_SAFE_LOG_RATIO,
    )
    ratio = jnp.exp(safe_log_ratio)
    low = 1.0 - cliprange
    high = 1.0 + cliprange
    use_clipped = ((advantages > 0.0) & (ratio > high)) | ((advantages < 0.0) & (ratio < low))
    safe_unclipped_ratio = jnp.where(mask_bool & ~use_clipped, ratio, 1.0)
    unclipped = -advantages * safe_unclipped_ratio
    clipped_bound = jnp.where(advantages > 0.0, high, low)
    clipped = -advantages * clipped_bound
    surrogate = jnp.where(mask_bool, jnp.where(use_clipped, clipped, unclipped), 0.0)
    loss = jnp.sum(surrogate) / _normalizer(action_mask, normalizer)
    if return_clip_fraction:
        clip_fraction = jnp.sum(use_clipped.astype(jnp.float32) * action_mask) / _normalizer(action_mask, None)
        return loss, clip_fraction
    return loss


def direct_importance_sampling_mask(
    current_logps: jax.Array,
    behavior_logps: jax.Array,
    action_mask: jax.Array,
    *,
    epsilon_low: float,
    epsilon_high: float,
) -> jax.Array:
    """Return SAO's strict double-sided token acceptance mask in log space.

    Args:
        current_logps: Current-policy token log probabilities.
        behavior_logps: Rollout-policy token log probabilities.
        action_mask: Binary mask selecting generated action tokens.
        epsilon_low: Lower rejection width in ``(0, 1]``.
        epsilon_high: Positive upper rejection width.

    Returns:
        Float action mask rejecting ratios on or outside either boundary.

    Raises:
        ValueError: If either rejection width is invalid.
    """
    if not 0.0 < epsilon_low <= 1.0:
        raise ValueError("`epsilon_low` must be in (0, 1].")
    if epsilon_high <= 0.0:
        raise ValueError("`epsilon_high` must be positive.")
    action_mask = action_mask.astype(jnp.float32)
    log_ratio = (current_logps - behavior_logps).astype(jnp.float32)
    lower = jnp.log1p(-jnp.asarray(epsilon_low, dtype=jnp.float32))
    upper = jnp.log1p(jnp.asarray(epsilon_high, dtype=jnp.float32))
    return ((log_ratio > lower) & (log_ratio < upper)).astype(jnp.float32) * action_mask


def sao_policy_loss(
    current_logps: jax.Array,
    behavior_logps: jax.Array,
    advantages: jax.Array,
    action_mask: jax.Array,
    *,
    epsilon_low: float,
    epsilon_high: float,
    normalizer: jax.Array | float | None = None,
    return_keep_mask: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute SAO's detached-ratio score-function surrogate.

    Args:
        current_logps: Current-policy token log probabilities.
        behavior_logps: Rollout-policy token log probabilities.
        advantages: Fixed token advantages, detached from differentiation.
        action_mask: Mask selecting generated action tokens.
        epsilon_low: Lower rejection width in ``(0, 1]``.
        epsilon_high: Positive upper rejection width.
        normalizer: Optional accumulation-correct global token denominator.
            Defaults to local action count, including rejected action tokens.
        return_keep_mask: Also return the DIS mask used by this objective.

    Returns:
        Scalar policy loss, or ``(loss, keep_mask)`` when requested. The
        importance weight is detached to retain the intended policy gradient.
    """
    action_mask = action_mask.astype(jnp.float32)
    keep = direct_importance_sampling_mask(
        current_logps,
        behavior_logps,
        action_mask,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
    )
    keep_bool = keep > 0.0
    log_ratio = (current_logps - behavior_logps).astype(jnp.float32)
    # Mask rejected/non-action positions before exp() to prevent exp(large) * 0 = NaN.
    safe_log_ratio = jnp.clip(
        jnp.where(keep_bool, log_ratio, 0.0),
        _MIN_SAFE_LOG_RATIO,
        _MAX_SAFE_LOG_RATIO,
    )
    ratio = jnp.where(keep_bool, jnp.exp(safe_log_ratio), 0.0)
    weight = jax.lax.stop_gradient(ratio)
    advantages = jax.lax.stop_gradient(jnp.where(keep_bool, advantages.astype(jnp.float32), 0.0))
    safe_current_logps = jnp.where(keep_bool, current_logps.astype(jnp.float32), 0.0)
    loss = -jnp.sum(weight * advantages * safe_current_logps) / _normalizer(action_mask, normalizer)
    return (loss, keep) if return_keep_mask else loss
