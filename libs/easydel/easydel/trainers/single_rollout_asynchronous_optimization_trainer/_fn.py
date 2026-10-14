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

"""Numerical primitives for Single-Rollout Asynchronous Optimization."""

from __future__ import annotations

import jax
from jax import numpy as jnp


def direct_importance_sampling_mask(
    current_logps: jax.Array,
    behavior_logps: jax.Array,
    action_mask: jax.Array,
    *,
    epsilon_low: float,
    epsilon_high: float,
) -> jax.Array:
    """Return SAO's strict, double-sided token acceptance mask.

    Comparisons are performed in log space.  A ratio exactly equal to either
    boundary is rejected, matching the strict interval in the paper.

    Args:
        current_logps: Current-policy token log probabilities.
        behavior_logps: Rollout-policy token log probabilities.
        action_mask: Binary mask selecting model-generated action tokens.
        epsilon_low: Lower rejection width in ``(0, 1]``.
        epsilon_high: Positive upper rejection width.

    Returns:
        Float mask with the same shape as ``current_logps``.
    """
    if not 0.0 < epsilon_low <= 1.0:
        raise ValueError("`epsilon_low` must be in (0, 1].")
    if epsilon_high <= 0.0:
        raise ValueError("`epsilon_high` must be positive.")
    log_ratio = current_logps - behavior_logps
    lower = jnp.log1p(-jnp.asarray(epsilon_low, dtype=log_ratio.dtype))
    upper = jnp.log1p(jnp.asarray(epsilon_high, dtype=log_ratio.dtype))
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
) -> jax.Array:
    """Compute the detached-ratio SAO score-function surrogate.

    The calibrated ratio is detached deliberately.  Differentiating through
    it would add an extra factor not present in the importance-weighted policy
    gradient.
    """
    action_mask = action_mask.astype(jnp.float32)
    keep = direct_importance_sampling_mask(
        current_logps,
        behavior_logps,
        action_mask,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
    )
    ratio = jnp.exp(current_logps - behavior_logps)
    weight = jax.lax.stop_gradient(ratio * keep)
    advantages = jax.lax.stop_gradient(advantages)
    if normalizer is None:
        normalizer = jnp.maximum(jnp.sum(action_mask), 1.0)
    else:
        normalizer = jnp.maximum(jnp.asarray(normalizer, dtype=jnp.float32), 1.0)
    return -jnp.sum(weight * advantages * current_logps * action_mask) / normalizer


__all__ = ("direct_importance_sampling_mask", "sao_policy_loss")
