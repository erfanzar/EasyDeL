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
"""Numerical primitives shared by online actor-critic trainers."""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp


def target_aligned_token_ids(input_ids: jax.Array) -> jax.Array:
    """Return causal-language-model targets aligned with next-token logits.

    A model invocation over ``input_ids[..., :-1]`` produces logits for
    ``input_ids[..., 1:]``. Keeping this shift in one helper prevents rollout
    code from accidentally optimizing the token at the same position as the
    input that contains it.

    Args:
        input_ids: Integer token IDs with sequence length on the last axis.

    Returns:
        ``input_ids[..., 1:]``.

    Raises:
        ValueError: If ``input_ids`` is scalar.
    """

    input_ids = jnp.asarray(input_ids)
    if input_ids.ndim == 0:
        raise ValueError("input_ids must have a sequence axis")
    return input_ids[..., 1:]


def target_aligned_action_mask(
    action_mask: jax.Array,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
    """Shift a token-role mask onto next-token policy targets.

    ``action_mask[..., t]`` describes the role of token ``input_ids[..., t]``.
    The corresponding policy loss is emitted by the logit at position
    ``t - 1``, so the returned mask drops position zero. When an attention mask
    is supplied, both the source and target positions must be valid. This also
    prevents the first real token of a left-padded sequence from being trained
    from a padding-position logit.

    Args:
        action_mask: Token-aligned mask with sequence length on the last axis.
        attention_mask: Optional same-shape valid-token mask.

    Returns:
        A mask with shape ``action_mask.shape[:-1] + (length - 1,)`` and the
        same dtype as ``action_mask``.

    Raises:
        ValueError: If a sequence axis is missing or the masks differ in shape.
    """

    action_mask = jnp.asarray(action_mask)
    if action_mask.ndim == 0:
        raise ValueError("action_mask must have a sequence axis")

    aligned = action_mask[..., 1:].astype(jnp.bool_)
    if attention_mask is not None:
        attention_mask = jnp.asarray(attention_mask)
        if attention_mask.shape != action_mask.shape:
            raise ValueError(
                "attention_mask and action_mask must have identical shapes; "
                f"got {attention_mask.shape} and {action_mask.shape}"
            )
        valid_source = attention_mask[..., :-1].astype(jnp.bool_)
        valid_target = attention_mask[..., 1:].astype(jnp.bool_)
        aligned = aligned & valid_source & valid_target
    return aligned.astype(action_mask.dtype)


def length_adaptive_lambda(
    optimized_token_count: jax.Array | int,
    *,
    alpha: float = 1.5,
) -> jax.Array:
    """Compute the length-adaptive GAE lambda used by long agentic rollouts.

    The schedule is ``lambda(l) = 1 - 1 / (alpha * l)``. Empty trajectories
    receive zero and the result is clipped to the valid GAE interval. The
    clipping only affects degenerate very-short schedules; for the published
    ``alpha=1.5`` and positive lengths it leaves the formula unchanged.

    Args:
        optimized_token_count: Number of policy-optimized tokens per
            trajectory.
        alpha: Positive length scale in the schedule.

    Returns:
        A float32 array matching ``optimized_token_count``.

    Raises:
        ValueError: If ``alpha`` is not positive.
    """

    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    lengths = jnp.asarray(optimized_token_count, dtype=jnp.float32)
    safe_lengths = jnp.maximum(lengths, 1.0)
    scheduled = 1.0 - 1.0 / (float(alpha) * safe_lengths)
    return jnp.where(lengths > 0, jnp.clip(scheduled, 0.0, 1.0), 0.0)


def global_token_denominator(
    token_mask: jax.Array,
    *,
    axis_name: tp.Hashable | None = None,
) -> jax.Array:
    """Return a globally reduced, non-zero token-count denominator.

    Args:
        token_mask: Mask selecting tokens that contribute to a loss.
        axis_name: Optional named JAX axis over which to sum the local count.

    Returns:
        The global float32 token count, clamped to at least one.
    """

    denominator = jnp.sum(jnp.asarray(token_mask, dtype=jnp.float32))
    if axis_name is not None:
        denominator = jax.lax.psum(denominator, axis_name)
    return jnp.maximum(denominator, 1.0)


def global_token_mean(
    values: jax.Array,
    token_mask: jax.Array,
    *,
    axis_name: tp.Hashable | None = None,
) -> jax.Array:
    """Average values with one denominator across all optimized tokens.

    This is intentionally a token-level reduction. It does not average each
    segment or trajectory independently before reducing the batch.

    Args:
        values: Per-token values.
        token_mask: Same-shape optimized-token mask.
        axis_name: Optional named JAX axis over which numerator and denominator
            are summed.

    Returns:
        A scalar float32 mean, or zero when the mask is empty.

    Raises:
        ValueError: If ``values`` and ``token_mask`` differ in shape.
    """

    values = jnp.asarray(values, dtype=jnp.float32)
    token_mask = jnp.asarray(token_mask, dtype=jnp.float32)
    if values.shape != token_mask.shape:
        raise ValueError(f"values and token_mask must have identical shapes; got {values.shape} and {token_mask.shape}")
    numerator = jnp.sum(values * token_mask)
    if axis_name is not None:
        numerator = jax.lax.psum(numerator, axis_name)
    return numerator / global_token_denominator(token_mask, axis_name=axis_name)


def _broadcast_per_trajectory(value: jax.Array | float, leading_shape: tuple[int, ...], name: str) -> jax.Array:
    """Broadcast a scalar or per-trajectory value over leading dimensions."""

    value = jnp.asarray(value, dtype=jnp.float32)
    if value.ndim == 0:
        return jnp.broadcast_to(value, leading_shape)
    try:
        return jnp.broadcast_to(value, leading_shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} with shape {value.shape} cannot broadcast to trajectory shape {leading_shape}"
        ) from exc


def skip_observation_gae(
    rewards: jax.Array,
    values: jax.Array,
    action_mask: jax.Array,
    *,
    gamma: float = 1.0,
    lam: jax.Array | float = 0.95,
    transition_mask: jax.Array | None = None,
    bootstrap_values: jax.Array | float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE on the action-token chain while skipping observations.

    Non-action positions neither emit an advantage nor reset the backward
    carry. Consequently, an action immediately before an environment
    observation bootstraps from the first later action, matching the logical
    policy transition rather than the adjacent token position.

    ``transition_mask`` is action-aligned: zero means that the action is a
    terminal boundary, while one permits bootstrapping to the next action (or
    ``bootstrap_values`` when no later action is present). If omitted,
    transitions continue exactly when a later action exists in the row.

    Args:
        rewards: Per-position rewards with time on the last axis.
        values: Per-position critic values, aligned with ``rewards``.
        action_mask: Same-shape mask selecting optimized action tokens.
        gamma: Discount factor between consecutive action tokens.
        lam: Scalar or per-trajectory GAE lambda.
        transition_mask: Optional same-shape continuation mask.
        bootstrap_values: Scalar or per-trajectory value beyond the final
            in-buffer action.

    Returns:
        ``(advantages, returns)`` with zeros at non-action positions.

    Raises:
        ValueError: If input shapes are incompatible or ``gamma`` is negative.
    """

    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")

    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    values = jnp.asarray(values, dtype=jnp.float32)
    action_mask = jnp.asarray(action_mask).astype(jnp.bool_)
    if rewards.ndim == 0:
        raise ValueError("rewards must have a sequence axis")
    if rewards.shape != values.shape or rewards.shape != action_mask.shape:
        raise ValueError(
            "rewards, values, and action_mask must have identical shapes; "
            f"got {rewards.shape}, {values.shape}, and {action_mask.shape}"
        )

    explicit_transitions = transition_mask is not None
    if transition_mask is None:
        transition_mask_array = jnp.zeros_like(action_mask)
    else:
        transition_mask_array = jnp.asarray(transition_mask).astype(jnp.bool_)
        if transition_mask_array.shape != rewards.shape:
            raise ValueError(
                f"transition_mask must match rewards; got {transition_mask_array.shape} and {rewards.shape}"
            )

    leading_shape = rewards.shape[:-1]
    lambda_array = _broadcast_per_trajectory(lam, leading_shape, "lam")
    bootstrap_array = _broadcast_per_trajectory(bootstrap_values, leading_shape, "bootstrap_values")

    def scan_step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        next_value, next_advantage, seen_action = carry
        reward_t, value_t, is_action_t, transition_t = inputs
        continues = transition_t if explicit_transitions else seen_action
        delta_t = reward_t + float(gamma) * next_value * continues - value_t
        advantage_t = delta_t + float(gamma) * lambda_array * next_advantage * continues
        output = jnp.where(is_action_t, advantage_t, 0.0)
        next_carry = (
            jnp.where(is_action_t, value_t, next_value),
            jnp.where(is_action_t, advantage_t, next_advantage),
            seen_action | is_action_t,
        )
        return next_carry, output

    reverse_inputs = (
        jnp.moveaxis(rewards[..., ::-1], -1, 0),
        jnp.moveaxis(values[..., ::-1], -1, 0),
        jnp.moveaxis(action_mask[..., ::-1], -1, 0),
        jnp.moveaxis(transition_mask_array[..., ::-1], -1, 0),
    )
    initial_carry = (
        bootstrap_array,
        jnp.zeros(leading_shape, dtype=jnp.float32),
        jnp.zeros(leading_shape, dtype=jnp.bool_),
    )
    _, reverse_advantages = jax.lax.scan(scan_step, initial_carry, reverse_inputs)
    advantages = jnp.moveaxis(reverse_advantages, 0, -1)[..., ::-1]
    advantages = jnp.where(action_mask, advantages, 0.0)
    returns = jnp.where(action_mask, advantages + values, 0.0)
    return advantages, returns


def following_segment_token_counts(
    segment_ids: jax.Array,
    action_mask: jax.Array,
) -> jax.Array:
    """Count optimized tokens in chronologically later contiguous segments.

    Args:
        segment_ids: Integer segment ID at each token position.
        action_mask: Same-shape optimized-token mask.

    Returns:
        An int32 count per action token and zero elsewhere.

    Raises:
        ValueError: If shapes differ or a sequence axis is missing.
    """

    segment_ids = jnp.asarray(segment_ids, dtype=jnp.int32)
    action_mask = jnp.asarray(action_mask).astype(jnp.bool_)
    if action_mask.ndim == 0:
        raise ValueError("action_mask must have a sequence axis")
    if segment_ids.shape != action_mask.shape:
        raise ValueError(
            f"segment_ids and action_mask must have identical shapes; got {segment_ids.shape} and {action_mask.shape}"
        )

    leading_shape = action_mask.shape[:-1]

    def scan_step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
        current_segment, current_count, finished_count, seen_action = carry
        segment_t, is_action_t = inputs
        changed_segment = is_action_t & seen_action & (segment_t != current_segment)
        updated_finished = finished_count + jnp.where(changed_segment, current_count, 0)
        updated_current_count = jnp.where(
            is_action_t,
            jnp.where(changed_segment | ~seen_action, 1, current_count + 1),
            current_count,
        )
        updated_segment = jnp.where(is_action_t, segment_t, current_segment)
        output = jnp.where(is_action_t, updated_finished, 0)
        return (
            updated_segment,
            updated_current_count,
            updated_finished,
            seen_action | is_action_t,
        ), output

    reverse_inputs = (
        jnp.moveaxis(segment_ids[..., ::-1], -1, 0),
        jnp.moveaxis(action_mask[..., ::-1], -1, 0),
    )
    initial_carry = (
        jnp.zeros(leading_shape, dtype=jnp.int32),
        jnp.zeros(leading_shape, dtype=jnp.int32),
        jnp.zeros(leading_shape, dtype=jnp.int32),
        jnp.zeros(leading_shape, dtype=jnp.bool_),
    )
    _, reverse_counts = jax.lax.scan(scan_step, initial_carry, reverse_inputs)
    return jnp.moveaxis(reverse_counts, 0, -1)[..., ::-1]


def cross_segment_advantage_correction(
    advantages: jax.Array,
    segment_ids: jax.Array,
    action_mask: jax.Array,
    *,
    gamma: float = 1.0,
    lam: jax.Array | float = 0.95,
) -> jax.Array:
    """Apply CompactionRL credit correction across trajectory segments.

    For segment ``s``, every local advantage is multiplied by
    ``(gamma * lambda) ** N_after_s``, where ``N_after_s`` counts optimized
    tokens in all later segments. Observation, prompt, and copied context
    tokens do not contribute to the exponent.

    Segment IDs are interpreted in action-token order and may have arbitrary
    integer values, but each logical segment must occupy one contiguous region
    of the trajectory.

    Args:
        advantages: Local per-token advantages.
        segment_ids: Same-shape integer segment IDs.
        action_mask: Same-shape optimized-token mask.
        gamma: Discount factor.
        lam: Scalar or per-trajectory GAE lambda.

    Returns:
        Corrected advantages, zero outside ``action_mask``.

    Raises:
        ValueError: If shapes are incompatible or ``gamma`` is negative.
    """

    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    advantages = jnp.asarray(advantages, dtype=jnp.float32)
    segment_ids = jnp.asarray(segment_ids, dtype=jnp.int32)
    action_mask = jnp.asarray(action_mask).astype(jnp.bool_)
    if advantages.ndim == 0:
        raise ValueError("advantages must have a sequence axis")
    if advantages.shape != segment_ids.shape or advantages.shape != action_mask.shape:
        raise ValueError(
            "advantages, segment_ids, and action_mask must have identical shapes; "
            f"got {advantages.shape}, {segment_ids.shape}, and {action_mask.shape}"
        )

    counts = following_segment_token_counts(segment_ids, action_mask)
    lambda_array = _broadcast_per_trajectory(lam, advantages.shape[:-1], "lam")
    factor = jnp.power(float(gamma) * lambda_array[..., None], counts)
    return jnp.where(action_mask, advantages * factor, 0.0)


__all__ = (
    "cross_segment_advantage_correction",
    "following_segment_token_counts",
    "global_token_denominator",
    "global_token_mean",
    "length_adaptive_lambda",
    "skip_observation_gae",
    "target_aligned_action_mask",
    "target_aligned_token_ids",
)
