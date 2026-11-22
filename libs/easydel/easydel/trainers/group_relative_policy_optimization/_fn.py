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

"""Internal functions for Group Relative Policy Optimization training.

This module contains the core computational functions used by the GRPO trainer,
implementing group-based relative policy optimization for RLHF. GRPO improves
training stability by normalizing rewards within groups of samples rather than
across the entire batch, reducing variance in gradient estimates.

The module provides functions for:
- Computing per-token log probabilities from model outputs
- Calculating KL divergence penalties between policy and reference models
- Group-based reward normalization and advantage estimation
- Policy gradient loss computation with various clipping strategies

All functions are JAX-compatible and support distributed training through sharding.
"""

import collections.abc
import math
import typing as tp

import jax
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import common_types, with_sharding_constraint

from easydel.caching import TransformerCache
from easydel.caching.shared_prefix import (
    PREFIX_SHARED_ATTENTION_MECHANISMS,
    PREFIX_SHARED_RECURRENT_MODEL_TYPES,
    PrefixCaptureView,
    PrefixForkView,
    PrefixRecurrentCaptureView,
    PrefixRecurrentForkView,
)
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.trainers._logprob_utils import (
    compute_per_token_logps_and_entropies_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)

from .._shared.off_policy import (
    SUM_PREFIX,
    acceptance_ok,
    add_behavior_sums,
    behavior_parity_ok,
    behavior_policy_terms,
    split_behavior_sums,
    summarize_behavior_sums,
)
from ..reward_protocol import RewardProtocol
from ..training_utils import (
    SCHEDULED_UPDATE_GATE_KEY,
    ScheduledLossAdapter,
    compact_generation_model_kwargs,
    extract_generation_model_kwargs,
    make_assertions_and_get_sizes,
    make_scheduled_state_shell,
    minibatch_call,
    normalize_generation_model_kwargs,
    prepare_generation_model_kwargs_for_call,
    register_scheduled_loss_adapter,
    repeat_prompt_aligned_model_kwargs,
    scheduled_loss_cache_key,
    slice_prompt_aligned_model_kwargs,
    update_metrics,
    update_state_respectfully,
)

RewardFunc = EasyDeLState | RewardProtocol | tp.Callable[[list, list], list[float]]


def _masked_sum_and_count(x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the (numerator, denominator) used by the chunked masked mean.

    Mirrors the ``masked_mean`` helper defined inside :func:`grpo_step`:
    when ``x`` has a trailing-singleton sequence axis (the sequence-level
    importance-sampling path) the per-row mask is ignored and the numerator
    is the plain sum; otherwise both numerator and denominator are taken
    against the completion mask.

    Args:
        x: Per-token tensor of shape ``[batch, seq]`` (or ``[batch, 1]``
            for sequence-level diagnostics).
        mask: Completion mask of shape ``[batch, seq]``. Ignored when the
            sequence axis is singleton.

    Returns:
        ``(numerator, denominator)`` where ``denominator`` is always
        clamped to ``>= 1`` to avoid divide-by-zero downstream.
    """

    if x.shape[1] == 1:
        return jnp.sum(x), jnp.array(x.shape[0], dtype=jnp.float32)
    return jnp.sum(x * mask), jnp.maximum(jnp.sum(mask), 1.0).astype(jnp.float32)


def _global_token_normalizer(
    num_items_in_batch: jax.Array | None,
    local_token_count: jax.Array,
    was_truncated: bool,
    gradient_accumulation_steps: int,
    *,
    is_training: bool = True,
) -> jax.Array:
    """Normalizer for globally token-normalized losses (``dapo``/``cispo``/``vespo``/``dppo``).

    These loss types divide ``sum(per_token_loss * mask)`` by the *full-batch*
    completion-token count carried in ``num_items_in_batch`` — a 0-dim scalar
    that :func:`minibatch_call` deliberately does **not** slice per microbatch.
    Under gradient accumulation (``gradient_accumulation_steps = K > 1``) during
    training, ``minibatch_call`` averages the accumulated gradients by ``1/K``.
    Dividing ``max(N_global, 1.0)`` by ``K`` here makes each microbatch loss
    ``S_k / (N_global/K) = K·S_k / N_global`` so the ``1/K`` average recombines
    to the exact ``S_total / N_global`` — even when ``0 < N_global < K`` (clamping
    before division avoids post-division ``max(N_global / K, 1.0)`` distortion).

    During evaluation (``is_training=False``), ``loss_fn`` runs once on the full
    batch without ``minibatch_call``'s ``1/K`` averaging, so ``num_items_in_batch``
    must not be divided by ``K``.

    A truncated window or a missing ``num_items_in_batch`` falls back to the
    microbatch-local token count, which is already correct under ``1/K``
    averaging (a per-microbatch mean) and must **not** be divided.
    """
    if was_truncated or num_items_in_batch is None:
        return jnp.maximum(local_token_count, 1.0)
    effective_ga = max(int(gradient_accumulation_steps), 1) if is_training else 1
    return jnp.maximum(num_items_in_batch, 1.0) / effective_ga


def _grpo_weighted_loss_denominator(
    batch: tp.Mapping[str, jax.Array],
    loss_type: str,
    max_loss_completion_tokens: int | None,
) -> jax.Array:
    """Full-batch loss denominator of a GRPO batch (weighted or unweighted).

    Matches the quantity each loss type divides by per minibatch: ``sum(w)`` for
    ``grpo``/``sapo``/``gmpo``/``luspo``, ``sum(w) * T`` for ``dr_grpo``, and
    ``sum(mask * w)`` for token-mean losses (with ``w = 1`` when ``difficulty_weights``
    is absent), with ``T`` and ``mask`` truncated to ``max_loss_completion_tokens``.

    Args:
        batch: Step or microbatch mapping with ``completion_mask`` and optional ``difficulty_weights``.
        loss_type: GRPO loss variant.
        max_loss_completion_tokens: Optional completion-token cap used by the loss.

    Returns:
        A 0-dim float32 denominator.
    """
    mask = jnp.asarray(batch["completion_mask"])
    if max_loss_completion_tokens is not None and mask.shape[1] > max_loss_completion_tokens:
        mask = mask[:, :max_loss_completion_tokens]
    mask = mask.astype(jnp.float32)
    weights = batch.get("difficulty_weights")
    if weights is not None:
        weights = jnp.asarray(weights).astype(jnp.float32)
        if weights.ndim == 1:
            weights = weights[:, None]
        if loss_type in {"grpo", "sapo", "gmpo", "luspo"}:
            return jnp.sum(weights)
        if loss_type == "dr_grpo":
            return jnp.sum(weights) * mask.shape[1]
        return jnp.sum(mask * weights)
    if loss_type in {"grpo", "sapo", "gmpo", "luspo"}:
        return jnp.asarray(mask.shape[0], dtype=jnp.float32)
    if loss_type == "dr_grpo":
        return jnp.asarray(mask.shape[0] * mask.shape[1], dtype=jnp.float32)
    return jnp.sum(mask)


def _compute_log_importance_weights(
    *,
    per_token_logps: jax.Array,
    old_per_token_logps: jax.Array,
    completion_mask: jax.Array,
    importance_sampling_level: str,
) -> jax.Array:
    """Compute token- or sequence-level GRPO log importance weights."""
    log_ratio = per_token_logps - old_per_token_logps
    if importance_sampling_level == "token":
        return jnp.where(completion_mask > 0, log_ratio, 0.0)
    if importance_sampling_level == "sequence":
        masked_log_ratio = jnp.where(completion_mask > 0, log_ratio, 0.0)
        return (masked_log_ratio.sum(axis=-1) / jnp.maximum(completion_mask.sum(axis=-1), 1.0))[:, None]
    if importance_sampling_level == "sequence_token":
        masked_log_ratio = jnp.where(completion_mask > 0, log_ratio, 0.0)
        sequence_log_weight = (masked_log_ratio.sum(axis=-1) / jnp.maximum(completion_mask.sum(axis=-1), 1.0))[:, None]
        return per_token_logps - jax.lax.stop_gradient(per_token_logps) + jax.lax.stop_gradient(sequence_log_weight)
    raise ValueError(
        f"Unknown importance sampling level: {importance_sampling_level}. "
        "Possible values are 'token', 'sequence', and 'sequence_token'."
    )


def _compute_importance_weights(
    *,
    per_token_logps: jax.Array,
    old_per_token_logps: jax.Array,
    completion_mask: jax.Array,
    importance_sampling_level: str,
) -> jax.Array:
    """Compute GRPO importance weights without loss-specific clipping/capping."""
    return jnp.exp(
        _compute_log_importance_weights(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            completion_mask=completion_mask,
            importance_sampling_level=importance_sampling_level,
        )
    )


def _compute_per_token_kl(
    per_token_logps: jax.Array,
    ref_per_token_logps: jax.Array,
    kl_estimator: str = "k3",
    completion_mask: jax.Array | None = None,
) -> jax.Array:
    """Estimate ``KL(pi || pi_ref)`` per token from sampled-token log-probs.

    With ``log_r = log pi - log pi_ref`` the estimators are ``"k1"`` -> ``log_r``,
    ``"k2"`` -> ``0.5 * log_r**2``, ``"k3"`` -> ``exp(-log_r) - 1 + log_r`` (the
    non-negative unbiased GRPO default) and ``"l2"`` -> ``log_r**2``.

    Args:
        per_token_logps: Current-policy log-probs, shape ``[batch, seq]``.
        ref_per_token_logps: Reference-policy log-probs, same shape.
        kl_estimator: One of ``"k1"``, ``"k2"``, ``"k3"``, ``"l2"``.
        completion_mask: Optional completion mask; when supplied, padded positions
            are zeroed before exponentiation so non-finite/extreme PAD values
            cannot poison gradients.

    Returns:
        Per-token KL estimate with the shape of ``per_token_logps``.

    Raises:
        ValueError: If ``kl_estimator`` is unknown.
    """
    ref_minus_policy = ref_per_token_logps - per_token_logps
    if completion_mask is not None:
        ref_minus_policy = jnp.where(completion_mask > 0, ref_minus_policy, 0.0)
    if kl_estimator == "k3":
        clamped = jnp.clip(ref_minus_policy, -80.0, 80.0)
        return jnp.exp(clamped) - clamped - 1
    if kl_estimator == "k1":
        return -ref_minus_policy
    if kl_estimator == "k2":
        return 0.5 * jnp.square(ref_minus_policy)
    if kl_estimator == "l2":
        return jnp.square(ref_minus_policy)
    raise ValueError(f"Unknown kl_estimator: {kl_estimator!r}. Possible values are 'k1', 'k2', 'k3', and 'l2'.")


def _gmpo_sequence_ratio(
    *,
    per_token_logps: jax.Array,
    old_per_token_logps: jax.Array,
    advantages: jax.Array,
    completion_mask: jax.Array,
    epsilon: float,
    epsilon_high: float,
    delta: float | None = None,
    token_is_weights: jax.Array | None = None,
) -> jax.Array:
    """Geometric mean of the advantage-sign-aware clipped token ratios (GMPO).

    Each token ratio ``r_t`` is clipped as ``min(r_t, clip(r_t))`` for positive
    advantages (capped at ``1 + epsilon_high``) and ``max(r_t, clip(r_t))`` for
    non-positive ones (floored at ``1 - epsilon``), clamped at ``1e-6``, and the
    sequence ratio is ``exp(mean_t log(clipped r_t))`` over completion tokens.
    Everything is computed in log space, so padded tokens with extreme
    log-ratios cannot overflow; clipped tokens receive no gradient.

    Token-level rollout importance-sampling weights ``w_t`` (``token_*`` eSurge
    correction modes) are folded *into* the geometric mean: tokens with
    ``w_t = 0`` leave the log-mean mask entirely (so they receive no gradient
    through the sequence ratio), and the remaining tokens add the constant
    ``stop_gradient(log w_t)`` to their clipped log ratio:
    ``exp(mean_{t: w_t > 0} [log clip(r_t) + log w_t])``.

    Args:
        per_token_logps: Current-policy log-probs ``[batch, seq]``.
        old_per_token_logps: Behaviour-policy log-probs ``[batch, seq]``.
        advantages: Per-sequence advantages ``[batch, 1]``.
        completion_mask: Completion-token mask ``[batch, seq]``.
        epsilon: Lower clip offset; must satisfy ``0 < epsilon < 1``.
        epsilon_high: Upper clip offset.
        delta: Optional absolute cap on every token ratio.
        token_is_weights: Optional token-level rollout importance weights
            ``[batch, seq]`` (treated as constants).

    Returns:
        Sequence geometric-mean ratio of shape ``[batch, 1]``.
    """
    mask = completion_mask > 0
    is_log_weight = None
    if token_is_weights is not None:
        weights = jax.lax.stop_gradient(token_is_weights).astype(jnp.float32)
        keep = jnp.isfinite(weights) & (weights > 0)
        mask = mask & keep
        is_log_weight = jnp.log(jnp.where(keep, weights, 1.0))
    log_ratio = jnp.where(mask, per_token_logps - old_per_token_logps, 0.0)
    clipped = jnp.where(
        advantages > 0,
        jnp.minimum(log_ratio, math.log1p(epsilon_high)),
        jnp.maximum(log_ratio, math.log1p(-epsilon)),
    )
    if delta is not None:
        clipped = jnp.minimum(clipped, math.log(delta))
    clipped = jnp.maximum(clipped, math.log(1e-6))
    if is_log_weight is not None:
        clipped = clipped + is_log_weight
    kept = jnp.sum(mask, axis=-1, keepdims=True).astype(jnp.float32)
    token_count = jnp.maximum(kept, 1.0)
    sequence_log_ratio = jnp.sum(jnp.where(mask, clipped, 0.0), axis=-1, keepdims=True) / token_count
    # A row with no kept token (every IS weight zero/non-finite) leaves the objective: ratio 0, not exp(0) = 1.
    return jnp.where(kept > 0, jnp.exp(sequence_log_ratio), 0.0)


def _is_token_level_ratio(ratio: jax.Array | None, per_token_logps: jax.Array) -> bool:
    """Whether a rollout importance-sampling ratio is per token (``[batch, seq]``) rather than per sequence."""
    return ratio is not None and ratio.ndim == 2 and int(ratio.shape[-1]) == int(per_token_logps.shape[-1])


def _apply_rollout_is_ratio_post_hoc(loss_type: str, ratio: jax.Array | None, per_token_logps: jax.Array) -> bool:
    """Whether the rollout IS ratio is multiplied into the per-token loss after the surrogate.

    VESPO folds the ratio into its sequence weight and GMPO folds token-level ratios into its
    geometric mean, so neither multiplies them in again.
    """
    if ratio is None or loss_type == "vespo":
        return False
    return not (loss_type == "gmpo" and _is_token_level_ratio(ratio, per_token_logps))


def _sanitize_rollout_ratio(ratio: jax.Array, accept: jax.Array) -> jax.Array:
    """Zero rollout IS ratios of rejected rows and non-finite entries so they cannot poison gradients."""
    return jnp.where(jnp.isfinite(ratio) & accept, ratio, 0.0)


def _finalize_behavior_metrics(
    metrics: LossMetrics,
    accumulation_steps: int,
) -> tuple[LossMetrics, dict[str, jax.Array] | None]:
    """Replace the additive behaviour sums in ``metrics`` with the ``off_policy/*`` diagnostics.

    Returns:
        ``(metrics, summary)``; ``summary`` is ``None`` when the batch carried no behaviour logprobs.
    """
    remaining, sums = split_behavior_sums(metrics.other_metrics)
    if not sums:
        return metrics, None
    summary = summarize_behavior_sums(sums, accumulation_steps=accumulation_steps)
    remaining.update(summary)
    return metrics.replace(other_metrics=remaining), summary


def _compute_off_policy_sequence_mask(
    *,
    per_token_logps: jax.Array,
    sampling_per_token_logps: jax.Array,
    advantages: jax.Array,
    completion_mask: jax.Array,
    threshold: float,
) -> jax.Array:
    """Keep positive-advantage rows and low-drift negative-advantage rows.

    The forward-KL estimate is sequence-level: sampling log-probs minus current
    policy log-probs, averaged over completion tokens. Negative-advantage rows
    whose mean drift exceeds ``threshold`` are removed from the policy objective.
    """
    forward_kl = jax.lax.stop_gradient(sampling_per_token_logps - per_token_logps)
    sequence_kl = jnp.sum(forward_kl * completion_mask, axis=-1, keepdims=True) / jnp.maximum(
        jnp.sum(completion_mask, axis=-1, keepdims=True),
        1.0,
    )
    return ((advantages >= 0) | (sequence_kl <= threshold)).astype(completion_mask.dtype)


def _compute_dppo_divergence_mask(
    *,
    per_token_logps: jax.Array,
    sampling_per_token_logps: jax.Array,
    advantages: jax.Array,
    completion_mask: jax.Array,
    divergence_type: str,
    epsilon_low: float,
    epsilon_high: float,
    current_topk_logps: jax.Array | None = None,
    sampling_topk_logps: jax.Array | None = None,
) -> jax.Array:
    """Compute DPPO trust-region masks for binary or top-k TV/KL.

    Binary variants compare only the generated token probability. Top-k
    variants compare normalized distributions over the generation-time top-k
    support and still use the sampled-token probability to decide whether the
    current policy moved in the positive- or negative-advantage direction.
    """
    per_token_logps_f32 = per_token_logps.astype(jnp.float32)
    sampling_per_token_logps_f32 = sampling_per_token_logps.astype(jnp.float32)
    prob = jnp.exp(per_token_logps_f32)
    sampling_prob = jnp.exp(sampling_per_token_logps_f32)

    if divergence_type == "binary_tv":
        divergence = jnp.abs(prob - sampling_prob)
    elif divergence_type == "binary_kl":
        safe_prob = jnp.clip(prob, 1e-7, 1.0 - 1e-7)
        safe_sampling_prob = jnp.clip(sampling_prob, 1e-7, 1.0 - 1e-7)
        divergence = safe_sampling_prob * (jnp.log(safe_sampling_prob) - jnp.log(safe_prob)) + (
            1.0 - safe_sampling_prob
        ) * (jnp.log1p(-safe_sampling_prob) - jnp.log1p(-safe_prob))
    elif divergence_type in {"topk_tv", "topk_kl"}:
        if current_topk_logps is None or sampling_topk_logps is None:
            raise ValueError("Top-k DPPO divergence requires current and sampling top-k log-prob tensors.")
        current_logq = jax.nn.log_softmax(current_topk_logps.astype(jnp.float32), axis=-1)
        sampling_logq = jax.nn.log_softmax(jax.lax.stop_gradient(sampling_topk_logps).astype(jnp.float32), axis=-1)
        current_q = jnp.exp(current_logq)
        sampling_q = jnp.exp(sampling_logq)
        if divergence_type == "topk_tv":
            divergence = 0.5 * jnp.sum(jnp.abs(current_q - sampling_q), axis=-1)
        else:
            divergence = jnp.sum(sampling_q * (sampling_logq - current_logq), axis=-1)
    else:
        raise ValueError(f"Unknown DPPO divergence_type: {divergence_type}")

    invalid_pos = (divergence > epsilon_high) & (prob > sampling_prob)
    invalid_neg = (divergence > epsilon_low) & (prob < sampling_prob)
    keep_mask = jnp.where(advantages > 0, ~invalid_pos, ~invalid_neg)
    return keep_mask.astype(completion_mask.dtype) * completion_mask


def _compute_grpo_policy_loss_terms(
    *,
    per_token_logps: jax.Array,
    old_per_token_logps: jax.Array,
    advantages: jax.Array,
    completion_mask: jax.Array,
    loss_type: str,
    epsilon: float,
    epsilon_high: float,
    delta: float | None,
    importance_sampling_level: str,
    sapo_temperature_pos: float,
    sapo_temperature_neg: float,
    vespo_k_pos: float,
    vespo_lambda_pos: float,
    vespo_k_neg: float,
    vespo_lambda_neg: float,
    importance_sampling_ratio: jax.Array | None = None,
    entropy_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute TRL-compatible GRPO/CISPO per-token surrogate terms."""
    log_ratio = per_token_logps - old_per_token_logps
    log_importance_weights = _compute_log_importance_weights(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_per_token_logps,
        completion_mask=completion_mask,
        importance_sampling_level=importance_sampling_level,
    )

    coef_1 = jnp.exp(log_importance_weights)
    if loss_type == "cispo":
        clamped_ratios = jax.lax.stop_gradient(jnp.minimum(coef_1, 1.0 + epsilon_high))
        return -clamped_ratios * advantages * per_token_logps, coef_1
    if loss_type == "sapo":
        sapo_temperature = jnp.where(advantages > 0, sapo_temperature_pos, sapo_temperature_neg)
        sapo_multiplier = jax.nn.sigmoid(sapo_temperature * (coef_1 - 1.0)) * 4.0 / sapo_temperature
        return -sapo_multiplier * advantages, coef_1
    if loss_type == "vespo":
        phi_seq = _vespo_gamma_weights(
            advantages=advantages,
            log_ratio_per_token=log_ratio,
            mask=completion_mask,
            importance_sampling_ratio=importance_sampling_ratio,
            k_pos=vespo_k_pos,
            lambda_pos=vespo_lambda_pos,
            k_neg=vespo_k_neg,
            lambda_neg=vespo_lambda_neg,
        )
        return -jax.lax.stop_gradient(phi_seq) * advantages * per_token_logps, coef_1
    if loss_type == "gmpo":
        gmpo_mask = completion_mask if entropy_mask is None else completion_mask * entropy_mask
        sequence_ratio = _gmpo_sequence_ratio(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            completion_mask=gmpo_mask,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            delta=delta,
            token_is_weights=(
                importance_sampling_ratio if _is_token_level_ratio(importance_sampling_ratio, per_token_logps) else None
            ),
        )
        return jnp.broadcast_to(-advantages * sequence_ratio, per_token_logps.shape), coef_1
    if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
        coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon_high)
        if delta is not None:
            coef_1 = jnp.minimum(coef_1, delta)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        return -jnp.minimum(per_token_loss1, per_token_loss2), coef_1
    raise ValueError(f"Unknown loss type: {loss_type}")


def _vespo_gamma_weights(
    *,
    advantages: jax.Array,
    log_ratio_per_token: jax.Array,
    mask: jax.Array,
    importance_sampling_ratio: jax.Array | None,
    k_pos: float = 2.0,
    lambda_pos: float = 3.0,
    k_neg: float = 3.0,
    lambda_neg: float = 2.0,
) -> jax.Array:
    """Compute VESPO sequence weights from clipped policy/reference ratios.

    Positive and negative advantages use separate ``k`` and ``lambda`` shaping
    parameters. The result is finite by construction and can be multiplied into
    the per-token GRPO-style objective under JIT.
    """
    lower_clamp = math.log(1e-8)
    log_ratio_clamped = jnp.clip(log_ratio_per_token, -20.0, 20.0)
    seq_log_ratio = jnp.sum(log_ratio_clamped * mask, axis=-1, keepdims=True)
    if importance_sampling_ratio is not None:
        log_is_ratio = jnp.clip(jnp.log(jnp.maximum(importance_sampling_ratio, 1e-8)), lower_clamp, 20.0)
        if log_is_ratio.shape[-1] == 1:
            has_active = jnp.sum(mask, axis=-1, keepdims=True) > 0
            seq_log_ratio = seq_log_ratio + jnp.where(has_active, log_is_ratio, 0.0)
        else:
            seq_log_ratio = seq_log_ratio + jnp.sum(jnp.where(mask > 0, log_is_ratio, 0.0), axis=-1, keepdims=True)

    log_w_seq = jnp.clip(seq_log_ratio, lower_clamp, 20.0)
    w_seq = jnp.exp(log_w_seq)
    is_nonnegative_advantage = advantages >= 0
    k_seq = jnp.where(is_nonnegative_advantage, k_pos, k_neg)
    lambda_seq = jnp.maximum(jnp.where(is_nonnegative_advantage, lambda_pos, lambda_neg), 1e-4)
    log_phi = lambda_seq + k_seq * log_w_seq - lambda_seq * w_seq
    return jnp.nan_to_num(jnp.exp(log_phi), nan=0.0, posinf=0.0, neginf=0.0)


def get_per_token_logps(
    model,
    input_ids,
    attention_mask,
    prompt_length,
    model_kwargs=None,
    logprob_vocab_chunk_size: int | None = None,
    vocab_shard_stage: int | None = None,
):
    """Compute per-token log probabilities for generated sequences.

    This function extracts log probabilities for each token in the completion
    portion of the sequence (after the prompt). It's used to compute likelihood
    ratios between policy and reference models for GRPO training.

    Args:
        model: The language model (EasyDeLBaseModule) to compute log probabilities.
        input_ids: Input token IDs including prompt and completion.
            Shape: [batch_size, seq_len]
        attention_mask: Binary mask indicating valid tokens (1) vs padding (0).
            Shape: [batch_size, seq_len]
        prompt_length: Number of tokens in the prompt portion. Log probabilities
            are only computed for tokens after this position.
        model_kwargs: Optional dictionary of extra model inputs (e.g. multimodal
            tensors like ``pixel_values`` or ``inputs_embeds``). Defaults to None.
        logprob_vocab_chunk_size: When set to a positive value, the log-softmax over
            the vocabulary is computed in chunks of this size to reduce peak
            memory. ``None`` disables vocabulary chunking and computes the
            full softmax in one pass.

    Returns:
        Array: Per-token log probabilities for the completion portion.
            Shape: [batch_size, seq_len - prompt_length]

    Note:
        The function shifts logits by one position to align with the autoregressive
        nature of language models, where each position predicts the next token.
        When the model's ``lmhead_chunksize`` is configured, the forward pass
        is run with ``apply_lm_head=False`` and log probabilities are computed
        directly from hidden states in a chunked fashion, avoiding
        materialization of the full logit tensor.
    """

    model_kwargs = compact_generation_model_kwargs(
        normalize_generation_model_kwargs(model_kwargs, model_callable=getattr(model, "forward", model)),
    )
    model_kwargs = prepare_generation_model_kwargs_for_call(
        model_kwargs,
        target_sequence_length=input_ids.shape[-1],
        prompt_length=prompt_length,
    )
    model_kwargs = _maybe_extend_inputs_embeds_for_scoring(
        model,
        input_ids,
        model_kwargs,
        prompt_length=prompt_length,
    )
    call_kwargs = {
        "attention_mask": attention_mask,
        **model_kwargs,
    }
    if model_kwargs.get("inputs_embeds", None) is None:
        call_kwargs["input_ids"] = input_ids
    lmhead_chunksize = resolve_lmhead_chunksize(model)
    if lmhead_chunksize is not None:
        call_kwargs["apply_lm_head"] = False
    outputs = model(**call_kwargs)
    targets = input_ids[:, prompt_length:]
    if outputs.logits is None and lmhead_chunksize is not None:
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise TypeError(
                f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
            )
        hidden_states = hidden_states[:, prompt_length - 1 : -1, :]
        token_log_probs, _ = compute_per_token_logps_and_entropies_from_hidden_states(
            model,
            hidden_states,
            targets,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
            return_entropy=False,
            vocab_shard_stage=vocab_shard_stage,
        )
        return token_log_probs
    logits = outputs.logits
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits.")
    logits = logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]
    token_log_probs, _ = compute_token_logps_and_entropies_chunked(
        logits,
        targets,
        return_entropy=False,
        chunk_size=logprob_vocab_chunk_size,
    )
    return token_log_probs


_PREFIX_SHARED_UNSUPPORTED_MODEL_TYPES = frozenset(
    {
        "mamba",
        "mamba2",
        "falcon_mamba",
        "falcon_h1",
        "rwkv",
        "mosaic_mpt",
        "mpt",
        "bloom",
    }
)
_PREFIX_SHARED_DROPOUT_ATTRS = (
    "attention_dropout",
    "attn_pdrop",
    "resid_pdrop",
    "embd_pdrop",
    "hidden_dropout",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
)


def validate_prefix_shared_model(model) -> None:
    """Raise when ``model`` cannot run the prefix-shared group forward.

    Args:
        model: The policy module (exposes ``config``).

    Raises:
        NotImplementedError: For an unverified attention operation, quantized KV, dropout,
            pipeline parallelism, sliding-window layers, unsupported SSM/ALiBi architectures,
            or layer types other than full attention (plus gated-delta ``linear_attention``
            for model types in ``PREFIX_SHARED_RECURRENT_MODEL_TYPES``).
    """
    config = getattr(model, "config", None)
    if config is None:
        raise NotImplementedError("Prefix-shared forward requires a model exposing `config`.")
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    mechanism = str(getattr(config, "attn_mechanism", ""))
    if mechanism not in PREFIX_SHARED_ATTENTION_MECHANISMS:
        raise NotImplementedError(
            f"Prefix-shared forward supports attn_mechanism in {sorted(PREFIX_SHARED_ATTENTION_MECHANISMS)}; "
            f"got {mechanism!r}."
        )
    if getattr(config, "kv_cache_quantization_config", None) is not None:
        raise NotImplementedError("Prefix-shared forward does not support a quantized KV cache.")
    for attr in _PREFIX_SHARED_DROPOUT_ATTRS:
        if float(getattr(text_config, attr, 0.0) or 0.0) > 0.0:
            raise NotImplementedError(f"Prefix-shared forward requires dropout to be disabled (`{attr}` == 0).")
    attn_config = getattr(text_config, "attn_config", None)
    if attn_config is not None and float(getattr(attn_config, "attn_pdrop", 0.0) or 0.0) > 0.0:
        raise NotImplementedError(
            "Prefix-shared forward requires dropout to be disabled (`attn_config.attn_pdrop` == 0)."
        )

    mesh = getattr(model, "mesh", None)
    mesh_shape = getattr(mesh, "shape", None) if mesh is not None else None
    if isinstance(mesh_shape, collections.abc.Mapping) and int(mesh_shape.get("pp", 1)) > 1:
        raise NotImplementedError("Prefix-shared forward does not support pipeline parallelism (pp > 1).")
    axis_dims = getattr(config, "sharding_axis_dims", None)
    if axis_dims is not None:
        pp_dim: int | None = None
        other_prod = 1
        if isinstance(axis_dims, collections.abc.Mapping):
            raw_pp = axis_dims.get("pp")
            if raw_pp is not None:
                pp_dim = int(raw_pp)
                for axis_name, dim_val in axis_dims.items():
                    if axis_name != "pp" and int(dim_val) > 0:
                        other_prod *= int(dim_val)
        elif len(axis_dims) > 0:
            axis_names = getattr(config, "sharding_axis_names", None)
            if axis_names is None:
                pp_idx: int | None = 0
            elif "pp" in axis_names:
                pp_idx = list(axis_names).index("pp")
            else:
                pp_idx = None
            if pp_idx is not None and pp_idx < len(axis_dims):
                pp_dim = int(axis_dims[pp_idx])
                for idx, dim_val in enumerate(axis_dims):
                    if idx != pp_idx and int(dim_val) > 0:
                        other_prod *= int(dim_val)
        if pp_dim is not None:
            if pp_dim > 1:
                raise NotImplementedError("Prefix-shared forward does not support pipeline parallelism (pp > 1).")
            if pp_dim == -1 and mesh_shape is None and jax.device_count() // max(other_prod, 1) > 1:
                raise NotImplementedError("Prefix-shared forward does not support pipeline parallelism (pp > 1).")

    if getattr(text_config, "use_sliding_window", False):
        raise NotImplementedError("Prefix-shared forward does not support sliding-window attention.")
    model_type = str(getattr(text_config, "model_type", "") or "").lower()
    if model_type in _PREFIX_SHARED_UNSUPPORTED_MODEL_TYPES:
        raise NotImplementedError(
            f"Prefix-shared forward does not support model_type {model_type!r} "
            "(recurrent/SSM and ALiBi architectures are unsupported)."
        )
    if getattr(text_config, "alibi", False) or (attn_config is not None and getattr(attn_config, "alibi", False)):
        raise NotImplementedError("Prefix-shared forward does not support ALiBi positional bias.")
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        allowed = {"full_attention"}
        if model_type in PREFIX_SHARED_RECURRENT_MODEL_TYPES:
            allowed.add("linear_attention")
        kinds = set(map(str, layer_types))
        if not kinds <= allowed:
            raise NotImplementedError(
                "Prefix-shared forward supports full-attention layers, plus gated-delta `linear_attention` layers "
                f"for model types {sorted(PREFIX_SHARED_RECURRENT_MODEL_TYPES)}; got model_type {model_type!r} "
                f"with layer_types {sorted(kinds)}."
            )


def _prefix_shared_linear_layers(model, num_layers: int) -> list[bool]:
    """Per layer, whether it is a gated-delta ``linear_attention`` layer (recurrent prefix views)."""
    layer_types = getattr(model.config.get_text_config(), "layer_types", None)
    if layer_types is None:
        return [False] * num_layers
    return [str(layer_type) == "linear_attention" for layer_type in layer_types]


def get_per_token_logps_prefix_shared(
    model,
    prompt_ids,
    prompt_mask,
    completion_ids,
    completion_attention_mask,
    num_generations: int,
    model_kwargs=None,
    logprob_vocab_chunk_size: int | None = None,
):
    """Per-token completion log-probabilities with each prompt run once per group.

    A capture pass runs the left-padded prompts at batch ``B`` and records every full-attention layer's
    post-RoPE key/value states and every gated-delta linear-attention layer's end-of-prompt conv window
    and recurrent state; a fork pass runs the completions at batch ``B * G`` against the gathered prefix
    or starting from the gathered state (row ``r`` reads prompt ``r // G``). The prompt's last column
    (``P - 1``) predicts completion token 0.
    Equal, up to float rounding, to :func:`get_per_token_logps` on ``concat([prompt.repeat(G), completion])``;
    gradients reach the prompt activations through the gather.

    Args:
        model: Policy module.
        prompt_ids: ``[B, P]`` left-padded prompt ids; every prompt must have a valid last column. With
            linear-attention layers ``P`` and ``C`` must both be at least 2.
        prompt_mask: ``[B, P]`` prompt validity mask.
        completion_ids: ``[B * G, C]`` completion ids, grouped by prompt.
        completion_attention_mask: ``[B * G, C]`` completion attention mask (right padded).
        num_generations: ``G``.
        model_kwargs: Must be empty: prompt-aligned extra inputs (e.g. multimodal) are not supported.
        logprob_vocab_chunk_size: Optional vocabulary chunk for the log-softmax.

    Returns:
        ``[B * G, C]`` per-token log-probabilities of ``completion_ids``.

    Raises:
        NotImplementedError: For unsupported models or extra model inputs.
        ValueError: When the completion batch is not ``B * G``.
    """
    validate_prefix_shared_model(model)
    if model_kwargs and any(value is not None for value in dict(model_kwargs).values()):
        raise NotImplementedError(
            f"Prefix-shared forward does not support prompt-aligned model inputs: {sorted(model_kwargs)}."
        )
    prompt_batch, prefix_len = prompt_ids.shape
    fork_batch, completion_len = completion_ids.shape
    if fork_batch != prompt_batch * num_generations:
        raise ValueError(
            f"completion batch {fork_batch} must equal prompt batch {prompt_batch} x num_generations {num_generations}."
        )
    num_layers = int(model.config.get_text_config().num_hidden_layers)
    linear_layers = _prefix_shared_linear_layers(model, num_layers)
    if any(linear_layers) and (prefix_len < 2 or completion_len < 2):
        raise NotImplementedError(
            "Prefix-shared forward with linear-attention layers requires prompt and completion lengths >= 2 "
            f"(single-token gated-delta steps are not supported); got {prefix_len} and {completion_len}."
        )

    capture_views = [PrefixRecurrentCaptureView() if linear else PrefixCaptureView() for linear in linear_layers]
    capture = model(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        past_key_values=TransformerCache(views=capture_views),
        mode=common_types.MODE_TRAIN,
        apply_lm_head=False,
    )
    prompt_valid = prompt_mask.astype(jnp.bool_)
    prefix_segment_ids = jnp.where(prompt_valid, 0, -1).astype(jnp.int32)
    fork_views = []
    for view in capture.past_key_values.views:
        if isinstance(view, PrefixRecurrentCaptureView):
            if view.conv_state is None or view.recurrent_state is None:
                raise RuntimeError("A linear-attention layer did not record its prefix state.")
            fork_views.append(
                PrefixRecurrentForkView(
                    prefix_conv_state=view.conv_state,
                    prefix_recurrent_state=view.recurrent_state,
                    num_generations=num_generations,
                )
            )
        else:
            fork_views.append(
                PrefixForkView(
                    prefix_key=view.key,
                    prefix_value=view.value,
                    prefix_segment_ids=prefix_segment_ids,
                    num_generations=num_generations,
                )
            )
    rows = jnp.arange(fork_batch, dtype=jnp.int32) // num_generations
    # RoPE positions continue from each prompt's valid length, as in the concatenated sequence.
    prompt_lengths = jnp.sum(prompt_valid, axis=1).astype(jnp.int32)
    position_ids = jnp.take(prompt_lengths, rows)[:, None] + jnp.arange(completion_len, dtype=jnp.int32)[None, :]
    fork = model(
        input_ids=completion_ids,
        attention_mask=completion_attention_mask,
        position_ids=position_ids,
        past_key_values=TransformerCache(views=fork_views),
        mode=common_types.MODE_TRAIN,
        apply_lm_head=False,
    )
    last_prompt_hidden = jnp.take(capture.last_hidden_state[:, prefix_len - 1 : prefix_len, :], rows, axis=0)
    hidden_states = jnp.concatenate([last_prompt_hidden, fork.last_hidden_state[:, :-1, :]], axis=1)
    lmhead_chunksize = resolve_lmhead_chunksize(model)
    if lmhead_chunksize is not None:
        token_log_probs, _ = compute_per_token_logps_and_entropies_from_hidden_states(
            model,
            hidden_states,
            completion_ids,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
            return_entropy=False,
        )
        return token_log_probs
    logits = model.compute_lm_logits(model.prepare_lm_head_inputs(hidden_states))
    token_log_probs, _ = compute_token_logps_and_entropies_chunked(
        logits,
        completion_ids,
        return_entropy=False,
        chunk_size=logprob_vocab_chunk_size,
    )
    return token_log_probs


def get_per_token_logps_and_selected_logps(
    model,
    input_ids,
    attention_mask,
    prompt_length,
    selected_indices,
    model_kwargs=None,
    logprob_vocab_chunk_size: int | None = None,
):
    """Return generated-token log-probs plus log-probs at selected vocab indices.

    ``selected_indices`` is shaped ``[batch, completion_len, k]`` and is
    interpreted as the fixed support from the sampling policy. The returned
    selected log-probs are current-policy log-probs gathered at exactly those
    indices, so top-k trust-region losses compare like-for-like supports.
    """
    model_kwargs = compact_generation_model_kwargs(
        normalize_generation_model_kwargs(model_kwargs, model_callable=getattr(model, "forward", model)),
    )
    model_kwargs = prepare_generation_model_kwargs_for_call(
        model_kwargs,
        target_sequence_length=input_ids.shape[-1],
        prompt_length=prompt_length,
    )
    model_kwargs = _maybe_extend_inputs_embeds_for_scoring(
        model,
        input_ids,
        model_kwargs,
        prompt_length=prompt_length,
    )
    call_kwargs = {
        "attention_mask": attention_mask,
        **model_kwargs,
    }
    if model_kwargs.get("inputs_embeds", None) is None:
        call_kwargs["input_ids"] = input_ids
    outputs = model(**call_kwargs)
    logits = outputs.logits
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits; top-k scoring requires full logits.")
    logits = logits[:, prompt_length - 1 : -1, :]
    targets = input_ids[:, prompt_length:]
    # Gather targets and selected support against the same directly computed
    # normalizer. The singleton support axis broadcasts only the gathers, not
    # the vocabulary tensor. Recovering log_z from target_logit - target_logp
    # loses precision for very negative targets and is NaN for masked targets.
    scoring_indices = jnp.concatenate((targets[..., None], selected_indices), axis=-1)
    support_logps, _ = compute_token_logps_and_entropies_chunked(
        logits[..., None, :], scoring_indices, return_entropy=False, chunk_size=logprob_vocab_chunk_size
    )
    return support_logps[..., 0].astype(logits.dtype), support_logps[..., 1:].astype(logits.dtype)


def get_per_token_logps_and_topk(
    model,
    input_ids,
    attention_mask,
    prompt_length,
    topk: int,
    model_kwargs=None,
):
    """Return generated-token log-probs and top-k log-probs for each position.

    This is used during DPPO preprocessing to snapshot the generation-time
    policy distribution over a compact support. The support is then carried in
    the batch so the train step can compare the current policy against the same
    tokens.
    """
    model_kwargs = compact_generation_model_kwargs(
        normalize_generation_model_kwargs(model_kwargs, model_callable=getattr(model, "forward", model)),
    )
    model_kwargs = prepare_generation_model_kwargs_for_call(
        model_kwargs,
        target_sequence_length=input_ids.shape[-1],
        prompt_length=prompt_length,
    )
    model_kwargs = _maybe_extend_inputs_embeds_for_scoring(
        model,
        input_ids,
        model_kwargs,
        prompt_length=prompt_length,
    )
    call_kwargs = {
        "attention_mask": attention_mask,
        **model_kwargs,
    }
    if model_kwargs.get("inputs_embeds", None) is None:
        call_kwargs["input_ids"] = input_ids
    outputs = model(**call_kwargs)
    logits = outputs.logits
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits; top-k scoring requires full logits.")
    logits = logits[:, prompt_length - 1 : -1, :]
    targets = input_ids[:, prompt_length:]
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    topk_logps, topk_indices = jax.lax.top_k(log_probs, int(topk))
    return token_log_probs.astype(logits.dtype), topk_indices, topk_logps.astype(logits.dtype)


def compute_per_token_logps(logits, input_ids, prompt_length):
    """Compute per-token log probabilities in a vectorized way.

    Converts raw logits to log-softmax probabilities, then gathers the
    log probability corresponding to each actual target token in the
    completion portion of the sequence.

    Args:
        logits: Pre-trimmed logits of shape ``[batch_size, seq_len, vocab_size]``.
        input_ids: Full input token IDs of shape ``[batch_size, seq_len]``.
        prompt_length: Number of prompt tokens. Targets are extracted from
            ``input_ids[:, prompt_length:]``.

    Returns:
        jax.Array: Per-token log probabilities for the completion portion,
            shape ``[batch_size, completion_len]``.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_ids = input_ids[:, prompt_length:]
    token_log_probs = jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(target_ids, axis=-1),
        axis=-1,
    )
    token_log_probs = jnp.squeeze(token_log_probs, axis=-1)
    return token_log_probs


def get_per_token_logps_and_entropies(
    model,
    input_ids,
    attention_mask,
    prompt_length,
    model_kwargs=None,
    logprob_vocab_chunk_size: int | None = None,
    vocab_shard_stage: int | None = None,
):
    """Compute per-token log probabilities and entropy for the completion portion.

    Similar to ``get_per_token_logps``, but also returns the per-token entropy
    of the predicted distribution. Entropy is used by GRPO variants that apply
    entropy-based filtering (e.g., top-entropy quantile masking).

    Args:
        model: The language model to run the forward pass on.
        input_ids: Input token IDs including prompt and completion.
            Shape: ``[batch_size, seq_len]``.
        attention_mask: Binary mask indicating valid tokens (1) vs padding (0).
            Shape: ``[batch_size, seq_len]``.
        prompt_length: Number of tokens in the prompt. Log probabilities and
            entropies are computed only for tokens after this position.
        model_kwargs: Optional dictionary of extra model inputs (e.g. multimodal
            tensors like ``pixel_values`` or ``inputs_embeds``). Defaults to None.
        logprob_vocab_chunk_size: When set to a positive value, the log-softmax and
            entropy computations over the vocabulary are performed in chunks
            of this size to reduce peak memory usage. ``None`` disables
            vocabulary chunking and computes the full softmax in a single pass.

    Returns:
        tuple[jax.Array, jax.Array]: A pair of arrays:
            - Per-token log probabilities, shape ``[batch_size, completion_len]``.
            - Per-token entropy of the predicted distribution, same shape.

    Note:
        When the model's ``lmhead_chunksize`` is configured, the forward
        pass is run with ``apply_lm_head=False`` and both log probabilities
        and entropies are computed directly from hidden states in a chunked
        fashion, avoiding materialization of the full logit tensor.
    """
    model_kwargs = compact_generation_model_kwargs(
        normalize_generation_model_kwargs(model_kwargs, model_callable=getattr(model, "forward", model)),
    )
    model_kwargs = prepare_generation_model_kwargs_for_call(
        model_kwargs,
        target_sequence_length=input_ids.shape[-1],
        prompt_length=prompt_length,
    )
    model_kwargs = _maybe_extend_inputs_embeds_for_scoring(
        model,
        input_ids,
        model_kwargs,
        prompt_length=prompt_length,
    )
    call_kwargs = {
        "attention_mask": attention_mask,
        **model_kwargs,
    }
    if model_kwargs.get("inputs_embeds", None) is None:
        call_kwargs["input_ids"] = input_ids
    lmhead_chunksize = resolve_lmhead_chunksize(model)
    if lmhead_chunksize is not None:
        call_kwargs["apply_lm_head"] = False
    outputs = model(**call_kwargs)
    targets = input_ids[:, prompt_length:]
    if outputs.logits is None and lmhead_chunksize is not None:
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise TypeError(
                f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
            )
        hidden_states = hidden_states[:, prompt_length - 1 : -1, :]
        token_log_probs, entropies = compute_per_token_logps_and_entropies_from_hidden_states(
            model,
            hidden_states,
            targets,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
            return_entropy=True,
            vocab_shard_stage=vocab_shard_stage,
        )
        return token_log_probs, entropies
    logits = outputs.logits
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits.")
    logits = logits[:, prompt_length - 1 :]
    logits = logits[:, :-1, :]
    token_log_probs, entropies = compute_token_logps_and_entropies_chunked(
        logits,
        targets,
        return_entropy=True,
        chunk_size=logprob_vocab_chunk_size,
    )
    return token_log_probs, entropies


def _maybe_extend_inputs_embeds_for_scoring(
    model,
    input_ids,
    model_kwargs,
    *,
    prompt_length: int,
):
    """Pad ``inputs_embeds`` to cover the completion when scoring multimodal prompts.

    GRPO generation may carry ``inputs_embeds`` only for the prompt
    (e.g. when the prompt embeds visual tokens). When the policy is
    later scored on the prompt+completion sequence we need a matching
    embedding tensor; this helper concatenates the model's own token
    embeddings for the completion ids onto the existing prompt
    embeddings. If ``inputs_embeds`` is absent or already full-length
    the kwargs are returned untouched.

    Args:
        model: Bound module exposing :meth:`compute_embedding` for the
            completion-side ids.
        input_ids: Full ``[batch, prompt + completion]`` id array used
            to derive the completion ids.
        model_kwargs: Generation kwargs that may carry ``inputs_embeds``.
        prompt_length: Number of prompt tokens at the start of
            ``input_ids``; used to slice the completion segment.

    Returns:
        A possibly-updated ``model_kwargs`` dict with ``inputs_embeds``
        extended to length ``input_ids.shape[-1]``.

    Raises:
        ValueError: If ``inputs_embeds`` has a length that is neither
            the prompt length nor the target full length, or if the
            model returns embeddings with a different rank than the
            existing prompt embeddings.
    """

    inputs_embeds = model_kwargs.get("inputs_embeds", None)
    if inputs_embeds is None:
        return model_kwargs

    current_length = int(inputs_embeds.shape[-2])
    target_length = int(input_ids.shape[-1])
    if current_length == target_length:
        return model_kwargs
    if current_length != int(prompt_length):
        raise ValueError(
            "GRPO scoring with `inputs_embeds` requires either full-sequence embeddings "
            f"or prompt-length embeddings. Got sequence axis {current_length} for target length {target_length}."
        )

    completion_input_ids = input_ids[:, prompt_length:target_length]
    if completion_input_ids.shape[-1] == 0:
        return model_kwargs

    completion_embeds = model.compute_embedding(completion_input_ids)
    if completion_embeds.ndim != inputs_embeds.ndim:
        raise ValueError(
            "Model `compute_embedding` returned embeddings with an unexpected rank for GRPO scoring: "
            f"{completion_embeds.ndim} vs prompt embeddings rank {inputs_embeds.ndim}."
        )
    if completion_embeds.dtype != inputs_embeds.dtype:
        completion_embeds = completion_embeds.astype(inputs_embeds.dtype)

    updated_model_kwargs = dict(model_kwargs)
    updated_model_kwargs["inputs_embeds"] = jnp.concatenate(
        [inputs_embeds, completion_embeds],
        axis=-2,
    )
    return updated_model_kwargs


def _expand_grpo_prompt_batch(
    batch: collections.abc.Mapping[str, jax.Array],
    *,
    model_callable: tp.Callable[..., tp.Any] | None = None,
) -> dict[str, jax.Array]:
    """Align prompt-shaped leaves with completion rows before microbatch slicing.

    ``GRPOTrainer`` emits ``prompt_ids``/``prompt_mask`` (and prompt-aligned
    multimodal kwargs) with leading dim ``B`` while completion-level tensors
    have leading dim ``B * G``. Expanding prompt leaves up front ensures
    ``minibatch_call`` and MPMD microbatching slice every leaf consistently and
    pair each completion with its true prompt.
    """
    expanded = dict(batch)
    prompt_ids = expanded.get("prompt_ids")
    completion_ids = expanded.get("completion_ids")
    if prompt_ids is None or completion_ids is None:
        return expanded
    prompt_rows = int(prompt_ids.shape[0])
    completion_rows = int(completion_ids.shape[0])
    if prompt_rows == completion_rows:
        return expanded
    if prompt_rows <= 0 or completion_rows % prompt_rows != 0:
        raise ValueError(
            f"GRPO prompt rows ({prompt_rows}) must match or evenly divide completion rows ({completion_rows})."
        )
    effective_num_generations = completion_rows // prompt_rows
    expanded["prompt_ids"] = prompt_ids.repeat(effective_num_generations, axis=0)
    if "prompt_mask" in expanded and expanded["prompt_mask"] is not None:
        expanded["prompt_mask"] = expanded["prompt_mask"].repeat(effective_num_generations, axis=0)
    prompt_model_kwargs = extract_generation_model_kwargs(expanded, model_callable=model_callable)
    if prompt_model_kwargs:
        repeated_kwargs = repeat_prompt_aligned_model_kwargs(
            prompt_model_kwargs,
            effective_num_generations,
            prompt_batch_size=prompt_rows,
        )
        expanded.update(repeated_kwargs)
    return expanded


def grpo_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    num_generations: int,
    beta: float,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    loss_type: str = "dapo",
    epsilon: float = 0.2,
    epsilon_high: float = 0.2,
    delta: float | None = None,
    importance_sampling_level: str = "token",
    top_entropy_quantile: float = 1.0,
    completion_chunk_size: int | None = None,
    max_loss_completion_tokens: int | None = None,
    logprob_vocab_chunk_size: int | None = None,
    sapo_temperature_pos: float = 1.0,
    sapo_temperature_neg: float = 1.05,
    vespo_k_pos: float = 2.0,
    vespo_lambda_pos: float = 3.0,
    vespo_k_neg: float = 3.0,
    vespo_lambda_neg: float = 2.0,
    off_policy_mask_threshold: float | None = None,
    use_bias_correction_kl: bool = False,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
    dppo_divergence_type: str | None = None,
    dppo_clip_ratio_c: float = 20.0,
    kl_estimator: str = "k3",
    off_policy_k3_threshold: float | None = None,
    off_policy_min_acceptance: float | None = None,
    startup_behavior_k3_max: float | None = None,
    startup_behavior_tail_fraction_max: float = 0.01,
    prefix_shared_forward: bool = False,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Perform a single GRPO training or evaluation step.

    With ``prefix_shared_forward=True`` the non-chunked policy forward runs each prompt once per group
    (see :func:`get_per_token_logps_prefix_shared`); the chunked path, top-k DPPO divergences and
    entropy masking raise ``NotImplementedError`` under that flag.

    Computes the group-relative policy optimization loss on a batch of
    pre-processed data (prompt IDs, completion IDs, advantages, and
    optionally reference log-probs). During training the function also
    computes gradients via minibatch accumulation and updates the model state.

    The function supports several loss variants controlled by ``loss_type``:
        - ``"grpo"``: Standard GRPO with per-sequence normalization.
        - ``"bnpo"``: Batch-normalized policy optimization.
        - ``"dr_grpo"``: Denominator-regularized GRPO.
        - ``"dapo"``: Dynamic advantage policy optimization (default).
        - ``"cispo"``: Clipped importance-sampling policy optimization.
        - ``"sapo"``: Soft advantage policy optimization.
        - ``"luspo"``: Length-unbiased sequence policy optimization.
        - ``"vespo"``: Variational sequence-level soft policy optimization.
        - ``"gmpo"``: Geometric-mean policy optimization (sequence-level
          geometric mean of sign-aware clipped token ratios).

    Args:
        state: Current model state including parameters and optimizer.
        batch: Mapping containing at minimum ``prompt_ids``, ``prompt_mask``,
            ``completion_ids``, ``completion_mask``, ``advantages``, and
            optionally ``ref_per_token_logps`` and ``old_per_token_logps``.
        num_generations: Number of completions generated per prompt.
        beta: KL divergence penalty coefficient. Set to 0.0 to disable.
        loss_config: Optional loss configuration for gradient clipping etc.
        learning_rate_fn: Learning rate schedule function.
        partition_spec: Sharding specification for the batch.
        gradient_accumulation_steps: Number of minibatch accumulation steps.
        is_training: If True, compute and apply gradients. If False, only
            compute metrics (evaluation mode).
        loss_type: Which loss variant to use (see above).
        epsilon: Lower clipping bound for importance-sampling ratios.
        epsilon_high: Upper clipping bound for importance-sampling ratios.
        delta: Optional upper cap on un-clipped importance weights.
        importance_sampling_level: ``"token"`` for per-token or ``"sequence"``
            for per-sequence importance weighting.
        top_entropy_quantile: Fraction of highest-entropy tokens to keep in
            the loss. 1.0 disables filtering.
        completion_chunk_size: Chunk size for memory-saving chunked completion
            loss. Set to ``None`` to disable chunking.
        max_loss_completion_tokens: Optional cap on completion tokens used by
            the GRPO loss. Set to ``None`` to disable truncation.
        sapo_temperature_pos: Soft-clipping temperature for positive SAPO advantages.
        sapo_temperature_neg: Soft-clipping temperature for negative SAPO advantages.
        vespo_k_pos: Gamma-weight exponent for positive VESPO advantages.
        vespo_lambda_pos: Gamma-weight decay for positive VESPO advantages.
        vespo_k_neg: Gamma-weight exponent for negative VESPO advantages.
        vespo_lambda_neg: Gamma-weight decay for negative VESPO advantages.
        off_policy_mask_threshold: Optional sequence-level forward-KL
            threshold for masking high-drift negative-advantage rows from
            the policy objective.
        use_bias_correction_kl: If True, multiply the KL penalty by the
            un-clipped importance weights.
        straight_through_emulator: Optional function for quantization-aware
            straight-through gradient estimation.
        dppo_divergence_type: DPPO trust-region divergence; ``None`` for
            non-DPPO losses.
        dppo_clip_ratio_c: Upper cap on DPPO importance ratios.
        kl_estimator: Per-token KL estimator for the ``beta`` penalty:
            ``"k1"``, ``"k2"``, ``"k3"`` (default), or ``"l2"``.
        off_policy_k3_threshold: Optional per-sequence mean behaviour-K3 bound. When the batch
            carries ``behavior_per_token_logps`` (rollout-engine sampling log-probs), sequences
            above the bound, or with a non-finite behaviour log ratio, have their loss zeroed
            while every normaliser keeps its pre-rejection token count. Non-finite rows are
            rejected even when this is ``None``.
        off_policy_min_acceptance: Optional minimum ``accepted / eligible`` row fraction. Below
            it the step leaves parameters and optimizer state unchanged and reports
            ``off_policy/step_skipped = 1``. A batch without behaviour logprobs is skipped.
        startup_behavior_k3_max: Optional bound on the token-mean behaviour K3 for steps whose
            batch carries ``behavior_parity_gate > 0`` (the trainer's first step); a violation
            skips the update.
        startup_behavior_tail_fraction_max: Bound on the fraction of tokens with
            ``|log ratio| > 0.25`` for the same startup gate.

    Returns:
        tuple[EasyDeLState, LossMetrics] | LossMetrics: When ``is_training``
            is True, returns the updated state and loss metrics. When False,
            returns only the loss metrics.
    """
    scope_root = "easydel/trainer/grpo/" + ("train_step" if is_training else "eval_step")
    effective_ga_steps = max(int(gradient_accumulation_steps), 1) if is_training else 1
    with jax.named_scope(scope_root + "/prepare_batch"):
        state_model = getattr(state, "model", None)
        state_mesh = getattr(state_model, "mesh", None)
        model_callable = getattr(state_model, "forward", state_model) if state_model is not None else None
        prompt_group_size: int | None = None
        if not prefix_shared_forward:
            batch = _expand_grpo_prompt_batch(batch, model_callable=model_callable)
        elif effective_ga_steps > 1:
            prompt_ids_full = batch.get("prompt_ids")
            completion_ids_full = batch.get("completion_ids")
            if prompt_ids_full is not None and completion_ids_full is not None:
                prompt_rows = int(prompt_ids_full.shape[0])
                completion_rows = int(completion_ids_full.shape[0])
                if prompt_rows <= 0 or completion_rows % prompt_rows != 0:
                    raise ValueError(
                        f"GRPO prompt rows ({prompt_rows}) must match or evenly divide "
                        f"completion rows ({completion_rows})."
                    )
                prompt_group_size = completion_rows // prompt_rows
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=effective_ga_steps,
            batch_partition_spec=partition_spec,
        )
        if prefix_shared_forward and effective_ga_steps > 1 and prompt_group_size is not None and prompt_group_size > 1:
            completion_rows = int(batch["completion_ids"].shape[0])
            if _batch_size != completion_rows:
                raise ValueError(
                    f"`prefix_shared_forward=True` with `gradient_accumulation_steps={effective_ga_steps}` "
                    f"inferred batch size {_batch_size}, expected completion rows {completion_rows}."
                )
            if minibatch_size % prompt_group_size != 0:
                prompt_rows = int(batch["prompt_ids"].shape[0])
                raise ValueError(
                    f"`prefix_shared_forward=True` with `gradient_accumulation_steps={effective_ga_steps}` "
                    f"requires whole prompt groups per microbatch: `minibatch_size` ({minibatch_size}) must be "
                    f"divisible by group size ({prompt_group_size}), i.e. prompt batch ({prompt_rows}) must be "
                    f"divisible by `gradient_accumulation_steps` ({effective_ga_steps})."
                )
        batch = with_sharding_constraint(batch, partition_spec, mesh=state_mesh, ignore_mpmd=True)
        # With ``difficulty_weights`` (dataset weights or discarded unscorable groups) every loss type
        # normalises by a weighted count. Under gradient accumulation a minibatch-local count would give a
        # fully or mostly discarded microbatch a full ``1/K`` share of the averaged gradient, so each
        # microbatch divides by the full-batch count instead, scaled by ``1/K`` to undo the uniform average.
        accumulated_weighted_denominator = None
        if is_training and effective_ga_steps > 1 and "difficulty_weights" in batch:
            accumulated_weighted_denominator = jnp.maximum(
                _grpo_weighted_loss_denominator(batch, loss_type, max_loss_completion_tokens), 1.0
            ) / float(effective_ga_steps)

    def loss_fn(tree, minibatch):
        """Compute the GRPO surrogate loss for one minibatch.

        Concatenates prompts and completions, calls the policy forward
        in chunked mode (using ``completion_chunk_size`` /
        ``logprob_vocab_chunk_size`` for memory), masks the
        per-token log-probabilities by the completion mask, applies
        the importance-sampling clip given by ``epsilon`` /
        ``epsilon_high`` / ``delta`` and the configured ``loss_type``
        (``grpo``, ``grpo_token``, ``dr_grpo``, ``bnpo``, ...), folds
        in the KL penalty against the reference policy and the
        optional ``top_entropy_quantile`` mask.

        Args:
            tree: Policy graphstate to differentiate against.
            minibatch: Dict carrying ``prompt_ids``, ``prompt_mask``,
                ``completion_ids``, ``completion_mask``,
                ``advantages``, ``ref_per_token_logps``, and any
                generation-time model kwargs.

        Returns:
            ``(loss, metrics)`` where ``metrics`` is a populated
            :class:`LossMetrics` recording surrogate loss components,
            clip fractions, KL diagnostics, and any straight-through
            quantizer signals.
        """
        if is_training and straight_through_emulator is not None:
            with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                tree = straight_through_emulator(tree)
        with jax.named_scope(scope_root + "/loss_fn/merge_state"):
            module = state.merge(tree)

        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            advantages,
        ) = (
            minibatch["prompt_ids"],
            minibatch["prompt_mask"],
            minibatch["completion_ids"],
            minibatch["completion_mask"],
            minibatch["advantages"],
        )
        completion_attention_mask = minibatch.get("completion_attention_mask", completion_mask)

        completion_was_truncated = False
        if max_loss_completion_tokens is not None and completion_ids.shape[1] > max_loss_completion_tokens:
            completion_ids = completion_ids[:, :max_loss_completion_tokens]
            completion_mask = completion_mask[:, :max_loss_completion_tokens]
            completion_attention_mask = completion_attention_mask[:, :max_loss_completion_tokens]
            completion_was_truncated = True

        # Use runtime batch shapes so filtered-group trainers (e.g. GFPO) can
        # train with a different effective generation count than sampling-time.
        effective_num_generations = completion_ids.shape[0] // max(prompt_ids.shape[0], 1)
        effective_num_generations = max(effective_num_generations, 1)

        input_ids = jnp.concatenate([prompt_ids.repeat(effective_num_generations, 0), completion_ids], axis=1)
        attention_mask = jnp.concatenate(
            [prompt_mask.repeat(effective_num_generations, 0), completion_attention_mask],
            axis=1,
        )
        prompt_len = prompt_ids.shape[-1]
        prompt_model_kwargs = extract_generation_model_kwargs(
            minibatch,
            model_callable=getattr(module, "forward", module),
        )
        completion_model_kwargs = repeat_prompt_aligned_model_kwargs(
            prompt_model_kwargs,
            effective_num_generations,
            prompt_batch_size=prompt_ids.shape[0],
        )

        advantages = minibatch["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        elif advantages.shape[1] != 1 and advantages.shape[1] != completion_ids.shape[1]:
            advantages = advantages[:, : completion_ids.shape[1]]
        has_difficulty_weights = "difficulty_weights" in minibatch
        difficulty_weights = minibatch.get("difficulty_weights")
        if not has_difficulty_weights:
            difficulty_weights = jnp.ones((completion_ids.shape[0], 1), dtype=jnp.float32)
        elif difficulty_weights.ndim == 1:
            difficulty_weights = difficulty_weights[:, None]
        difficulty_weights = difficulty_weights.astype(jnp.float32)

        def weighted_denominator(local_denominator: jax.Array) -> jax.Array:
            """Loss denominator of a weighted batch: global under accumulation, else this minibatch's own."""
            if accumulated_weighted_denominator is not None:
                return accumulated_weighted_denominator
            return jnp.maximum(local_denominator, 1.0)

        old_per_token_logps = minibatch.get("old_per_token_logps")
        if old_per_token_logps is not None and old_per_token_logps.shape[1] != completion_ids.shape[1]:
            old_per_token_logps = old_per_token_logps[:, : completion_ids.shape[1]]
        sampling_topk_indices = minibatch.get("sampling_topk_indices")
        if sampling_topk_indices is not None and sampling_topk_indices.shape[1] != completion_ids.shape[1]:
            sampling_topk_indices = sampling_topk_indices[:, : completion_ids.shape[1]]
        sampling_topk_logps = minibatch.get("sampling_topk_logps")
        if sampling_topk_logps is not None and sampling_topk_logps.shape[1] != completion_ids.shape[1]:
            sampling_topk_logps = sampling_topk_logps[:, : completion_ids.shape[1]]
        completion_token_count = jnp.sum(completion_mask)
        completion_lengths = jnp.sum(completion_mask, axis=1)
        importance_sampling_ratio = minibatch.get("importance_sampling_ratio")
        if importance_sampling_ratio is not None:
            if importance_sampling_ratio.ndim == 1:
                importance_sampling_ratio = importance_sampling_ratio[:, None]
            if importance_sampling_ratio.shape[1] != 1 and importance_sampling_ratio.shape[1] != completion_ids.shape[1]:
                importance_sampling_ratio = importance_sampling_ratio[:, : completion_ids.shape[1]]
        behavior_per_token_logps = minibatch.get("behavior_per_token_logps")
        if behavior_per_token_logps is not None:
            behavior_per_token_logps = behavior_per_token_logps[:, : completion_ids.shape[1]]

        use_chunked_completion_loss = (
            completion_chunk_size is not None
            and completion_ids.shape[0] > completion_chunk_size
            and top_entropy_quantile >= 1.0
        )
        if use_chunked_completion_loss and prefix_shared_forward:
            raise NotImplementedError("prefix_shared_forward does not support the chunked completion loss in v1.")
        if use_chunked_completion_loss:
            expanded_prompt_ids = prompt_ids.repeat(effective_num_generations, 0)
            expanded_prompt_mask = prompt_mask.repeat(effective_num_generations, 0)
            completion_batch_size = int(completion_ids.shape[0])
            normalizer = _global_token_normalizer(
                minibatch.get("num_items_in_batch"),
                completion_token_count,
                completion_was_truncated,
                effective_ga_steps,
                is_training=is_training,
            )

            loss_numerator = jnp.array(0.0, dtype=jnp.float32)
            mean_kl_num = jnp.array(0.0, dtype=jnp.float32)
            mean_kl_den = jnp.array(0.0, dtype=jnp.float32)
            ref_logps_num = jnp.array(0.0, dtype=jnp.float32)
            ref_logps_den = jnp.array(0.0, dtype=jnp.float32)
            low_clip_num = jnp.array(0.0, dtype=jnp.float32)
            low_clip_den = jnp.array(0.0, dtype=jnp.float32)
            high_clip_num = jnp.array(0.0, dtype=jnp.float32)
            high_clip_den = jnp.array(0.0, dtype=jnp.float32)
            region_clip_num = jnp.array(0.0, dtype=jnp.float32)
            region_clip_den = jnp.array(0.0, dtype=jnp.float32)
            cispo_clip_num = jnp.array(0.0, dtype=jnp.float32)
            cispo_clip_den = jnp.array(0.0, dtype=jnp.float32)
            dppo_mask_num = jnp.array(0.0, dtype=jnp.float32)
            dppo_mask_den = jnp.array(0.0, dtype=jnp.float32)
            off_policy_keep_num = jnp.array(0.0, dtype=jnp.float32)
            off_policy_keep_den = jnp.array(0.0, dtype=jnp.float32)
            grpo_weight_den = jnp.sum(difficulty_weights)
            behavior_sums = None

            for start in range(0, completion_batch_size, completion_chunk_size):
                end = min(start + completion_chunk_size, completion_batch_size)
                chunk_completion_ids = completion_ids[start:end]
                chunk_completion_mask = completion_mask[start:end]
                chunk_completion_attention_mask = completion_attention_mask[start:end]
                chunk_prompt_ids = expanded_prompt_ids[start:end]
                chunk_prompt_mask = expanded_prompt_mask[start:end]
                chunk_input_ids = jnp.concatenate([chunk_prompt_ids, chunk_completion_ids], axis=1)
                chunk_attention_mask = jnp.concatenate(
                    [chunk_prompt_mask, chunk_completion_attention_mask],
                    axis=1,
                )
                chunk_model_kwargs = slice_prompt_aligned_model_kwargs(
                    completion_model_kwargs,
                    start,
                    end,
                    prompt_batch_size=completion_batch_size,
                )
                with jax.named_scope(scope_root + "/loss_fn/chunked/policy_logps"):
                    chunk_current_topk_logps = None
                    if dppo_divergence_type in {"topk_tv", "topk_kl"} and sampling_topk_indices is not None:
                        chunk_per_token_logps, chunk_current_topk_logps = get_per_token_logps_and_selected_logps(
                            module,
                            chunk_input_ids,
                            chunk_attention_mask,
                            prompt_len,
                            sampling_topk_indices[start:end],
                            model_kwargs=chunk_model_kwargs,
                            logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                        )
                    else:
                        chunk_per_token_logps = get_per_token_logps(
                            module,
                            chunk_input_ids,
                            chunk_attention_mask,
                            prompt_len,
                            model_kwargs=chunk_model_kwargs,
                            logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                        )
                chunk_old_per_token_logps = (
                    old_per_token_logps[start:end]
                    if old_per_token_logps is not None
                    else jax.lax.stop_gradient(chunk_per_token_logps)
                )
                chunk_ref_per_token_logps = (
                    minibatch["ref_per_token_logps"][start:end, : completion_ids.shape[1]]
                    if beta != 0.0
                    else jnp.zeros_like(chunk_per_token_logps)
                )
                if beta != 0.0:
                    chunk_per_token_kl = _compute_per_token_kl(
                        chunk_per_token_logps,
                        chunk_ref_per_token_logps,
                        kl_estimator,
                        completion_mask=chunk_completion_mask,
                    )
                    if use_bias_correction_kl:
                        chunk_importance_weights = _compute_importance_weights(
                            per_token_logps=chunk_per_token_logps,
                            old_per_token_logps=chunk_old_per_token_logps,
                            completion_mask=chunk_completion_mask,
                            importance_sampling_level=importance_sampling_level,
                        )
                        chunk_per_token_kl = chunk_per_token_kl * chunk_importance_weights
                else:
                    chunk_per_token_kl = jnp.zeros_like(chunk_per_token_logps)
                chunk_advantages = advantages[start:end]
                chunk_difficulty_weights = difficulty_weights[start:end]
                chunk_importance_sampling_ratio = (
                    importance_sampling_ratio[start:end] if importance_sampling_ratio is not None else None
                )
                chunk_behavior_accept = None
                if behavior_per_token_logps is not None:
                    chunk_behavior = behavior_policy_terms(
                        chunk_per_token_logps,
                        behavior_per_token_logps[start:end],
                        chunk_completion_mask,
                        k3_threshold=off_policy_k3_threshold,
                    )
                    behavior_sums = add_behavior_sums(behavior_sums, chunk_behavior.sums)
                    chunk_behavior_accept = chunk_behavior.accept
                    if chunk_importance_sampling_ratio is not None:
                        chunk_importance_sampling_ratio = _sanitize_rollout_ratio(
                            chunk_importance_sampling_ratio, chunk_behavior_accept
                        )

                chunk_divergence_mask = None
                if dppo_divergence_type is not None:
                    chunk_sampling_per_token_logps = minibatch.get("sampling_per_token_logps")
                    if chunk_sampling_per_token_logps is None:
                        chunk_sampling_per_token_logps = chunk_old_per_token_logps
                    else:
                        chunk_sampling_per_token_logps = chunk_sampling_per_token_logps[
                            start:end, : completion_ids.shape[1]
                        ]
                    log_ratio = jnp.where(
                        chunk_completion_mask > 0,
                        chunk_per_token_logps - chunk_sampling_per_token_logps,
                        0.0,
                    )
                    coef_1 = jax.lax.stop_gradient(jnp.exp(jnp.minimum(log_ratio, math.log(dppo_clip_ratio_c))))
                    chunk_divergence_mask = _compute_dppo_divergence_mask(
                        per_token_logps=chunk_per_token_logps,
                        sampling_per_token_logps=chunk_sampling_per_token_logps,
                        advantages=chunk_advantages,
                        completion_mask=chunk_completion_mask,
                        divergence_type=dppo_divergence_type,
                        epsilon_low=epsilon,
                        epsilon_high=epsilon_high,
                        current_topk_logps=chunk_current_topk_logps,
                        sampling_topk_logps=(
                            sampling_topk_logps[start:end] if sampling_topk_logps is not None else None
                        ),
                    )
                    chunk_per_token_loss = -chunk_advantages * coef_1 * chunk_divergence_mask * chunk_per_token_logps
                else:
                    chunk_per_token_loss, coef_1 = _compute_grpo_policy_loss_terms(
                        per_token_logps=chunk_per_token_logps,
                        old_per_token_logps=chunk_old_per_token_logps,
                        advantages=chunk_advantages,
                        completion_mask=chunk_completion_mask,
                        loss_type=loss_type,
                        epsilon=epsilon,
                        epsilon_high=epsilon_high,
                        delta=delta,
                        importance_sampling_level=importance_sampling_level,
                        sapo_temperature_pos=sapo_temperature_pos,
                        sapo_temperature_neg=sapo_temperature_neg,
                        vespo_k_pos=vespo_k_pos,
                        vespo_lambda_pos=vespo_lambda_pos,
                        vespo_k_neg=vespo_k_neg,
                        vespo_lambda_neg=vespo_lambda_neg,
                        importance_sampling_ratio=chunk_importance_sampling_ratio,
                    )

                if _apply_rollout_is_ratio_post_hoc(loss_type, chunk_importance_sampling_ratio, chunk_per_token_logps):
                    chunk_per_token_loss = chunk_per_token_loss * jax.lax.stop_gradient(chunk_importance_sampling_ratio)

                if off_policy_mask_threshold is not None:
                    chunk_sampling_per_token_logps = minibatch.get("sampling_per_token_logps")
                    if chunk_sampling_per_token_logps is None:
                        chunk_sampling_per_token_logps = chunk_old_per_token_logps
                    else:
                        chunk_sampling_per_token_logps = chunk_sampling_per_token_logps[
                            start:end, : completion_ids.shape[1]
                        ]
                    chunk_off_policy_mask = _compute_off_policy_sequence_mask(
                        per_token_logps=chunk_per_token_logps,
                        sampling_per_token_logps=chunk_sampling_per_token_logps,
                        advantages=chunk_advantages,
                        completion_mask=chunk_completion_mask,
                        threshold=off_policy_mask_threshold,
                    )
                    per_chunk_keep_num, per_chunk_keep_den = _masked_sum_and_count(
                        chunk_off_policy_mask,
                        jnp.ones_like(chunk_off_policy_mask),
                    )
                    off_policy_keep_num = off_policy_keep_num + per_chunk_keep_num
                    off_policy_keep_den = off_policy_keep_den + per_chunk_keep_den
                    chunk_per_token_loss = chunk_per_token_loss * chunk_off_policy_mask

                if beta != 0.0:
                    chunk_per_token_loss = chunk_per_token_loss + beta * chunk_per_token_kl
                chunk_per_token_loss = chunk_per_token_loss * chunk_difficulty_weights
                if chunk_behavior_accept is not None:
                    chunk_per_token_loss = jnp.where(chunk_behavior_accept, chunk_per_token_loss, 0.0)

                if loss_type in {"grpo", "sapo", "gmpo"}:
                    loss_numerator = loss_numerator + jnp.sum(
                        jnp.sum(chunk_per_token_loss * chunk_completion_mask, axis=1)
                        / jnp.maximum(jnp.sum(chunk_completion_mask, axis=1), 1.0)
                    )
                    grpo_weight_den = jnp.sum(difficulty_weights)
                elif loss_type == "luspo":
                    loss_numerator = loss_numerator + jnp.sum(chunk_per_token_loss * chunk_completion_mask)
                else:
                    loss_numerator = loss_numerator + jnp.sum(chunk_per_token_loss * chunk_completion_mask)

                if beta != 0.0:
                    chunk_mean_kl_num, chunk_mean_kl_den = _masked_sum_and_count(
                        chunk_per_token_kl, chunk_completion_mask
                    )
                    mean_kl_num = mean_kl_num + chunk_mean_kl_num
                    mean_kl_den = mean_kl_den + chunk_mean_kl_den
                    chunk_ref_num, chunk_ref_den = _masked_sum_and_count(
                        chunk_ref_per_token_logps,
                        chunk_completion_mask,
                    )
                    ref_logps_num = ref_logps_num + chunk_ref_num
                    ref_logps_den = ref_logps_den + chunk_ref_den

                if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo", "gmpo"]:
                    is_low_clipped = (coef_1 < 1 - epsilon) & (chunk_advantages < 0)
                    is_high_clipped = (coef_1 > 1 + epsilon_high) & (chunk_advantages > 0)
                    is_region_clipped = is_low_clipped | is_high_clipped
                    chunk_low_num, chunk_low_den = _masked_sum_and_count(
                        is_low_clipped.astype(jnp.float32),
                        chunk_completion_mask,
                    )
                    chunk_high_num, chunk_high_den = _masked_sum_and_count(
                        is_high_clipped.astype(jnp.float32),
                        chunk_completion_mask,
                    )
                    chunk_region_num, chunk_region_den = _masked_sum_and_count(
                        is_region_clipped.astype(jnp.float32),
                        chunk_completion_mask,
                    )
                    low_clip_num = low_clip_num + chunk_low_num
                    low_clip_den = low_clip_den + chunk_low_den
                    high_clip_num = high_clip_num + chunk_high_num
                    high_clip_den = high_clip_den + chunk_high_den
                    region_clip_num = region_clip_num + chunk_region_num
                    region_clip_den = region_clip_den + chunk_region_den
                elif loss_type == "cispo":
                    is_cispo_clipped = (coef_1 > 1.0 + epsilon_high) & (chunk_advantages > 0)
                    chunk_cispo_num, chunk_cispo_den = _masked_sum_and_count(
                        is_cispo_clipped.astype(jnp.float32),
                        chunk_completion_mask,
                    )
                    cispo_clip_num = cispo_clip_num + chunk_cispo_num
                    cispo_clip_den = cispo_clip_den + chunk_cispo_den
                elif loss_type == "dppo" and chunk_divergence_mask is not None:
                    is_masked = (chunk_divergence_mask == 0) & (chunk_completion_mask > 0)
                    chunk_dppo_num, chunk_dppo_den = _masked_sum_and_count(
                        is_masked.astype(jnp.float32),
                        chunk_completion_mask,
                    )
                    dppo_mask_num = dppo_mask_num + chunk_dppo_num
                    dppo_mask_den = dppo_mask_den + chunk_dppo_den

            if loss_type in {"grpo", "sapo", "gmpo"}:
                loss = loss_numerator / weighted_denominator(grpo_weight_den)
            elif loss_type == "luspo":
                loss = loss_numerator / weighted_denominator(jnp.sum(difficulty_weights))
            elif loss_type == "bnpo":
                weighted_token_count = jnp.sum(completion_mask * difficulty_weights)
                loss = loss_numerator / weighted_denominator(weighted_token_count)
            elif loss_type == "dr_grpo":
                loss = loss_numerator / weighted_denominator(jnp.sum(difficulty_weights) * completion_ids.shape[1])
            elif loss_type in ["cispo", "dapo", "vespo", "dppo"]:
                if has_difficulty_weights:
                    loss = loss_numerator / weighted_denominator(jnp.sum(completion_mask * difficulty_weights))
                else:
                    loss = loss_numerator / normalizer
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            other_metrics: dict[str, jax.Array] = {
                "mean_entropy": jnp.array(jnp.nan, dtype=jnp.float32),
                "advantages": jnp.mean(advantages),
            }
            if beta != 0.0:
                mean_kl = mean_kl_num / jnp.maximum(mean_kl_den, 1.0)
                other_metrics["mean_kl"] = mean_kl
                other_metrics["ref_per_token_logps"] = ref_logps_num / jnp.maximum(ref_logps_den, 1.0)
            else:
                mean_kl = None
            if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo", "gmpo"]:
                other_metrics["clip_ratio/low_mean"] = low_clip_num / jnp.maximum(low_clip_den, 1.0)
                other_metrics["clip_ratio/high_mean"] = high_clip_num / jnp.maximum(high_clip_den, 1.0)
                other_metrics["clip_ratio/region_mean"] = region_clip_num / jnp.maximum(region_clip_den, 1.0)
            elif loss_type == "cispo":
                other_metrics["cispo_clip_ratio"] = cispo_clip_num / jnp.maximum(cispo_clip_den, 1.0)
            elif loss_type == "dppo" and dppo_divergence_type is not None:
                other_metrics["dppo_mask_ratio/overall_mean"] = dppo_mask_num / jnp.maximum(dppo_mask_den, 1.0)
            if off_policy_mask_threshold is not None:
                other_metrics["off_policy_keep_ratio"] = off_policy_keep_num / jnp.maximum(off_policy_keep_den, 1.0)
            if behavior_sums is not None:
                other_metrics.update({SUM_PREFIX + key: value for key, value in behavior_sums.items()})

            return loss, LossMetrics(
                loss=loss,
                accuracy=1,
                other_metrics=other_metrics,
            )

        entropies = None
        current_topk_logps = None
        with jax.named_scope(scope_root + "/loss_fn/policy_logps"):
            if prefix_shared_forward:
                if (dppo_divergence_type in {"topk_tv", "topk_kl"} and sampling_topk_indices is not None) or (
                    top_entropy_quantile < 1.0
                ):
                    raise NotImplementedError(
                        "prefix_shared_forward supports neither top-k DPPO divergences nor top_entropy_quantile < 1."
                    )
                per_token_logps = get_per_token_logps_prefix_shared(
                    module,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_attention_mask,
                    effective_num_generations,
                    model_kwargs=completion_model_kwargs,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                )
            elif dppo_divergence_type in {"topk_tv", "topk_kl"} and sampling_topk_indices is not None:
                per_token_logps, current_topk_logps = get_per_token_logps_and_selected_logps(
                    module,
                    input_ids,
                    attention_mask,
                    prompt_len,
                    sampling_topk_indices,
                    model_kwargs=completion_model_kwargs,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                )
            elif top_entropy_quantile < 1.0:
                per_token_logps, entropies = get_per_token_logps_and_entropies(
                    module,
                    input_ids,
                    attention_mask,
                    prompt_len,
                    model_kwargs=completion_model_kwargs,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                )
            else:
                per_token_logps = get_per_token_logps(
                    module,
                    input_ids,
                    attention_mask,
                    prompt_len,
                    model_kwargs=completion_model_kwargs,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                )

        if old_per_token_logps is None:
            old_per_token_logps = jax.lax.stop_gradient(per_token_logps)

        with jax.named_scope(scope_root + "/loss_fn/kl_to_reference"):
            ref_per_token_logps = None
            if beta != 0.0:
                ref_per_token_logps = minibatch["ref_per_token_logps"][:, : completion_ids.shape[1]]
                per_token_kl = _compute_per_token_kl(
                    per_token_logps,
                    ref_per_token_logps,
                    kl_estimator,
                    completion_mask=completion_mask,
                )
                if use_bias_correction_kl:
                    importance_weights = _compute_importance_weights(
                        per_token_logps=per_token_logps,
                        old_per_token_logps=old_per_token_logps,
                        completion_mask=completion_mask,
                        importance_sampling_level=importance_sampling_level,
                    )
                    per_token_kl = per_token_kl * importance_weights
            else:
                per_token_kl = jnp.zeros_like(per_token_logps)

        behavior_terms = None
        if behavior_per_token_logps is not None:
            behavior_terms = behavior_policy_terms(
                per_token_logps,
                behavior_per_token_logps,
                completion_mask,
                k3_threshold=off_policy_k3_threshold,
            )
            if importance_sampling_ratio is not None:
                importance_sampling_ratio = _sanitize_rollout_ratio(importance_sampling_ratio, behavior_terms.accept)
        entropy_mask = None
        if top_entropy_quantile < 1.0 and entropies is not None:
            masked_entropies = jnp.where(completion_mask > 0, entropies, jnp.nan)
            entropy_threshold = jnp.nanquantile(masked_entropies, 1 - top_entropy_quantile)
            entropy_mask = (entropies >= entropy_threshold).astype(completion_mask.dtype) * completion_mask

        with jax.named_scope(scope_root + "/loss_fn/policy_objective"):
            dppo_divergence_mask = None
            if dppo_divergence_type is not None:
                sampling_per_token_logps = minibatch.get("sampling_per_token_logps")
                if sampling_per_token_logps is None:
                    sampling_per_token_logps = old_per_token_logps
                elif sampling_per_token_logps.shape[1] != completion_ids.shape[1]:
                    sampling_per_token_logps = sampling_per_token_logps[:, : completion_ids.shape[1]]
                log_ratio = jnp.where(completion_mask > 0, per_token_logps - sampling_per_token_logps, 0.0)
                coef_1 = jax.lax.stop_gradient(jnp.exp(jnp.minimum(log_ratio, math.log(dppo_clip_ratio_c))))
                dppo_divergence_mask = _compute_dppo_divergence_mask(
                    per_token_logps=per_token_logps,
                    sampling_per_token_logps=sampling_per_token_logps,
                    advantages=advantages,
                    completion_mask=completion_mask,
                    divergence_type=dppo_divergence_type,
                    epsilon_low=epsilon,
                    epsilon_high=epsilon_high,
                    current_topk_logps=current_topk_logps,
                    sampling_topk_logps=sampling_topk_logps,
                )
                per_token_loss = -advantages * coef_1 * dppo_divergence_mask * per_token_logps
            else:
                per_token_loss, coef_1 = _compute_grpo_policy_loss_terms(
                    per_token_logps=per_token_logps,
                    old_per_token_logps=old_per_token_logps,
                    advantages=advantages,
                    completion_mask=completion_mask,
                    loss_type=loss_type,
                    epsilon=epsilon,
                    epsilon_high=epsilon_high,
                    delta=delta,
                    importance_sampling_level=importance_sampling_level,
                    sapo_temperature_pos=sapo_temperature_pos,
                    sapo_temperature_neg=sapo_temperature_neg,
                    vespo_k_pos=vespo_k_pos,
                    vespo_lambda_pos=vespo_lambda_pos,
                    vespo_k_neg=vespo_k_neg,
                    vespo_lambda_neg=vespo_lambda_neg,
                    importance_sampling_ratio=importance_sampling_ratio,
                    entropy_mask=entropy_mask,
                )

        if _apply_rollout_is_ratio_post_hoc(loss_type, importance_sampling_ratio, per_token_logps):
            per_token_loss = per_token_loss * jax.lax.stop_gradient(importance_sampling_ratio)

        off_policy_mask = None
        if off_policy_mask_threshold is not None:
            sampling_per_token_logps = minibatch.get("sampling_per_token_logps")
            if sampling_per_token_logps is None:
                sampling_per_token_logps = old_per_token_logps
            elif sampling_per_token_logps.shape[1] != completion_ids.shape[1]:
                sampling_per_token_logps = sampling_per_token_logps[:, : completion_ids.shape[1]]
            off_policy_mask = _compute_off_policy_sequence_mask(
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                advantages=advantages,
                completion_mask=completion_mask,
                threshold=off_policy_mask_threshold,
            )
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if beta != 0.0:
            per_token_loss = per_token_loss + beta * per_token_kl
        per_token_loss = per_token_loss * difficulty_weights
        if behavior_terms is not None:
            # Rejected rows contribute zero; normalisers below still count their tokens.
            per_token_loss = jnp.where(behavior_terms.accept, per_token_loss, 0.0)

        with jax.named_scope(scope_root + "/loss_fn/reduce_loss"):
            if loss_type in {"grpo", "sapo", "gmpo"}:
                sequence_loss = jnp.sum(per_token_loss * completion_mask, axis=1) / jnp.maximum(
                    completion_lengths,
                    1.0,
                )
                loss = jnp.sum(sequence_loss) / weighted_denominator(jnp.sum(difficulty_weights))
            elif loss_type == "luspo":
                loss = jnp.sum(per_token_loss * completion_mask) / weighted_denominator(jnp.sum(difficulty_weights))
            elif loss_type == "bnpo":
                weighted_token_count = jnp.sum(completion_mask * difficulty_weights)
                loss = jnp.sum(per_token_loss * completion_mask) / weighted_denominator(weighted_token_count)
            elif loss_type == "dr_grpo":
                loss = jnp.sum(per_token_loss * completion_mask) / weighted_denominator(
                    jnp.sum(difficulty_weights) * completion_ids.shape[1]
                )
            elif loss_type in ["cispo", "dapo", "vespo", "dppo"]:
                if has_difficulty_weights:
                    loss = jnp.sum(per_token_loss * completion_mask) / weighted_denominator(
                        jnp.sum(completion_mask * difficulty_weights)
                    )
                else:
                    normalizer = _global_token_normalizer(
                        minibatch.get("num_items_in_batch"),
                        completion_token_count,
                        completion_was_truncated,
                        effective_ga_steps,
                        is_training=is_training,
                    )
                    loss = jnp.sum(per_token_loss * completion_mask) / normalizer
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

        def masked_mean(x):
            """Average ``x`` over masked completion tokens.

            For sequence-level (``shape[1] == 1``) tensors falls back
            to a plain mean since there is nothing to mask.

            Args:
                x: Tensor of shape ``[batch, seq_len]`` or ``[batch, 1]``.

            Returns:
                A scalar mean over the masked positions.
            """
            if x.shape[1] == 1:
                return jnp.mean(x)
            return jnp.sum(x * completion_mask) / jnp.maximum(completion_token_count, 1.0)

        other_metrics: dict[str, jax.Array] = {
            "mean_entropy": (
                masked_mean(entropies) if entropies is not None else jnp.array(jnp.nan, dtype=per_token_logps.dtype)
            ),
            "advantages": jnp.mean(advantages),
        }

        if beta != 0.0:
            mean_kl = masked_mean(per_token_kl)
            other_metrics["mean_kl"] = mean_kl
            if ref_per_token_logps is not None:
                other_metrics["ref_per_token_logps"] = masked_mean(ref_per_token_logps)
        else:
            mean_kl = None

        if loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo", "gmpo"]:
            is_low_clipped = (coef_1 < 1 - epsilon) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            other_metrics["clip_ratio/low_mean"] = masked_mean(is_low_clipped.astype(jnp.float32))
            other_metrics["clip_ratio/high_mean"] = masked_mean(is_high_clipped.astype(jnp.float32))
            other_metrics["clip_ratio/region_mean"] = masked_mean(is_region_clipped.astype(jnp.float32))
        elif loss_type == "cispo":
            is_cispo_clipped = (coef_1 > 1.0 + epsilon_high) & (advantages > 0)
            other_metrics["cispo_clip_ratio"] = masked_mean(is_cispo_clipped.astype(jnp.float32))
        elif loss_type == "dppo" and dppo_divergence_mask is not None:
            is_masked = (dppo_divergence_mask == 0) & (completion_mask > 0)
            other_metrics["dppo_mask_ratio/overall_mean"] = masked_mean(is_masked.astype(jnp.float32))
        if off_policy_mask is not None:
            other_metrics["off_policy_keep_ratio"] = jnp.mean(off_policy_mask)
        if behavior_terms is not None:
            other_metrics.update({SUM_PREFIX + key: value for key, value in behavior_terms.sums.items()})

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics=other_metrics,
        )

    if is_training:
        with jax.named_scope(scope_root + "/grad_and_minibatch"):
            gradients, metrics = minibatch_call(
                state=state,
                batch=batch,
                minibatch_size=minibatch_size,
                grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
                prompt_group_size=prompt_group_size if prefix_shared_forward else None,
            )
        metrics, behavior_summary = _finalize_behavior_metrics(metrics, effective_ga_steps)
        contract_active = off_policy_min_acceptance is not None or startup_behavior_k3_max is not None
        update_gate = None
        if behavior_summary is not None and contract_active:
            update_gate = acceptance_ok(behavior_summary["off_policy/accepted_fraction"], off_policy_min_acceptance)
            parity_flag = batch.get("behavior_parity_gate")
            if startup_behavior_k3_max is not None and parity_flag is not None:
                parity = behavior_parity_ok(
                    behavior_summary["off_policy/behavior_k3"],
                    behavior_summary["off_policy/tail_fraction"],
                    k3_max=startup_behavior_k3_max,
                    tail_fraction_max=startup_behavior_tail_fraction_max,
                )
                update_gate = update_gate & (parity | (jnp.asarray(parity_flag) <= 0))
        elif off_policy_min_acceptance is not None:
            # Fail closed: the acceptance contract is configured but the batch carries no behaviour logprobs.
            update_gate = jnp.asarray(False)
        elif startup_behavior_k3_max is not None and batch.get("behavior_parity_gate") is not None:
            # The first-step parity check cannot run without behaviour logprobs: skip that step only.
            update_gate = jnp.asarray(batch["behavior_parity_gate"]) <= 0
        metrics = update_metrics(
            metrics=metrics,
            learning_rate_fn=learning_rate_fn,
            step=state.step,
            gradients=gradients,
        )
        if update_gate is not None:
            other_metrics = dict(metrics.other_metrics or {})
            other_metrics["off_policy/step_skipped"] = 1.0 - update_gate.astype(jnp.float32)
            metrics = metrics.replace(other_metrics=other_metrics)
        with jax.named_scope(scope_root + "/update_state"):
            if update_gate is None:
                state = update_state_respectfully(
                    state=state,
                    gradients=gradients,
                    loss_config=loss_config,
                    metrics=metrics,
                )
            else:
                # Gated update: the skip branch returns the state untouched, so parameters,
                # optimizer moments and the step counter are all preserved bit-for-bit.
                state = jax.lax.cond(
                    update_gate,
                    lambda current: update_state_respectfully(
                        state=current,
                        gradients=gradients,
                        loss_config=loss_config,
                        metrics=metrics,
                    ),
                    lambda current: current,
                    state,
                )
        return state, metrics
    else:
        with jax.named_scope(scope_root + "/eval_call"):
            _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
        metrics, _ = _finalize_behavior_metrics(metrics, 1)
        return metrics


def _grpo_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build the scheduled-loss cache key for GRPO-family trainers.

    The key fingerprints all algorithmic knobs that change the loss
    graph (``num_generations``, ``beta``, ``loss_type``, clip bounds,
    importance-sampling level, chunk sizes) plus the object identities
    of the loss config, learning-rate schedule, and any QAT
    straight-through emulator so that recompilation is triggered only
    when those actually change.

    Args:
        call: The :class:`ScheduledStepCall` being compiled by the
            SpectraX VJP pipeline.

    Returns:
        A tuple suitable for use as a SpectraX scheduled-loss cache key.
    """

    key = scheduled_loss_cache_key(
        call,
        value_fields=(
            "num_generations",
            "beta",
            "partition_spec",
            "gradient_accumulation_steps",
            "loss_type",
            "epsilon",
            "epsilon_high",
            "delta",
            "importance_sampling_level",
            "top_entropy_quantile",
            "completion_chunk_size",
            "max_loss_completion_tokens",
            "logprob_vocab_chunk_size",
            "sapo_temperature_pos",
            "sapo_temperature_neg",
            "vespo_k_pos",
            "vespo_lambda_pos",
            "vespo_k_neg",
            "vespo_lambda_neg",
            "off_policy_mask_threshold",
            "use_bias_correction_kl",
            "dppo_divergence_type",
            "dppo_clip_ratio_c",
            "kl_estimator",
            "off_policy_k3_threshold",
            "off_policy_min_acceptance",
            "startup_behavior_k3_max",
            "startup_behavior_tail_fraction_max",
            "prefix_shared_forward",
        ),
        object_fields=("loss_config", "learning_rate_fn", "straight_through_emulator"),
    )
    # The microbatch weight callable (and the weighted loss/gate-sum closure) is built only on a
    # cache miss, so whether microbatches are denominator-weighted must be part of the key.
    return (*key, ("difficulty_weights", _grpo_scheduled_weighted(call)))


def _grpo_scheduled_weighted(call) -> bool:
    """Whether the scheduled GRPO call weights its microbatches by their loss denominators."""
    if "difficulty_weights" in call.batch:
        return True
    return call.schedule is not None and call.get("loss_type") in {"bnpo", "cispo", "dapo", "vespo", "dppo"}


def _grpo_local_denominator_correction(denominator: jax.Array) -> jax.Array:
    """Factor turning a microbatch loss ``num / max(d, 1)`` into ``num / d`` (``1`` when ``d <= 0``).

    ``grpo_step`` clamps its loss denominator at ``1``; the weighted scheduled reduction
    ``sum_i (d_i / D) * loss_i`` needs the unclamped ``num_i / d_i`` to equal ``num / D``. The factor is
    ``1`` whenever ``d >= 1`` and only differs for fractional weights.
    """
    denominator = jnp.asarray(denominator, jnp.float32)
    positive = denominator > 0
    return jnp.where(positive, jnp.maximum(denominator, 1.0) / jnp.where(positive, denominator, 1.0), 1.0)


def _make_grpo_microbatch_weight(call):
    """Build microbatch weight function matching GRPO's loss normalizer across MPMD microbatches."""
    loss_type = call.get("loss_type")
    max_loss_completion_tokens = call.get("max_loss_completion_tokens")

    def weight_fn(_tree, batch):
        return _grpo_weighted_loss_denominator(batch, loss_type, max_loss_completion_tokens)

    return weight_fn


def _make_grpo_scheduled_weight(call):
    """Per-microbatch loss weight of a scheduled GRPO step when denominator weighting is active.

    The weight of a microbatch is its own loss denominator ``d_i``
    (:func:`_grpo_weighted_loss_denominator`): ``sum(w)``, ``sum(w) * T``, or
    ``sum(mask * w)`` depending on ``loss_type``. The scheduled runtime scales each microbatch by
    ``d_i / D``, so ``sum_i (d_i / D) * (num_i / d_i) = num / D``: the full-batch objective the SPMD
    step uses, and a fully discarded microbatch (``d_i = 0``) contributes no gradient. Unweighted
    row-normalised batches return ``None`` and keep the uniform microbatch mean unchanged.

    Args:
        call: The :class:`ScheduledStepCall` being compiled.

    Returns:
        ``(tree, batch) -> scalar`` weight callable, or ``None`` for uniform microbatches.
    """
    if not _grpo_scheduled_weighted(call):
        return None
    return _make_grpo_microbatch_weight(call)


def _grpo_scheduled_prepare_batch(call):
    """Prepare the GRPO batch before scheduled MPMD microbatching.

    * Under a real MPMD schedule (``call.schedule is not None``), expands prompt-aligned
      leaves to completion-row granularity (preserving prompt/completion pairing across any
      microbatch count) and strips ``num_items_in_batch`` so each microbatch normalizes by
      its local denominator while ``make_microbatch_weight`` recombines exact full-batch sums.
    * Strips the host-only ``behavior_parity_gate`` flag from pipelined loss inputs
      (``_grpo_scheduled_finalize_aux`` reads it from ``call.batch``).
    """
    prepared = dict(call.batch)
    if call.schedule is not None:
        state_model = getattr(call.state, "model", None)
        model_callable = getattr(state_model, "forward", state_model) if state_model is not None else None
        prepared = _expand_grpo_prompt_batch(prepared, model_callable=model_callable)
        prepared.pop("num_items_in_batch", None)
    prepared.pop("behavior_parity_gate", None)
    return prepared


def _make_grpo_scheduled_loss(call):
    """Build a SpectraX-scheduled scalar GRPO-loss closure for ``call``.

    Captures the algorithmic hyperparameters (``num_generations``,
    ``beta``, ``loss_type``, clip bounds, importance-sampling level,
    chunk sizes, ...) and any QAT straight-through emulator from the
    :class:`ScheduledStepCall`, then returns a function
    ``(tree, batch) -> scalar`` which evaluates :func:`grpo_step` in
    eval-mode with ``gradient_accumulation_steps=1`` (since MPMD has
    already sliced the microbatch) and returns only the loss field.

    Args:
        call: The :class:`ScheduledStepCall` carrying loss config and
            hyperparameters.

    Returns:
        A closure ``loss_fn(tree, batch) -> jax.Array`` returning the
        scalar GRPO loss for one scheduled microbatch.
    """

    num_generations = call.get("num_generations")
    beta = call.get("beta")
    loss_config = call.get("loss_config")
    learning_rate_fn = call.get("learning_rate_fn")
    partition_spec = call.get("partition_spec")
    loss_type = call.get("loss_type")
    epsilon = call.get("epsilon")
    epsilon_high = call.get("epsilon_high")
    delta = call.get("delta")
    importance_sampling_level = call.get("importance_sampling_level")
    top_entropy_quantile = call.get("top_entropy_quantile")
    completion_chunk_size = call.get("completion_chunk_size")
    max_loss_completion_tokens = call.get("max_loss_completion_tokens")
    logprob_vocab_chunk_size = call.get("logprob_vocab_chunk_size")
    sapo_temperature_pos = call.get("sapo_temperature_pos")
    sapo_temperature_neg = call.get("sapo_temperature_neg")
    vespo_k_pos = call.get("vespo_k_pos")
    vespo_lambda_pos = call.get("vespo_lambda_pos")
    vespo_k_neg = call.get("vespo_k_neg")
    vespo_lambda_neg = call.get("vespo_lambda_neg")
    off_policy_mask_threshold = call.get("off_policy_mask_threshold")
    use_bias_correction_kl = call.get("use_bias_correction_kl")
    straight_through_emulator = call.get("straight_through_emulator")
    dppo_divergence_type = call.get("dppo_divergence_type")
    dppo_clip_ratio_c = call.get("dppo_clip_ratio_c", 20.0)
    kl_estimator = call.get("kl_estimator", "k3")
    off_policy_k3_threshold = call.get("off_policy_k3_threshold", None)
    prefix_shared_forward = call.get("prefix_shared_forward", False)
    # With an update gate configured the loss also returns additive gate sums as aux; the
    # adapter's ``finalize_aux`` turns them into the scheduled update gate.
    gated = _grpo_update_gate_active(call)
    # With denominator weighting active the runtime weights microbatches by their loss denominators
    # (:func:`_make_grpo_scheduled_weight`); the loss and gate sums are adjusted to match.
    weighted = _grpo_scheduled_weighted(call)
    state_shell = make_scheduled_state_shell(call)

    def scheduled_loss(tree, batch):
        """Compute the scalar GRPO loss inside the SpectraX scheduled VJP.

        Applies the optional straight-through emulator to the policy
        graphstate, rebuilds an :class:`EasyDeLState` for ``grpo_step``,
        and dispatches in eval mode so only the metrics path runs.

        Args:
            tree: Policy graphstate to differentiate against.
            batch: Scheduled microbatch dict consumed by ``grpo_step``
                (``prompt_ids`` / ``completion_ids`` / ``advantages`` /
                ``ref_per_token_logps`` / generation kwargs).

        Returns:
            The scalar GRPO loss for the microbatch.
        """

        with jax.named_scope("easydel/trainer/grpo/scheduled_loss"):
            if straight_through_emulator is not None:
                tree = straight_through_emulator(tree)
            scheduled_state = state_shell.replace(graphstate=tree)
            metrics = grpo_step(
                scheduled_state,
                batch,
                num_generations,
                beta,
                loss_config=loss_config,
                learning_rate_fn=learning_rate_fn,
                partition_spec=partition_spec,
                gradient_accumulation_steps=1,
                is_training=False,
                loss_type=loss_type,
                epsilon=epsilon,
                epsilon_high=epsilon_high,
                delta=delta,
                importance_sampling_level=importance_sampling_level,
                top_entropy_quantile=top_entropy_quantile,
                completion_chunk_size=completion_chunk_size,
                max_loss_completion_tokens=max_loss_completion_tokens,
                logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                sapo_temperature_pos=sapo_temperature_pos,
                sapo_temperature_neg=sapo_temperature_neg,
                vespo_k_pos=vespo_k_pos,
                vespo_lambda_pos=vespo_lambda_pos,
                vespo_k_neg=vespo_k_neg,
                vespo_lambda_neg=vespo_lambda_neg,
                off_policy_mask_threshold=off_policy_mask_threshold,
                use_bias_correction_kl=use_bias_correction_kl,
                straight_through_emulator=None,
                dppo_divergence_type=dppo_divergence_type,
                dppo_clip_ratio_c=dppo_clip_ratio_c,
                kl_estimator=kl_estimator,
                off_policy_k3_threshold=off_policy_k3_threshold,
                prefix_shared_forward=prefix_shared_forward,
            )
            loss = metrics.loss
            denominator = None
            if weighted:
                denominator = _grpo_weighted_loss_denominator(batch, loss_type, max_loss_completion_tokens)
                loss = loss * _grpo_local_denominator_correction(denominator).astype(loss.dtype)
            if not gated:
                return loss
            return loss, _grpo_scheduled_gate_sums(
                batch, metrics, max_loss_completion_tokens, loss, microbatch_weight=denominator
            )

    return scheduled_loss


_GRPO_GATE_AUX_PREFIX = "grpo_gate_sums/"
"""Aux-key prefix of the additive sums the gated scheduled GRPO loss reports per microbatch."""

_GRPO_GATE_SUM_KEYS = ("has_behavior", "eligible_rows", "accepted_rows", "tokens", "k3_sum", "tail_count")


def _grpo_update_gate_active(call) -> bool:
    """Whether the GRPO call configures a whole-update gate (acceptance or first-step parity)."""
    return call.get("off_policy_min_acceptance") is not None or call.get("startup_behavior_k3_max") is not None


def _grpo_scheduled_gate_sums(
    batch,
    metrics: LossMetrics,
    max_loss_completion_tokens: int | None,
    loss: jax.Array,
    microbatch_weight: jax.Array | None = None,
) -> dict:
    """Additive per-microbatch sums from which the whole-batch update gate is rebuilt.

    ``grpo_step`` (eval mode) reports the ratio diagnostics of its microbatch; multiplying them back
    by their denominators (eligible rows, action tokens — both recomputed from the same truncated
    completion mask ``grpo_step`` uses) yields additive sums. Without microbatch weights the scheduled
    runtime reduces aux with one common scale per microbatch, so the ratios of the reduced sums equal
    the whole-batch ratios the SPMD step gates on.

    With ``microbatch_weight = d_i`` the runtime (pipeline and outer accumulation alike) scales aux by
    ``d_i / D`` instead. Each sum is then divided by ``d_i`` so its reduced value is ``sum / D``: the
    common ``1 / D`` cancels in every ratio. ``has_behavior`` is left undivided (it reduces to ``1``)
    and a microbatch with ``d_i = 0`` is left as is (its scale is ``0``, or uniform when ``D = 0``).

    Args:
        batch: Scheduled microbatch.
        metrics: ``grpo_step`` eval-mode metrics of the microbatch.
        max_loss_completion_tokens: Completion-token cap used by the loss.
        loss: Scalar microbatch loss the sums are anchored on.
        microbatch_weight: Optional scalar weight ``d_i`` the runtime reduces this microbatch with.

    Returns:
        Mapping of ``grpo_gate_sums/*`` aux keys to scalars.
    """
    mask = jnp.asarray(batch["completion_mask"])
    if max_loss_completion_tokens is not None and mask.shape[1] > max_loss_completion_tokens:
        mask = mask[:, :max_loss_completion_tokens]
    mask_f = (mask > 0).astype(jnp.float32)
    tokens = jnp.sum(mask_f)
    eligible = jnp.sum((jnp.sum(mask_f, axis=-1) > 0).astype(jnp.float32))
    other = metrics.other_metrics or {}
    zero = jnp.zeros((), jnp.float32)
    if "off_policy/accepted_fraction" in other:
        sums = {
            "has_behavior": jnp.ones((), jnp.float32),
            "eligible_rows": eligible,
            "accepted_rows": jnp.asarray(other["off_policy/accepted_fraction"], jnp.float32) * eligible,
            "tokens": tokens,
            "k3_sum": jnp.asarray(other["off_policy/behavior_k3"], jnp.float32) * tokens,
            "tail_count": jnp.asarray(other["off_policy/tail_fraction"], jnp.float32) * tokens,
        }
    else:
        sums = {key: zero for key in _GRPO_GATE_SUM_KEYS}
    if microbatch_weight is not None:
        weight = jnp.asarray(microbatch_weight, jnp.float32)
        inverse = jnp.where(weight > 0, 1.0 / jnp.where(weight > 0, weight, 1.0), 1.0)
        sums = {key: value if key == "has_behavior" else value * inverse for key, value in sums.items()}
    # The pipeline runtime requires every aux output to be computed in the terminal (loss) stage;
    # values derived only from batch inputs or constants could be clustered elsewhere, so anchor
    # each sum on the (finite-sanitised) loss with a zero coefficient.
    anchor = 0.0 * jnp.nan_to_num(jax.lax.stop_gradient(jnp.asarray(loss, jnp.float32)))
    return {_GRPO_GATE_AUX_PREFIX + key: jax.lax.stop_gradient(value) + anchor for key, value in sums.items()}


def _grpo_scheduled_finalize_aux(call, aux):
    """Turn the reduced gate sums into ``off_policy/*`` diagnostics and the scheduled update gate.

    Mirrors the SPMD gate in :func:`grpo_step`: ``accepted_fraction >= off_policy_min_acceptance``
    and, while the batch's ``behavior_parity_gate`` flag is set, the first-step parity bounds
    ``behavior_k3 <= startup_behavior_k3_max`` and ``tail_fraction <= startup_behavior_tail_fraction_max``.
    Without behaviour log-probs the acceptance gate fails closed and the parity-only gate skips
    exactly the flagged step. The inputs are host-replicated, so every pipeline rank agrees.
    """
    if not aux or not any(key.startswith(_GRPO_GATE_AUX_PREFIX) for key in aux):
        return aux
    aux = dict(aux)
    sums = {
        key: float(np.asarray(jax.device_get(aux.pop(_GRPO_GATE_AUX_PREFIX + key)), np.float64).reshape(()))
        for key in _GRPO_GATE_SUM_KEYS
    }
    min_acceptance = call.get("off_policy_min_acceptance")
    k3_max = call.get("startup_behavior_k3_max")
    tail_max = call.get("startup_behavior_tail_fraction_max", 0.01)
    parity_flag = call.batch.get("behavior_parity_gate")
    parity_pending = parity_flag is not None and int(np.asarray(jax.device_get(parity_flag))) > 0
    has_behavior = sums["has_behavior"] > 0.5 or (call.batch.get("behavior_per_token_logps") is not None)
    if has_behavior:
        eligible, tokens = sums["eligible_rows"], sums["tokens"]
        accepted_fraction = sums["accepted_rows"] / eligible if eligible > 0 else 1.0
        behavior_k3 = sums["k3_sum"] / tokens if tokens > 0 else 0.0
        tail_fraction = sums["tail_count"] / tokens if tokens > 0 else 0.0
        aux["off_policy/accepted_fraction"] = jnp.asarray(accepted_fraction, jnp.float32)
        aux["off_policy/behavior_k3"] = jnp.asarray(behavior_k3, jnp.float32)
        aux["off_policy/tail_fraction"] = jnp.asarray(tail_fraction, jnp.float32)
        # Compare in float32, as the SPMD gate does on device.
        gate = min_acceptance is None or bool(np.float32(accepted_fraction) >= np.float32(min_acceptance))
        if k3_max is not None and parity_pending:
            parity = bool(np.float32(behavior_k3) <= np.float32(k3_max)) and bool(
                np.float32(tail_fraction) <= np.float32(tail_max)
            )
            gate = gate and parity
    elif min_acceptance is not None:
        gate = False
    elif k3_max is not None and parity_flag is not None:
        gate = not parity_pending
    else:
        gate = True
    aux["off_policy/step_skipped"] = jnp.asarray(0.0 if gate else 1.0, jnp.float32)
    aux[SCHEDULED_UPDATE_GATE_KEY] = jnp.asarray(1.0 if gate else 0.0, jnp.float32)
    return aux


register_scheduled_loss_adapter(
    step_fn=grpo_step,
    adapter=ScheduledLossAdapter(
        name="grpo",
        make_loss=_make_grpo_scheduled_loss,
        make_cache_key=_grpo_scheduled_loss_cache_key,
        has_aux=_grpo_update_gate_active,
        prepare_batch=_grpo_scheduled_prepare_batch,
        make_microbatch_weight=_make_grpo_scheduled_weight,
        finalize_aux=_grpo_scheduled_finalize_aux,
    ),
)
