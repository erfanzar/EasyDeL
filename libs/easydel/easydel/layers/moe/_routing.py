# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Shared top-k expert selection for MoE routers.

The families listed in :func:`moe_group_topk_select` route through this one implementation, installed as a
:attr:`MoeFusedHooks.select_hook`.

The pipeline, in order:

1. Turn router logits into scores (``sigmoid``, ``softmax``, or pass through
   already-scored logits).
2. Add the selection bias, if any (per expert, or per token and expert).
   **The bias steers selection only** — the returned weights are gathered from
   the *unbiased* scores. This is what makes the routing auxiliary-loss-free:
   the bias moves load between experts without perturbing the values that
   reach the combine.
3. Optionally restrict the candidates to the ``topk_group`` best expert groups,
   where a group scores as the sum of its top-``group_topk_k`` members (or as
   its single best member). Experts outside the kept groups rank below every
   eligible expert and carry zero weight, so a ``k`` above the kept groups'
   capacity yields zero-weight slots.
4. Take a flat top-``k`` over the survivors, gather the unbiased scores, and
   optionally renormalise and rescale them.

Every knob exists because some family needed it; none is speculative. See
:func:`moe_group_topk_select` for which family uses which setting.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

ScoreFn = tp.Literal["sigmoid", "softmax", "none"]
GroupScore = tp.Literal["topk_sum", "max"]

__all__ = ("fp32_router_logits", "moe_group_topk_select", "normalize_topk_weights")


def fp32_router_logits(hidden_states: Array, weight: Array, hidden_size: int) -> Float[Array, "tokens experts"]:
    """Router projection in float32, flattened to ``[tokens, experts]`` and tagged ``moe_router_logits``.

    The GLM routers keep this matmul in fp32 whatever the activation dtype: the gap between selected and rejected
    experts can fall below bfloat16 resolution near initialisation. Default matmul precision.

    Args:
        hidden_states: Token activations whose last axis is ``hidden_size``.
        weight: Router weight ``[hidden_size, num_experts]`` (any dtype).
        hidden_size: Model width used to flatten ``hidden_states``.

    Returns:
        fp32 logits ``[tokens, num_experts]``.
    """
    hidden_states = hidden_states.reshape(-1, hidden_size)
    return checkpoint_name(
        jnp.matmul(hidden_states.astype(jnp.float32), weight.astype(jnp.float32)),
        "moe_router_logits",
    )


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _normalize(weights: Array, eps: float) -> Array:
    """Row-normalize ``weights``; an all-zero row keeps the constant denominator ``1``."""
    mass = jnp.sum(weights, axis=-1, keepdims=True)
    return weights / jnp.where(mass == 0, 1.0, mass + eps)


@_normalize.defjvp
def _normalize_jvp(eps: float, primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    """Differentiate the quotient through its normalized output instead of ``denominator ** -2``.

    The division rule forms ``denominator ** -2``, which overflows float32 for a selected mass below about ``1e-19``
    (saturated sigmoid scores). This equivalent linear rule stays finite there; an all-zero row keeps its constant
    denominator, so its mass tangent is dropped.
    """
    (weights,), (weights_dot,) = primals, tangents
    mass = jnp.sum(weights, axis=-1, keepdims=True)
    denominator = jnp.where(mass == 0, 1.0, mass + eps)
    normalized = _normalize(weights, eps)
    mass_dot = jnp.where(mass == 0, 0.0, jnp.sum(weights_dot, axis=-1, keepdims=True))
    return normalized, (weights_dot - normalized * mass_dot) / denominator


def normalize_topk_weights(weights: Float[Array, "tokens k"], eps: float = 1e-20) -> Float[Array, "tokens k"]:
    """Divide each row of selected expert weights by its sum plus ``eps``; an all-zero row stays zero.

    The derivative is taken through the normalized output, which keeps it finite for a tiny selected mass, where
    differentiating the plain quotient overflows float32.

    Args:
        weights: Selected expert weights ``(num_tokens, k)``.
        eps: Added to each row sum. ``0.0`` divides by the bare sum.

    Returns:
        The normalized weights.
    """
    return _normalize(weights, float(eps))


def _take_topk(scores: Array, indices: Array) -> Array:
    """``jnp.take_along_axis(scores, indices, axis=-1)`` as a one-hot select-and-sum.

    Each output is one selected score plus exact zeros, so values match the gather. TPU lowers the
    ``[tokens, experts] -> [tokens, k]`` lookup as a batched gather, and its transpose as a scatter-add.
    The select-and-sum avoids both: the select fuses into the sum, and its transpose is a masked broadcast.

    Args:
        scores: ``(tokens, experts)`` scores.
        indices: ``(tokens, k)`` integer expert ids.

    Returns:
        ``(tokens, k)`` scores at ``indices``.
    """
    hit = indices[..., None] == jnp.arange(scores.shape[-1], dtype=indices.dtype)
    return jnp.sum(jnp.where(hit, scores[..., None, :], jnp.zeros((), scores.dtype)), axis=-1)


# Picks made one ``argmax`` at a time (see :func:`_topk_indices`); a larger ``k`` uses ``lax.top_k``.
_ARGMAX_TOPK_MAX_K = 8


def _topk_indices(scores: Array, k: int) -> Int[Array, "tokens k"]:
    """Indices of the ``k`` largest scores per row, in descending order and the lower index first on ties.

    On TPU, ``jax.lax.top_k`` over an operand sharded along the token axis outside ``shard_map`` (an auxiliary loss,
    the unfused MoE path, a direct call) returns wrong indices with correct values at ``k == 1`` and at small ``k``
    on wide rows. ``argmax`` does not, so up to :data:`_ARGMAX_TOPK_MAX_K` entries are picked one ``argmax`` at a
    time, each pick removed before the next. ``-inf`` scores are first raised to the lowest finite value, so they rank
    after every finite score and before removed entries: slots past a row's finite entries take the excluded experts
    in index order, as ``top_k`` does.

    Args:
        scores: ``(tokens, experts)`` floating-point selection scores.
        k: Entries kept per row.

    Returns:
        ``(tokens, k)`` int32 indices.
    """
    if k > _ARGMAX_TOPK_MAX_K:
        return jax.lax.top_k(scores, k=k)[1]
    ids = jnp.arange(scores.shape[-1], dtype=jnp.int32)
    key = jnp.maximum(scores, jnp.finfo(scores.dtype).min)
    picks = []
    for _ in range(k):
        pick = jnp.argmax(key, axis=-1, keepdims=True).astype(jnp.int32)
        picks.append(pick)
        key = jnp.where(ids == pick, -jnp.inf, key)
    return jnp.concatenate(picks, axis=-1)


def _score(gate_logits: Array, score_fn: ScoreFn) -> Array:
    """Map raw router logits to per-expert scores in fp32."""
    logits = gate_logits.astype(jnp.float32)
    if score_fn == "sigmoid":
        return jax.nn.sigmoid(logits)
    if score_fn == "softmax":
        return jax.nn.softmax(logits, axis=-1)
    if score_fn == "none":
        return logits
    raise ValueError(f"unknown score_fn {score_fn!r}; expected 'sigmoid', 'softmax' or 'none'")


def _group_eligibility(
    scores_for_choice: Array,
    *,
    n_routed_experts: int,
    n_group: int,
    topk_group: int,
    group_topk_k: int,
    group_score: GroupScore,
) -> Bool[Array, "tokens experts"]:
    """Mark the experts inside each token's ``topk_group`` best groups.

    Args:
        scores_for_choice: Selection scores ``(num_tokens, n_routed_experts)``.
        n_routed_experts: Total routed experts; must divide by ``n_group``.
        n_group: Number of equal-sized expert groups.
        topk_group: Groups kept per token.
        group_topk_k: Members summed to score a group (``group_score='topk_sum'``).
        group_score: How a group is scored from its members.

    Returns:
        Boolean eligibility ``(num_tokens, n_routed_experts)``.
    """
    num_tokens = scores_for_choice.shape[0]
    group_size = n_routed_experts // n_group
    grouped = scores_for_choice.reshape(num_tokens, n_group, group_size)

    if group_score == "max":
        group_scores = jnp.max(grouped, axis=-1)
    elif group_score == "topk_sum":
        group_scores = jnp.sum(jax.lax.top_k(grouped, k=min(group_topk_k, group_size))[0], axis=-1)
    else:
        raise ValueError(f"unknown group_score {group_score!r}; expected 'topk_sum' or 'max'")

    # A stable sort keeps each group paired with its score on token-sharded operands, where ``top_k`` indices are
    # unreliable (see :func:`_topk_indices`); ties keep the lower group first, as ``top_k`` does.
    group_ids = jnp.broadcast_to(jnp.arange(n_group, dtype=jnp.int32), group_scores.shape)
    _, ranked = jax.lax.sort_key_val(-group_scores, group_ids, dimension=-1, is_stable=True)
    group_idx = ranked[:, : min(topk_group, n_group)]
    group_mask = jnp.sum(jax.nn.one_hot(group_idx, n_group, dtype=scores_for_choice.dtype), axis=1)
    return jnp.repeat(group_mask > 0, group_size, axis=1)


def moe_group_topk_select(
    gate_logits: Float[Array, "tokens experts"],
    pre_bias_logits: Array | None,
    k: int,
    *,
    n_routed_experts: int,
    score_fn: ScoreFn = "sigmoid",
    e_score_correction_bias: Array | None = None,
    n_group: int = 1,
    topk_group: int = 1,
    group_topk_k: int = 2,
    group_score: GroupScore = "topk_sum",
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    norm_eps: float = 1e-20,
) -> tuple[Float[Array, "tokens k"], Int[Array, "tokens k"]]:
    """Select ``k`` experts per token and return their combine weights.

    Usable directly as a :attr:`MoeFusedHooks.select_hook` via
    ``functools.partial``. Bind ``e_score_correction_bias`` to the *live*
    parameter value inside ``forward`` — binding it once at construction time
    freezes the zero-initialised tensor and silently disables load balancing.

    Settings by family:

    - GLM-4-MoE / GLM-4-MoE-Lite / GLM-MoE-DSA / GLM-5-Next, DeepSeek-V3:
      ``score_fn='sigmoid'``, ``group_score='topk_sum'`` (grouped when ``n_group > 1``).
    - Kimi-Linear: ``score_fn`` from the config, ``group_score='topk_sum'``.
    - DeepSeek-V2: ``score_fn='softmax'``, ``group_score='max'`` (grouped for
      ``group_limited_greedy``).
    - Mistral-4: ``score_fn='none'`` (the MoE softmaxes the logits first), grouped.
    - DeepSeek-V4 / V4.1: ``score_fn='none'`` (the gate already scores), flat;
      V4.1 vision tokens pass a per-token bias.
    - Hunyuan-V3: flat, ``score_fn='sigmoid'``.
    - MiniMax-M3-VL: flat, ``score_fn='sigmoid'``, ``norm_eps=0.0``.

    Args:
        gate_logits: Router logits ``(num_tokens, n_routed_experts)``.
        pre_bias_logits: Accepted and ignored; part of the ``select_hook``
            signature that some callers pass positionally.
        k: Routed experts selected per token.
        n_routed_experts: Total routed experts.
        score_fn: ``'sigmoid'``, ``'softmax'``, or ``'none'`` when the caller
            already produced scores.
        e_score_correction_bias: Selection bias ``(n_routed_experts,)`` or per
            token ``(num_tokens, n_routed_experts)``, or ``None`` for no bias.
            Added to the selection scores only; never to the returned weights.
        n_group: Expert groups. ``1`` disables the group stage entirely.
        topk_group: Groups kept per token.
        group_topk_k: Members summed to score a group under ``'topk_sum'``.
        group_score: ``'topk_sum'`` or ``'max'``.
        norm_topk_prob: Divide the selected weights by their sum (see
            :func:`normalize_topk_weights`).
        routed_scaling_factor: Final multiplier on the weights.
        norm_eps: Added to the normalisation denominator. ``1e-20`` guards a
            vanishing sum; pass ``0.0`` to divide by the bare sum. An all-zero
            row stays zero either way.

    Returns:
        ``(topk_weights, topk_indices)``, both ``(num_tokens, k)``. Slots past
        the kept groups' capacity carry zero weight.
    """
    del pre_bias_logits

    scores = _score(gate_logits, score_fn)

    if e_score_correction_bias is None:
        scores_for_choice = scores
    else:
        scores_for_choice = scores + e_score_correction_bias.astype(jnp.float32)

    eligible = None
    if n_group > 1:
        eligible = _group_eligibility(
            scores_for_choice,
            n_routed_experts=n_routed_experts,
            n_group=n_group,
            topk_group=topk_group,
            group_topk_k=group_topk_k,
            group_score=group_score,
        )
        scores_for_choice = jnp.where(eligible, scores_for_choice, -jnp.inf)

    topk_indices = _topk_indices(scores_for_choice, k)
    combine = scores if eligible is None else jnp.where(eligible, scores, jnp.zeros((), scores.dtype))
    topk_weights = _take_topk(combine, topk_indices)

    if norm_topk_prob:
        topk_weights = normalize_topk_weights(topk_weights, norm_eps)

    return topk_weights * routed_scaling_factor, topk_indices
