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

"""Shared top-k expert selection for auxiliary-loss-free (``noaux_tc``) routing.

Ten model families implemented the same routing pipeline independently — three
of them byte-identically. This module holds the one implementation they all
delegate to, as a :attr:`MoeFusedHooks.select_hook`.

The pipeline, in order:

1. Turn router logits into scores (``sigmoid``, ``softmax``, or pass through
   already-scored logits).
2. Add the per-expert selection bias, if any. **The bias steers selection
   only** — the returned weights are gathered from the *unbiased* scores. This
   is what makes the routing auxiliary-loss-free: the bias moves load between
   experts without perturbing the values that reach the combine.
3. Optionally restrict the candidates to the ``topk_group`` best expert groups,
   where a group scores as the sum of its top-``group_topk_k`` members (or as
   its single best member).
4. Take a flat top-``k`` over the survivors, gather the unbiased scores, and
   optionally renormalise and rescale them.

Every knob exists because some family needed it; none is speculative. See
:func:`moe_group_topk_select` for which family uses which setting.
"""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

ScoreFn = tp.Literal["sigmoid", "softmax", "none"]
GroupScore = tp.Literal["topk_sum", "max"]

__all__ = ("moe_group_topk_select",)


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


def _group_restrict(
    scores_for_choice: Array,
    *,
    n_routed_experts: int,
    n_group: int,
    topk_group: int,
    group_topk_k: int,
    group_score: GroupScore,
) -> Array:
    """Zero every expert outside the ``topk_group`` best groups.

    Args:
        scores_for_choice: Selection scores ``(num_tokens, n_routed_experts)``.
        n_routed_experts: Total routed experts; must divide by ``n_group``.
        n_group: Number of equal-sized expert groups.
        topk_group: Groups kept per token.
        group_topk_k: Members summed to score a group (``group_score='topk_sum'``).
        group_score: How a group is scored from its members.

    Returns:
        ``scores_for_choice`` with non-surviving groups set to ``-inf``.
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

    group_idx = jax.lax.top_k(group_scores, k=min(topk_group, n_group))[1]
    group_mask = jnp.sum(jax.nn.one_hot(group_idx, n_group, dtype=scores_for_choice.dtype), axis=1)
    score_mask = jnp.repeat(group_mask, group_size, axis=1)
    return jnp.where(score_mask > 0, scores_for_choice, -jnp.inf)


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

    - GLM-4-MoE / GLM-4-MoE-Lite / GLM-MoE-DSA, DeepSeek-V3, Kimi-Linear:
      ``score_fn='sigmoid'``, grouped, ``group_score='topk_sum'``.
    - DeepSeek-V2: ``score_fn='softmax'``, grouped, ``group_score='max'``.
    - Mistral-4: ``score_fn='none'`` (the gate already softmaxed), grouped.
    - Hunyuan-V3, DeepSeek-V4: flat (``n_group=1``), ``score_fn='sigmoid'``.
    - MiniMax-M3-VL: flat, ``score_fn='sigmoid'``, ``norm_eps=0.0``.

    Args:
        gate_logits: Router logits ``(num_tokens, n_routed_experts)``.
        pre_bias_logits: Accepted and ignored; part of the ``select_hook``
            signature that some callers pass positionally.
        k: Routed experts selected per token.
        n_routed_experts: Total routed experts.
        score_fn: ``'sigmoid'``, ``'softmax'``, or ``'none'`` when the caller
            already produced scores.
        e_score_correction_bias: Per-expert selection bias, or ``None`` for no
            bias. Added to the selection scores only; never to the returned
            weights.
        n_group: Expert groups. ``1`` disables the group stage entirely.
        topk_group: Groups kept per token.
        group_topk_k: Members summed to score a group under ``'topk_sum'``.
        group_score: ``'topk_sum'`` or ``'max'``.
        norm_topk_prob: Divide the selected weights by their sum.
        routed_scaling_factor: Final multiplier on the weights.
        norm_eps: Added to the normalisation denominator. ``1e-20`` guards a
            degenerate all-zero row; pass ``0.0`` to reproduce a bare sum.

    Returns:
        ``(topk_weights, topk_indices)``, both ``(num_tokens, k)``.
    """
    del pre_bias_logits

    scores = _score(gate_logits, score_fn)

    if e_score_correction_bias is None:
        scores_for_choice = scores
    else:
        scores_for_choice = scores + e_score_correction_bias.astype(jnp.float32)

    if n_group > 1:
        scores_for_choice = _group_restrict(
            scores_for_choice,
            n_routed_experts=n_routed_experts,
            n_group=n_group,
            topk_group=topk_group,
            group_topk_k=group_topk_k,
            group_score=group_score,
        )

    topk_indices = jax.lax.top_k(scores_for_choice, k=k)[1]
    topk_weights = jnp.take_along_axis(scores, topk_indices, axis=-1)

    if norm_topk_prob:
        topk_weights = topk_weights / (jnp.sum(topk_weights, axis=-1, keepdims=True) + norm_eps)

    return topk_weights * routed_scaling_factor, topk_indices
