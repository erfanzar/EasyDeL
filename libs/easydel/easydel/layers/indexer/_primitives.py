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

"""Shared selection primitives for sparse-attention indexers.

Everything here is jit-safe and fixed-shape: causal/padding visibility,
top-k with ``-1`` padding, selection-to-mask conversion, the
straight-through score proxy used to train indexers through hard top-k, and
the vacuous-selection guard every family re-implements slightly differently.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

__all__ = (
    "indices_to_bool_mask",
    "is_vacuous_selection",
    "ste_score_proxy",
    "topk_select",
    "visible_causal_mask",
)


def visible_causal_mask(
    q_length: int,
    kv_length: int,
    key_valid: Bool[Array, "batch kv"] | None = None,
) -> Bool[Array, "batch q kv"]:
    """Combine causal visibility with optional key validity.

    Args:
        q_length: Number of query positions in this step.
        kv_length: Total KV length (cache included).
        key_valid: Boolean key validity (padding/cache flag); ``None`` means
            every position is valid.

    Returns:
        Boolean ``(batch, q_length, kv_length)`` visibility: position ``t``
        is visible to query ``q`` iff ``t <= kv_length - q_length + q`` (and
        valid).
    """
    kv_positions = jnp.arange(kv_length)
    q_positions = kv_length - q_length + jnp.arange(q_length)
    causal = kv_positions[None, None, :] <= q_positions[None, :, None]
    if key_valid is None:
        batch = 1
        return jnp.broadcast_to(causal, (batch, q_length, kv_length))
    return causal & key_valid[:, None, :]


def topk_select(
    scores: Float[Array, "batch q n"],
    k: int,
    invalid_value: float | None = None,
    valid: Bool[Array, "batch q n"] | None = None,
) -> Int[Array, "batch q k"]:
    """Top-k over the last axis with ``-1`` padding for invalid picks.

    Args:
        scores: Candidate scores; ``-inf``/very-negative entries are treated
            as unselectable.
        k: Number of selections (clamped to ``n``).
        invalid_value: Optional sentinel already present in ``scores``; when
            given together with ``valid`` the explicit ``valid`` map wins.
        valid: Optional boolean candidate validity; picks failing it are
            rewritten to ``-1``.

    Returns:
        Integer indices ``(batch, q, k)``, ``-1`` where a pick is invalid.
    """
    n = scores.shape[-1]
    k = min(k, n)
    _, idx = jax.lax.top_k(scores, k)
    if valid is not None:
        picked_valid = jnp.take_along_axis(valid.astype(jnp.bool_), idx, axis=-1)
        idx = jnp.where(picked_valid, idx, -1)
    elif invalid_value is not None:
        picked = jnp.take_along_axis(scores, idx, axis=-1)
        idx = jnp.where(picked > invalid_value, idx, -1)
    return idx.astype("i4")


def indices_to_bool_mask(
    topk_indices: Int[Array, "batch q k"],
    kv_length: int,
) -> Bool[Array, "batch q kv"]:
    """Expand ``-1``-padded selected indices into a dense boolean mask.

    Args:
        topk_indices: Selected positions per query (``-1`` = none).
        kv_length: Width of the dense KV axis.

    Returns:
        Boolean ``(batch, q, kv_length)``; ``True`` where the position is
        selected. Duplicate selections collapse (``any``).
    """
    safe = jnp.clip(topk_indices, 0, kv_length - 1)
    one_hot = jax.nn.one_hot(safe, kv_length, dtype=jnp.bool_)
    # ``-1`` slots must select nothing: clip alone would map them to
    # position 0, so mask those lanes out before the reduction.
    valid = topk_indices >= 0
    one_hot = jnp.where(valid[..., None], one_hot, False)
    return jnp.any(one_hot, axis=-2)


def ste_score_proxy(
    score_proxy: Float[Array, "..."],
    selected: Bool[Array, "..."] | Int[Array, "..."],
) -> Float[Array, "..."]:
    """Straight-through score surrogate added to an attention bias.

    Zero in the primal, identity in the tangent: ``s - stop_grad(s)`` lets LM
    loss gradients train the indexer through the hard top-k.

    Args:
        score_proxy: Indexer scores at the selected positions.
        selected: Boolean selection map (or 0/1 proxy).

    Returns:
        The surrogate term, same shape as ``score_proxy``.
    """
    gate = score_proxy - jax.lax.stop_gradient(score_proxy)
    return jnp.where(selected, gate, 0.0)


def is_vacuous_selection(index_topk: int, n_slots: int) -> bool:
    """Whether the indexer cannot exclude anything (static, trace-time).

    Args:
        index_topk: Configured selection budget.
        n_slots: Number of selectable slots (static: ``max_model_len``
            derived, not data-dependent).

    Returns:
        ``True`` when ``index_topk >= n_slots`` — every candidate is always
        selectable and the whole indexer can be skipped.
    """
    return index_topk >= n_slots
