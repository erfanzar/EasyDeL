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

import math
import typing as tp

import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

__all__ = (
    "CompactLayout",
    "compact_valid_layout",
    "indices_to_bool_mask",
    "is_vacuous_selection",
    "mask_to_indices",
    "permute_tokens",
    "ste_score_proxy",
    "topk_select",
    "topk_selection_mask",
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


def topk_selection_mask(
    scores: Float[Array, "... n"],
    values: Float[Array, "... k"],
    indices: Int[Array, "... k"],
) -> Bool[Array, "... n"]:
    """Convert a ``lax.top_k`` selection to a boolean mask without scatter.

    Resolve the shared operation helper only when called: cache construction
    imports indexer layers before the caching package has finished exporting
    the cache-view types consumed by operations.

    Args:
        scores: The exact scores ranked by ``top_k``, shaped ``[..., n]``.
        values: Descending selected scores ``[..., k]``, with ``k >= 1``.
        indices: Selected indices of the same shape as ``values``.

    Returns:
        Boolean mask ``[..., n]`` matching the selected indices, including
        IEEE total-order ties.
    """
    from easydel.operations.kernels._mask_info import topk_selection_mask as selection_mask

    return selection_mask(scores=scores, values=values, indices=indices)


def mask_to_indices(mask: Bool[Array, "... n"], k: int) -> Int[Array, "... k"]:
    """Compact a boolean selection into ``-1``-padded indices without a sort.

    The inverse of :func:`indices_to_bool_mask` for a selection of at most
    ``k`` items per row: selected positions are written in ascending index
    order (not score order) and the remaining slots hold ``-1``. Selected
    positions beyond the ``k``-th are dropped. Under jit the scatter is dead
    code when the caller only consumes the mask.

    Args:
        mask: Boolean selection ``[..., n]``.
        k: Static output width, non-negative.

    Returns:
        Int32 indices ``[..., k]``.

    Raises:
        ValueError: If ``k`` is negative.
    """
    if k < 0:
        raise ValueError("Selection width must be non-negative.")
    shape = mask.shape[:-1]
    width = mask.shape[-1]
    rows = math.prod(shape)
    flat = mask.reshape(rows, width).astype(jnp.bool_)
    # Slot of each selected position; unselected positions target slot k, which is out of range and dropped.
    slot = jnp.where(flat, jnp.cumsum(flat, axis=-1, dtype=jnp.int32) - 1, k)
    index = jnp.broadcast_to(jnp.arange(width, dtype=jnp.int32), flat.shape)
    out = jnp.full((rows, k), -1, dtype=jnp.int32)
    out = out.at[jnp.arange(rows)[:, None], slot].set(index, mode="drop")
    return out.reshape(*shape, k)


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
    # One scatter into a ``[rows, kv + 1]`` buffer instead of a ``[..., k, kv]`` one-hot cube. ``-1`` lanes land on
    # the extra drop column (select nothing); other indices clip into range; duplicates collapse.
    valid = topk_indices >= 0
    safe = jnp.where(valid, jnp.clip(topk_indices, 0, kv_length - 1), kv_length)
    shape = topk_indices.shape[:-1]
    rows = math.prod(shape)
    flat = safe.reshape(rows, topk_indices.shape[-1])
    mask = jnp.zeros((rows, kv_length + 1), dtype=jnp.bool_)
    mask = mask.at[jnp.arange(rows)[:, None], flat].set(True)
    return mask[:, :kv_length].reshape(*shape, kv_length)


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


class CompactLayout(tp.NamedTuple):
    """Valid-first token order of every row (see :func:`compact_valid_layout`).

    Attributes:
        order: ``[batch, seq]`` int32; ``order[b, j]`` is the physical index of compact slot ``j`` (valid tokens
            first, then padding, each in position order).
        inverse: ``[batch, seq]`` int32 inverse permutation (``inverse[b, order[b, j]] == j``).
        rank: ``[batch, seq]`` int32 compact position of every valid token (valid tokens before it), ``-1`` for
            padding.
        identity: Scalar bool, ``True`` when every row is already valid-first (no padding or right padding only).
    """

    order: Array
    inverse: Array
    rank: Array
    identity: Array

    def gather(self, x: Array) -> Array:
        """``x[b, order[b, j]]`` along the sequence axis of ``x [batch, seq, ...]`` (see :func:`permute_tokens`).

        Skips the permutation at run time (``lax.cond``) when the layout is the identity, the common case of
        unpadded or right-padded training batches.
        """
        return jax.lax.cond(
            self.identity,
            lambda rows: rows,
            lambda rows: permute_tokens(rows, self.order, self.inverse),
            x,
        )


def compact_valid_layout(valid: Bool[Array, "batch seq"]) -> CompactLayout:
    """Valid-first token order of every row.

    Strategies that group consecutive *valid* tokens (learned pools and their incomplete tail) define their key grid
    over the compacted row: valid tokens in position order, padding after them. A kernel that reads contiguous key
    ranges needs the keys in that order (:meth:`CompactLayout.gather`), while each query keeps its own slot and uses
    its compact rank as its causal position. For rows whose valid tokens form a prefix the layout is the identity.

    Args:
        valid: Token validity ``[batch, seq]``.

    Returns:
        The :class:`CompactLayout` of every row.
    """
    seq_len = valid.shape[1]
    is_valid = valid.astype(jnp.int32)
    is_pad = 1 - is_valid
    valid_through = jnp.cumsum(is_valid, axis=1)
    pad_through = jnp.cumsum(is_pad, axis=1)
    before_valid = valid_through - is_valid
    before_pad = pad_through - is_pad
    num_valid = valid_through[:, -1:]
    inverse = jnp.where(is_valid > 0, before_valid, num_valid + before_pad).astype(jnp.int32)
    slots = jnp.arange(seq_len, dtype=jnp.int32)
    # A permutation is the identity exactly when its inverse is, so unpadded and right-padded batches are recognised
    # from ``inverse`` and skip the searches. On TPU v5p the searches took 0.55 s per training step of a 45-layer
    # GLM-5.3-Flash at 131072 tokens per sequence.
    identity = jnp.all(inverse == slots[None, :])

    def search(counts):
        # Compact slot j holds the (j)-th valid token (j < num_valid) or the (j - num_valid)-th padding token: the
        # first index whose running count exceeds it. A binary search per slot instead of an argsort of ``inverse``.
        # ``method="scan"``: log2(seq) vectorised steps, never the [seq, seq] comparison of ``"compare_all"``.
        valid_counts, pad_counts, n_valid = counts
        first_valid = jax.vmap(lambda c: jnp.searchsorted(c, slots, side="right", method="scan"))(valid_counts)
        first_pad = jax.vmap(lambda c, n: jnp.searchsorted(c, slots - n, side="right", method="scan"))(
            pad_counts, n_valid[:, 0]
        )
        order = jnp.where(slots[None, :] < n_valid, first_valid, first_pad)
        return jnp.clip(order, 0, seq_len - 1).astype(jnp.int32)

    def unchanged(counts):
        return jnp.broadcast_to(slots[None, :], counts[0].shape)

    order = jax.lax.cond(identity, unchanged, search, (valid_through, pad_through, num_valid))
    rank = jnp.where(is_valid > 0, before_valid, -1).astype(jnp.int32)
    return CompactLayout(order=order, inverse=inverse, rank=rank, identity=identity)


def segment_pool_max_documents(seq_len: int) -> int:
    """Documents per row a :func:`segment_pool_layout` key grid holds: one per 128 tokens, at least 64."""
    return max(int(seq_len) // 128, 64)


@jax.custom_vjp
def _segment_pool_gather(order: Array, rank: Array, x: Array) -> Array:
    """Gather ``x[b, order[b, g]]`` with an exact gather-by-``rank`` reverse-mode VJP.

    ``SegmentPoolLayout`` places each fitted valid token into at most one pool-grid slot
    (``order[b, rank[b, s]] == s`` when ``rank[b, s] >= 0``, and ``rank[b, s] == -1`` otherwise),
    so the transpose of gathering by ``order`` is one gather of ``dy`` by ``rank`` masked on
    ``rank >= 0`` rather than a ``[batch, grid, ...]`` scatter-add.
    """
    idx = jnp.maximum(order, 0)
    out = jax.vmap(lambda row, i: jnp.take(row, i, axis=0))(x, idx)
    filled = (order >= 0).reshape(order.shape + (1,) * (x.ndim - 2))
    return jnp.where(filled, out, jnp.zeros((), out.dtype))


def _segment_pool_gather_fwd(order: Array, rank: Array, x: Array):
    return _segment_pool_gather(order, rank, x), (order.shape, rank)


def _segment_pool_gather_bwd(residuals, dy: Array):
    order_shape, rank = residuals
    r_idx = jnp.maximum(rank, 0)
    dx = jax.vmap(lambda drow, i: jnp.take(drow, i, axis=0))(dy, r_idx)
    valid_r = (rank >= 0).reshape(rank.shape + (1,) * (dy.ndim - 2))
    dx = jnp.where(valid_r, dx, jnp.zeros((), dx.dtype))
    return (
        jnp.zeros(order_shape, dtype=jax.dtypes.float0),
        jnp.zeros(rank.shape, dtype=jax.dtypes.float0),
        dx,
    )


_segment_pool_gather.defvjp(_segment_pool_gather_fwd, _segment_pool_gather_bwd)


class SegmentPoolLayout(tp.NamedTuple):
    """Packed documents on a pool-aligned key grid (see :func:`segment_pool_layout`).

    Attributes:
        order: ``[batch, grid]`` int32 physical token in each grid slot, ``-1`` for an empty slot.
        rank: ``[batch, seq]`` int32 grid slot of every valid token, ``-1`` for padding and for tokens past the grid.
        key_segments: ``[batch, grid]`` int32 document of each grid slot, ``-1`` for an empty slot.
        query_segments: ``[batch, seq]`` int32 document of every token with a grid slot, ``-1`` otherwise.
        overflow: ``[batch]`` int32 valid tokens that did not fit the grid (their queries attend to nothing).
    """

    order: Array
    rank: Array
    key_segments: Array
    query_segments: Array
    overflow: Array

    def gather(self, x: Array) -> Array:
        """``x[b, order[b, g]]`` along the sequence axis of ``x [batch, seq, ...]``, zeros in empty slots."""
        return _segment_pool_gather(self.order, self.rank, x)


def segment_pool_layout(
    segment_ids: Int[Array, "batch seq"],
    kpool: int,
    max_documents: int | None = None,
) -> SegmentPoolLayout:
    """Lay packed documents out on a key grid where every document starts on a pool boundary.

    Valid tokens keep their order (padding removed, as in :func:`compact_valid_layout`); after each document the grid
    skips to the next multiple of ``kpool``. A pool therefore never holds keys of two documents, pools restart at every
    document start (the dense strategy's ``_pooled_states`` rule), and a query's incomplete tail pool
    ``kpool * (pos // kpool) .. pos`` lies inside its own document. A new document starts wherever the segment id
    changes between consecutive valid tokens.

    Args:
        segment_ids: Document ids ``[batch, seq]``, negative for padding.
        kpool: Tokens per pool.
        max_documents: Documents per row the grid has room for; ``None`` = :func:`segment_pool_max_documents`. The
            grid holds ``seq + (kpool - 1) * (max_documents - 1)`` slots rounded up to a pool. Tokens past it are
            dropped and counted in ``overflow``.

    Returns:
        The :class:`SegmentPoolLayout` of every row.
    """
    segment_ids = jnp.asarray(segment_ids, jnp.int32)
    batch, seq = segment_ids.shape
    kpool = int(kpool)
    if max_documents is None:
        max_documents = segment_pool_max_documents(seq)
    grid = -(-(seq + (kpool - 1) * max(int(max_documents) - 1, 0)) // kpool) * kpool
    valid = segment_ids >= 0
    base = compact_valid_layout(valid)
    cseg = jnp.take_along_axis(segment_ids, base.order, axis=1)
    slots = jnp.arange(seq, dtype=jnp.int32)[None, :]
    num_valid = jnp.sum(valid, axis=1, keepdims=True, dtype=jnp.int32)
    cvalid = slots < num_valid
    edge = jnp.full((batch, 1), -2, jnp.int32)
    prev = jnp.concatenate([edge, cseg[:, :-1]], axis=1)
    nxt = jnp.concatenate([cseg[:, 1:], edge], axis=1)
    nxt_valid = jnp.concatenate([cvalid[:, 1:], jnp.zeros((batch, 1), jnp.bool_)], axis=1)
    start = cvalid & (cseg != prev)
    end = cvalid & (~nxt_valid | (nxt != cseg))
    doc_start = jax.lax.cummax(jnp.where(start, slots, 0), axis=1)
    pad = jnp.where(end, (-(slots - doc_start + 1)) % kpool, 0).astype(jnp.int32)
    gpos = slots + jnp.cumsum(pad, axis=1, dtype=jnp.int32) - pad
    fits = cvalid & (gpos < grid)
    target = jnp.where(fits, gpos, grid)
    order = jax.vmap(lambda t, o: jnp.full((grid,), -1, jnp.int32).at[t].set(o, mode="drop"))(target, base.order)
    key_segments = jax.vmap(lambda t, s: jnp.full((grid,), -1, jnp.int32).at[t].set(s, mode="drop"))(target, cseg)
    rank = jnp.take_along_axis(jnp.where(fits, gpos, -1), base.inverse, axis=1)
    rank = jnp.where(valid, rank, -1).astype(jnp.int32)
    query_segments = jnp.where(rank >= 0, segment_ids, -1)
    overflow = jnp.sum(cvalid & ~fits, axis=1, dtype=jnp.int32)
    return SegmentPoolLayout(
        order=order, rank=rank, key_segments=key_segments, query_segments=query_segments, overflow=overflow
    )


def permute_tokens(x: Array, order: Int[Array, "batch seq"], inverse: Int[Array, "batch seq"]) -> Array:
    """``out[b, j] = x[b, order[b, j]]`` along the sequence axis, with gather-only derivatives.

    Posed as the linear solve ``out[:, inverse] = x`` (as the MoE token sort): the forward is one gather with
    ``order`` and, a permutation being orthogonal, reverse mode is one gather with ``inverse`` instead of the
    scatter-add JAX derives for a gather. Forward mode, batching and higher derivatives follow from
    :func:`jax.lax.custom_linear_solve`.

    Args:
        x: ``[batch, seq, ...]`` rows.
        order: Permutation of ``range(seq)`` per row (e.g. from :func:`compact_valid_layout`).
        inverse: Its inverse permutation.

    Returns:
        ``x`` with every row's sequence axis permuted by ``order``.
    """

    def take(rows: Array, index: Array) -> Array:
        return jnp.take_along_axis(rows, index.reshape(index.shape + (1,) * (rows.ndim - 2)), axis=1)

    return jax.lax.custom_linear_solve(
        lambda permuted: take(permuted, inverse),
        x,
        solve=lambda _, rows: take(rows, order),
        transpose_solve=lambda _, rows: take(rows, inverse),
    )
