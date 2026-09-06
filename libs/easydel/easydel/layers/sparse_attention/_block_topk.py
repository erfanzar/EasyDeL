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

"""Block-granular budgeted top-k token indexer (Qwen4-Exp QSA).

Reference semantics (HF ``Qwen4ExpTextQSAIndexer``), restated vectorized:

1. A fused ``index_qk_proj`` maps hidden states to indexer queries
   (``n_heads`` heads) and one raw key per token (``kv_heads == 1``).
2. Queries are per-head RMSNormed and RoPEd at their *own* positions; raw keys
   are cached **unnormed and unroped** -- compression happens at read time.
3. For each query, the visible prefix (causal + padding) is grouped into
   complete blocks of ``compress_ratio`` consecutive visible tokens; each block
   is mean-pooled, RMSNormed, and RoPEd at the block's *start* position.
4. Scores are ``sum_h relu(q_h . k_block) / sqrt(head_dim)``; the top
   ``budget // compress_ratio`` blocks are expanded back to their member
   tokens, and the trailing incomplete block is always appended.
5. The result is a boolean mask ``[batch, 1, q_len, kv_len]`` the attention
   layer ANDs with its causal mask.

The vectorization assumes the visible set per row is a contiguous prefix
(positions ``first_visible..q``), which is exactly what causal masking with
optional padding produces. Everything is fixed-shape and jit-safe; invalid
selections are ``-1``-padded and scattered to a drop slot, matching the
reference.

Weight layout matches the checkpoint: ``index_qk_proj.weight``
``[(n_heads + kv_heads) * head_dim, hidden]`` (EasyDeL stores the transpose),
``q_layernorm.weight`` / ``k_layernorm.weight`` ``[head_dim]`` with the
zero-centred ``(1 + w)`` convention.
"""

from __future__ import annotations

import math

import jax
import spectrax as spx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ..linears import ColumnParallelLinear
from ..norms import RMSNorm

__all__ = ("BlockTopKIndexer", "apply_partial_rope")


def apply_partial_rope(
    x: Float[Array, "... dim"],
    cos: Float[Array, "... rotary"],
    sin: Float[Array, "... rotary"],
) -> Float[Array, "... dim"]:
    """Apply NeoX-style (split-half) RoPE to the leading ``cos.shape[-1]`` channels.

    The trailing channels past the rotary width pass through unchanged. This is
    the ``apply_rotary_pos_emb`` of the reference: ``rotate_half`` pairs
    channel ``i`` with ``i + rotary/2`` inside the rotated prefix only.

    Args:
        x: Tensor whose trailing axis carries the rotated prefix.
        cos: Cosines, broadcastable to ``x[..., :rotary_dim]``.
        sin: Sines, same shape rules as ``cos``.

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    rotary_dim = cos.shape[-1]
    x_rope, x_nope = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    # cos/sin arrive pre-doubled ([half | half]); this is exactly
    # (x1 + i x2) * (cos + i sin) in split-half (NeoX) form.
    rotated = jnp.concatenate(
        [x1 * cos[..., :half] - x2 * sin[..., :half], x2 * cos[..., half:] + x1 * sin[..., half:]], axis=-1
    )
    if rotary_dim == x.shape[-1]:
        return rotated
    return jnp.concatenate([rotated, x_nope], axis=-1)


class BlockTopKIndexer(spx.Module):
    """Qwen4-Exp QSA indexer: block top-k token selection for sparse attention.

    Attributes:
        index_qk_proj: Fused projection to indexer queries + raw token keys.
        q_layernorm: Per-head RMSNorm on queries (before RoPE).
        k_layernorm: RMSNorm on pooled block keys (before RoPE).
    """

    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_kv_heads: int,
        index_head_dim: int,
        indexer_budget: int,
        indexer_compress_ratio: int,
        eps: float = 1e-6,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: spx.Rngs,
    ) -> None:
        """Build the indexer.

        Args:
            hidden_size: Model hidden width (input width of ``index_qk_proj``).
            index_n_heads: Number of indexer query heads.
            index_kv_heads: Number of indexer key heads; QSA requires ``1``.
            index_head_dim: Per-head indexer width.
            indexer_budget: Maximum number of tokens selected from complete
                blocks per query; must divide by ``indexer_compress_ratio``.
            indexer_compress_ratio: Tokens mean-pooled into one block key.
            eps: RMSNorm epsilon.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.

        Raises:
            ValueError: On the same invariants the reference config validates.
        """
        if index_kv_heads != 1:
            raise ValueError(f"BlockTopKIndexer requires indexer_kv_heads=1, got {index_kv_heads}.")
        if indexer_budget % indexer_compress_ratio:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio.")
        self.hidden_size = hidden_size
        self.index_n_heads = index_n_heads
        self.index_kv_heads = index_kv_heads
        self.index_head_dim = index_head_dim
        self.token_budget = indexer_budget
        self.compress_ratio = indexer_compress_ratio
        self.block_topk = indexer_budget // indexer_compress_ratio
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.index_qk_proj = ColumnParallelLinear(
            hidden_size,
            (index_n_heads + index_kv_heads) * index_head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(0.02),
            rngs=rngs,
        )
        norm_kwargs = dict(
            eps=eps,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            scale_offset=1.0,
            kernel_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.q_layernorm = RMSNorm(index_head_dim, **norm_kwargs)
        self.k_layernorm = RMSNorm(index_head_dim, **norm_kwargs)

    def project(self, hidden_states: Float[Array, "batch seq hidden"]) -> tuple[Array, Array]:
        """Project hidden states to indexer queries and raw token keys.

        Args:
            hidden_states: ``[batch, seq, hidden]`` (the current tokens only).

        Returns:
            ``(q, raw_k)``: queries ``[batch, seq, n_heads, head_dim]``
            (normed, not yet roped) and raw keys ``[batch, seq, head_dim]``
            (neither normed nor roped -- cache these verbatim).
        """
        batch, seq = hidden_states.shape[:2]
        qk = self.index_qk_proj(hidden_states)
        q, k = jnp.split(qk, [self.index_n_heads * self.index_head_dim], axis=-1)
        q = q.reshape(batch, seq, self.index_n_heads, self.index_head_dim)
        q = self.q_layernorm(q)
        return q, k.reshape(batch, seq, self.index_head_dim)

    def select(
        self,
        q: Float[Array, "batch qseq heads dim"],
        raw_k: Float[Array, "batch kv dim"],
        *,
        q_cos: Float[Array, "batch qseq rotary"],
        q_sin: Float[Array, "batch qseq rotary"],
        k_cos: Float[Array, "batch kv rotary"],
        k_sin: Float[Array, "batch kv rotary"],
        visible: Bool[Array, "batch kv"] | None = None,
        kv_positions: Int[Array, "batch kv"] | None = None,
        q_indices: Int[Array, "batch qseq"] | None = None,
        q_segment_ids: Int[Array, "batch qseq"] | None = None,
        kv_segment_ids: Int[Array, "batch kv"] | None = None,
        return_blocks: bool = False,
        return_score_proxy: bool = False,
    ) -> Int[Array, "batch qseq budget+ratio-1"]:
        """Select up to ``token_budget`` visible tokens per query.

        Args:
            q: Indexer queries (normed, unroped), current tokens only.
            raw_k: Raw indexer keys for the *full* prefix (cached + current).
            q_cos, q_sin: RoPE tables at the current query positions.
            k_cos, k_sin: RoPE tables at every prefix position.
            visible: Boolean prefix visibility (padding); ``None`` means all
                positions visible. Causality is handled here from positions, so
                pass the padding mask only.
            kv_positions: Position id of every prefix token (defaults to
                ``arange``); needed when the prefix is not zero-based.
            q_indices: Absolute sequence index of each query (defaults to the
                trailing ``q_len`` positions, i.e. an unpadded prefill). Pass
                explicitly when decoding into a pre-sized buffer.

        Returns:
            Token indices ``[batch, qseq, token_budget + compress_ratio - 1]``,
            ``-1``-padded, reference-compatible.
        """
        batch, q_len, _n_heads, head_dim = q.shape
        kv_len = raw_k.shape[1]
        ratio = self.compress_ratio
        max_blocks = (kv_len + ratio - 1) // ratio

        q_roped = apply_partial_rope(q, q_cos[:, :, None, :], q_sin[:, :, None, :])

        if visible is None:
            visible = jnp.ones((batch, kv_len), jnp.bool_)
        if q_indices is None:
            q_indices = jnp.arange(q_len, dtype=jnp.int32)[None, :] + (kv_len - q_len)
        q_idx = jnp.broadcast_to(q_indices, (batch, q_len))

        if q_segment_ids is not None or kv_segment_ids is not None:
            if q_segment_ids is None or kv_segment_ids is None:
                raise ValueError("q_segment_ids and kv_segment_ids must be provided together")
            raise NotImplementedError(
                "packed-document QSA is not supported until selection preserves "
                "the model's segment-local pooled-block ranking semantics"
            )
        if kv_positions is None:
            kv_positions = jnp.broadcast_to(jnp.arange(kv_len, dtype=jnp.int32)[None, :], (batch, kv_len))
        # ``kv_positions`` carries absolute RoPE positions, but block
        # membership is defined by compact buffer indices. Mixing the two made
        # non-zero-based prefixes generate out-of-range block IDs.
        compact_positions = jnp.arange(kv_len, dtype=jnp.int32)[None, :]

        # First visible position per row; blocks are counted from it so the
        # grouping matches the reference's compaction of the visible list.
        first_visible = jnp.argmax(visible.astype(jnp.int32), axis=1)  # 0 when all-false
        has_visible = visible.any(axis=1)
        first_visible = jnp.where(has_visible, first_visible, 0)

        # Block id per compact buffer position. Causal-contiguity makes
        # membership exact even when the corresponding RoPE positions start at
        # a non-zero absolute offset.
        rel = compact_positions - first_visible[:, None]
        block_id = jnp.where(rel >= 0, rel // ratio, -1)  # [B, K]

        # Pooled block keys via scatter reduction. This is O(B*K*D) storage and
        # avoids the former [B,K,ceil(K/R)] one-hot (quadratic in cache length).
        member = (block_id >= 0) & visible  # [B, K]
        safe_block_id = jnp.clip(block_id, 0, max_blocks - 1)
        batch_idx = jnp.arange(batch, dtype=jnp.int32)[:, None]
        block_sum = jnp.zeros((batch, max_blocks, raw_k.shape[-1]), jnp.float32)
        block_sum = block_sum.at[batch_idx, safe_block_id].add(
            jnp.where(member[..., None], raw_k.astype(jnp.float32), 0.0)
        )
        block_cnt = jnp.zeros((batch, max_blocks), jnp.float32)
        block_cnt = block_cnt.at[batch_idx, safe_block_id].add(member.astype(jnp.float32))
        pooled = block_sum / jnp.maximum(block_cnt, 1.0)[..., None]
        pooled = self.k_layernorm(pooled.astype(raw_k.dtype))
        complete = block_cnt >= ratio  # [B, L]

        # Block start positions -> rope the pooled keys there.
        start_pos = first_visible[:, None] + jnp.arange(max_blocks, dtype=jnp.int32)[None, :] * ratio  # [B, L]
        start_pos = jnp.clip(start_pos, 0, kv_len - 1)
        blk_cos = jnp.take_along_axis(k_cos, start_pos[..., None], axis=1)  # [B, L, R]
        blk_sin = jnp.take_along_axis(k_sin, start_pos[..., None], axis=1)
        block_keys = apply_partial_rope(pooled, blk_cos, blk_sin)  # [B, L, D]

        # Scores: sum_h relu(q . k) / sqrt(d), fp32 like the reference.
        scores = jnp.einsum("bqhd,bld->bqhl", q_roped.astype(jnp.float32), block_keys.astype(jnp.float32))
        scores = jnp.sum(jax.nn.relu(scores), axis=2) / math.sqrt(head_dim)  # [B, Q, L]

        # A block is open to a query iff it is complete and its last member is
        # causally reachable from the query's absolute sequence index.
        block_end = start_pos + (ratio - 1)
        open_mask = complete[:, None, :] & (block_end[:, None, :] <= q_idx[..., None])  # [B, Q, L]
        scores = jnp.where(open_mask, scores, -jnp.inf)

        k_pick = min(self.block_topk, max_blocks)
        top_scores, top_blocks = jax.lax.top_k(scores, k_pick)  # [B, Q, k]
        picked_valid = top_scores > -jnp.inf
        q_live = (q_idx - first_visible[:, None]) >= 0  # [B, Q]
        selected = self._expand_selection(top_blocks, picked_valid, first_visible, q_idx, q_live)
        if return_blocks:
            return selected, block_keys, complete
        if return_score_proxy:
            # Expand block scores to token positions. The proxy is used only as
            # a zero-valued straight-through bias on hard-selected tokens, so
            # inference numerics stay exact while LM loss trains the indexer.
            token_block_ids = jnp.broadcast_to(safe_block_id[:, None, :], (batch, q_len, kv_len))
            token_scores = jnp.take_along_axis(scores, token_block_ids, axis=-1)
            token_scores = jnp.where(member[:, None, :] & jnp.isfinite(token_scores), token_scores, 0.0)
            return selected, token_scores
        return selected

    def _expand_selection(
        self,
        top_blocks: Int[Array, "batch qseq k"],
        picked_valid: Bool[Array, "batch qseq k"],
        first_visible: Int[Array, "batch"],  # noqa: F821
        q_idx: Int[Array, "batch qseq"],
        q_live: Bool[Array, "batch qseq"],
    ) -> Int[Array, "batch qseq budget+ratio-1"]:
        """Expand picked block ids to member token indices plus the live tail.

        Block members are disjoint by construction and the incomplete tail
        block is always fully visible to its query, so the returned indices
        are duplicate-free, ``-1``-padded to ``token_budget + ratio - 1``.
        """
        batch, q_len, k_pick = top_blocks.shape
        ratio = self.compress_ratio
        picked_start = first_visible[:, None, None] + top_blocks * ratio  # [B, Q, k]
        members = picked_start[..., None] + jnp.arange(ratio, dtype=jnp.int32)  # [B, Q, k, R]
        members = jnp.where(picked_valid[..., None], members, -1)
        members = members.reshape(batch, q_len, k_pick * ratio)

        rel_q = q_idx - first_visible[:, None]  # [B, Q]
        n_complete = jnp.where(q_live, (rel_q + 1) // ratio, 0)
        tail_start = first_visible[:, None] + n_complete * ratio  # [B, Q]
        tail = tail_start[..., None] + jnp.arange(ratio - 1, dtype=jnp.int32)  # [B, Q, R-1]
        tail_valid = (tail <= q_idx[..., None]) & q_live[..., None]
        tail = jnp.where(tail_valid, tail, -1)

        selected = jnp.concatenate([members, tail], axis=-1)  # [B, Q, k*R + R-1]
        width = self.token_budget + ratio - 1
        if selected.shape[-1] > width:
            selected = selected[..., :width]
        elif selected.shape[-1] < width:
            selected = jnp.pad(selected, ((0, 0), (0, 0), (0, width - selected.shape[-1])), constant_values=-1)
        return selected.astype(jnp.int32)

    def select_step(
        self,
        q: Float[Array, "batch 1 heads dim"],
        *,
        q_cos: Float[Array, "batch 1 rotary"],
        q_sin: Float[Array, "batch 1 rotary"],
        key_buffer: Float[Array, "batch kv dim"],
        block_keys: Float[Array, "batch blocks dim"],
        blocks_complete: Bool[Array, "batch blocks"],
        visible: Bool[Array, "batch kv"],
        open_cos: Float[Array, "batch 1 rotary"],
        open_sin: Float[Array, "batch 1 rotary"],
        write_at: Int[Array, "batch"],  # noqa: F821
    ) -> tuple[
        Int[Array, "batch 1 budget+ratio-1"],
        Float[Array, "batch blocks dim"],
        Bool[Array, "batch blocks"],
    ]:
        """Single-token decode selection with incremental block pooling.

        Only the block the current token lands in is re-pooled (its ``<=
        ratio`` raw members are re-meant exactly as the full path pools them);
        every earlier block keeps its frozen roped key. This replaces the
        full-buffer ``one_hot`` pooling pass — O(max_model_len) per step —
        with O(ratio) work, while the block-score ranking over the pooled-key
        buffer is unchanged.

        Args:
            q: Indexer queries for the current token (normed, unroped).
            q_cos, q_sin: RoPE tables at the current position.
            key_buffer: Raw indexer-key buffer (current token already written).
            block_keys: Roped pooled keys per block (state, updated here).
            blocks_complete: Closed-block flags (state, updated here).
            visible: Padding-visibility history.
            open_cos, open_sin: RoPE tables at the open block's start position.
            write_at: Absolute sequence index of the current token per row.

        Returns:
            ``(selected, block_keys, blocks_complete)`` — selected token
            indices ``[B, 1, budget + ratio - 1]`` and the updated state.
        """
        batch = q.shape[0]
        ratio = self.compress_ratio
        max_blocks = block_keys.shape[1]

        q_roped = apply_partial_rope(q, q_cos[:, :, None, :], q_sin[:, :, None, :])

        first_visible = jnp.argmax(visible.astype(jnp.int32), axis=1)
        has_visible = visible.any(axis=1)
        first_visible = jnp.where(has_visible, first_visible, 0)

        rel = write_at.astype(jnp.int32) - first_visible  # [B]
        q_live = rel >= 0
        b_open = jnp.where(q_live, rel // ratio, 0)
        open_start = first_visible + b_open * ratio
        cnt = jnp.where(q_live, rel - b_open * ratio + 1, 0)  # members incl. current

        # Re-pool the open block from its raw members (identical mean math to
        # the full path's fp32 one_hot einsum over the same <=ratio values).
        member_pos = open_start[:, None] + jnp.arange(ratio, dtype=jnp.int32)[None, :]  # [B, R]
        member_keys = jnp.take_along_axis(key_buffer, member_pos[..., None], axis=1)  # [B, R, D]
        member_vis = jnp.take_along_axis(visible, member_pos, axis=1)  # [B, R]
        member_valid = (jnp.arange(ratio, dtype=jnp.int32)[None, :] < cnt[:, None]) & member_vis
        pooled = (
            jnp.sum(member_keys.astype(jnp.float32) * member_valid[..., None], axis=1) / jnp.maximum(cnt, 1)[:, None]
        )
        pooled = self.k_layernorm(pooled.astype(key_buffer.dtype))
        pooled_roped = apply_partial_rope(pooled[:, None, :], open_cos, open_sin)[:, 0]

        rows = jnp.arange(batch, dtype=jnp.int32)
        block_keys = block_keys.at[rows, b_open].set(pooled_roped.astype(block_keys.dtype))
        closing = q_live & (cnt == ratio)
        blocks_complete = blocks_complete.at[rows, b_open].set(blocks_complete[rows, b_open] | closing)

        k_pick = min(self.block_topk, max_blocks)
        block_end = first_visible[:, None] + jnp.arange(max_blocks, dtype=jnp.int32)[None, :] * ratio + (ratio - 1)

        def select_all_blocks(_):
            top_blocks = jnp.broadcast_to(
                jnp.arange(k_pick, dtype=jnp.int32)[None, None, :],
                (batch, 1, k_pick),
            )
            picked_valid = blocks_complete[:, None, :k_pick] & (
                block_end[:, None, :k_pick] <= write_at.astype(jnp.int32)[:, None, None]
            )
            picked_valid = picked_valid & q_live[:, None, None]
            return self._expand_selection(
                top_blocks,
                picked_valid,
                first_visible,
                write_at[:, None].astype(jnp.int32),
                q_live[:, None],
            )

        def rank_blocks(_):
            # Rank blocks exactly as the full path: sum_h relu(q . k) / sqrt(d), fp32.
            scores = jnp.einsum("bqhd,bld->bqhl", q_roped.astype(jnp.float32), block_keys.astype(jnp.float32))
            scores = jnp.sum(jax.nn.relu(scores), axis=2) / math.sqrt(self.index_head_dim)
            open_mask = blocks_complete[:, None, :] & (
                block_end[:, None, :] <= write_at.astype(jnp.int32)[:, None, None]
            )
            scores = jnp.where(open_mask & q_live[:, None, None], scores, -jnp.inf)
            top_scores, top_blocks = jax.lax.top_k(scores, k_pick)
            return self._expand_selection(
                top_blocks,
                top_scores > -jnp.inf,
                first_visible,
                write_at[:, None].astype(jnp.int32),
                q_live[:, None],
            )

        # Before the live prefix exceeds the token budget, QSA selection is
        # vacuous: every complete block plus the open tail is retained. Avoid
        # scoring and top-k ranking the full max-context block buffer.
        all_within_budget = jnp.all(jnp.maximum(rel + 1, 0) <= self.token_budget)
        selected = jax.lax.cond(all_within_budget, select_all_blocks, rank_blocks, operand=None)
        return selected, block_keys, blocks_complete

    def build_mask(
        self,
        selected: Int[Array, "batch qseq width"],
        kv_len: int,
    ) -> Bool[Array, "batch 1 qseq kv"]:
        """Scatter selected token indices into a boolean attention mask.

        ``-1`` padding is absorbed into a drop slot one past the vocabulary of
        positions and sliced away, so no ``[batch, qseq, width, kv]`` one-hot
        is ever materialized.

        Args:
            selected: Output of :meth:`select`.
            kv_len: Prefix length the mask spans.

        Returns:
            Boolean mask ``[batch, 1, qseq, kv_len]``; ``True`` = attendable.
        """
        batch, q_len = selected.shape[:2]
        scatter_idx = jnp.where(selected >= 0, selected, kv_len)
        mask = jnp.zeros((batch, q_len, kv_len + 1), jnp.bool_)
        rows = jnp.broadcast_to(jnp.arange(batch)[:, None, None], scatter_idx.shape)
        qpos = jnp.broadcast_to(jnp.arange(q_len)[None, :, None], scatter_idx.shape)
        mask = mask.at[rows, qpos, scatter_idx].set(True, mode="drop")
        return mask[:, None, :, :kv_len]

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        *,
        q_cos: Array,
        q_sin: Array,
        k_cos: Array,
        k_sin: Array,
        cached_raw_k: Float[Array, "batch prefix dim"] | None = None,
        visible: Bool[Array, "batch kv"] | None = None,
        kv_positions: Int[Array, "batch kv"] | None = None,
        q_indices: Int[Array, "batch seq"] | None = None,
        q_segment_ids: Int[Array, "batch seq"] | None = None,
        kv_segment_ids: Int[Array, "batch kv"] | None = None,
        return_score_proxy: bool = False,
    ) -> tuple[Bool[Array, "batch 1 qseq kv"], Array] | tuple[Bool[Array, "batch 1 qseq kv"], Array, Array]:
        """Project, select, and build the sparse attention mask.

        Args:
            hidden_states: Current tokens ``[batch, seq, hidden]``.
            q_cos, q_sin: RoPE tables at the current positions.
            k_cos, k_sin: RoPE tables at every prefix position.
            cached_raw_k: Raw indexer keys from the cache (unnormed, unroped);
                the current tokens' keys are appended to them.
            visible: Padding visibility over the full prefix.
            kv_positions: Position ids over the full prefix.
            q_indices: Absolute sequence indices of the current queries.

        Returns:
            ``(mask, raw_k_full)``: boolean mask ``[batch, 1, seq, kv]`` and
            the full raw-key sequence (for the caller to write back to cache).
        """
        q, raw_k = self.project(hidden_states)
        if cached_raw_k is not None:
            raw_k_full = jnp.concatenate([cached_raw_k, raw_k], axis=1)
        else:
            raw_k_full = raw_k
        kv_len = raw_k_full.shape[1]
        selected_out = self.select(
            q,
            raw_k_full,
            q_cos=q_cos,
            q_sin=q_sin,
            k_cos=k_cos,
            k_sin=k_sin,
            visible=visible,
            kv_positions=kv_positions,
            q_indices=q_indices,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            return_score_proxy=return_score_proxy,
        )
        if return_score_proxy:
            selected, score_proxy = selected_out
            return self.build_mask(selected, kv_len), raw_k_full, score_proxy[:, None, :, :]
        return self.build_mask(selected_out, kv_len), raw_k_full
