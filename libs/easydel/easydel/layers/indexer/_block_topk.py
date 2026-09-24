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
from ._selection import BaseIndexer, IndexerSelection, SelectionSpec

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


class BlockTopKIndexer(BaseIndexer):
    """Qwen4-Exp QSA indexer: block top-k token selection for sparse attention.

    Attributes:
        index_qk_proj: Fused projection to indexer queries + raw token keys.
        q_layernorm: Per-head RMSNorm on queries (before RoPE).
        k_layernorm: RMSNorm on pooled block keys (before RoPE).
    """

    selection_spec = SelectionSpec(candidate_unit="block", output_unit="token")

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
        self.precision = precision

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
        # Split on the head axis, not the lane axis: slicing the lane axis
        # before the f32 norm aborts XLA:TPU compilation (jax 0.11.2, libtpu
        # 0.0.48) for bf16 inputs at seq >= 4096. The values are identical.
        qk = qk.reshape(batch, seq, self.index_n_heads + 1, self.index_head_dim)
        q = self.q_layernorm(qk[:, :, : self.index_n_heads])
        return q, qk[:, :, self.index_n_heads]

    def score_blocks(self, q: Array, block_keys: Array) -> Array:
        """Score dense roped blocks, preserving the dense head-reduction order.

        Args:
            q: Roped queries ``[batch, queries, heads, dim]``.
            block_keys: Normed, roped keys ``[batch, blocks, dim]``.

        Returns:
            Float32 block scores ``[batch, queries, blocks]``.
        """
        scores = jnp.einsum(
            "bqhd,bld->bqhl",
            q.astype(jnp.float32),
            block_keys.astype(jnp.float32),
            # Older serialized GraphDefs predate the scorer precision field.
            precision=getattr(self, "precision", None),
        )
        return jnp.sum(jax.nn.relu(scores), axis=2) / math.sqrt(self.index_head_dim)

    def score_paged_blocks(self, q: Array, physical_keys: Array, physical_block: Array) -> Array:
        """Score mapped physical blocks without a token-by-head-by-page tensor.

        Args:
            q: Roped queries ``[tokens, heads, dim]``.
            physical_keys: Normed, roped keys ``[physical_blocks, dim]``.
            physical_block: Logical-to-physical block map ``[tokens, blocks]``.

        Returns:
            Float32 scores in logical block order ``[tokens, blocks]``.

        Heads accumulate sequentially; the dense path deliberately retains its
        own reduction order.
        """
        q_score = q.astype(jnp.float32)
        k_score = physical_keys.astype(jnp.float32)

        def score_head(head, accum):
            head_scores = jnp.einsum("td,pd->tp", q_score[:, head], k_score, precision=getattr(self, "precision", None))
            return accum + jax.nn.relu(head_scores)

        physical_scores = jax.lax.fori_loop(
            0,
            self.index_n_heads,
            score_head,
            jnp.zeros((q.shape[0], physical_keys.shape[0]), dtype=jnp.float32),
        )
        scores = jnp.take_along_axis(physical_scores, physical_block, axis=1)
        return scores / math.sqrt(self.index_head_dim)

    def select_blocks(
        self,
        scores: Array,
        *,
        first_visible: Array,
        q_indices: Array,
        q_live: Array,
        kv_len: int | None = None,
    ) -> IndexerSelection:
        """Rank masked block scores and expand them to tokens plus the tail.

        Args:
            scores: ``[batch, queries, blocks]``; invalid blocks are ``-inf``.
            first_visible: First compact visible offset per batch row.
            q_indices: Compact query offsets ``[batch, queries]``.
            q_live: Whether each query has a visible prefix.
            kv_len: When given (``blocks * ratio >= kv_len``), also return the
                selection as a dense ``[batch, queries, kv_len]`` mask.

        Returns:
            Token selection with fixed width ``token_budget + ratio - 1``.
            Positive infinity is a valid score; negative infinity is not.
        """
        picked = self.select_candidates(scores, min(self.block_topk, scores.shape[-1]), mesh_source=self.mesh_source)
        selected = self._expand_selection(picked.indices, picked.indices >= 0, first_visible, q_indices, q_live)
        if kv_len is None or picked.mask is None:
            return IndexerSelection(selected)
        mask = self._expand_selection_mask(picked.mask, first_visible, q_indices, q_live, kv_len)
        return IndexerSelection(selected, mask=mask)

    def select_paged(
        self, q: Array, physical_keys: Array, physical_block: Array, q_indices: Array, valid: Array
    ) -> IndexerSelection:
        """Select logical token offsets from already prepared paged block keys.

        Physical page writes, mapping and RoPE preparation belong to the model;
        scoring, causal block eligibility, ranking and expansion belong here.

        Args:
            q: Roped queries ``[tokens, heads, dim]``.
            physical_keys: Normed, roped physical block keys.
            physical_block: Physical block id for each logical block per query.
            q_indices: Logical query token offsets ``[tokens]``.
            valid: Live query flags ``[tokens]``.

        Returns:
            Logical token indices ``[tokens, token_budget + ratio - 1]``.
        """
        scores = self.score_paged_blocks(q, physical_keys, physical_block)
        block = jnp.arange(scores.shape[-1], dtype=jnp.int32)[None, :]
        complete = block * self.compress_ratio + self.compress_ratio - 1 <= q_indices[:, None]
        scores = jnp.where(complete & valid[:, None], scores, -jnp.inf)
        selection = self.select_blocks(
            scores[:, None, :],
            first_visible=jnp.zeros_like(q_indices),
            q_indices=q_indices[:, None],
            q_live=(q_indices >= 0)[:, None],
        )
        return IndexerSelection(selection.indices[:, 0])

    def select_prefix(self, q_indices: Array) -> IndexerSelection:
        """Return every logical prefix token when all queries are within budget.

        Args:
            q_indices: Logical query offsets of any leading shape.

        Returns:
            Prefix tokens with a fixed trailing width and ``-1`` padding.
        """
        token = jnp.arange(self.token_budget + self.compress_ratio - 1, dtype=jnp.int32)
        return IndexerSelection(jnp.where(token <= q_indices[..., None], token, -1))

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
        return_selection: bool = False,
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
            kv_positions: Accepted for API compatibility; block membership uses
                compact buffer indices, not RoPE positions.
            q_indices: Absolute sequence index of each query (defaults to the
                trailing ``q_len`` positions, i.e. an unpadded prefill). Pass
                explicitly when decoding into a pre-sized buffer.
            return_selection: Return an :class:`IndexerSelection` that also
                carries the dense ``[batch, qseq, kv]`` mask in place of the
                bare indices.

        Returns:
            Token indices ``[batch, qseq, token_budget + compress_ratio - 1]``,
            ``-1``-padded, reference-compatible.
        """
        batch, q_len, _n_heads, _head_dim = q.shape
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
        # ``kv_positions`` carries absolute RoPE positions, but block
        # membership is defined by compact buffer indices.
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

        # Pooled block keys via scatter reduction: O(B*K*D) storage, no
        # [B,K,ceil(K/R)] one-hot.
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
        scores = self.score_blocks(q_roped, block_keys)

        # A block is open to a query iff it is complete and its last member is
        # causally reachable from the query's absolute sequence index.
        block_end = start_pos + (ratio - 1)
        open_mask = complete[:, None, :] & (block_end[:, None, :] <= q_idx[..., None])  # [B, Q, L]
        scores = jnp.where(open_mask, scores, -jnp.inf)

        q_live = (q_idx - first_visible[:, None]) >= 0  # [B, Q]
        selection = self.select_blocks(
            scores,
            first_visible=first_visible,
            q_indices=q_idx,
            q_live=q_live,
            kv_len=kv_len if return_selection else None,
        )
        selected = selection if return_selection else selection.indices
        if return_blocks:
            return selected, block_keys, complete
        if return_score_proxy:
            # Expand block scores to token positions. The proxy is used only as
            # a zero-valued straight-through bias on hard-selected tokens, so
            # inference numerics stay exact while LM loss trains the indexer.
            token_scores = self._blocks_to_tokens(scores, first_visible, kv_len)
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

    def _expand_selection_mask(
        self,
        block_mask: Bool[Array, "batch qseq blocks"],
        first_visible: Int[Array, "batch"],  # noqa: F821
        q_idx: Int[Array, "batch qseq"],
        q_live: Bool[Array, "batch qseq"],
        kv_len: int,
    ) -> Bool[Array, "batch qseq kv"]:
        """Dense form of :meth:`_expand_selection`, without scattering indices.

        Block ``b`` covers tokens ``first_visible + b*ratio + [0, ratio)``, so
        the block mask is repeated over its members and shifted by
        ``first_visible``; the tail is a range compare. This avoids scattering
        ``token_budget`` indices per query, which is slow on TPU.
        """
        ratio = self.compress_ratio
        members = self._blocks_to_tokens(block_mask, first_visible, kv_len)
        rel_q = q_idx - first_visible[:, None]
        n_complete = jnp.where(q_live, (rel_q + 1) // ratio, 0)
        tail_start = (first_visible[:, None] + n_complete * ratio)[..., None]
        position = jax.lax.broadcasted_iota(jnp.int32, members.shape, 2)
        tail = (position >= tail_start) & (position < tail_start + ratio - 1) & (position <= q_idx[..., None])
        return members | (tail & q_live[..., None])

    def _blocks_to_tokens(self, per_block: Array, first_visible: Array, kv_len: int) -> Array:
        """Spread ``[batch, q, blocks]`` values to ``[batch, q, kv_len]`` tokens.

        Token ``t >= first_visible`` takes its block's value
        ``per_block[..., (t - first_visible) // ratio]``; earlier tokens get
        zero. Repeat-and-shift avoids an element gather (and, in reverse mode,
        a scatter-add), both slow on TPU.
        ``blocks * ratio >= kv_len`` is required.
        """
        tokens = jnp.repeat(per_block, self.compress_ratio, axis=-1)
        tokens = jnp.pad(tokens, ((0, 0), (0, 0), (kv_len, 0)))
        return jax.vmap(lambda row, start: jax.lax.dynamic_slice_in_dim(row, start, kv_len, axis=-1))(
            tokens, kv_len - first_visible
        )

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
        every earlier block keeps its frozen roped key. This is O(ratio) work
        per step instead of re-pooling the full buffer; the block-score
        ranking over the pooled-key buffer is unchanged.

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

        # Re-pool the open block from its raw members (same fp32 mean as the
        # full path).
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
            scores = self.score_blocks(q_roped, block_keys)
            open_mask = blocks_complete[:, None, :] & (
                block_end[:, None, :] <= write_at.astype(jnp.int32)[:, None, None]
            )
            scores = jnp.where(open_mask & q_live[:, None, None], scores, -jnp.inf)
            return self.select_blocks(
                scores,
                first_visible=first_visible,
                q_indices=write_at[:, None].astype(jnp.int32),
                q_live=q_live[:, None],
            ).indices

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
        return IndexerSelection(selected).to_mask(kv_len)[:, None, :, :]

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
            return_selection=True,
        )
        if return_score_proxy:
            selection, score_proxy = selected_out
            return selection.to_mask(kv_len)[:, None, :, :], raw_k_full, score_proxy[:, None, :, :]
        return selected_out.to_mask(kv_len)[:, None, :, :], raw_k_full
