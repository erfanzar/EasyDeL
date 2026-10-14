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

"""Shared-key lightning indexer with a candidate-block prefilter.

Index keys are ``k_norm(wk(latent))`` of a compressed-KV source's pre-RoPE latent rather than the output of an
indexer-private compressor, so several layers can score against one key set. Rotary application is supplied by the
caller as callables, keeping this layer independent of any rope convention.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import jax
import spectrax as spx
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int

from easydel.layers.compressed_window import MASK_MIN
from easydel.layers.linears import ColumnParallelLinear
from easydel.layers.norms import RMSNorm

from ._paged import bounded_block_max_topk, slot_entry_topk

__all__ = (
    "IndexerScoreInputs",
    "SharedKeyIndexer",
    "SharedKeyIndexerConfig",
    "select_candidate_blocks",
    "selection_bias",
    "shared_key_scores",
)


def select_candidate_blocks(
    scores: Float[Array, "batch seq entries"],
    compress_lens: Int[Array, "batch seq"],
    topk_blocks: int,
    block_size: int,
) -> Bool[Array, "batch seq entries"]:
    """Level one of a two-level top-k: keep each query's ``topk_blocks`` best blocks of entries.

    A block scores the max of its entries (unreachable entries are already ``-inf``); the block holding the query's
    newest reachable entry is always kept; picks that come back ``-inf`` are dropped.

    Args:
        scores: Masked indexer scores ``[B, S, T]`` (``-inf`` where unreachable).
        compress_lens: Reachable entry count per query ``[B, S]``.
        topk_blocks: Number of blocks to keep.
        block_size: Entries per block.

    Returns:
        Boolean entry mask ``[B, S, T]``.
    """
    batch, seq, width = scores.shape
    pad = (-width) % block_size
    padded = jnp.pad(scores, ((0, 0), (0, 0), (0, pad)), constant_values=-jnp.inf)
    num_blocks = (width + pad) // block_size
    block_scores = padded.reshape(batch, seq, num_blocks, block_size).max(axis=-1)
    last = (compress_lens - 1) // block_size
    block_scores = jnp.where(jnp.arange(num_blocks)[None, None, :] == last[..., None], jnp.inf, block_scores)
    values, indices = jax.lax.top_k(block_scores, min(topk_blocks, num_blocks))
    keep = jnp.zeros((batch, seq, num_blocks), dtype=bool)
    keep = keep.at[jnp.arange(batch)[:, None, None], jnp.arange(seq)[None, :, None], indices].set(values > -jnp.inf)
    return jnp.repeat(keep, block_size, axis=-1)[..., :width]


def shared_key_scores(
    q: Float[Array, "batch seq heads dim"],
    weights: Float[Array, "batch seq heads"],
    index_k: Float[Array, "batch entries dim"],
    positions: Int[Array, "batch seq"],
    ratio: int,
    precision: jax.lax.PrecisionLike = None,
) -> Float[Array, "batch seq entries"]:
    """Differentiable shared-key indexer scores ``sum_h relu(q_h . k_t) * w_h`` (``-inf`` beyond each query's reach).

    Rows are independent, so callers may pass any query slice (``q`` / ``weights`` / ``positions`` cut along ``seq``)
    to bound memory; a position of ``-1`` reaches nothing.

    Args:
        q: Rotated indexer queries ``[B, S, H, D]``.
        weights: Per-head weights ``[B, S, H]`` (scaling folded in).
        index_k: Index keys ``[B, T, D]``.
        positions: Query positions ``[B, S]``.
        ratio: Source tokens per entry (query ``s`` reaches entries ``< (s + 1) // ratio``).
        precision: Matmul precision.

    Returns:
        fp32 scores ``[B, S, T]``.
    """
    logits = jnp.einsum("bshd,btd->bsht", q.astype(jnp.float32), index_k.astype(jnp.float32), precision=precision)
    scores = jnp.sum(jax.nn.relu(logits) * weights[..., None], axis=2)
    compress_lens = (positions.astype(jnp.int32) + 1) // ratio
    reachable = jnp.arange(index_k.shape[1], dtype=jnp.int32)[None, None, :] < compress_lens[..., None]
    return jnp.where(reachable, scores, -jnp.inf)


class IndexerScoreInputs(tp.NamedTuple):
    """Lazy shared-key indexer scores: everything :func:`shared_key_scores` needs, without the ``[B, S, T]`` tensor.

    Consumers that need scores (an auxiliary indexer loss) compute them per query tile; :meth:`dense` materializes
    the full tensor for small inputs and tests.
    """

    q: Array
    weights: Array
    index_k: Array
    positions: Array
    ratio: int
    precision: tp.Any = None

    def rows(self, q: Array, weights: Array, positions: Array) -> Array:
        """Scores of a query slice (the slice of :attr:`q` / :attr:`weights` / :attr:`positions` given)."""
        return shared_key_scores(q, weights, self.index_k, positions, self.ratio, self.precision)

    def dense(self) -> Array:
        """Full ``[B, S, T]`` scores."""
        return self.rows(self.q, self.weights, self.positions)


def selection_bias(topk: Int[Array, "batch seq k"], num_entries: int) -> Float[Array, "batch 1 seq entries"]:
    """Additive fp32 bias opening the selected compressed entries (``-1`` = no entry).

    Args:
        topk: Selected entry offsets ``[B, S, k]``.
        num_entries: Width of the entry axis.

    Returns:
        ``[B, 1, S, T]`` bias: ``0`` at selected entries, the fp32 minimum elsewhere.
    """
    batch, seq, _ = topk.shape
    target = jnp.where(topk >= 0, topk, num_entries)
    opened = jnp.zeros((batch, seq, num_entries), dtype=bool)
    opened = opened.at[jnp.arange(batch)[:, None, None], jnp.arange(seq)[None, :, None], target].set(True, mode="drop")
    return jnp.where(opened, 0.0, MASK_MIN).astype(jnp.float32)[:, None]


@dataclass(frozen=True)
class SharedKeyIndexerConfig:
    """Scalar settings of :class:`SharedKeyIndexer`, independent of model configs.

    Args:
        hidden_size: Hidden width feeding the per-head weights.
        q_lora_rank: Width of the shared query residual.
        index_n_heads: Number of scoring heads.
        index_head_dim: Per-head query / key width.
        index_topk: Maximum entries selected per query.
        ratio: Source tokens per compressed entry (sets each query's reachable entry count).
        key_input_dim: Width of the KV source latent the index keys are projected from.
        owns_keys: Whether this indexer carries ``wk`` / ``k_norm`` (it is the key owner).
        is_candidate_source: Publish candidate blocks (level-one selection) for later indexers.
        uses_candidates: Restrict scores to previously published candidate blocks.
        candidate_topk_blocks: Blocks kept by the level-one selection.
        candidate_block_size: Entries per candidate block.
        rms_norm_eps: Epsilon of the index-key RMSNorm.
        initializer_range: Standard deviation of the projection initializers.
    """

    hidden_size: int
    q_lora_rank: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    ratio: int
    key_input_dim: int
    owns_keys: bool = True
    is_candidate_source: bool = False
    uses_candidates: bool = False
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 1
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {self.ratio}.")
        if self.index_n_heads < 1 or self.index_head_dim < 1 or self.index_topk < 1:
            raise ValueError("index_n_heads, index_head_dim and index_topk must be positive.")
        if (self.is_candidate_source or self.uses_candidates) and self.candidate_block_size < 1:
            raise ValueError("candidate_block_size must be >= 1 when candidates are used.")


class SharedKeyIndexer(spx.Module):
    """Chooses up to ``index_topk`` compressed entries per query against externally supplied index keys.

    ``score[t] = sum_h relu(q_h . k_t) * w_h`` with ``w = weights_proj(x) * index_head_dim^-0.5 * n_heads^-0.5``;
    entries beyond the query's reach ``(position + 1) // ratio`` are excluded. Selection is a hard top-k (no gradient
    flows through it); ``forward(..., return_scores=True)`` additionally returns the differentiable scores so a caller
    can train the indexer with an auxiliary objective. A candidate source also returns a block mask; a candidate user only scores inside it.

    Parameter leaves: ``q_b_proj.weight``, ``weights_proj.weight`` and, when ``owns_keys``, ``wk.weight`` and
    ``k_norm.weight``.
    """

    def __init__(
        self,
        config: SharedKeyIndexerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the indexer.

        Args:
            config: Scalar indexer settings.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators (consumed in q / weights / key / norm order).
        """
        self.precision = precision
        self.ratio = config.ratio
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.owns_keys = config.owns_keys
        self.is_candidate_source = config.is_candidate_source
        self.uses_candidates = config.uses_candidates
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        kwargs = dict(
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.q_b_proj = ColumnParallelLinear(config.q_lora_rank, self.n_heads * self.head_dim, **kwargs)
        self.weights_proj = ColumnParallelLinear(config.hidden_size, self.n_heads, **kwargs)
        if self.owns_keys:
            self.wk = ColumnParallelLinear(config.key_input_dim, self.head_dim, **kwargs)
            self.k_norm = RMSNorm(
                dim=self.head_dim, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
        else:
            self.wk = None
            self.k_norm = None

    def keys_from_latent(
        self,
        latent: Float[Array, "batch entries dim"],
        rotate_keys: tp.Callable[[Array], Array],
    ) -> Float[Array, "batch entries index_dim"]:
        """Index keys ``[B, T, index_head_dim]`` from the KV source's pre-RoPE latent.

        Args:
            latent: Pre-RoPE compressed latent ``[B, T, key_input_dim]``.
            rotate_keys: Rotary callable mapping keys ``[B, T, index_head_dim]`` to rotated keys of the same shape.

        Returns:
            Rotated index keys.
        """
        return rotate_keys(self.k_norm(self.wk(latent)))

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        index_k: Array | None,
        rotate_queries: tp.Callable[[Array], Array],
        candidates: Array | None,
        positions: Int[Array, "batch seq"],
        return_scores: bool = False,
        kernel_mesh: object | None = None,
        slot_axes: tp.Sequence[str] = (),
    ) -> tuple[Array, Array | None] | tuple[Array, Array | None, Array]:
        """Return ``(topk [B, S, k] int32 with -1 padding, published candidate block ids or None)``.

        ``index_k`` may be longer than the reachable range (a decode cache buffer). Selection runs through the shared
        context-bounded selectors (:func:`~easydel.layers.indexer.slot_entry_topk`,
        :func:`~easydel.layers.indexer.bounded_block_max_topk`): only entries below each query's
        ``(position + 1) // ratio`` are scored, so the cost follows the live context, not the buffer length, and no
        ``[B, S, T]`` score tensor is built. A candidate user whose candidates hold fewer reachable entries than
        ``index_topk`` returns ``-1`` for the rest (the reference keeps arbitrary ``-inf`` ties there; the released
        configuration never reaches that case).

        Args:
            hidden_states: Attention-input hidden states ``[B, S, hidden]``.
            q_residual: Normed query residual ``[B, S, q_lora]``.
            index_k: Rotated index keys ``[B, T, index_head_dim]`` (``None`` or empty selects nothing).
            rotate_queries: Rotary callable mapping queries ``[B, S, H, index_head_dim]`` to the same layout.
            candidates: Candidate block ids ``[B, S, KB]`` (ascending, ``-1``-padded) published by the candidate
                source (used when ``uses_candidates``).
            positions: Query positions ``[B, S]``.
            return_scores: Also return :class:`IndexerScoreInputs` for the differentiable reachability-masked scores
                (``-inf`` beyond each query's reach, before candidate masking), for an auxiliary training objective.
                The ``[B, S, T]`` tensor is never built here; consumers evaluate it per query tile (``None`` when
                there are no entries).
            kernel_mesh: Mesh for the replicated-key Pallas selector dispatch (``None`` keeps XLA on multi-device).
            slot_axes: Mesh axes ``index_k`` is slot-sharded over (a slot-sharded decode cache). When given, selection
                runs under one ``shard_map`` over those axes: every device scores only its own slots against its
                local key shard (fused Pallas on TPU), so no key buffer is gathered and the work is split ``N`` ways.

        Returns:
            The selection and, on a candidate source, its published candidate block ids ``[B, S, KB]``; with
            ``return_scores`` a third element, the scores.
        """
        batch, seq, _ = hidden_states.shape
        entries = 0 if index_k is None else index_k.shape[1]
        if entries == 0:
            empty_candidates = jnp.full((batch, seq, 0), -1, dtype=jnp.int32) if self.is_candidate_source else None
            empty = (jnp.full((batch, seq, 0), -1, dtype=jnp.int32), empty_candidates)
            return (*empty, None) if return_scores else empty
        q, weights = self.query_and_weights(hidden_states, q_residual, rotate_queries)
        compress_lens = (positions.astype(jnp.int32) + 1) // self.ratio
        # Selection carries no gradient; it reads only each query's reachable prefix (shared bounded selectors).
        sel_q = jax.lax.stop_gradient(q.astype(jnp.float32))
        sel_w = jax.lax.stop_gradient(weights)
        keys = jax.lax.stop_gradient(index_k)
        cand = candidates if self.uses_candidates else None
        mesh = getattr(kernel_mesh, "jax_mesh", kernel_mesh)
        if slot_axes and mesh is not None and getattr(mesh, "size", 1) > 1:
            topk, published = self._slot_local_select(sel_q, sel_w, keys, compress_lens, cand, mesh, tuple(slot_axes))
        else:
            topk, published = self._select(sel_q, sel_w, keys, compress_lens, cand, kernel_mesh)
        if return_scores:
            return topk, published, IndexerScoreInputs(q, weights, index_k, positions, self.ratio, self.precision)
        return topk, published

    def _select(
        self,
        q: Array,
        weights: Array,
        keys: Array,
        compress_lens: Array,
        candidates: Array | None,
        kernel_mesh: object | None,
    ) -> tuple[Array, Array | None]:
        """Published candidate blocks (candidate source only) and the entry top-k over each query's reach."""
        published = None
        if self.is_candidate_source:
            published = bounded_block_max_topk(
                q,
                weights,
                keys,
                compress_lens,
                self.candidate_topk_blocks,
                self.candidate_block_size,
                precision=self.precision,
            )
        topk = slot_entry_topk(
            q,
            weights,
            keys,
            compress_lens,
            self.index_topk,
            candidate_blocks=candidates,
            block_size=self.candidate_block_size,
            precision=self.precision,
            kernel_mesh=kernel_mesh,
        )
        return topk, published

    def _slot_local_select(
        self,
        q: Array,
        weights: Array,
        keys: Array,
        compress_lens: Array,
        candidates: Array | None,
        mesh: tp.Any,
        slot_axes: tuple[str, ...],
    ) -> tuple[Array, Array | None]:
        """:meth:`_select` under a ``shard_map`` that splits every slot-major operand over ``slot_axes``.

        Slots are independent, so each device selects for its own slots from its own key shard; inside the fully
        manual ``shard_map`` the paged selector runs its fused Pallas kernel per device.
        """
        row = PartitionSpec(slot_axes)
        operands = (q, weights, keys, compress_lens) + (() if candidates is None else (candidates,))
        with_published = self.is_candidate_source

        def body(q_, w_, k_, lens_, *rest):
            topk, published = self._select(q_, w_, k_, lens_, rest[0] if rest else None, None)
            return (topk, published) if with_published else topk

        out = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(row,) * len(operands),
            out_specs=(row, row) if with_published else row,
            check_vma=False,
        )(*operands)
        return out if with_published else (out, None)

    def query_and_weights(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        rotate_queries: tp.Callable[[Array], Array],
    ) -> tuple[Array, Array]:
        """Rotated indexer queries ``[B, S, H, D]`` and fp32 per-head weights ``[B, S, H]`` (scaling folded in)."""
        batch, seq, _ = hidden_states.shape
        q = rotate_queries(self.q_b_proj(q_residual).reshape(batch, seq, self.n_heads, self.head_dim))
        weights = self.weights_proj(hidden_states).astype(jnp.float32) * (self.head_dim**-0.5 * self.n_heads**-0.5)
        return q, weights

    def candidate_width(self, entries: int) -> int:
        """Width of the published candidate block-id list for an ``entries``-long key buffer."""
        return min(self.candidate_topk_blocks, -(-int(entries) // self.candidate_block_size))

    def reachable_scores(
        self, q: Array, weights: Array, index_k: Array, positions: Int[Array, "batch seq"]
    ) -> Float[Array, "batch seq entries"]:
        """Differentiable dense scores ``[B, S, T]`` (``-inf`` beyond reach); see :func:`shared_key_scores`."""
        return shared_key_scores(q, weights, index_k, positions, self.ratio, self.precision)
