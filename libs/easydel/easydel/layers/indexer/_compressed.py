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

"""Compressed-entry indexer projections, scoring, selection and cache orchestration.

Compression and rotary details are supplied by the owning architecture. Ranked
indices remain compressed-entry offsets; they must never be expanded to tokens.
"""

from __future__ import annotations

import functools
import typing as tp
from dataclasses import dataclass

import jax
import spectrax as spx
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int

from easydel.layers.compressed_window import (
    advance_compressor_stream,
    compress_windows,
    entry_visibility,
    trace_compressor_stream,
)
from easydel.layers.linears import ColumnParallelLinear
from easydel.layers.norms import RMSNorm

from ._paged import slot_entry_topk
from ._primitives import topk_selection_mask
from ._selection import BaseIndexer, IndexerSelection, SelectionSpec, top_k_values_indices


@dataclass(frozen=True)
class CompressedIndexerConfig:
    """Scalar projection and selection settings, independent of model configs.

    Args:
        hidden_size: Input hidden width.
        q_lora_rank: Width of the shared query residual.
        index_n_heads: Number of scoring heads.
        index_head_dim: Per-head query and compressed-key width.
        index_topk: Maximum number of compressed entries selected per query.
        compress_rate: Source tokens per emitted entry.
        rms_norm_eps: Epsilon for the compressed-key RMSNorm.
        initializer_range: Standard deviation of projection initializers.
    """

    hidden_size: int
    q_lora_rank: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    compress_rate: int
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02


class CompressedIndexerAdapter(tp.Protocol):
    """Architecture-owned compression/rotary/cache policy, without parameters.

    The indexer owns all parameter leaves directly, preserving native checkpoint
    paths and RNG order. Adapters receive that owner for window compression and
    implement the cache's emission and overlap rules rather than approximating
    compression with generic token pooling.
    """

    def finalize_entries(self, owner: CompressedIndexer, pooled: Array, entry: Array) -> Array:
        """Norm and rotate pooled groups ``[B, G, D]`` (``entry [B, G]``) into stored compressed keys.

        Pooling and the open-group / Ca-Cb overlap state machine are the registered ``gated_pool_compress``
        operation; only the architecture's entry norm and rotary placement live here.
        """
        ...

    def rotate_queries(self, q: Array, position_ids: Array, dtype: jnp.dtype) -> Array:
        """Rotate projected queries [B,S,H,D], returning the same layout."""
        ...

    def entry_visibility(self, positions: Array, cache_position: Array, capacity: int, rate: int) -> Array:
        """Return causal and emitted-entry visibility [B,S,capacity]."""
        ...

    def selection_is_vacuous(self, budget: int, capacity: int) -> bool:
        """Return whether scoring and all indexer cache work can be omitted."""
        ...


class CompressedIndexerScorer(spx.Module):
    """Lightning-indexer scoring head: ``sum_h w_{t,h} * relu(q_{t,h} . K^IComp_s)``."""

    def __init__(
        self,
        config: CompressedIndexerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the scorer.

        Args:
            config: Scalar indexer settings.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        self.softmax_scale = config.index_head_dim**-0.5
        self.weights_scaling = config.index_n_heads**-0.5
        self.precision = precision
        self.weights_proj = ColumnParallelLinear(
            config.hidden_size,
            config.index_n_heads,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )

    def forward(self, q: Array, compressed_kv: Array, hidden_states: Array) -> Array:
        """Score queries against compressed indexer keys (fp32).

        Args:
            q: Indexer queries ``[B, S, H_i, D_i]``.
            compressed_kv: Compressed indexer keys ``[B, T, D_i]``.
            hidden_states: Hidden states ``[B, S, D_model]`` for the per-head
                weighting.

        Returns:
            Per-query index scores ``[B, S, T]`` (fp32).
        """
        scores = jnp.einsum(
            "bshd,btd->bsht",
            q.astype(jnp.float32),
            compressed_kv.astype(jnp.float32),
            precision=self.precision,
        )
        scores = jax.nn.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).astype(jnp.float32) * self.weights_scaling
        return jnp.sum(scores * weights[..., None], axis=2)


class CompressedIndexer(BaseIndexer):
    """Lightning Indexer: picks the top-``index_topk`` compressed entries per query.

    Runs its own scaled-down two-series (Ca/Cb) compressor at
    ``index_head_dim`` over the same windows as the outer CSA compressor,
    scores queries (projected from the shared ``q_residual``) against the
    compressed keys, and returns the per-query top-k entry indices with
    ``-1`` marking invalid (causally unreachable) picks.
    """

    selection_spec = SelectionSpec("compressed_entry", "compressed_entry")

    def __init__(
        self,
        config: CompressedIndexerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        compressor_adapter: CompressedIndexerAdapter,
    ):
        """Initialize the indexer with checkpoint-native, direct projections.

        Args:
            config: Scalar projection and compressed-entry selection settings.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random streams consumed in kv/gate/bias/norm/query/scorer order.
            compressor_adapter: Parameter-free compression, rotary and cache
                policy supplied by the architecture; owns no projection math.
        """
        from easydel.infra.utils import ArrayParam

        self.compressor_adapter = compressor_adapter
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.compress_rate = config.compress_rate
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        linear_kwargs = dict(
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.kv_proj = ColumnParallelLinear(config.hidden_size, 2 * self.head_dim, **linear_kwargs)
        self.gate_proj = ColumnParallelLinear(config.hidden_size, 2 * self.head_dim, **linear_kwargs)
        self.position_bias = ArrayParam.bound(
            shape=(self.compress_rate, 2 * self.head_dim),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )
        self.kv_norm = RMSNorm(
            dim=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.q_b_proj = ColumnParallelLinear(config.q_lora_rank, self.num_heads * self.head_dim, **linear_kwargs)
        self.scorer = CompressedIndexerScorer(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        position_ids: Int[Array, "batch seq"],
        return_scores: bool = False,
        return_selection: bool = False,
    ) -> Array | tuple[Array, Array] | IndexerSelection | None:
        """Compute per-query top-k compressed-entry indices.

        Args:
            hidden_states: Attention-input hidden states ``[B, S, D_model]``.
            q_residual: Normed query LoRA residual shared with core attention.
            position_ids: Query positions ``[B, S]``.
            return_scores: Also return the differentiable full-entry scores for
                a zero-primal training bias. Static when this method is jitted.
            return_selection: Return an :class:`IndexerSelection` holding the
                indices, the scores as ``score_proxy`` and the dense selection
                mask, so ``to_bias`` needs no scatter. Takes precedence over
                ``return_scores``. Static when this method is jitted.

        Returns:
            Int array ``[B, S, k]`` of selected entry indices with ``-1`` for
            invalid picks, or ``None`` when no complete window fits. With
            ``return_scores=True``, returns ``(indices, scores)`` where scores
            have shape ``[B, S, n_entries]`` and retain projection gradients.
        """
        _batch, seq_len, _ = hidden_states.shape
        rate = self.compress_rate
        n_windows = seq_len // rate
        if n_windows == 0:
            return None
        usable = n_windows * rate
        kv = self.kv_proj(hidden_states)[:, :usable]
        gate = self.gate_proj(hidden_states)[:, :usable]
        compressed, _, _ = self._compress_windows(kv, gate)
        index_scores = self._score_queries(hidden_states, q_residual, position_ids, compressed)
        selection = self._select_causal(index_scores, position_ids, n_windows)
        indices = checkpoint_name(selection.indices, "indexer_topk")
        if return_selection:
            mask = None if selection.mask is None else checkpoint_name(selection.mask, "indexer_topk")
            return IndexerSelection(indices, index_scores, mask)
        return (indices, index_scores) if return_scores else indices

    def _compress_windows(self, kv: Array, gate: Array) -> tuple[Array, Array, Array]:
        """Two-series compressed keys of complete windows plus the last window's Ca overlap state."""
        out = compress_windows(
            kv, gate, self.compress_rate, position_bias=self.position_bias.value, series_dim=self.head_dim
        )
        batch, groups = out.pooled.shape[:2]
        entry = jnp.broadcast_to(jnp.arange(groups, dtype=jnp.int32)[None], (batch, groups))
        return self.compressor_adapter.finalize_entries(self, out.pooled, entry), out.overlap_kv, out.overlap_gate

    def _score_queries(
        self,
        hidden_states: Array,
        q_residual: Array,
        position_ids: Array,
        compressed: Array,
    ) -> Array:
        """Project + rotate indexer queries and score them against ``compressed``.

        Args:
            hidden_states: Hidden states ``[B, S, D_model]``.
            q_residual: Normed query LoRA residual ``[B, S, q_lora]``.
            position_ids: Query positions ``[B, S]``.
            compressed: Compressed indexer keys ``[B, T, index_head_dim]``.

        Returns:
            Index scores ``[B, S, T]`` (fp32).
        """
        batch, seq_len, _ = hidden_states.shape
        q = self.q_b_proj(q_residual).reshape(batch, seq_len, self.num_heads, self.head_dim)
        q = self.compressor_adapter.rotate_queries(q, position_ids, hidden_states.dtype)
        return self.scorer(q, compressed, hidden_states)

    def _select_causal(self, index_scores: Array, position_ids: Array, compressed_len: int) -> IndexerSelection:
        """:meth:`_select_causal_top_k` as a full selection (indices and mask)."""
        valid = entry_visibility(position_ids, compressed_len, self.compress_rate)
        return self.select_candidates(index_scores, self.index_topk, valid, self.mesh_source)

    def _select_causal_top_k(self, index_scores: Array, position_ids: Array, compressed_len: int) -> Array:
        """Causally mask scores and keep the per-query top-k entry indices.

        Args:
            index_scores: Scores ``[B, S, T]`` (fp32).
            position_ids: Query positions ``[B, S]``.
            compressed_len: Number of scored entries ``T``.

        Returns:
            Int indices ``[B, S, k]`` with ``-1`` marking invalid picks.
        """
        return self._select_causal(index_scores, position_ids, compressed_len).indices

    @staticmethod
    def select_candidates(
        scores: Array, k: int, valid: Array | None = None, mesh_source: object | None = None
    ) -> IndexerSelection:
        """Rank compressed entries, retaining the full score domain for training.

        Unlike token selectors, eligibility here is purely causal/cache validity:
        keep the original top-k tie and non-finite-score behavior exactly.
        No compressed-entry index is expanded to a token offset.
        """
        if k < 0:
            raise ValueError("Selection budget must be non-negative.")
        valid = jnp.ones_like(scores, dtype=jnp.bool_) if valid is None else jnp.broadcast_to(valid, scores.shape)
        masked = jnp.where(valid, scores, -jnp.inf)
        values, indices = top_k_values_indices(masked, min(k, scores.shape[-1]), mesh_source)
        picked = jnp.take_along_axis(valid, indices, axis=-1)
        mask = None
        if indices.shape[-1] > 0:
            # Unused masks are dead code under jit.
            mask = topk_selection_mask(masked, values, indices) & valid
        return IndexerSelection(jnp.where(picked, indices, -1), scores, mask)

    def cached_extend(
        self,
        hidden_states: Float[Array, "batch block hidden"],
        q_residual: Float[Array, "batch block q_lora"],
        positions: Int[Array, "batch block"],
        valid: Array,
        cache_view: tp.Any,
        kernel_mesh: object | None = None,
    ) -> tuple[Array | None, tp.Any]:
        """Block form of the cached decode step (batched packed prefill).

        Advances the indexer's own compressed-key stream by a block of tokens per row through the adapter's
        ``extend_step``, then selects each query's top-``index_topk`` entries below its reach ``(position + 1) //
        rate`` with the shared context-bounded :func:`~easydel.layers.indexer.slot_entry_topk`. Entries emitted later
        in the block lie beyond every earlier query's reach, so writing them first is causal.

        Args:
            hidden_states: ``[B, blk, D_model]``.
            q_residual: ``[B, blk, q_lora]``.
            positions: Stream positions ``[B, blk]`` (garbage at invalid cells).
            valid: ``[B, blk]`` bool, a prefix per row.
            cache_view: This layer's cache view (indexer stream fields).
            kernel_mesh: Mesh for the selector's replicated-page kernel dispatch.

        Returns:
            ``(top_k [B, blk, k] int32 with -1 padding, view)``; ``top_k`` is ``None`` when the selection is vacuous
            (every reachable entry is attended, the indexer stream is not maintained, as on the decode path).
        """
        rate = self.compress_rate
        n_slots = cache_view.num_entry_slots
        if n_slots == 0 or self.compressor_adapter.selection_is_vacuous(self.index_topk, n_slots):
            return None, cache_view
        batch, block, _ = hidden_states.shape
        kv_blk = self.kv_proj(hidden_states)
        gate_blk = self.gate_proj(hidden_states)
        cache_view = trace_compressor_stream(
            cache_view,
            "indexer",
            kv=kv_blk,
            gate=gate_blk,
            positions=positions,
            valid=valid,
            rate=rate,
            head_dim=self.head_dim,
            position_bias=self.position_bias.value,
        )
        entries, buffer_kv, buffer_gate, overlap_kv, overlap_gate = advance_compressor_stream(
            kv=kv_blk,
            gate=gate_blk,
            buffer_kv=cache_view.indexer_buffer_kv,
            buffer_gate=cache_view.indexer_buffer_gate,
            entries=cache_view.indexer_entries,
            position_bias=self.position_bias.value,
            positions=positions,
            valid=valid,
            rate=rate,
            overlap_kv=cache_view.indexer_overlap_kv,
            overlap_gate=cache_view.indexer_overlap_gate,
            finalize=functools.partial(self.compressor_adapter.finalize_entries, self),
        )
        cache_view = cache_view.replace(
            indexer_buffer_kv=buffer_kv,
            indexer_buffer_gate=buffer_gate,
            indexer_entries=entries,
            indexer_overlap_kv=overlap_kv,
            indexer_overlap_gate=overlap_gate,
        )
        q = self.q_b_proj(q_residual).reshape(batch, block, self.num_heads, self.head_dim)
        q = self.compressor_adapter.rotate_queries(q, positions, hidden_states.dtype)
        weights = self.scorer.weights_proj(hidden_states).astype(jnp.float32) * self.scorer.weights_scaling
        reach = jnp.where(valid, (positions.astype(jnp.int32) + 1) // rate, 0)
        # sum_h w_h * relu(sm * q.k) == the scorer's sum_h w_h * relu(q.k) * sm (sm > 0).
        top_k = slot_entry_topk(
            jax.lax.stop_gradient(q.astype(jnp.float32)),
            jax.lax.stop_gradient(weights),
            entries,
            reach,
            self.index_topk,
            sm_scale=float(self.scorer.softmax_scale),
            relu=True,
            precision=self.precision,
            kernel_mesh=kernel_mesh,
        )
        return top_k, cache_view

    def cached_forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        position_ids: Int[Array, "batch seq"],
        cache_view: tp.Any,
        valid: Array | None = None,
    ) -> tuple[Array | None, tp.Any]:
        """Run the indexer with cache state (prefill or decode).

        Args:
            hidden_states: Attention-input hidden states ``[B, S, D_model]``.
            q_residual: Normed query LoRA residual shared with core attention.
            position_ids: Query positions ``[B, S]``.
            cache_view: This layer's cache view (indexer stream fields).
            valid: Optional per-row bool mask ``[B]`` for decode steps; rows
                with ``False`` leave state untouched.

        Returns:
            Tuple ``(top_k_indices, updated_view)``. On prefill the indices
            cover the ``S // rate`` fresh entries (``None`` if no window
            fits); on decode they cover the padded entry axis. ``-1`` marks
            invalid picks in both cases.
        """
        seq_len = hidden_states.shape[1]
        if seq_len > 1:
            return self._cached_prefill(hidden_states, q_residual, position_ids, cache_view)
        return self._cached_decode(hidden_states, q_residual, position_ids, cache_view, valid=valid)

    def _cached_prefill(
        self,
        hidden_states: Array,
        q_residual: Array,
        position_ids: Array,
        cache_view: tp.Any,
    ) -> tuple[Array | None, tp.Any]:
        """Prefill-from-empty: stateless indexer math plus state writes."""
        seq_len = hidden_states.shape[1]
        rate = self.compress_rate
        n_windows = min(seq_len // rate, cache_view.num_entry_slots)
        usable = n_windows * rate
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        top_k_indices = None
        if n_windows > 0:
            compressed, ca_kv, ca_gate = self._compress_windows(kv[:, :usable], gate[:, :usable])
            cache_view = cache_view.replace(
                indexer_entries=cache_view.indexer_entries.at[:, :n_windows].set(
                    compressed.astype(cache_view.indexer_entries.dtype)
                ),
                indexer_overlap_kv=ca_kv.astype(cache_view.indexer_overlap_kv.dtype),
                indexer_overlap_gate=ca_gate.astype(jnp.float32),
            )
            index_scores = self._score_queries(hidden_states, q_residual, position_ids, compressed)
            top_k_indices = self._select_causal_top_k(index_scores, position_ids, n_windows)
        remainder = seq_len - usable
        if remainder > 0:
            cache_view = cache_view.replace(
                indexer_buffer_kv=cache_view.indexer_buffer_kv.at[:, :remainder].set(
                    kv[:, usable:].astype(cache_view.indexer_buffer_kv.dtype)
                ),
                indexer_buffer_gate=cache_view.indexer_buffer_gate.at[:, :remainder].set(
                    gate[:, usable:].astype(cache_view.indexer_buffer_gate.dtype)
                ),
            )
        return top_k_indices, cache_view

    def _cached_decode(
        self,
        hidden_states: Array,
        q_residual: Array,
        position_ids: Array,
        cache_view: tp.Any,
        valid: Array | None = None,
    ) -> tuple[Array | None, tp.Any]:
        """Single-token step over the padded indexer-entry axis.

        Note:
            When the selection is vacuous
            (``compressor_adapter.selection_is_vacuous``) the whole indexer is dead
            work, not just its sort, and the step short-circuits to plain
            visibility. Everything feeding the selection is then unreachable:
            the two projections, the windowed compression state machine, and the
            five indexer cache tensors it maintains -- whose only consumer is
            ``_score_queries``, which is skipped as well. Nothing outside the
            indexer reads that state.

            This is the common serving case whenever the entry capacity does
            not exceed ``index_topk``. Both operands are static, so the branch
            resolves at trace time.
        """
        rate = self.compress_rate
        n_slots = cache_view.num_entry_slots
        if n_slots == 0:
            return None, cache_view

        if self.compressor_adapter.selection_is_vacuous(self.index_topk, n_slots):
            visible = self.compressor_adapter.entry_visibility(position_ids, cache_view.cache_position, n_slots, rate)
            entry_ids = jnp.broadcast_to(jnp.arange(n_slots, dtype=jnp.int32), visible.shape)
            return jnp.where(visible, entry_ids, -1), cache_view

        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        cache_view = trace_compressor_stream(
            cache_view,
            "indexer",
            kv=kv,
            gate=gate,
            positions=cache_view.cache_position[:, None],
            valid=None if valid is None else valid[:, None],
            rate=rate,
            head_dim=self.head_dim,
            position_bias=self.position_bias.value,
        )
        entries, buffer_kv, buffer_gate, overlap_kv, overlap_gate = advance_compressor_stream(
            kv=kv,
            gate=gate,
            buffer_kv=cache_view.indexer_buffer_kv,
            buffer_gate=cache_view.indexer_buffer_gate,
            entries=cache_view.indexer_entries,
            position_bias=self.position_bias.value,
            position=cache_view.cache_position,
            rate=rate,
            overlap_kv=cache_view.indexer_overlap_kv,
            overlap_gate=cache_view.indexer_overlap_gate,
            valid=valid,
            finalize=functools.partial(self.compressor_adapter.finalize_entries, self),
        )
        cache_view = cache_view.replace(
            indexer_buffer_kv=buffer_kv,
            indexer_buffer_gate=buffer_gate,
            indexer_entries=entries,
            indexer_overlap_kv=overlap_kv,
            indexer_overlap_gate=overlap_gate,
        )
        visible = self.compressor_adapter.entry_visibility(position_ids, cache_view.cache_position, n_slots, rate)
        top_k = min(self.index_topk, n_slots)

        # Scoring reads the whole `indexer_entries` buffer -- [B, n_slots, head_dim] --
        # every decode step.
        #
        # Until the context passes `top_k`, every live entry is selected anyway:
        # `top_k` over `live <= top_k` candidates returns all of them, so the
        # ranking cannot exclude anything and the scores are computed only to be
        # thrown away. Returning the visible prefix is the same *set* of entries,
        # and attention over a key axis is permutation-invariant given a bias
        # gathered by the same indices.
        #
        # The state update above stays unconditional. Short-circuiting before it
        # -- which is what the `selection_is_vacuous` early return does --
        # would skip the entry write and leave the buffer stale.
        def _by_score(_):
            index_scores = self._score_queries(hidden_states, q_residual, position_ids, entries)
            return self.select_candidates(index_scores, top_k, visible, self.mesh_source).indices

        def _by_prefix(_):
            idx = jnp.broadcast_to(jnp.arange(top_k, dtype=jnp.int32), (*visible.shape[:-1], top_k))
            return jnp.where(visible[..., :top_k], idx, -1)

        live = jnp.max((cache_view.cache_position.astype(jnp.int32) + 1) // rate)
        top_k_indices = jax.lax.cond(live <= top_k, _by_prefix, _by_score, operand=None)
        return top_k_indices, cache_view
