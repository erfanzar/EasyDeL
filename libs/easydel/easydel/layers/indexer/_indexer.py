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

"""The unified dynamic sparse-attention indexer module."""

from __future__ import annotations

import typing as tp
from functools import partial

import jax
import spectrax as spx
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float

from easydel.layers.linears import ColumnParallelLinear

from ._config import IndexerConfig, IndexerKind
from ._primitives import indices_to_bool_mask, visible_causal_mask
from ._rope import apply_indexer_rope, truncate_cos_sin
from ._selection import BaseIndexer, IndexerSelection, SelectionSpec, top_k_indices

__all__ = ("IndexerLayerNorm", "IndexerOutput", "SparseIndexer")


class IndexerLayerNorm(spx.Module):
    """Standard LayerNorm with HF-compatible ``weight``/``bias`` parameters.

    Single shared copy of the byte-identical norm previously duplicated as
    ``GlmMoeDsaLayerNorm`` and ``Glm5NextLayerNorm``. Computed in float32 and
    returned in the input dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the norm.

        Args:
            hidden_size: Feature dim to normalise over.
            eps: Numerical stability epsilon.
            dtype: Computation dtype.
            param_dtype: Parameter storage dtype.
            rngs: Random number generator collection (unused; kept for the
                common module-construction signature).
        """
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.weight = spx.Parameter(jnp.ones((hidden_size,), dtype=param_dtype))
        self.bias = spx.Parameter(jnp.zeros((hidden_size,), dtype=param_dtype))

    def forward(self, hidden_states: Float[Array, "... hidden_size"]) -> Float[Array, "... hidden_size"]:
        """Apply LayerNorm in fp32 and return in the input dtype.

        Args:
            hidden_states: Tensor whose last axis has length ``hidden_size``.

        Returns:
            Normalised tensor with the same shape and dtype as the input.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        mean = jnp.mean(hidden_states, axis=-1, keepdims=True)
        variance = jnp.mean((hidden_states - mean) ** 2, axis=-1, keepdims=True)
        normed = (hidden_states - mean) * jax.lax.rsqrt(variance + self.eps)
        out = normed * self.weight.value.astype(jnp.float32) + self.bias.value.astype(jnp.float32)
        return out.astype(input_dtype)

    __call__ = forward


class IndexerOutput(tp.NamedTuple):
    """Everything an indexer can hand back to its attention layer.

    Attributes:
        topk_indices: Selected positions ``[batch, seq, width]`` int32,
            ``-1`` where a slot is unused/invalid. ``width`` is
            ``index_topk`` (plus ``kpool_size - 1`` when the tail pool is
            selected).
        packed_state: Per-token state covering the full scored KV range, or
            ``None`` for stateless indexers. Layout per ``packed_state``:
            ``"keys"`` -> ``[B, T, index_head_dim]`` normalized keys;
            ``"key_gate_valid"`` -> ``[B, T, 2*index_head_dim + 1]``
            (raw key | gate scores | validity flag).
        score_proxy: Indexer scores ``[batch, seq, n_candidates]`` float32
            for the straight-through training bias; ``None`` when selection
            ran under ``stop_gradient``.
        topk_mask: The selection as a dense ``[batch, seq, kv]`` boolean mask
            over the packed-state range, equal to
            ``selection.to_mask(kv)``. Built without scattering the token
            indices (pool strategy only); ``None`` otherwise.
    """

    topk_indices: Array
    packed_state: Array | None = None
    score_proxy: Array | None = None
    topk_mask: Array | None = None

    @property
    def selection(self) -> IndexerSelection:
        """Common attention-consumer view, without changing the cache tuple."""
        return IndexerSelection(self.topk_indices, self.score_proxy, self.topk_mask)


class SparseIndexer(BaseIndexer):
    """GLM-style token and learned-pool indexer strategies.

    Selection and cache packing are :class:`IndexerConfig`-driven. Other
    compression/projection layouts specialize :class:`BaseIndexer` instead
    of pretending to have the same checkpoint structure or pooling math.

    Submodule names match the GLM checkpoints (``wq_b`` / ``wk`` /
    ``k_norm`` / ``weights_proj`` / ``index_kpool_compress_ape`` /
    ``index_kpool_compress_gate``) so families can adopt the layer without
    renaming loaded weights.

    Attributes:
        wq_b: Indexer query projection.
        wk: Single shared index-key projection.
        k_norm: Index-key LayerNorm.
        weights_proj: Learned per-token head-importance projection
            (``head_reduction="weighted"``).
    """

    def __init__(
        self,
        config: IndexerConfig,
        layer_idx: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the indexer.

        Args:
            config: Declarative indexer configuration.
            layer_idx: Index of the owning decoder layer (informational; kept
                for parity with the family modules).
            dtype: Computation dtype.
            param_dtype: Parameter storage dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        if config.packed_state == "keys":
            self._packed_dim = self.head_dim
        elif config.packed_state == "key_gate_valid":
            self._packed_dim = 2 * self.head_dim + 1
        else:
            self._packed_dim = 0

        column_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.wq_b = column_linear(config.q_input_dim, self.n_heads * self.head_dim)
        self.wk = column_linear(config.hidden_size, self.head_dim)
        self.k_norm = IndexerLayerNorm(
            self.head_dim,
            eps=config.norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        if config.head_reduction == "weighted":
            self.weights_proj = column_linear(config.hidden_size, self.n_heads)

        if config.kind == IndexerKind.POOL:
            # Raw (Parameter, no `.weight` suffix) HF tensors, kept in torch
            # layout: the gate stays transposed for the matmul, the ape is
            # per-pool-slot/per-channel. ArrayParam.bound so the values
            # re-materialize through sequential_init / checkpoint resure.
            # Imported here (not module-level): `easydel.infra.utils` sits
            # above `layers` in the package graph and importing it eagerly
            # would close an import cycle.
            from easydel.infra.utils import ArrayParam

            self.index_kpool = config.kpool_size
            self.index_kpool_always_select_tail = config.select_tail
            self.index_kpool_compress_ape = ArrayParam.bound(
                shape=(config.kpool_size, self.head_dim),
                dtype=param_dtype,
                init_method="zeros",
                key=rngs.param,
            )
            self.index_kpool_compress_gate = ArrayParam.bound(
                shape=(self.head_dim, config.hidden_size),
                dtype=param_dtype,
                init_method="ones",
                key=rngs.param,
            )

    @property
    def selection_spec(self) -> SelectionSpec:
        """Static ranked/output units; learned pools expand to token offsets."""
        return SelectionSpec("block" if self.config.kind == IndexerKind.POOL else "token", "token")

    @property
    def packed_state_dim(self) -> int:
        """Width of the per-token packed state (0 when stateless)."""
        return self._packed_dim

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_resid: Float[Array, "batch seq q_input_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq"] | None = None,
        cached_packed: Float[Array, "batch cached_seq packed"] | None = None,
        position_ids: Array | None = None,
        frequencies: Float[Array, "batch seq 2 rotary"] | None = None,
        pairwise_mask: Array | None = None,
        prev_topk_indices: Array | None = None,
        *,
        cache_positions: Array | None = None,
        key_valid_mask: Array | None = None,
        query_positions: Array | None = None,
        key_positions: Array | None = None,
        segment_ids: Array | None = None,
        key_segment_ids: Array | None = None,
    ) -> IndexerOutput:
        """Score candidates and return the top-k selection.

        Args:
            hidden_states: Layer input ``[batch, seq, hidden]`` (K side; also
                the Q side when ``query_source="hidden"``).
            q_resid: MLA low-rank residual ``[batch, seq, q_input_dim]``
                (Q side when ``query_source="q_lora"``).
            attention_mask: Boolean padding mask ``[batch, seq]`` for the
                current tokens; all-valid when ``None``.
            cached_packed: Packed state of previously cached tokens
                ``[batch, cached_seq, packed_dim]``.
            position_ids: Positions of the current tokens (rope only).
            frequencies: Rotary tables ``[..., 2 * rotary]`` (rope only).
            pairwise_mask: Optional additive/boolean query-key mask
                ``[batch, seq, kv]`` restricting candidates (token kind).
            prev_topk_indices: Another layer's selection for ``shared``
                indexer layers; when given, it is returned unchanged and
                nothing is computed.
            cache_positions: Optional physical insertion slots ``[batch, seq]``
                in preallocated ``cached_packed`` storage. Writes preserve its
                shape; negative/out-of-range slots and padded queries are ignored.
                Without this argument new rows are appended.
            key_valid_mask: Optional post-update validity ``[batch, kv]``;
                required to distinguish unused slots in keys-only storage.
            query_positions: Logical causal positions ``[batch, seq]``. Defaults
                to ``cache_positions`` for indexed writes or append coordinates.
                These are independent of the RoPE-only ``position_ids``.
            key_positions: Logical positions ``[batch, kv]`` of stored rows;
                defaults to physical row numbers. Supply these for shifted caches.
            segment_ids: Current query document identities ``[batch, seq]``;
                negative identities are invalid. With no cached history, also
                supplies key identities. Pool grouping and APE offsets restart
                at every document boundary.
            key_segment_ids: Document identities over the complete post-update
                storage ``[batch, kv]``. Required for segmented cached calls;
                the owning cache retains these separately from packed channels.

        Returns:
            :class:`IndexerOutput` with the ``-1``-padded top-k indices and
            the packed state covering every scored token.
        """
        if prev_topk_indices is not None:
            # "shared" indexer layer: reuse another layer's selection.
            return IndexerOutput(topk_indices=prev_topk_indices, packed_state=None, score_proxy=None)

        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

        if cache_positions is not None:
            if cached_packed is None or self.config.packed_state == "none":
                raise ValueError("cache_positions requires preallocated cached_packed storage.")
            if cache_positions.shape != (batch_size, seq_len):
                raise ValueError("cache_positions must have shape [batch, seq].")
            if self.config.packed_state == "keys" and key_valid_mask is None:
                raise ValueError("Indexed keys-only storage requires a full key_valid_mask.")
        query, key, gate_scores = self._project(hidden_states, q_resid, frequencies, position_ids)
        packed_state = self._pack_state(key, gate_scores, attention_mask, cached_packed, cache_positions)
        if key_segment_ids is None and segment_ids is not None and cached_packed is None:
            key_segment_ids = segment_ids
        if query_positions is None:
            query_positions = cache_positions
        return self.select_from_state(
            hidden_states,
            query,
            packed_state,
            attention_mask=attention_mask,
            pairwise_mask=pairwise_mask,
            key_states=key,
            key_valid_mask=key_valid_mask,
            query_positions=query_positions,
            key_positions=key_positions,
            segment_ids=segment_ids,
            key_segment_ids=key_segment_ids,
        )

    def project_state(
        self,
        hidden_states: Array,
        q_resid: Array | None = None,
        attention_mask: Array | None = None,
        position_ids: Array | None = None,
        frequencies: Array | None = None,
    ) -> tuple[Array, Array]:
        """Project current queries and packed features for an external cache update.

        Args:
            hidden_states: Current inputs ``[batch, seq, hidden]``.
            q_resid: Optional MLA query residual.
            attention_mask: Optional current-token validity ``[batch, seq]``.
            position_ids: Optional RoPE positions (not cache insertion slots).
            frequencies: Optional RoPE tables.

        Returns:
            Current queries and packed state. Pass the state through the owning
            cache's public auxiliary update, then call :meth:`select_from_state`
            with its attention-time state and aligned visibility/segments. This
            also supports sliding windows without reproducing cache layout math.

        Raises:
            ValueError: If the strategy has no packed-state representation.
        """
        if self.config.packed_state == "none":
            raise ValueError("project_state requires a stateful packed layout.")
        if attention_mask is None:
            attention_mask = jnp.ones(hidden_states.shape[:2], dtype=jnp.bool_)
        query, key, gates = self._project(hidden_states, q_resid, frequencies, position_ids)
        return query, self._pack_state(key, gates, attention_mask, None)

    def select_from_state(
        self,
        hidden_states: Array,
        query: Array,
        packed_state: Array | None,
        *,
        attention_mask: Array | None = None,
        pairwise_mask: Array | None = None,
        key_states: Array | None = None,
        key_valid_mask: Array | None = None,
        query_positions: Array | None = None,
        key_positions: Array | None = None,
        segment_ids: Array | None = None,
        key_segment_ids: Array | None = None,
    ) -> IndexerOutput:
        """Rank an externally aligned indexer state without updating it again.

        Args:
            hidden_states: Current inputs supplying learned head weights.
            query: Current projected/rotated queries from :meth:`project_state`.
            packed_state: Full attention-time features in physical KV order.
            attention_mask: Current-token validity, all-valid by default.
            pairwise_mask: Optional boolean/additive aligned query-key mask.
            key_states: Current keys for stateless strategies only.
            key_valid_mask: Optional full-storage key validity.
            query_positions: Optional logical causal positions of queries.
            key_positions: Optional logical positions of stored keys.
            segment_ids: Optional query document identities.
            key_segment_ids: Optional full-storage key document identities.

        Returns:
            Selection over physical storage rows, and the supplied packed state.
            Segments, validity and causality are enforced before ranking.
        """
        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        key = key_states
        if packed_state is None and key is None:
            raise ValueError("select_from_state requires packed_state or stateless key_states.")
        kv_len = key.shape[1] if packed_state is None else packed_state.shape[1]
        if key_valid_mask is not None:
            if key_valid_mask.shape != (batch_size, kv_len):
                raise ValueError("key_valid_mask must cover the full post-update indexer storage.")
            key_valid_mask = key_valid_mask.astype(jnp.bool_)
            if self.config.packed_state == "key_gate_valid":
                packed_state = packed_state.at[..., -1].set(key_valid_mask.astype(packed_state.dtype))
        if (segment_ids is None) != (key_segment_ids is None):
            raise ValueError("Segmented indexing requires both segment_ids and full key_segment_ids.")
        if segment_ids is not None:
            if segment_ids.shape != (batch_size, seq_len) or key_segment_ids.shape != (batch_size, kv_len):
                raise ValueError("Segment identities must match current queries and full post-update storage.")
            attention_mask = attention_mask & (segment_ids >= 0)

        # Keep the legacy token scorer's caller-owned causality when no new
        # metadata is supplied; indexed storage and packed documents require
        # explicit positions/validity before ranking rather than after top-k.
        visibility = None
        if self.config.kind == IndexerKind.POOL or any(
            value is not None for value in (key_valid_mask, query_positions, key_positions, segment_ids)
        ):
            if query_positions is None:
                query_positions = jnp.broadcast_to(kv_len - seq_len + jnp.arange(seq_len), (batch_size, seq_len))
            if key_positions is None:
                key_positions = jnp.broadcast_to(jnp.arange(kv_len), (batch_size, kv_len))
            visibility = (
                (key_positions[:, None, :] <= query_positions[:, :, None])
                & (key_positions[:, None, :] >= 0)
                & (query_positions[:, :, None] >= 0)
            )
            visibility = visibility & attention_mask[..., None]
            if key_valid_mask is not None:
                visibility = visibility & key_valid_mask[:, None, :]
            elif self.config.packed_state == "key_gate_valid":
                visibility = visibility & (packed_state[:, None, :, -1] >= 0.5)
            elif kv_len == seq_len:
                visibility = visibility & attention_mask[:, None, :]
            if segment_ids is not None:
                visibility = visibility & (segment_ids[:, :, None] == key_segment_ids[:, None, :])

        if self.config.kind == IndexerKind.POOL and pairwise_mask is not None:
            mask = pairwise_mask[:, 0] if pairwise_mask.ndim == 4 else pairwise_mask
            if mask.dtype != jnp.bool_:
                mask = jnp.isfinite(mask) & (mask > jnp.finfo(mask.dtype).min)
            visibility = visibility & mask

        topk_mask = None
        if self.config.kind == IndexerKind.POOL:
            topk_indices, topk_mask = self._select_pool(
                hidden_states, query, packed_state, attention_mask, visibility, key_segment_ids
            )
            score_proxy = None
        else:
            if visibility is not None:
                if pairwise_mask is None:
                    pairwise_mask = visibility
                else:
                    mask = pairwise_mask[:, 0] if pairwise_mask.ndim == 4 else pairwise_mask
                    pairwise_mask = (
                        mask & visibility if mask.dtype == jnp.bool_ else jnp.where(visibility, mask, -jnp.inf)
                    )
            topk_indices, score_proxy = self._select_token(
                hidden_states, query, key, packed_state, attention_mask, pairwise_mask
            )

        if self.config.stop_gradient:
            topk_indices = jax.lax.stop_gradient(topk_indices)
            score_proxy = None
        # Small, but recomputing it replays scoring and the full top-k.
        topk_indices = checkpoint_name(topk_indices, "indexer_topk")
        if topk_mask is not None:
            topk_mask = checkpoint_name(topk_mask, "indexer_topk")
        return IndexerOutput(
            topk_indices=topk_indices,
            packed_state=packed_state,
            score_proxy=score_proxy,
            topk_mask=topk_mask,
        )

    def _project(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_resid: Array | None,
        frequencies: Array | None,
        position_ids: Array | None,
    ) -> tuple[Array, Array, Array | None]:
        """Project hidden states / residual into indexer queries, keys, gates.

        Args:
            hidden_states: Layer input ``[batch, seq, hidden]``.
            q_resid: Low-rank residual when ``query_source="q_lora"``.
            frequencies: Rotary tables (``rope_style != "none"``).
            position_ids: Positions of the current tokens.

        Returns:
            ``(query, key, gate_scores)`` — queries
            ``[batch, seq, heads, head_dim]`` (roped), keys
            ``[batch, seq, head_dim]`` (normed, roped), and the gate scores
            ``[batch, seq, head_dim]`` float32 for ``key_gate_valid`` layouts
            (``None`` otherwise).

        Raises:
            ValueError: If ``query_source="q_lora"`` and ``q_resid`` is
                missing.
        """
        batch_size, seq_len, _ = hidden_states.shape
        if self.config.query_source == "q_lora":
            if q_resid is None:
                raise ValueError("IndexerConfig.query_source='q_lora' requires q_resid.")
            q_input = q_resid
        else:
            q_input = hidden_states

        query = self.wq_b(q_input).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        key = self.k_norm(self.wk(hidden_states))

        if self.config.rope_style != "none" and frequencies is not None:
            freqs = frequencies[position_ids] if position_ids is not None else frequencies
            cos, sin = jnp.split(freqs.astype(jnp.float32), 2, axis=-1)
            if self.config.rope_style == "interleaved":
                requested_width = self.head_dim if self.config.rope_dim is None else self.config.rope_dim
                half_width = min(requested_width, self.head_dim, 2 * cos.shape[-1]) // 2
                cos_t, sin_t = truncate_cos_sin(cos, sin, half_width)
                query = apply_indexer_rope(query, cos_t[:, :, None, :], sin_t[:, :, None, :], style="interleaved")
                key = apply_indexer_rope(key, cos_t, sin_t, style="interleaved")
            else:
                cos_t, sin_t = truncate_cos_sin(cos, sin, self.config.rope_dim)
                query = apply_indexer_rope(query, cos_t[:, :, None, :], sin_t[:, :, None, :], style="split_half")
                key = apply_indexer_rope(key, cos_t, sin_t, style="split_half")

        gate_scores = None
        if self.config.packed_state == "key_gate_valid":
            gate_scores = jnp.einsum(
                "bsh,gh->bsg",
                hidden_states.astype(jnp.float32),
                self.index_kpool_compress_gate.value.astype(jnp.float32),
                precision=self.precision,
            )
        return query, key, gate_scores

    def _pack_state(
        self,
        key: Float[Array, "batch seq head_dim"],
        gate_scores: Array | None,
        attention_mask: Bool[Array, "batch seq"],
        cached_packed: Array | None,
        cache_positions: Array | None = None,
    ) -> Array | None:
        """Append current state or update explicitly indexed fixed-capacity rows.

        Args:
            key: Normalized index keys ``[batch, seq, head_dim]``.
            gate_scores: Gate projection scores (``key_gate_valid`` only).
            attention_mask: Current-token padding mask.
            cached_packed: Previously cached state.
            cache_positions: Optional per-row physical insertion slots.

        Returns:
            The packed state ``[batch, cached + seq, dim]``, or ``None`` for
            stateless indexers.
        """
        if self.config.packed_state == "none":
            return None
        if self.config.packed_state == "keys":
            current = key
        else:
            current = jnp.concatenate(
                [key, gate_scores.astype(key.dtype), attention_mask.astype(key.dtype)[..., None]],
                axis=-1,
            )
        if cache_positions is not None:
            capacity = cached_packed.shape[1]
            slots = jnp.where(
                attention_mask & (cache_positions >= 0) & (cache_positions < capacity), cache_positions, capacity
            )
            return jax.vmap(lambda old, pos, new: old.at[pos].set(new, mode="drop"))(
                cached_packed, slots, current.astype(cached_packed.dtype)
            )
        if cached_packed is not None:
            return jnp.concatenate([cached_packed, current], axis=1)
        return current

    def _reduce_heads(
        self,
        scores: Float[Array, "batch seq heads n"],
        hidden_states: Float[Array, "batch seq hidden"],
    ) -> Float[Array, "batch seq n"]:
        """Collapse the head axis of the per-dot scores.

        Args:
            scores: Per-head scores ``[batch, seq, heads, n]`` already
                scaled/activated.
            hidden_states: Layer input for the learned head weights.

        Returns:
            Per-candidate scores ``[batch, seq, n]``.
        """
        if self.config.head_reduction == "uniform":
            return jnp.sum(scores, axis=2)
        weights = self.weights_proj(hidden_states).astype(jnp.float32) * (self.n_heads**-0.5)
        return jnp.einsum("bsh,bshn->bsn", weights, scores, precision=self.precision)

    def _select_token(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        query: Float[Array, "batch seq heads head_dim"],
        key: Float[Array, "batch seq head_dim"],
        packed_state: Array | None,
        attention_mask: Bool[Array, "batch seq"],
        pairwise_mask: Array | None,
    ) -> tuple[Array, Array | None]:
        """Score every key token and keep the top ``index_topk``.

        Args:
            hidden_states: Layer input (head-weight source).
            query: Roped indexer queries.
            key: Normed/roped index keys (current tokens).
            packed_state: Full-range state including the cache.
            attention_mask: Current-token padding mask (unused; the mask
                arrives via ``pairwise_mask``).
            pairwise_mask: Optional caller-provided query-key mask.

        Returns:
            ``(topk_indices [batch, seq, k], score_proxy)``.
        """
        del attention_mask
        keys_f32 = packed_state if packed_state is not None else key
        if keys_f32.shape[-1] != self.head_dim:
            keys_f32 = keys_f32[..., : self.head_dim]
        keys_f32 = keys_f32.astype(jnp.float32)

        scores = jnp.einsum("bshd,btd->bsht", query.astype(jnp.float32), keys_f32, precision=self.precision)
        scores = scores * self.softmax_scale
        if self.config.score_activation == "relu":
            scores = jax.nn.relu(scores)
        index_scores = self._reduce_heads(scores, hidden_states)

        valid_candidates = None
        if pairwise_mask is not None:
            mask = pairwise_mask
            if mask.ndim == 4:
                mask = mask[:, 0]
            if mask.shape[-1] != index_scores.shape[-1]:
                mask = mask[..., : index_scores.shape[-1]]
            if mask.dtype == jnp.bool_:
                valid_candidates = jnp.broadcast_to(mask, index_scores.shape)
                index_scores = jnp.where(mask, index_scores, jnp.finfo(jnp.float32).min)
            else:
                index_scores = index_scores + mask.astype(jnp.float32)
                valid_candidates = jnp.isfinite(index_scores) & (mask > jnp.finfo(mask.dtype).min)

        select_k = min(self.index_topk, index_scores.shape[-1])
        if valid_candidates is None:
            topk_indices = top_k_indices(index_scores, select_k, self.mesh_source).astype("i4")
        else:
            topk_indices = self.select_candidates(index_scores, select_k, valid_candidates, self.mesh_source).indices
        return topk_indices, index_scores

    def _select_pool(
        self,
        hidden_states: Array,
        query: Array,
        packed_state: Array,
        attention_mask: Array,
        visibility: Array | None = None,
        key_segment_ids: Array | None = None,
    ) -> tuple[Array, Array]:
        """Group visible keys consistently for both ranked pools and their tail.

        Prefix visibility can reuse storage-wide summaries. Sliding windows or
        arbitrary pairwise masks can cut across those groups: in that case each
        query's visible keys must be compacted *before* learned pooling. Map
        queries sequentially to avoid a query-by-key-by-channel temporary.
        Returned indices always retain the original physical cache coordinates.
        """
        valid_keys = packed_state[..., -1] >= 0.5
        visible = (
            visible_causal_mask(query.shape[1], packed_state.shape[1], valid_keys)
            if visibility is None
            else visibility & valid_keys[:, None, :]
        )
        positions = jnp.arange(packed_state.shape[1])
        last_visible = jnp.max(jnp.where(visible, positions, -1), axis=-1)
        prefix = valid_keys[:, None, :] & (positions <= last_visible[..., None])
        if key_segment_ids is not None:
            last_segment = jnp.take_along_axis(key_segment_ids, jnp.maximum(last_visible, 0), axis=1)
            prefix = prefix & (key_segment_ids[:, None, :] == last_segment[..., None])

        def shared_pools(_):
            return self._select_storage_pools(
                hidden_states, query, packed_state, attention_mask, visible, key_segment_ids
            )

        def query_local_pools(_):
            def select_one(inputs):
                hidden, projected, current_valid, current_visible = inputs
                local_state = packed_state.at[..., -1].set(current_visible.astype(packed_state.dtype))
                indices, mask = self._select_storage_pools(
                    hidden[:, None],
                    projected[:, None],
                    local_state,
                    current_valid[:, None],
                    key_segment_ids=key_segment_ids,
                )
                return indices[:, 0], mask[:, 0]

            indices, mask = jax.lax.map(
                select_one,
                tuple(jnp.swapaxes(x, 0, 1) for x in (hidden_states, query, attention_mask, visible)),
            )
            return jnp.swapaxes(indices, 0, 1), jnp.swapaxes(mask, 0, 1)

        return jax.lax.cond(jnp.all(prefix == visible), shared_pools, query_local_pools, operand=None)

    def _select_storage_pools(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        query: Float[Array, "batch seq heads head_dim"],
        packed_state: Array,
        attention_mask: Bool[Array, "batch seq"],
        visibility: Array | None = None,
        key_segment_ids: Array | None = None,
    ) -> tuple[Array, Array | None]:
        """Score pool summaries, top-k pools, expand back to token indices.

        Args:
            hidden_states: Layer input (head-weight source).
            query: Roped indexer queries.
            packed_state: Packed ``[key | gate | valid]`` states over the
                full KV range.
            attention_mask: Current-token padding mask.
            visibility: Optional explicit causal/document visibility.
            key_segment_ids: Optional document identities for pool boundaries.

        Returns:
            ``(topk_indices [batch, seq, width], topk_mask [batch, seq, kv])``
            — ``width`` is ``index_topk`` plus ``kpool_size - 1`` when the
            tail pool is selected; ``-1`` marks invalid entries. The mask
            holds the same selection in dense form.
        """
        batch_size, seq_len, _ = hidden_states.shape
        kv_len = packed_state.shape[1]
        kpool = self.config.kpool_size

        valid_keys = packed_state[..., -1] >= 0.5
        visible_tokens = (
            visible_causal_mask(seq_len, kv_len, valid_keys)
            if visibility is None
            else visibility & valid_keys[:, None, :]
        )

        pool_keys, pool_indices, pool_valid = self._pooled_states(packed_state, key_segment_ids)
        num_pools = pool_keys.shape[1]

        scores = jnp.einsum(
            "bshd,bpd->bshp", query.astype(jnp.float32), pool_keys.astype(jnp.float32), precision=self.precision
        )
        scores = scores * self.softmax_scale
        if self.config.score_activation == "relu":
            scores = jax.nn.relu(scores)
        index_scores = self._reduce_heads(scores, hidden_states)

        # A pool is selectable only when its final token is visible
        # (causality + padding).
        pool_end = jnp.clip(pool_indices[..., -1], 0, kv_len - 1)
        pool_end_q = jnp.broadcast_to(pool_end[:, None, :], (batch_size, seq_len, num_pools))
        pool_visible = jnp.take_along_axis(visible_tokens, pool_end_q.astype("i4"), axis=2)
        # Explicit logical positions may differ from physical row order.
        # Every member must be visible, not just the final physical row.
        if visibility is not None:
            for member in range(kpool - 1):
                member_q = jnp.broadcast_to(
                    jnp.maximum(pool_indices[..., member], 0)[:, None, :], (batch_size, seq_len, num_pools)
                )
                pool_visible = pool_visible & jnp.take_along_axis(visible_tokens, member_q, axis=2)
        valid_candidates = pool_visible & pool_valid[:, None, :]

        min_score = jnp.finfo(jnp.float32).min
        index_scores = jnp.where(valid_candidates, index_scores, min_score)

        select_k = min(self.index_topk // kpool, num_pools)
        selection = self.select_candidates(index_scores, select_k, valid_candidates, self.mesh_source)
        selected = selection.indices

        selected_valid = selected >= 0
        expanded_pools = jnp.broadcast_to(pool_indices[:, None], (batch_size, seq_len, num_pools, kpool))
        safe_selected = jnp.maximum(selected, 0)
        selected_pool_indices = jnp.take_along_axis(expanded_pools, safe_selected[..., None], axis=2)
        selected_indices = selected_pool_indices.reshape(batch_size, seq_len, select_k * kpool)
        selected_valid_flat = jnp.broadcast_to(
            selected_valid[..., None], (batch_size, seq_len, select_k, kpool)
        ).reshape(batch_size, seq_len, select_k * kpool)

        topk_indices = jnp.where(selected_valid_flat, selected_indices, -1)

        # Map each physical token's compact valid-token rank back to its
        # selected pool. This preserves cache coordinates even when cached
        # right padding leaves holes before a live continuation, without a
        # scatter or a query-by-pool-by-token temporary.
        if key_segment_ids is not None:
            # Per-document incomplete groups are excluded from the ranked
            # candidate axis, so a global token-rank division is not a pool ID.
            topk_mask = indices_to_bool_mask(topk_indices, kv_len)
        else:
            pool_mask = selection.mask
            if pool_mask is None:
                pool_mask = jnp.zeros(index_scores.shape, dtype=jnp.bool_)
            token_pool = jnp.maximum(jnp.cumsum(valid_keys, axis=-1) - 1, 0) // kpool
            token_pool = jnp.broadcast_to(token_pool[:, None, :], (batch_size, seq_len, kv_len))
            topk_mask = jnp.take_along_axis(pool_mask, token_pool.astype("i4"), axis=-1)
            topk_mask = topk_mask & valid_keys[:, None, :]

        output_width = self.index_topk
        if self.config.select_tail:
            topk_indices = self._append_visible_tail(topk_indices, visible_tokens, valid_keys)
            topk_mask = topk_mask | self._visible_tail_mask(visible_tokens, valid_keys)
            output_width += kpool - 1

        pad_width = output_width - topk_indices.shape[-1]
        if pad_width > 0:
            topk_indices = jnp.pad(topk_indices, [(0, 0), (0, 0), (0, pad_width)], constant_values=-1)
        topk_indices = topk_indices[..., :output_width]
        topk_indices = jnp.where(attention_mask[..., None], topk_indices, -1)
        topk_mask = topk_mask & attention_mask[..., None]
        return topk_indices.astype("i4"), topk_mask

    @staticmethod
    def _compact_valid_indices(key_valid: Bool[Array, "batch kv"]) -> Array:
        """Map valid-token ranks to original cache indices, padded with ``-1``."""
        kv_length = key_valid.shape[-1]
        return jax.vmap(lambda row: jnp.nonzero(row, size=kv_length, fill_value=-1)[0])(key_valid)

    def _pooled_states(
        self,
        packed_state: Array,
        key_segment_ids: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """Build compressed k-pool candidates from the packed state.

        Pools group consecutive *valid* tokens, keeping their original
        cache indices. Padding at the start or between cached chunks never
        consumes a pool slot; incomplete pools are marked invalid.

        Args:
            packed_state: Packed ``[key | gate | valid]`` states
                ``(batch, kv, 2 * head_dim + 1)``.
            key_segment_ids: Optional document identities; complete pools
                never cross a boundary, and APE slot zero restarts there.

        Returns:
            Tuple of pool summary keys ``(batch, pools, head_dim)``, raw
            member token indices ``(batch, pools, kpool)`` (``-1`` invalid),
            and pool validity ``(batch, pools)``.
        """
        keys, gate_scores, valid_flags = jnp.split(packed_state, [self.head_dim, 2 * self.head_dim], axis=-1)
        valid_keys = valid_flags[..., 0] >= 0.5

        _, kv_len = valid_keys.shape
        kpool = self.config.kpool_size
        num_pools = (kv_len + kpool - 1) // kpool

        if key_segment_ids is None:
            compact_indices = self._compact_valid_indices(valid_keys)
            compact_indices = jnp.pad(compact_indices, [(0, 0), (0, num_pools * kpool - kv_len)], constant_values=-1)
            pool_indices = compact_indices.reshape(valid_keys.shape[0], num_pools, kpool)
        else:
            valid_keys = valid_keys & (key_segment_ids >= 0)
            compact_indices = self._compact_valid_indices(valid_keys)
            compact_segments = jnp.take_along_axis(key_segment_ids, jnp.maximum(compact_indices, 0), axis=1)
            ranks = jnp.broadcast_to(jnp.arange(kv_len), valid_keys.shape)
            boundary = jnp.concatenate(
                [
                    jnp.ones((valid_keys.shape[0], 1), dtype=jnp.bool_),
                    compact_segments[:, 1:] != compact_segments[:, :-1],
                ],
                axis=1,
            )
            run_start = jax.lax.associative_scan(jnp.maximum, jnp.where(boundary, ranks, 0), axis=1)
            local_rank = ranks - run_start
            complete_end = ((local_rank + 1) % kpool == 0) & (compact_indices >= 0)
            # At most ceil(kv/kpool) complete pools can exist, even when every
            # document leaves an incomplete tail. Compact only complete groups
            # instead of allocating a candidate for every possible segment.
            ends = jax.vmap(lambda row: jnp.nonzero(row, size=num_pools, fill_value=-1)[0])(complete_end)
            member_ranks = ends[..., None] - kpool + 1 + jnp.arange(kpool)
            pool_indices = jnp.take_along_axis(
                compact_indices[:, None, :], jnp.clip(member_ranks, 0, kv_len - 1), axis=2
            )
            pool_indices = jnp.where(ends[..., None] >= 0, pool_indices, -1)

        safe_indices = jnp.clip(pool_indices, 0, kv_len - 1)
        in_range = pool_indices >= 0

        def take(x, idx):
            # x: [B, n, d]; idx: [B, P, kpool] -> [B, P, kpool, d]
            flat = x.reshape(x.shape[0], x.shape[1], -1)
            gathered = jnp.take_along_axis(flat, idx.reshape(x.shape[0], -1)[..., None].astype("i4"), axis=1)
            return gathered.reshape((x.shape[0], *idx.shape[1:], x.shape[2]))

        grouped_keys = take(keys, safe_indices)
        grouped_gate_scores = take(gate_scores.astype(jnp.float32), safe_indices)
        grouped_valid = take(valid_keys[..., None].astype("i4"), safe_indices)[..., 0].astype(bool)
        grouped_valid = grouped_valid & in_range

        pool_valid = jnp.all(grouped_valid, axis=-1)
        pool_indices = jnp.where(grouped_valid, pool_indices, -1)

        # Softmax-gated learned average over each complete pool.
        ape = self.index_kpool_compress_ape.value.astype(jnp.float32)
        logits = grouped_gate_scores + ape[None, None]
        logits = jnp.where(grouped_valid[..., None], logits, -1e30)
        probabilities = jax.nn.softmax(logits, axis=2)
        probabilities = jnp.nan_to_num(probabilities, nan=0.0)
        pool_keys = jnp.sum(probabilities * grouped_keys.astype(jnp.float32), axis=2).astype(keys.dtype)

        return pool_keys, pool_indices, pool_valid

    def _append_visible_tail(
        self,
        topk_indices: Array,
        token_visible: Bool[Array, "batch q kv"],
        key_valid: Bool[Array, "batch kv"],
    ) -> Array:
        """Append the current incomplete pool as raw token indices.

        With ``kpool_size=4`` and visible keys ``[A B C D E F]`` the selected
        full pool covers ``[A B C D]``; the tail appends ``[E F]``.

        Args:
            topk_indices: Selected pool-expanded indices ``(batch, q, w)``.
            token_visible: Query visibility ``(batch, q, kv)``.
            key_valid: Key validity ``(batch, kv)``.

        Returns:
            Indices ``(batch, q, w + kpool_size - 1)`` with the tail appended
            (``-1`` where invalid).
        """
        max_tail_width = self.config.kpool_size - 1
        if max_tail_width == 0:
            return topk_indices

        tail_mask = self._visible_tail_mask(token_visible, key_valid)
        # The compact ranks are query/document-local, while returned indices
        # always remain physical cache coordinates.
        tail_indices = jax.vmap(jax.vmap(lambda row: jnp.nonzero(row, size=max_tail_width, fill_value=-1)[0]))(tail_mask)
        return jnp.concatenate([topk_indices, tail_indices], axis=-1)

    def _visible_tail_range(
        self,
        token_visible: Bool[Array, "batch q kv"],
        key_valid: Bool[Array, "batch kv"],
    ) -> tuple[Array, Array]:
        """Compact rank start and length of each query's incomplete tail pool."""
        visible_count = jnp.sum(token_visible & key_valid[:, None, :], axis=-1)
        tail_count = visible_count % self.config.kpool_size
        return visible_count - tail_count, tail_count

    def _visible_tail_mask(
        self,
        token_visible: Bool[Array, "batch q kv"],
        key_valid: Bool[Array, "batch kv"],
    ) -> Bool[Array, "batch q kv"]:
        """Dense form of :meth:`_append_visible_tail`'s appended indices."""
        if self.config.kpool_size == 1:
            return jnp.zeros_like(token_visible, dtype=jnp.bool_)
        tail_start, tail_count = self._visible_tail_range(token_visible, key_valid)
        rank = jnp.cumsum(token_visible & key_valid[:, None, :], axis=-1) - 1
        start = tail_start[..., None]
        in_tail = (rank >= start) & (rank < start + tail_count[..., None])
        return in_tail & token_visible.astype(jnp.bool_)
