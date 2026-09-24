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
from ._primitives import indices_to_bool_mask, is_vacuous_selection, visible_causal_mask
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

        query, key, gate_scores = self._project(hidden_states, q_resid, frequencies, position_ids)
        packed_state = self._pack_state(key, gate_scores, attention_mask, cached_packed)

        topk_mask = None
        if self.config.kind == IndexerKind.POOL:
            topk_indices, topk_mask = self._select_pool(hidden_states, query, packed_state, attention_mask)
            score_proxy = None
        else:
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
                rope_dim = cos.shape[-1]
                rope_dim = max(1, min(2 * rope_dim, self.head_dim) // 2)
                cos_t, sin_t = truncate_cos_sin(cos, sin, rope_dim)
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
            )
        return query, key, gate_scores

    def _pack_state(
        self,
        key: Float[Array, "batch seq head_dim"],
        gate_scores: Array | None,
        attention_mask: Bool[Array, "batch seq"],
        cached_packed: Array | None,
    ) -> Array | None:
        """Append the current tokens' state to the cached state.

        Args:
            key: Normalized index keys ``[batch, seq, head_dim]``.
            gate_scores: Gate projection scores (``key_gate_valid`` only).
            attention_mask: Current-token padding mask.
            cached_packed: Previously cached state.

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
        return jnp.einsum("bsh,bshn->bsn", weights, scores)

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

        scores = jnp.einsum("bshd,btd->bsht", query.astype(jnp.float32), keys_f32)
        scores = scores * self.softmax_scale
        if self.config.score_activation == "relu":
            scores = jax.nn.relu(scores)
        index_scores = self._reduce_heads(scores, hidden_states)

        if pairwise_mask is not None:
            mask = pairwise_mask
            if mask.ndim == 4:
                mask = mask[:, 0]
            if mask.shape[-1] != index_scores.shape[-1]:
                mask = mask[..., : index_scores.shape[-1]]
            if mask.dtype == jnp.bool_:
                index_scores = jnp.where(mask, index_scores, jnp.finfo(jnp.float32).min)
            else:
                index_scores = index_scores + mask.astype(jnp.float32)

        select_k = min(self.index_topk, index_scores.shape[-1])
        topk_indices = top_k_indices(index_scores, select_k, self.mesh_source).astype("i4")
        return topk_indices, index_scores

    def _select_pool(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        query: Float[Array, "batch seq heads head_dim"],
        packed_state: Array,
        attention_mask: Bool[Array, "batch seq"],
    ) -> tuple[Array, Array | None]:
        """Score pool summaries, top-k pools, expand back to token indices.

        Args:
            hidden_states: Layer input (head-weight source).
            query: Roped indexer queries.
            packed_state: Packed ``[key | gate | valid]`` states over the
                full KV range.
            attention_mask: Current-token padding mask.

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
        visible_tokens = visible_causal_mask(seq_len, kv_len, valid_keys)

        pool_keys, pool_indices, pool_valid = self._pooled_states(packed_state)
        num_pools = pool_keys.shape[1]

        scores = jnp.einsum("bshd,bpd->bshp", query.astype(jnp.float32), pool_keys.astype(jnp.float32))
        scores = scores * self.softmax_scale
        if self.config.score_activation == "relu":
            scores = jax.nn.relu(scores)
        index_scores = self._reduce_heads(scores, hidden_states)

        # A pool is selectable only when its final token is visible
        # (causality + padding).
        pool_end = jnp.clip(pool_indices[..., -1], 0, kv_len - 1)
        pool_end_q = jnp.broadcast_to(pool_end[:, None, :], (batch_size, seq_len, num_pools))
        pool_visible = jnp.take_along_axis(visible_tokens, pool_end_q.astype("i4"), axis=2)
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

        # Dense form of the same selection. Selected pools are complete, so
        # pool ``p`` covers tokens ``first_key + p*kpool + [0, kpool)``: mark
        # the pools, repeat each over its members and shift by ``first_key``.
        # This avoids scattering the token indices, which is slow on TPU.
        pool_mask = selection.mask
        if pool_mask is None:
            pool_mask = jnp.zeros(index_scores.shape, dtype=jnp.bool_)
        token_mask = jnp.repeat(pool_mask, kpool, axis=-1)
        token_mask = jnp.pad(token_mask, [(0, 0), (0, 0), (kv_len, 0)])
        first_key = self._first_valid_key(valid_keys)
        topk_mask = jax.vmap(lambda row, start: jax.lax.dynamic_slice_in_dim(row, start, kv_len, axis=-1))(
            token_mask, kv_len - first_key
        )

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
    def _first_valid_key(key_valid: Bool[Array, "batch kv"]) -> Array:
        """Index of each row's first valid key; ``kv`` for an all-invalid row."""
        kv_length = key_valid.shape[-1]
        return jnp.where(jnp.any(key_valid, axis=-1), jnp.argmax(key_valid, axis=-1), kv_length)

    def _pooled_states(
        self,
        packed_state: Array,
    ) -> tuple[Array, Array, Array]:
        """Build compressed k-pool candidates from the packed state.

        Pooling starts at the first valid token so padded prefixes do not
        misalign the groups; incomplete pools are marked invalid.

        Args:
            packed_state: Packed ``[key | gate | valid]`` states
                ``(batch, kv, 2 * head_dim + 1)``.

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

        first_key = self._first_valid_key(valid_keys)
        offsets = jnp.arange(num_pools * kpool).reshape(1, num_pools, kpool)
        pool_indices = first_key[:, None, None] + offsets

        safe_indices = jnp.clip(pool_indices, 0, kv_len - 1)
        in_range = pool_indices < kv_len

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

        _, _, kv_length = token_visible.shape
        tail_start, tail_count = self._visible_tail_range(token_visible, key_valid)
        tail_offsets = jnp.arange(max_tail_width)
        tail_indices = tail_start[..., None] + tail_offsets

        tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & (tail_indices < kv_length)
        safe_tail = jnp.clip(tail_indices, 0, kv_length - 1)
        tail_visible = jnp.take_along_axis(token_visible, safe_tail.astype("i4"), axis=2).astype(bool) & (
            tail_indices >= 0
        )
        tail_indices = jnp.where(tail_valid & tail_visible, tail_indices, -1)
        return jnp.concatenate([topk_indices, tail_indices], axis=-1)

    def _visible_tail_range(
        self,
        token_visible: Bool[Array, "batch q kv"],
        key_valid: Bool[Array, "batch kv"],
    ) -> tuple[Array, Array]:
        """Start and length ``(batch, q)`` of each query's incomplete tail pool."""
        first_key = self._first_valid_key(key_valid)
        visible_count = jnp.sum(token_visible, axis=-1)
        tail_count = visible_count % self.config.kpool_size
        return first_key[:, None] + visible_count - tail_count, tail_count

    def _visible_tail_mask(
        self,
        token_visible: Bool[Array, "batch q kv"],
        key_valid: Bool[Array, "batch kv"],
    ) -> Bool[Array, "batch q kv"]:
        """Dense form of :meth:`_append_visible_tail`'s appended indices."""
        if self.config.kpool_size == 1:
            return jnp.zeros_like(token_visible, dtype=jnp.bool_)
        tail_start, tail_count = self._visible_tail_range(token_visible, key_valid)
        position = jax.lax.broadcasted_iota(jnp.int32, token_visible.shape, 2)
        start = tail_start[..., None]
        in_tail = (position >= start) & (position < start + tail_count[..., None])
        return in_tail & token_visible.astype(jnp.bool_)


# convenience re-exports for callers that mask attention from indices
indices_to_mask = indices_to_bool_mask
vacuous_selection = is_vacuous_selection
