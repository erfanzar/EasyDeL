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

"""Operation-backed weighted token indexer with explicit key-cache ownership."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import spectrax as spx
from jax import numpy as jnp

from easydel.infra.sequence_packing import pairwise_attention_mask_from_mask_info
from easydel.layers.linears import ColumnParallelLinear
from easydel.layers.rotary import apply_rope_interleaved, apply_rope_split_half

from ._distill import indexer_distillation_loss
from ._paged import paged_token_select
from ._selection import BaseIndexer, IndexerSelection, SelectionSpec

if tp.TYPE_CHECKING:
    from ejkernel.types import MaskInfo
    from jaxtyping import Array

    from easydel.operations.kernels.glm_moe_dsa_indexer import GlmMoeDsaIndexerOutput


@dataclass(frozen=True)
class TokenIndexerConfig:
    """Dimensions and RoPE policy for weighted per-token operation dispatch."""

    hidden_size: int
    q_input_dim: int
    num_heads: int
    head_dim: int
    topk: int
    rope_dim: int
    rope_interleave: bool = False
    initializer_range: float = 0.02


class TokenIndexer(BaseIndexer):
    """Weighted token scores dispatched through the GLM DSA operation.

    This strategy deliberately retains that operation's mask, rotary and
    cache semantics rather than substituting the pooled indexer's scorer.
    The ``kernels_proj`` checkpoint name and HF ``weights_proj`` reform rule
    are part of this implementation's public parameter layout.
    """

    selection_spec = SelectionSpec(candidate_unit="token", output_unit="token")
    # Selected ids are token ids; there is no pooled ``dsa_pool_training`` selection.
    selection_chunk_size: tp.ClassVar[int] = 1
    supports_pool_training: tp.ClassVar[bool] = False

    def __init__(
        self,
        config: TokenIndexerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        base_config,
        norm_factory: tp.Callable[[], spx.Module],
    ):
        """Build checkpoint-compatible projections and operation metadata.

        ``base_config`` supplies the standard EasyDeL runtime/sharding policy;
        ``norm_factory`` supplies the key norm with its exact family layout.
        """
        # Operations import shared layers too; defer until construction so
        # exporting this class from the layers root cannot form an import cycle.
        from easydel.operations import OperationMetadata
        from easydel.operations.kernels.glm_moe_dsa_indexer import GlmMoeDsaIndexerOp

        self.indexer_config = config
        self.dtype, self.param_dtype, self.precision = dtype, param_dtype, precision
        self.index_n_heads, self.index_head_dim, self.index_topk = config.num_heads, config.head_dim, config.topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.indexer_rope_interleave = config.rope_interleave
        linear = partial(
            ColumnParallelLinear,
            rngs=rngs,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )
        self.wq_b = linear(config.q_input_dim, config.num_heads * config.head_dim)
        self.wk = linear(config.hidden_size, config.head_dim)
        self.k_norm = norm_factory()
        self.kernels_proj = linear(config.hidden_size, config.num_heads)
        self.indexer_op = GlmMoeDsaIndexerOp(
            OperationMetadata(runtime_dtype=dtype, runtime_softmax_dtype=jnp.float32, base_config=base_config)
        )

    @property
    def reform_param(self):
        """Map the HF head-weight projection to the historical runtime name."""
        return {
            "weights_proj.weight$": {
                "splits": [{"name": "kernels_proj.weight", "spliter": lambda x: x.swapaxes(-1, -2)}],
                "inverse_spliter": lambda x: x.swapaxes(-1, -2),
            },
        }

    @staticmethod
    def selection_from_output(output: GlmMoeDsaIndexerOutput) -> IndexerSelection:
        """View an existing output, preserving its pre-ranking visibility contract.

        The legacy operation indices can include masked fillers when fewer than
        topk keys are live. Its dense mask is authoritative for those lanes;
        expose them as -1 through the common indexer interface as well.
        """
        indices, mask = output.topk_indices, output.topk_mask
        if mask is not None:
            valid = (indices >= 0) & (indices < mask.shape[-1])
            live = jnp.take_along_axis(mask, jnp.clip(indices, 0, mask.shape[-1] - 1), axis=-1)
            indices = jnp.where(valid & live, indices, -1)
        return IndexerSelection(indices, mask=mask)

    def forward(
        self,
        hidden_states,
        q_resid,
        position_ids,
        frequencies=None,
        attention_mask=None,
        cached_keys=None,
        use_cache=False,
        *,
        cache_positions=None,
        key_valid_mask=None,
        query_positions=None,
        key_positions=None,
        segment_ids=None,
        key_segment_ids=None,
        reset_cache=None,
    ) -> GlmMoeDsaIndexerOutput:
        """Return selected token IDs and an optional updated key cache.

        Args:
            hidden_states: Current layer inputs ``[batch, seq, hidden]``.
            q_resid: Optional low-rank query inputs; ``None`` uses hidden states.
            position_ids: Current RoPE positions ``[batch, seq]``.
            frequencies: Optional RoPE table.
            attention_mask: Optional pairwise boolean/additive score mask.
            cached_keys: Historical keys or preallocated indexed key storage.
            use_cache: Update and return the key cache when true.
            cache_positions: Optional physical write slots ``[batch, seq]``
                in fixed-capacity ``cached_keys``. Multi-token continuations
                retain history instead of resetting it in this mode.
            key_valid_mask: Optional post-update validity ``[batch, kv]``.
            query_positions: Optional logical causal positions ``[batch, seq]``;
                separate from RoPE positions and physical insertion slots.
            key_positions: Optional logical positions ``[batch, kv]`` of keys.
            segment_ids: Optional query document identities ``[batch, seq]``.
            key_segment_ids: Optional post-update key identities ``[batch, kv]``.
            reset_cache: Explicit reset policy. ``None`` preserves the legacy
                multi-token reset/single-token append behavior unless physical
                insertion slots are supplied, in which case history is kept.

        Rotary transformation, indexed writes and visibility are owned by the
        registered operation. The owning cache retains validity and segment
        metadata separately from the checkpoint-compatible key channels.
        """
        q_input = q_resid if q_resid is not None else hidden_states
        query_states = self.wq_b(q_input).reshape(
            hidden_states.shape[0], hidden_states.shape[1], self.index_n_heads, self.index_head_dim
        )
        key_states = self.k_norm(self.wk(hidden_states))
        head_weights = self.kernels_proj(hidden_states).astype(jnp.float32) * (self.index_n_heads**-0.5)
        indexed_metadata = {
            name: value
            for name, value in {
                "cache_positions": cache_positions,
                "key_valid_mask": key_valid_mask,
                "query_positions": query_positions,
                "key_positions": key_positions,
                "segment_ids": segment_ids,
                "key_segment_ids": key_segment_ids,
            }.items()
            if value is not None
        }
        if reset_cache is None:
            reset_cache = cache_positions is None and hidden_states.shape[1] > 1
        return self.indexer_op(
            query_states=query_states,
            key_states=key_states,
            head_weights=head_weights,
            position_ids=position_ids,
            qk_rope_head_dim=self.indexer_config.rope_dim,
            index_topk=self.index_topk,
            softmax_scale=self.softmax_scale,
            frequencies=frequencies,
            attention_mask=attention_mask,
            cached_keys=cached_keys,
            use_cache=use_cache,
            reset_cache=reset_cache,
            indexer_rope_interleave=self.indexer_rope_interleave,
            precision=self.precision,
            **indexed_metadata,
        )

    def select_uncached(
        self,
        hidden_states: Array,
        q_resid: Array | None,
        position_ids: Array,
        frequencies: Array | None,
        mask_info: MaskInfo,
    ) -> IndexerSelection:
        """Selection for a call without a cache: the keys are the current tokens.

        Args:
            hidden_states: Layer input ``[batch, seq, hidden]``.
            q_resid: Low-rank query input, or ``None`` to use ``hidden_states``.
            position_ids: RoPE positions ``[batch, seq]``.
            frequencies: RoPE table, or ``None``.
            mask_info: Self-attention mask metadata.

        Returns:
            The selection, with lanes the operation filled past the visible keys set to ``-1``.
        """
        q_len = hidden_states.shape[1]
        output = self(
            hidden_states=hidden_states,
            q_resid=q_resid,
            position_ids=position_ids,
            frequencies=frequencies,
            attention_mask=pairwise_attention_mask_from_mask_info(mask_info, q_len, q_len),
        )
        return self.selection_from_output(output)

    def select_with_cache(
        self,
        hidden_states: Array,
        q_resid: Array | None,
        position_ids: Array,
        frequencies: Array | None,
        cache_view: tp.Any,
        mask_info: MaskInfo,
        cache_metadata: tp.Any,
        *,
        causal: bool,
        mode: tp.Any,
        runtime_sharding_resolver: tp.Any,
    ) -> tuple[IndexerSelection, tp.Any]:
        """Selection over a transformer cache whose auxiliary channel stores the rotated indexer keys.

        The cache first reports its attention-time layout (placeholder rows at the current physical slots); the
        operation writes the current keys into those slots, and the same public append stores them, so sliding
        caches keep only their fixed-capacity window.

        Args:
            hidden_states: Layer input ``[batch, seq, hidden]``.
            q_resid: Low-rank query input, or ``None``.
            position_ids: RoPE positions ``[batch, seq]``.
            frequencies: RoPE table, or ``None``.
            cache_view: A ``TransformerCacheView`` with ``auxiliary_dim == head_dim``.
            mask_info: Attention mask metadata of the current step.
            cache_metadata: Cache metadata.
            causal: Whether the owning attention is causal.
            mode: Runtime mode.
            runtime_sharding_resolver: Sharding resolver for the cache update.

        Returns:
            ``(selection over the cache rows, updated cache_view)``.
        """

        def update(states, view):
            return view.update_auxiliary(
                states,
                mask_info,
                cache_metadata,
                causal=causal,
                mode=mode,
                runtime_sharding_resolver=runtime_sharding_resolver,
            )

        bsz, q_len, _ = hidden_states.shape
        aligned_keys, index_mask, _ = update(jnp.zeros((bsz, q_len, self.index_head_dim), jnp.float32), cache_view)
        pairwise_mask = index_mask.get_or_compute_attention_mask(dtype=jnp.bool_)[:, 0]
        slots = index_mask.q_positions - index_mask.kv_positions[:, :1]
        slots = jnp.where(jnp.any(pairwise_mask, axis=-1), slots, -1)
        segment_ids, key_segment_ids = index_mask.get_or_compute_segment_ids()
        output = self(
            hidden_states=hidden_states,
            q_resid=q_resid,
            position_ids=position_ids,
            frequencies=frequencies,
            attention_mask=pairwise_mask,
            cached_keys=aligned_keys,
            use_cache=True,
            cache_positions=slots,
            key_valid_mask=jnp.any(pairwise_mask, axis=-2),
            query_positions=index_mask.q_positions,
            key_positions=index_mask.kv_positions,
            segment_ids=segment_ids,
            key_segment_ids=key_segment_ids,
        )
        current_keys = jnp.take_along_axis(output.cached_keys, jnp.maximum(slots, 0)[..., None], axis=1)
        _, _, cache_view = update(current_keys, cache_view)
        return self.selection_from_output(output), cache_view

    def select_with_pages(
        self,
        hidden_states: Array,
        q_resid: Array | None,
        frequencies: Array | None,
        cache_view: tp.Any,
        cache_metadata: tp.Any,
    ) -> tuple[Array, tp.Any]:
        """Token selection over an MLA ragged-pages cache with indexer pages (:func:`~._paged.paged_token_select`).

        Args:
            hidden_states: Packed layer input ``[1, tokens, hidden]`` (or ``[batch, seq, hidden]``, flattened).
            q_resid: Low-rank query input of the same layout, or ``None``.
            frequencies: RoPE table, or ``None``.
            cache_view: An ``MLARaggedPagesCacheView`` carrying indexer token pages.
            cache_metadata: Ragged-pages metadata of the step.

        Returns:
            ``(token ids [tokens, index_topk], cache_view with updated indexer pages)``.
        """
        from easydel.operations.kernels.multi_latent_ragged_page_attention import _resolve_distribution

        tokens = hidden_states.shape[0] * hidden_states.shape[1]
        selected, token_pages = paged_token_select(
            self,
            hidden_states.reshape(tokens, hidden_states.shape[-1]),
            None if q_resid is None else q_resid.reshape(tokens, q_resid.shape[-1]),
            frequencies,
            cache_view.indexer_token_pages,
            jnp.asarray(cache_metadata.pages_tables, jnp.int32),
            jnp.asarray(cache_metadata.context_lens, jnp.int32),
            jnp.asarray(cache_metadata.query_start_loc, jnp.int32),
            # ``[..., 2]``: the live count, per rank for rank-major ([dp, 3]) distributions.
            _resolve_distribution(cache_metadata)[..., 2],
            constrain_rows=cache_view.constrain_indexer_rows,
            constrain_pages=cache_view.constrain_indexer_pages,
            dp_sharding=cache_view.indexer_dp_sharding(),
        )
        return selected, cache_view.replace(indexer_token_pages=token_pages)

    def distillation_scores(
        self,
        hidden_states: Array,
        q_resid: Array | None,
        position_ids: Array | None,
        frequencies: Array | None,
        visible: Array,
    ) -> Array:
        """Differentiable pre-top-k scores, with unreachable tokens at ``-inf``.

        Reproduces the operation's fp32 prefix RoPE, per-head ReLU and learned head weighting with the same
        parameters. Inputs are detached so only indexer parameters train.

        Args:
            hidden_states: Layer inputs ``[B, S, hidden]``.
            q_resid: Optional low-rank query inputs ``[B, S, rank]``.
            position_ids: RoPE positions ``[B, S]``.
            frequencies: Optional concatenated half-width cosine/sine table.
            visible: Boolean causal/padding/document reachability ``[B, S, S]``.

        Returns:
            Float32 indexer scores ``[B, S, S]``.
        """
        hidden_states = jax.lax.stop_gradient(hidden_states)
        q_input = hidden_states if q_resid is None else jax.lax.stop_gradient(q_resid)
        query = self.wq_b(q_input).reshape(*hidden_states.shape[:2], self.index_n_heads, self.index_head_dim)
        key = self.k_norm(self.wk(hidden_states))
        weights = self.kernels_proj(hidden_states).astype(jnp.float32) * (self.index_n_heads**-0.5)
        query, key = query.astype(jnp.float32), key.astype(jnp.float32)
        if frequencies is not None:
            cos, sin = jnp.split(frequencies[position_ids], 2, axis=-1)
            rope_dim = min(self.indexer_config.rope_dim, self.index_head_dim, cos.shape[-1] * 2)
            rope_dim -= rope_dim % 2
            if rope_dim > 0:
                cos, sin = cos[..., : rope_dim // 2], sin[..., : rope_dim // 2]
                q_pe, k_pe = query[..., :rope_dim], key[..., :rope_dim]
                if self.indexer_rope_interleave:
                    q_pe = apply_rope_interleaved(q_pe, cos[..., None, :], sin[..., None, :])
                    k_pe = apply_rope_interleaved(k_pe, cos, sin)
                else:
                    cos, sin = jnp.concatenate((cos, cos), -1), jnp.concatenate((sin, sin), -1)
                    q_pe = apply_rope_split_half(q_pe, cos, sin, unsqueeze_dim=-2)
                    k_pe = apply_rope_split_half(k_pe, cos, sin, unsqueeze_dim=None)
                query = jnp.concatenate((q_pe, query[..., rope_dim:]), -1)
                key = jnp.concatenate((k_pe, key[..., rope_dim:]), -1)
        scores = jnp.einsum("bshd,btd->bsht", query, key, precision=self.precision) * self.softmax_scale
        scores = jnp.einsum("bsht,bsh->bst", jax.nn.relu(scores), weights, precision=self.precision)
        return jnp.where(visible, scores, -jnp.inf)

    def distillation_loss(
        self,
        hidden_states: Array,
        q_resid: Array | None,
        attention_query: Array,
        attention_key: Array,
        visibility: Array,
        query_segment_ids: Array,
        key_segment_ids: Array,
        *,
        scaling: float,
        sinks: Array | None = None,
        position_ids: Array | None = None,
        frequencies: Array | None = None,
    ) -> Array:
        """Indexer warm-up loss: dense attention distilled onto :meth:`distillation_scores`.

        Args:
            hidden_states: Layer inputs ``[B, S, hidden]``.
            q_resid: Optional low-rank query inputs ``[B, S, rank]``.
            attention_query: The attention's queries ``[B, S, H, Dq]``.
            attention_key: The attention's per-head keys ``[B, S, H, Dq]``.
            visibility: Pre-selection causal/padding/document mask ``[B, S, S]``.
            query_segment_ids: Query document identities ``[B, S]``.
            key_segment_ids: Key document identities ``[B, S]``; negative means padding.
            scaling: Attention softmax scale.
            sinks: Optional attention sinks, one per attention head.
            position_ids: RoPE positions ``[B, S]``.
            frequencies: RoPE table, or ``None``.

        Returns:
            Unscaled scalar loss.
        """
        scores = self.distillation_scores(hidden_states, q_resid, position_ids, frequencies, visibility)
        batch, seq = hidden_states.shape[:2]
        positions = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (batch, seq))
        return indexer_distillation_loss(
            attention_query.transpose(0, 2, 1, 3),
            attention_key.transpose(0, 2, 1, 3),
            scores,
            scaling=scaling,
            sinks=sinks,
            precision=self.precision,
            entry_valid=jnp.any(visibility, axis=-2),
            q_positions=positions,
            entry_positions=positions,
            q_segment_ids=query_segment_ids,
            entry_segment_ids=key_segment_ids,
        )
