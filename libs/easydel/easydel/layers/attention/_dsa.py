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

"""DeepSeek Sparse Attention (DSA): Multi-head Latent Attention restricted to a learned indexer's selection.

:class:`DeepSeekSparseAttention` is the single implementation of the DSA attention block. A model family subclasses it
to build its indexer (:meth:`DeepSeekSparseAttention._create_indexer`) and, where its reference differs, the MLA norm
epsilon (:meth:`~easydel.layers.attention.UnifiedAttention._mla_norm_eps`) or the score scale
(:meth:`~easydel.layers.attention.UnifiedAttention._mla_softmax_scale`):

- GLM-5-Next (``Glm5NextDSAAttention``): NoPE MLA with the learned k-pool indexer (pooled
  :class:`~easydel.layers.indexer.SparseIndexer`).
- GLM-MoE-DSA (``GlmMoeDsaAttention``): split-RoPE MLA with the per-token
  :class:`~easydel.layers.indexer.TokenIndexer`.

The indexer owns selection and its cache protocol; the layer routes by cache type:

- no cache: ``indexer.select_uncached`` (training, stateless prefill), or the ``dsa_pool_training`` path for pooled
  indexers;
- :class:`TransformerCacheView`: ``indexer.select_with_cache`` keeps indexer state position-aligned with the KV;
- :class:`MLARaggedPagesCacheView` with indexer pages: ``indexer.select_with_pages`` and the sparse MLA kernel;
- any other cache: dense attention, with a one-time warning.

A view wrapped in a :class:`ParallelHybridCacheView` is routed by its transformer view.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.loggings import get_logger
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float, Int
from spectrax import common_types

from easydel.caching import (
    MLARaggedPagesCacheView,
    OperationsMetadata,
    ParallelHybridCacheView,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.infra.modeling_outputs import SparseAttentionLayerOutput
from easydel.infra.sequence_packing import document_segment_ids, pairwise_attention_mask_from_mask_info
from easydel.layers.indexer import (
    IndexerSelection,
    ScoreRowSplit,
    compact_valid_layout,
    pool_bitmap_tail_error,
    use_sparse_training_attention,
)
from easydel.operations.kernels import DsaGatherAttn, DsaPoolSparseAttn
from easydel.operations.kernels.dsa_sparse_attention import sparse_selected_training_attention

from ..rotary import apply_mla_rope
from ._unified import UnifiedAttention

logger = get_logger(__name__)

__all__ = ("DeepSeekSparseAttention",)


def _with_rope(nope: Array, rope: Array) -> Array:
    """Concatenate the non-positional and rotary head components; the rotary part may be empty."""
    return nope if rope.shape[-1] == 0 else jnp.concatenate([nope, rope], axis=-1)


class DeepSeekSparseAttention(UnifiedAttention):
    """MLA attention whose queries attend only to the keys their indexer selects.

    Queries factor through ``q_lora_rank`` (when set), keys and values through the compressed ``kv_lora_rank``
    latent; each head splits into a ``qk_nope_head_dim`` part and a ``qk_rope_head_dim`` rotary part (possibly
    empty). The selection becomes a boolean mask ANDed into the pairwise attention mask on the dense paths, the
    ``sparse_selected`` operand of the MLA ragged-pages kernel when serving, or the input of the tile-skipping
    ``sparse_selected_training_attention`` / pool kernels in training.

    Layers with ``config.indexer_types[layer_idx] == "shared"`` keep an indexer (checkpoints carry its parameters)
    but reuse the preceding full layer's selection, passed as ``previous_topk_indices``.

    The indexer returned by :meth:`_create_indexer` provides:

    - ``select_uncached(hidden_states, q_resid, position_ids, frequencies, mask_info)`` ->
      :class:`~easydel.layers.indexer.IndexerSelection`;
    - ``select_with_cache(hidden_states, q_resid, position_ids, frequencies, cache_view, mask_info, cache_metadata, *,
      causal, mode, runtime_sharding_resolver)`` -> ``(selection, cache_view)``;
    - ``select_with_pages(hidden_states, q_resid, frequencies, cache_view, cache_metadata)`` -> ``(ids [tokens,
      width], cache_view)`` in units of ``selection_chunk_size`` tokens;
    - ``distillation_loss(...)`` for ``indexer_aux_loss_coef > 0``;
    - ``supports_pool_training`` and, when true, ``select_pools_chunked`` for ``dsa_pool_training``.

    The pooled :class:`~easydel.layers.indexer.SparseIndexer` and :class:`~easydel.layers.indexer.TokenIndexer`
    implement it.
    """

    def __init__(
        self,
        config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Build the MLA projections, the attention performer and the indexer.

        Args:
            config: Model configuration with the MLA dimensions, ``index_topk`` and ``indexer_types``.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
            layer_idx: Index of this layer in the decoder stack.
        """
        self.config = config
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.index_topk = config.index_topk
        # Resolved at init: ``layer_idx`` is scan-safe, so a scanned segment runs every layer through one template
        # whose ``layer_idx`` is the segment's first.
        self.uses_shared_indexer = config.indexer_types[layer_idx] == "shared"
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="mla",
            causal=True,
            use_mla_lora=bool(config.q_lora_rank),
        )
        self.head_dim = self.v_head_dim

    def define_network(self, config, dtype, param_dtype, precision, rngs):
        """Build the MLA projections, the attention performer and the indexer.

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.
        """
        self._build_mla_projections(config, dtype, param_dtype, precision, rngs)
        self.indexer = self._create_indexer(config, dtype, param_dtype, precision, rngs)

    def _create_indexer(self, config, dtype, param_dtype, precision, rngs) -> spx.Module:
        """Build this layer's indexer (see the class docstring for the interface it provides).

        Args:
            config: Model configuration.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: JAX matmul precision.
            rngs: Random number generator collection.

        Returns:
            The indexer module.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _create_indexer.")

    def _create_attention_performer(self, config, rngs):
        """The MLA performer, holding the RNG only when attention dropout draws from it.

        Holding the model-wide ``Rngs`` without dropout makes every layer share an RNG variable, which forces the layer
        scan onto its sequential-carry lowering.

        Args:
            config: Model configuration.
            rngs: Random number generator collection.

        Returns:
            The attention performer.
        """
        dropout = float(getattr(config, "attention_dropout", 0.0) or 0.0)
        return super()._create_attention_performer(config, rngs if dropout > 0.0 else None)

    def _mla_cache_rope_width(self) -> int:
        """Rotary width of the MLA ragged-pages cache: ``mla_cache_rope_width`` when the config sets it.

        NoPE models may request a zero-padded rotary width because the Pallas MLA kernel needs it 128-aligned; zero
        channels add nothing to the scores.

        Raises:
            ValueError: If the requested width is smaller than ``qk_rope_head_dim``.
        """
        width = getattr(self.config, "mla_cache_rope_width", None)
        if width is None:
            return self.qk_rope_head_dim
        if int(width) < self.qk_rope_head_dim:
            raise ValueError(
                f"mla_cache_rope_width ({width}) is smaller than qk_rope_head_dim ({self.qk_rope_head_dim})."
            )
        return int(width)

    @property
    def _attention_dtype(self):
        """Compute dtype shared by the dense, sparse-selected and pool paths: the dense performer's.

        That is ``mla_attn_dtype`` when set, else ``attn_dtype``, resolved when the performer was built, so every
        path of a layer computes in the same dtype.
        """
        return self.attention_performer.metadata.runtime_dtype

    def _split_kv_b(self, weight: Array) -> tuple[Array, Array]:
        """Split the ``kv_b_proj`` kernel ``[kv_lora_rank, heads * (qk_nope + v)]`` into ``W_k`` and ``W_v``.

        Args:
            weight: The kernel; ``heads`` is read from its width, so a head-sharded local kernel splits correctly.

        Returns:
            ``(w_k [kv_lora_rank, heads, qk_nope_head_dim], w_v [kv_lora_rank, heads, v_head_dim])``.
        """
        width = self.qk_nope_head_dim + self.v_head_dim
        weight = weight.reshape(self.kv_lora_rank, weight.shape[1] // width, width)
        return weight[:, :, : self.qk_nope_head_dim], weight[:, :, self.qk_nope_head_dim :]

    def _project_mla_inputs(self, hidden_states: Array) -> tuple[Array | None, Array, Array, Array, Array]:
        """Query components, the normed KV latent and the shared rotary key component.

        Args:
            hidden_states: Layer input ``[batch, seq, hidden]``.

        Returns:
            ``(q_resid, q_nope, q_pe, compressed_kv, k_pe)``: the normed low-rank query residual (``None`` without
            ``q_lora_rank``), ``q_nope [B, H, S, qk_nope]``, ``q_pe [B, H, S, qk_rope]``, the normed latent
            ``[B, S, kv_lora_rank]`` and ``k_pe [B, 1, S, qk_rope]``.
        """
        bsz, q_len, _ = hidden_states.shape
        q_resid = None
        if self.use_mla_lora:
            q_resid = self.mla_q_a_layernorm(checkpoint_name(self.mla_q_a_proj(hidden_states), name="attn_query_a"))
            q = checkpoint_name(self.mla_q_b_proj(q_resid), name="attn_query")
        else:
            q = checkpoint_name(self.mla_q_proj(hidden_states), name="attn_query")
        q = q.reshape(bsz, q_len, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        latent = self.mla_kv_a_proj_with_mqa(hidden_states)
        k_pe = latent[..., self.kv_lora_rank :].reshape(bsz, q_len, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        compressed_kv = self.mla_kv_a_layernorm(latent[..., : self.kv_lora_rank])
        return q_resid, q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim :], compressed_kv, k_pe

    def _shared_selection(self, previous_topk_indices: Array | None) -> IndexerSelection:
        """The preceding full layer's selection, for a shared-indexer layer.

        Raises:
            ValueError: If no selection was passed.
        """
        if previous_topk_indices is None:
            raise ValueError(
                f"{type(self).__name__} layer {self.layer_idx} reuses a shared indexer selection and needs the "
                "preceding full layer's previous_topk_indices."
            )
        return IndexerSelection(previous_topk_indices)

    @staticmethod
    def _has_paged_indexer(attention_view, cache_metadata) -> bool:
        """Whether a ragged-pages MLA view and its metadata carry paged indexer state for this step.

        Decided from the cache config, not this view's buffers: shared-indexer layers own no indexer pages but must
        still route the reused selection to the sparse kernel.
        """
        qsl = None if cache_metadata is None else getattr(cache_metadata, "query_start_loc", None)
        return (
            isinstance(attention_view, MLARaggedPagesCacheView)
            and bool(getattr(attention_view.metadata, "has_indexer_pages", False))
            and getattr(cache_metadata, "pages_tables", None) is not None
            and qsl is not None
            # 1-D (single replica) or rank-major [dp, rows + 1] (data-parallel serving).
            and qsl.ndim in (1, 2)
        )

    def _indexer_distillation_loss(
        self,
        hidden_states: Array,
        q_resid: Array | None,
        query_states: Array,
        key_states: Array,
        mask_info: MaskInfo,
        position_ids: Array,
        frequencies: Array | None,
    ) -> Array:
        """Unscaled indexer warm-up loss: this layer's dense attention distilled onto the indexer scores.

        Visibility is causal in physical token order (packed RoPE positions may restart) and restricted to valid
        tokens of the same document; documents of pairwise-only masks are recovered from the mask.

        Args:
            hidden_states: Layer input ``[batch, seq, hidden]``.
            q_resid: Normed low-rank query residual, or ``None``.
            query_states: Dense attention queries ``[batch, seq, heads, q_head_dim]``.
            key_states: Dense attention keys ``[batch, seq, heads, q_head_dim]``.
            mask_info: Attention mask metadata.
            position_ids: RoPE positions ``[batch, seq]``.
            frequencies: MLA RoPE table, or ``None``.

        Returns:
            The scalar loss.
        """
        bsz, q_len = hidden_states.shape[:2]
        positions = jnp.arange(q_len)
        visible = jnp.broadcast_to(positions[None, None, :] <= positions[None, :, None], (bsz, q_len, q_len))
        pairwise = pairwise_attention_mask_from_mask_info(mask_info, q_len, q_len)
        if pairwise is not None:
            visible = visible & pairwise
        query_segments, key_segments = document_segment_ids(mask_info, pairwise)
        visible = visible & (query_segments[:, :, None] == key_segments[:, None, :])
        visible = visible & (query_segments[:, :, None] >= 0) & (key_segments[:, None, :] >= 0)
        return self.indexer.distillation_loss(
            hidden_states,
            q_resid,
            query_states,
            key_states,
            visible,
            query_segments,
            key_segments,
            scaling=self.attention_performer.softmax_scale,
            sinks=self._softmax_aux(),
            position_ids=position_ids,
            frequencies=frequencies,
        )

    def _project_output(self, attn_output: Array) -> Array:
        """Output projection between the hidden-state sharding constraints.

        Args:
            attn_output: Merged heads ``[batch, seq, heads * v_head_dim]``.

        Returns:
            The projected output ``[batch, seq, hidden]``.
        """
        attn_output = self.shard_attention_prod(attn_output, pre_projection=True)
        attn_output = checkpoint_name(self.output_projection(attn_output), name="attn_output")
        return self.shard_attention_prod(attn_output)

    def _use_dsa_pool_training(
        self,
        *,
        cache_view,
        mask_info: MaskInfo,
        shared_indexer: bool,
        aux_active: bool,
        output_attentions: bool,
        return_topk_indices: bool,
        mode,
    ) -> bool:
        """Whether this forward takes the ``dsa_pool_training`` path (chunked pool selection + pool kernel).

        Returns ``False`` with a one-time warning naming the reasons when the flag is on but the layer or the call
        needs what only the dense path provides. The pool path handles any per-row padding; packed documents and custom
        pairwise masks look like padding as traced values, so they are recognised from the static
        ``MaskInfo.padding_only`` promise.
        """
        if not self.config.dsa_pool_training:
            return False
        if not self.indexer.supports_pool_training or self.qk_rope_head_dim > 0:
            logger.warning_once(
                "%s: dsa_pool_training needs a pooled indexer and attention without rotary components "
                "(this layer has a %s and qk_rope_head_dim=%s); its DSA layers keep the dense / sparse-selected path.",
                type(self).__name__,
                type(self.indexer).__name__,
                self.qk_rope_head_dim,
            )
            return False
        reasons = []
        if cache_view is not None:
            reasons.append("a cache is attached")
        if not mask_info.padding_only:
            reasons.append(
                "the mask carries packed documents or a custom pairwise mask (MaskInfo.padding_only is False; "
                "the pool path supports per-row padding only)"
            )
        if shared_indexer:
            reasons.append("the layer reuses a shared indexer selection")
        if aux_active:
            reasons.append("indexer_aux_loss_coef > 0")
        if output_attentions:
            reasons.append("output_attentions is requested")
        if return_topk_indices:
            reasons.append("a later shared indexer needs this layer's top-k indices")
        if self.sliding_window is not None:
            reasons.append("a sliding window is set")
        if self._softmax_aux() is not None:
            reasons.append("softmax sinks are set")
        deterministic = self.attention_performer.deterministic
        if deterministic is None:
            deterministic = mode != common_types.MODE_TRAIN or not self.attention_performer.training
        if not deterministic and self.attention_performer.dropout_prob > 0.0:
            reasons.append("attention dropout is active")
        if reasons:
            logger.warning_once(
                "%s: dsa_pool_training is enabled but layer %s uses the dense DSA path because %s.",
                type(self).__name__,
                self.layer_idx,
                "; ".join(reasons),
            )
            return False
        return True

    def _pool_training_output(self, attn_output: Array) -> SparseAttentionLayerOutput:
        """Wrap the pool path's merged heads ``[batch, seq, heads * v_head_dim]`` after the output projection."""
        return SparseAttentionLayerOutput(
            attention_output=self._project_output(attn_output),
            attention_weight=None,
            cache_view=None,
            indexer_aux_loss=None,
        )

    def _forward_dsa_pool_training(
        self,
        hidden_states: Array,
        q_resid: Array | None,
        query_states: Array,
        compressed_kv: Array,
        q_mask: Array | None,
    ) -> SparseAttentionLayerOutput:
        """No-cache DSA forward without ``O(seq^2)`` tensors.

        The indexer selects pools query-chunk by query-chunk (``select_pools_chunked``); attention runs in the
        weight-absorbed MLA form (``q_nope @ W_k`` against the shared ``kv_lora_rank`` latent, ``K == V``) through the
        pool-bitmap ``dsa_pool_sparse_attention`` operation, and the latent output is mapped back with ``W_v``. Pools
        group consecutive valid tokens as in the dense path: the latent is read in each row's valid-first order
        (:func:`~easydel.layers.indexer.compact_valid_layout`), the order the selection indexes, while queries stay in
        place. A selected key's compact position never exceeds its query's compact rank, which never exceeds the
        query's physical slot, so the kernel's physical causal cut keeps every selected key. Numerically the same
        attention as the dense top-k-mask path for unpacked rows with any per-row padding.
        ``dsa_pool_kernel="gather"`` delegates to :meth:`_forward_dsa_pool_gather`.

        Args:
            hidden_states: Layer input ``[batch, seq, hidden]``.
            q_resid: Normed low-rank query residual ``[batch, seq, q_lora_rank]``.
            query_states: Per-head NoPE queries ``[batch, seq, heads, qk_nope_head_dim]``.
            compressed_kv: Normed KV latent ``[batch, seq, kv_lora_rank]``.
            q_mask: Token validity ``[batch, seq]`` or ``None``.

        Returns:
            :class:`SparseAttentionLayerOutput` with the projected attention output.

        Raises:
            ValueError: If the bitmap kernel cannot take the indexer's tail keys
                (:func:`~easydel.layers.indexer.pool_bitmap_tail_error`).
        """
        if self.config.dsa_pool_kernel == "gather":
            return self._forward_dsa_pool_gather(hidden_states, q_resid, query_states, compressed_kv, q_mask)
        kpool = self.indexer.config.kpool_size
        tail_error = pool_bitmap_tail_error(kpool, self.indexer.config.select_tail)
        if tail_error is not None:
            raise ValueError(tail_error)
        bsz, q_len, _ = hidden_states.shape
        layout = None if q_mask is None else compact_valid_layout(q_mask)
        pool_mask, tails = self.indexer.select_pools_chunked(
            hidden_states,
            q_resid,
            q_mask,
            query_chunk_size=int(self.config.dsa_pool_query_chunk),
            packed=True,
            layout=layout,
        )
        w_k, w_v = self._split_kv_b(self.mla_kv_b_proj.weight.value)
        compute_dtype = self._attention_dtype
        q_absorbed = jnp.einsum(
            "bsnd,knd->bnsk",
            query_states.astype(compute_dtype),
            w_k.astype(compute_dtype),
            preferred_element_type=jnp.float32,
        ).astype(compute_dtype)
        latent = compressed_kv.astype(compute_dtype)
        if layout is not None:
            latent = layout.gather(latent)
        out_latent = DsaPoolSparseAttn(self.attention_performer.metadata)(
            q_absorbed,
            latent,
            pool_mask,
            tails,
            None,
            kpool=kpool,
            softmax_scale=self.attention_performer.softmax_scale,
        ).attention_outputs
        attn = jnp.einsum(
            "bnsk,knd->bsnd",
            out_latent.astype(compute_dtype),
            w_v.astype(compute_dtype),
            preferred_element_type=jnp.float32,
        )
        return self._pool_training_output(attn.astype(hidden_states.dtype).reshape(bsz, q_len, -1))

    def _forward_dsa_pool_gather(
        self,
        hidden_states: Array,
        q_resid: Array | None,
        query_states: Array,
        compressed_kv: Array,
        q_mask: Array | None,
    ) -> SparseAttentionLayerOutput:
        """``dsa_pool_kernel="gather"`` variant of :meth:`_forward_dsa_pool_training`.

        The indexer returns the selected pool ids directly (no mask) and attention runs through the per-query
        ``dsa_gather_attention`` operation: every query DMA-gathers exactly its ``index_topk // kpool`` pools plus its
        own incomplete pool, so the cost does not depend on how spread the selection is. On a TPU mesh the call splits
        the query sequence over the tensor-parallel axes (all heads local); the split is applied to the
        ``qk_nope_head_dim``-wide queries before the weight absorption (narrower than the absorbed
        ``kv_lora_rank``-wide queries) and undone on the ``v_head_dim``-wide output before the output projection. As in
        the bitmap variant the latent is read in each row's valid-first order and every query's compact rank is its
        position.

        Args:
            hidden_states: Layer input ``[batch, seq, hidden]``.
            q_resid: Normed low-rank query residual ``[batch, seq, q_lora_rank]``.
            query_states: Per-head NoPE queries ``[batch, seq, heads, qk_nope_head_dim]``.
            compressed_kv: Normed KV latent ``[batch, seq, kv_lora_rank]``.
            q_mask: Token validity ``[batch, seq]`` or ``None``.

        Returns:
            :class:`SparseAttentionLayerOutput` with the projected attention output.
        """
        bsz, q_len, _ = hidden_states.shape
        weight = self.mla_kv_b_proj.weight.value
        heads = weight.shape[1] // (self.qk_nope_head_dim + self.v_head_dim)
        op = DsaGatherAttn(self.attention_performer.metadata)
        layout = op.layout(batch=bsz, seq=q_len, num_heads=heads)
        split = layout is not None and bool(layout.seq)
        # Split the indexer's per-chunk score rows like the attention's queries; the activation layout replicates
        # them over the tensor axes, so every tensor-parallel device would repeat the whole selection.
        row_split = ScoreRowSplit(mesh=layout.mesh, rows=layout.query_row_entries) if split else None
        key_layout = None if q_mask is None else compact_valid_layout(q_mask)
        kpool = self.indexer.config.kpool_size
        pool_ids, _ = self.indexer.select_pools_chunked(
            hidden_states,
            q_resid,
            q_mask,
            query_chunk_size=int(self.config.dsa_pool_query_chunk),
            as_indices=True,
            row_split=row_split,
            layout=key_layout,
        )
        if key_layout is None:
            positions = jnp.broadcast_to(jnp.arange(q_len, dtype=jnp.int32)[None, :], (bsz, q_len))
        else:
            positions = key_layout.rank  # compact rank of each valid query, -1 for padding
        if split:
            # The kv_b kernel is small, so every shard holds it whole.
            query_states = layout.constrain(query_states, layout.query_spec)
            weight = layout.constrain(weight, PartitionSpec(None, None))
        w_k, w_v = self._split_kv_b(weight)
        compute_dtype = self._attention_dtype
        q_absorbed = jnp.einsum(
            "bsnd,knd->bsnk",
            query_states.astype(compute_dtype),
            w_k.astype(compute_dtype),
            preferred_element_type=jnp.float32,
        ).astype(compute_dtype)
        if split:
            q_absorbed = layout.constrain(q_absorbed, layout.query_spec)
        latent = compressed_kv.astype(compute_dtype)
        if key_layout is not None:
            latent = key_layout.gather(latent)
        out_latent = op(
            q_absorbed,
            latent,
            pool_ids,
            positions,
            kpool=kpool,
            softmax_scale=self.attention_performer.softmax_scale,
            include_tail=bool(self.indexer.config.select_tail) and kpool > 1,
        ).attention_outputs
        attn = jnp.einsum(
            "bsnk,knd->bsnd",
            out_latent.astype(compute_dtype),
            w_v.astype(compute_dtype),
            preferred_element_type=jnp.float32,
        ).astype(hidden_states.dtype)
        if split:
            attn = layout.constrain(attn, layout.head_parallel_spec)
        return self._pool_training_output(attn.reshape(bsz, q_len, -1))

    def forward_mla(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | ParallelHybridCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len rope_dim"] | None = None,
        alibi: Array | None = None,
        previous_topk_indices: Array | None = None,
        return_topk_indices: bool = False,
    ) -> SparseAttentionLayerOutput | tuple[SparseAttentionLayerOutput, Array | None]:
        """Run MLA attention restricted to the indexer's selection.

        Args:
            hidden_states: Layer input ``[batch, seq_len, hidden_size]``.
            mask_info: Attention mask metadata.
            position_ids: RoPE positions ``[batch, seq_len]``.
            mode: Runtime mode.
            cache_view: KV cache view, optionally wrapped in a :class:`ParallelHybridCacheView`.
            cache_metadata: Cache metadata.
            output_attentions: Return the attention weights (dense paths only).
            frequencies: MLA RoPE table; unused when ``qk_rope_head_dim == 0``.
            alibi: Unused.
            previous_topk_indices: The preceding full layer's selection, required by shared-indexer layers.
            return_topk_indices: Also return this layer's selection (``None`` when the layer attends densely).

        Returns:
            :class:`SparseAttentionLayerOutput`, or ``(output, topk_indices)`` with ``return_topk_indices``.

        Raises:
            ValueError: If a shared-indexer layer gets no ``previous_topk_indices``.
        """
        del alibi
        bsz, q_len, _ = hidden_states.shape
        hybrid = isinstance(cache_view, ParallelHybridCacheView)
        attention_view = cache_view.transformer if hybrid else cache_view
        ragged = isinstance(attention_view, MLARaggedPagesCacheView)

        q_resid, q_nope, q_pe, compressed_kv, k_pe = self._project_mla_inputs(hidden_states)
        table = None if frequencies is None else getattr(frequencies, "value", frequencies)
        if table is not None and self.qk_rope_head_dim > 0:
            q_pe, k_pe = apply_mla_rope(
                q_pe, k_pe, table, position_ids, interleave=bool(getattr(self.config, "rope_interleave", False))
            )
        q_nope, q_pe = self._postprocess_mla_queries(q_nope, q_pe, position_ids)

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info, input_ids=None, inputs_embeds=hidden_states, attention_mask=None
        )
        causal_for_kernel = self.causal and not getattr(mask_info, "_causal_baked", False)
        sliding_window_for_kernel = None if mask_info.sliding_window_baked_in else self.sliding_window

        shared = self.uses_shared_indexer
        aux_coef = float(getattr(self.config, "indexer_aux_loss_coef", 0.0) or 0.0)
        aux_active = cache_view is None and not shared and aux_coef > 0.0
        if self._use_dsa_pool_training(
            cache_view=cache_view,
            mask_info=mask_info,
            shared_indexer=shared,
            aux_active=aux_active,
            output_attentions=output_attentions,
            return_topk_indices=return_topk_indices,
            mode=mode,
        ):
            return self._forward_dsa_pool_training(
                hidden_states,
                q_resid,
                q_nope.transpose(0, 2, 1, 3),
                compressed_kv,
                mask_info.q_attention_mask[:, :q_len].astype(jnp.bool_),
            )

        if attention_view is None:
            source = "uncached"
        elif isinstance(attention_view, TransformerCacheView):
            source = "cached"
        elif self._has_paged_indexer(attention_view, cache_metadata):
            source = "paged"
        else:
            source = None
            logger.warning_once(
                "%s: the %s cache (or its metadata) carries no indexer state, so DSA layers attend densely; this "
                "diverges from the reference once the context exceeds index_topk=%s tokens.",
                type(self).__name__,
                type(attention_view).__name__,
                self.index_topk,
            )
        selection = None
        if source is not None and shared:
            selection = self._shared_selection(previous_topk_indices)
        elif source == "uncached":
            selection = self.indexer.select_uncached(
                jax.lax.stop_gradient(hidden_states) if aux_active else hidden_states,
                jax.lax.stop_gradient(q_resid) if aux_active and q_resid is not None else q_resid,
                position_ids,
                table,
                mask_info,
            )
        elif source == "cached":
            selection, attention_view = self.indexer.select_with_cache(
                hidden_states,
                q_resid,
                position_ids,
                table,
                attention_view,
                mask_info,
                cache_metadata,
                causal=self.causal,
                mode=mode,
                runtime_sharding_resolver=self.config.runtime_sharding_resolver,
            )
        elif source == "paged":
            selected, attention_view = self.indexer.select_with_pages(
                hidden_states, q_resid, table, attention_view, cache_metadata
            )
            selection = IndexerSelection(selected)
        if source in ("cached", "paged") and not shared:
            cache_view = cache_view.replace(transformer=attention_view) if hybrid else attention_view

        indexer_aux_loss = None
        mla_kwargs: dict[str, tp.Any] = {}
        w_v = None
        if ragged:
            # The ragged-pages MLA kernel takes the weight-absorbed form (``q_nope @ W_k`` against the latent); the
            # cache stores the latent plus the rotary key component, padded to the cache's rotary width.
            w_k, w_v = self._split_kv_b(self.mla_kv_b_proj.weight.value)
            query_states = jnp.einsum("bhsd,khd->bshk", q_nope.astype(jnp.float32), w_k.astype(jnp.float32)).astype(
                q_nope.dtype
            )
            pad = self._mla_cache_rope_width() - self.qk_rope_head_dim
            queries_pe = q_pe.transpose(0, 2, 1, 3)
            keys_pe = k_pe[:, 0]
            if pad:
                queries_pe = jnp.pad(queries_pe, ((0, 0), (0, 0), (0, 0), (0, pad)))
                keys_pe = jnp.pad(keys_pe, ((0, 0), (0, 0), (0, pad)))
            key_states = value_states = jnp.concatenate([compressed_kv[:, :, None, :], keys_pe[:, :, None, :]], -1)
            mla_kwargs = {
                "queries_nope": query_states,
                "queries_pe": queries_pe,
                "keys_values": compressed_kv,
                "keys_pe": keys_pe,
                "softmax_scale": self.attention_performer.softmax_scale,
            }
        else:
            kv = (
                self.mla_kv_b_proj(compressed_kv)
                .reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(0, 2, 1, 3)
            )
            k_nope = kv[..., : self.qk_nope_head_dim]
            rope_keys = jnp.broadcast_to(k_pe, (*k_nope.shape[:3], self.qk_rope_head_dim))
            query_states = _with_rope(q_nope, q_pe).transpose(0, 2, 1, 3)
            key_states = _with_rope(k_nope, rope_keys).transpose(0, 2, 1, 3)
            value_states = kv[..., self.qk_nope_head_dim :].transpose(0, 2, 1, 3)
            if aux_active:
                indexer_aux_loss = self._indexer_distillation_loss(
                    hidden_states, q_resid, query_states, key_states, mask_info, position_ids, table
                )

        (
            key_states,
            value_states,
            mask_info,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
            causal=self.causal,
        )

        combined_mask = None
        if selection is not None and ragged:
            mla_kwargs["sparse_selected"] = selection.indices.reshape(bsz * q_len, -1)
            mla_kwargs["sparse_chunk_size"] = self.indexer.selection_chunk_size
        elif selection is not None:
            kv_len = key_states.shape[1]
            topk_mask = selection.to_mask(kv_len)
            attention_mask = pairwise_attention_mask_from_mask_info(mask_info, q_len, kv_len)
            combined_mask = topk_mask if attention_mask is None else attention_mask & topk_mask
            # The selection already applies absolute-position causality, also when reused by a shared layer. An
            # absolute history mask need not carry the cache's baked-causal flag; a second, chunk-relative triangle
            # would hide live historical keys in continuation blocks.
            mask_info = mask_info.replace(
                attention_mask=combined_mask[:, None, :, :],
                causal_mask_baked_in=mask_info.causal_mask_baked_in or self.causal,
            )
            causal_for_kernel = False

        # The sparse training operation has no dropout/RNG contract; match the dispatcher's training state.
        deterministic = self.attention_performer.deterministic
        if deterministic is None:
            deterministic = mode != common_types.MODE_TRAIN or not self.attention_performer.training
        attention_dropout_active = not deterministic and self.attention_performer.dropout_prob > 0.0
        topk_indices = None if selection is None else selection.indices
        if (
            combined_mask is not None
            and not attention_dropout_active
            and use_sparse_training_attention(
                topk_indices=topk_indices,
                cache_view=cache_view,
                mask_info=mask_info,
                output_attentions=output_attentions,
                sliding_window=sliding_window_for_kernel,
                softmax_aux=self.get_softmax_aux(),
                sparse_training=self.config.dsa_sparse_training,
                seq_len=q_len,
            )
        ):
            # Same per-head q/k/v and selection mask as the dense path, through a tile-skipping flash kernel instead
            # of materialized [B, H, S, S] masked scores.
            attn_heads = sparse_selected_training_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                selection_mask=combined_mask,
                mask_info=None,
                softmax_scale=self.attention_performer.softmax_scale,
                causal=bool(causal_for_kernel),
                dtype=self._attention_dtype,
                metadata=self.attention_performer.metadata,
            )
            result = SparseAttentionLayerOutput(
                attention_output=self._project_output(attn_heads.astype(hidden_states.dtype).reshape(bsz, q_len, -1)),
                attention_weight=None,
                cache_view=cache_view,
                indexer_aux_loss=indexer_aux_loss,
            )
            return (result, topk_indices) if return_topk_indices else result

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=causal_for_kernel,
            sliding_window=sliding_window_for_kernel,
            softmax_aux=self._softmax_aux(),
            output_attentions=output_attentions,
            **mla_kwargs,
        )
        if attentions.cache_view is not None:
            cache_view = attentions.cache_view
        if ragged:
            # Native MLA kernels return [tokens, H, kv_lora_rank], the paged adapter [B, S, H, kv_lora_rank].
            latent_out = attentions.attention_outputs.reshape(bsz * q_len, w_v.shape[1], self.kv_lora_rank)
            attn_output = jnp.einsum("thk,khv->thv", latent_out.astype(jnp.float32), w_v.astype(jnp.float32)).astype(
                latent_out.dtype
            )
            attn_output = attn_output.reshape(bsz, q_len, -1)
        else:
            attn_output = self._merge_heads(attentions.attention_outputs)

        result = SparseAttentionLayerOutput(
            attention_output=self._project_output(attn_output),
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
            indexer_aux_loss=indexer_aux_loss,
        )
        return (result, topk_indices) if return_topk_indices else result
