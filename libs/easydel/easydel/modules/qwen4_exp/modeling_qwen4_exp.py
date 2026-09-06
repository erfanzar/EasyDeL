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

"""Qwen4-Exp (Qwen3.8-Flash-Next) — hybrid GatedDeltaNet/QSA MoE with hyper-connections.

JAX port of the reference (``transformers.models.qwen4_exp``), composed from
shared EasyDeL layers rather than model-local reimplementations:

- GatedDeltaNet linear layers: :class:`Qwen3NextLinearAttention` in
  split-proj mode (the checkpoint stores ``in_proj_qkv`` / ``in_proj_z`` /
  ``in_proj_b`` / ``in_proj_a``), with the output-gate norm switched to
  ``output_gate_type`` (sigmoid here).
- Sparse full attention: :class:`Qwen3NextFullAttention` (fused q+gate
  projection, per-head QK norm, sigmoid output gate, partial interleaved
  mRoPE) plus the shared :class:`BlockTopKIndexer` for QSA token selection.
- MoE: fused-expert SwiGLU stack + softmax top-k router + sigmoid-gated
  shared expert.
- Hyper-connections: :class:`GatedResidual` (``layers/residual``). The
  model-level ``hyper_connection_mixer`` doubles as the final norm/collapse —
  the checkpoint carries no separate final norm.
- PLE: hashed n-gram embeddings (:class:`NGramEmbed`) injected on
  ``ple_layer_ids`` (1-indexed, linear-attention layers only).
- MTP: auxiliary one-layer hybrid head (``mtp.*``). HF ships these weights
  but ignores them; the head is reconstructed from the checkpoint shapes
  after the Qwen3.5 MTP pattern, for speculative-decoding use.

Cache: GDN layers ride the standard hybrid conv/recurrent state; QSA layers
additionally cache raw (unnormed, unroped) indexer keys, the mRoPE position
history and the visibility history; PLE layers carry a dilated conv state and
the trailing n-gram token context (:class:`Qwen4ExpQSAView` / :class:`Qwen4ExpLinearView`).
"""

from __future__ import annotations

import typing as tp
from functools import cached_property, partial

import jax
import numpy as np
import spectrax as spx
from eformer.jaximus import ImplicitArray
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import (
    HybridCache,
    HybridCacheConfig,
    HybridCacheView,
    OperationsMetadata,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    RecurrentCacheView,
    TransformerCacheConfig,
    TransformerCacheView,
    TransformerMetadata,
    UnifiedAttentionCacheView,
)
from easydel.caching.ragged_page.cache import kv_pair_shares_head_dim_axis
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    DecoderLayerOutput,
    ModelOutput,
)
from easydel.infra.sharding import resolve_stage_mesh
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import (
    BaseMoeModule,
    BlockTopKIndexer,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    FusedExpertLayout,
    GatedResidual,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RMSNorm,
    RMSNormGated,
    RowParallelLinear,
    RowParallelMoELinear,
    expand_streams,
    inject_streams,
    moe_down_projection_reform_param,
    split_fused_gate_up_projection,
)
from easydel.layers.embeddings import NGramEmbed
from easydel.layers.norms import lowfloats
from easydel.layers.sparse_attention import apply_partial_rope
from easydel.modules._base import BaseCausalLMModule, BaseVisionLanguageModule
from easydel.modules.qwen3_5.modeling_qwen3_5 import _get_rope_index_from_mm_token_types
from easydel.modules.qwen3_next.modeling_qwen3_next import (
    Qwen3NextFullAttention,
    Qwen3NextLinearAttention,
)
from easydel.modules.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VisionTransformerPretrainedModel,
    merge_multimodal_embeddings,
)
from easydel.operations import OperationMetadata
from easydel.operations.kernels import GatedDeltaRuleOp, RaggedGatedDeltaRule

from .qwen4_exp_configuration import Qwen4ExpConfig, Qwen4ExpTextConfig, Qwen4ExpVisionConfig

logger = get_logger(__name__)

QWEN4_FULL = "qwen_sparse_attention"
QWEN4_LINEAR = "linear_attention"


def _dynamic_update_rows(buffer: Array, values: Array, starts: Array, axis: int = 0) -> Array:
    """Update each batch row at its own dynamic offset.

    ``axis`` is relative to a single row (the leading batch axis is removed by
    ``vmap``). Keeping this operation shared prevents continuous-batching cache
    writes from accidentally reusing row zero's offset.
    """
    if buffer.shape[0] != values.shape[0]:
        raise ValueError(f"buffer/value batch mismatch: {buffer.shape[0]} != {values.shape[0]}")

    def update(row, value, start):
        indices = [0] * row.ndim
        indices[axis] = start
        return jax.lax.dynamic_update_slice(row, value, tuple(indices))

    return jax.vmap(update)(buffer, values, starts.astype(jnp.int32))


def _packed_depthwise_causal_conv(
    hidden_states: Array,
    kernel: Array,
    segment_ids: Array,
    dilation: int,
) -> Array:
    """Depthwise causal convolution that never crosses packed documents."""
    batch, seq_len, channels = hidden_states.shape
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    out = jnp.zeros((batch, seq_len, channels), jnp.float32)
    width = kernel.shape[0]
    for kernel_idx in range(width):
        lag = dilation * (width - 1 - kernel_idx)
        source = positions - lag
        gather = jnp.clip(source, 0, seq_len - 1)[None, :, None]
        shifted = jnp.take_along_axis(hidden_states, jnp.broadcast_to(gather, hidden_states.shape), axis=1)
        source_segments = jnp.take_along_axis(segment_ids, jnp.broadcast_to(gather[..., 0], segment_ids.shape), axis=1)
        valid = (source[None, :] >= 0) & (segment_ids >= 0) & (source_segments == segment_ids)
        out = out + jnp.where(valid[..., None], shifted.astype(jnp.float32), 0.0) * kernel[kernel_idx, 0]
    return out


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class Qwen4ExpRMSNorm(RMSNorm):
    """Qwen4-Exp RMSNorm: zero-centred ``(1 + w)`` scale, fp32 reduction, optional grouping.

    Every norm in this family (hyper-connection norms, per-head Q/K norms,
    indexer layernorms, PLE norms, MTP fuser norms) is the reference
    ``Qwen4ExpTextRMSNorm``: the mean-square is computed in float32 (per
    ``group_size`` group when set) and the stored weight is zero-centred so
    init is the identity.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.bfloat16,
        group_size: int | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the norm; see :class:`easydel.layers.norms.RMSNorm`."""
        super().__init__(
            dim,
            eps=eps,
            dtype=dtype,
            param_dtype=param_dtype,
            scale_offset=1.0,
            group_size=group_size,
            kernel_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )


# ---------------------------------------------------------------------------
# Cache: QSA layers ride TransformerCacheView (+indexer state), linear layers
# ride HybridCacheView (+PLE state), both under one heterogeneous container.
# ---------------------------------------------------------------------------


@auto_pytree
class Qwen4ExpQSAView(TransformerCacheView):
    """KV cache view for QSA layers, extended with indexer state.

    Extra attributes:
        indexer_key: Raw (unnormed, unroped) indexer keys ``[B, S, idx_dim]``;
            compression happens at selection time, so the cache holds the
            per-token keys verbatim.
        indexer_visible: Padding-visibility history ``[B, S]``.
        mrope_positions: mRoPE position history ``[B, 3, S]`` (block keys are
            roped at block-start positions during selection, which needs the
            full history for vision-containing prefixes).
    """

    indexer_key: Float[Array, "batch seq idx_dim"] | None = None
    indexer_visible: Bool[Array, "batch seq"] | None = None
    mrope_positions: Int[Array, "batch 3 seq"] | None = None
    indexer_block_keys: Float[Array, "batch blocks idx_dim"] | None = None
    indexer_blocks_complete: Bool[Array, "batch blocks"] | None = None

    @classmethod
    def init(
        cls,
        config: TransformerCacheConfig,
        indexer_head_dim: int,
        layer_index: int | None = None,
        *,
        indexer_max_blocks: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        **view_kwargs,
    ) -> "Qwen4ExpQSAView":
        """Allocate the KV cache plus the indexer buffers."""
        import dataclasses as _dc

        base = TransformerCacheView.init(config, layer_index=layer_index, dtype=dtype, **view_kwargs)
        fields = {f.name: getattr(base, f.name) for f in _dc.fields(TransformerCacheView)}
        batch, seq = config.batch_size, config.sequence_length
        max_blocks = max(indexer_max_blocks, 1)
        return cls(
            **fields,
            indexer_key=jnp.zeros((batch, seq, indexer_head_dim), dtype=dtype),
            indexer_visible=jnp.zeros((batch, seq), jnp.bool_),
            mrope_positions=jnp.zeros((batch, 3, seq), jnp.int32),
            indexer_block_keys=jnp.zeros((batch, max_blocks, indexer_head_dim), dtype=dtype),
            indexer_blocks_complete=jnp.zeros((batch, max_blocks), jnp.bool_),
        )


@auto_pytree
class Qwen4ExpPagedQSAView(RaggedPagesCacheView):
    """Paged KV view with physical-page-aligned raw QSA indexer history."""

    indexer_key_pages: Float[Array, "num_pages page_size index_dim"] | None = None
    mrope_position_pages: Int[Array, "num_pages page_size 3"] | None = None

    @classmethod
    def init(
        cls,
        config,
        layer_index: int | None = None,
        *,
        mesh=None,
        runtime_sharding_resolver=None,
        quantizer=None,
    ) -> "Qwen4ExpPagedQSAView":
        import dataclasses as _dc

        base = RaggedPagesCacheView.init(
            config,
            layer_index=layer_index,
            mesh=mesh,
            runtime_sharding_resolver=runtime_sharding_resolver,
            quantizer=quantizer,
        )
        fields = {f.name: getattr(base, f.name) for f in _dc.fields(RaggedPagesCacheView)}
        index_dim = int(getattr(config, "qwen4_indexer_head_dim", 0))
        if index_dim <= 0:
            raise ValueError("Qwen4 paged QSA cache config is missing qwen4_indexer_head_dim")
        base_sharding = getattr(base.kv_pages, "sharding", None)
        sidecar_sharding = None
        if isinstance(base_sharding, NamedSharding):
            page_axis = base_sharding.spec[0] if len(base_sharding.spec) else None
            sidecar_sharding = NamedSharding(base_sharding.mesh, PartitionSpec(page_axis, None, None))
        return cls(
            **fields,
            indexer_key_pages=jnp.zeros(
                (config.num_pages, config.page_size, index_dim),
                dtype=jnp.bfloat16,
                device=sidecar_sharding,
            ),
            mrope_position_pages=jnp.zeros(
                (config.num_pages, config.page_size, 3),
                dtype=jnp.int32,
                device=sidecar_sharding,
            ),
        )

    def reset(self) -> "Qwen4ExpPagedQSAView":
        return self.replace(
            kv_pages=jnp.zeros_like(self.kv_pages),
            indexer_key_pages=jnp.zeros_like(self.indexer_key_pages),
            mrope_position_pages=jnp.zeros_like(self.mrope_position_pages),
        )


def _paged_qsa_token_map(cache_metadata: RaggedPagesMetadata, total_tokens: int, num_pages: int):
    """Map packed current tokens to request rows and physical page slots."""
    qsl = cache_metadata.query_start_loc.astype(jnp.int32).reshape(-1)
    num_rows = qsl.shape[0] - 1
    token_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    row = jnp.clip(jnp.searchsorted(qsl, token_idx, side="right") - 1, 0, num_rows - 1)
    live_rows = jnp.clip(cache_metadata.num_seqs.reshape(-1)[0].astype(jnp.int32), 0, num_rows)
    valid = token_idx < jnp.take(qsl, live_rows)
    scheduled = qsl[1:] - qsl[:-1]
    old_len = cache_metadata.context_lens.astype(jnp.int32)[:num_rows] - scheduled
    logical = jnp.take(old_len, row) + token_idx - jnp.take(qsl, row)
    page_size = int(cache_metadata.page_size)
    logical_page = logical // page_size
    page_offset = logical % page_size
    tables = cache_metadata.pages_tables.reshape(num_rows, -1)
    physical_page = tables[row, jnp.clip(logical_page, 0, tables.shape[1] - 1)]
    physical_page = jnp.where(valid, physical_page, num_pages)
    return row, logical, physical_page, page_offset, valid


@auto_pytree
class Qwen4ExpLinearView(HybridCacheView):
    """Linear-attention (GDN) view, extended with PLE state on PLE layers.

    Extra attributes:
        ple_conv_state: Dilated PLE conv input window ``[B, C, state_len]``.
        ple_token_context: Last ``ngram_size - 1`` token ids ``[B, ctx]``.
    """

    ple_conv_state: Float[Array, "batch channels state_len"] | None = None
    ple_token_context: Int[Array, "batch ctx"] | None = None
    ple_segment_context: Int[Array, "batch ctx"] | None = None

    @classmethod
    def init(
        cls,
        config: HybridCacheConfig,
        layer_index: int | None = None,
        *,
        with_ple: bool = False,
        ple_conv_dim: int = 0,
        ple_conv_state_len: int = 0,
        ple_context_len: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_specs=None,
    ) -> "Qwen4ExpLinearView":
        """Allocate the GDN conv/recurrent state plus PLE state when asked."""
        import dataclasses as _dc

        with_ple = with_ple or bool(getattr(config, "qwen4_with_ple", False))
        ple_conv_dim = ple_conv_dim or int(getattr(config, "qwen4_ple_conv_dim", 0))
        ple_conv_state_len = ple_conv_state_len or int(getattr(config, "qwen4_ple_conv_state_len", 0))
        ple_context_len = ple_context_len or int(getattr(config, "qwen4_ple_context_len", 0))
        view = HybridCacheView.init(config, layer_index, dtype=dtype, partition_specs=partition_specs)
        fields = {f.name: getattr(view, f.name) for f in _dc.fields(HybridCacheView)}
        return cls(
            **fields,
            ple_conv_state=(
                jnp.zeros((config.batch_size, ple_conv_dim, ple_conv_state_len), dtype=jnp.float32) if with_ple else None
            ),
            ple_token_context=(jnp.zeros((config.batch_size, ple_context_len), jnp.int32) if with_ple else None),
            ple_segment_context=(jnp.full((config.batch_size, ple_context_len), -1, jnp.int32) if with_ple else None),
        )


@auto_pytree
class Qwen4ExpOperationsLinearView(RecurrentCacheView):
    """eSurge recurrent view carrying Qwen4 PLE continuation state."""

    ple_conv_state: Float[Array, "batch channels state_len"] | None = None
    ple_token_context: Int[Array, "batch ctx"] | None = None
    ple_segment_context: Int[Array, "batch ctx"] | None = None

    @classmethod
    def init(cls, config, layer_index=None, *, dtype=jnp.bfloat16, partition_specs=None):
        import dataclasses as _dc

        view = RecurrentCacheView.init(config, layer_index, dtype=dtype, partition_specs=partition_specs)
        fields = {f.name: getattr(view, f.name) for f in _dc.fields(RecurrentCacheView)}
        with_ple = bool(getattr(config, "qwen4_with_ple", False))
        conv_dim = int(getattr(config, "qwen4_ple_conv_dim", 0))
        state_len = int(getattr(config, "qwen4_ple_conv_state_len", 0))
        context_len = int(getattr(config, "qwen4_ple_context_len", 0))
        return cls(
            **fields,
            ple_conv_state=(jnp.zeros((config.batch_size, conv_dim, state_len), jnp.float32) if with_ple else None),
            ple_token_context=(jnp.zeros((config.batch_size, context_len), jnp.int32) if with_ple else None),
            ple_segment_context=(jnp.full((config.batch_size, context_len), -1, jnp.int32) if with_ple else None),
        )


class Qwen4ExpCache(HybridCache):
    """Heterogeneous per-layer cache: QSA views + linear (GDN/PLE) views."""

    @classmethod
    def init_cache(
        cls,
        *,
        transformer_config: TransformerCacheConfig,
        hybrid_config: HybridCacheConfig,
        qsa_layers: tuple[int, ...],
        indexer_head_dim: int,
        indexer_max_blocks: int = 0,
        ple_layers: tuple[int, ...],
        ple_conv_dim: int,
        ple_conv_state_len: int,
        ple_context_len: int,
        dtype: jnp.dtype = jnp.bfloat16,
        recurrent_dtype: jnp.dtype | None = None,
        **view_kwargs,
    ) -> "Qwen4ExpCache":
        """Allocate one view per decoder layer by layer kind."""
        views = []
        for i in range(transformer_config.num_hidden_layers):
            if i in qsa_layers:
                views.append(
                    Qwen4ExpQSAView.init(
                        transformer_config,
                        indexer_head_dim,
                        layer_index=i,
                        indexer_max_blocks=indexer_max_blocks,
                        dtype=dtype,
                        **view_kwargs,
                    )
                )
            else:
                views.append(
                    Qwen4ExpLinearView.init(
                        hybrid_config,
                        layer_index=i,
                        with_ple=i in ple_layers,
                        ple_conv_dim=ple_conv_dim,
                        ple_conv_state_len=ple_conv_state_len,
                        ple_context_len=ple_context_len,
                        dtype=recurrent_dtype or dtype,
                    )
                )
        return cls(views=views)


# ---------------------------------------------------------------------------
# GatedDeltaNet (linear_attention layers)
# ---------------------------------------------------------------------------


class Qwen4ExpGatedDeltaNet(Qwen3NextLinearAttention):
    """Qwen4-Exp GatedDeltaNet: split projections + configurable output-gate activation.

    The released checkpoint stores separate ``in_proj_qkv`` / ``in_proj_z`` /
    ``in_proj_b`` / ``in_proj_a`` (plain contiguous ``[q | k | v]``, no
    per-head interleave) — exactly the shared module's
    ``linear_attention_separate_proj`` mode; the only delta is the
    output-gate norm activation (``output_gate_type``, sigmoid here).
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize the GDN block for layer ``layer_idx``."""
        if not config.linear_attention_separate_proj:
            raise ValueError("Qwen4-Exp requires linear_attention_separate_proj=True.")
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            activation=config.output_gate_type or config.hidden_act,
            rngs=rngs,
        )
        state_dtype = jnp.dtype(config.mamba_ssm_dtype)
        metadata = OperationMetadata(
            runtime_dtype=state_dtype,
            runtime_softmax_dtype=jnp.float32,
            base_config=config,
        )
        self.gdr_op = GatedDeltaRuleOp(metadata)
        self.ragged_gdr_op = RaggedGatedDeltaRule(metadata)


# ---------------------------------------------------------------------------
# Sparse full attention (QSA)
# ---------------------------------------------------------------------------


class Qwen4ExpAttention(Qwen3NextFullAttention):
    """Qwen4-Exp sparse full attention: sigmoid-gated attention + QSA indexer.

    The indexer scores mean-pooled key blocks and produces an additive
    per-query bias (0 for selected, -inf otherwise) that is passed to the
    attention kernel alongside the standard mask path — the cache/mask
    machinery is never modified, so this composes with paged/prefill/decode
    exactly like the dense parent. Raw indexer keys are cached (unnormed,
    unroped) alongside the KV cache so decode steps score the whole prefix.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize the sparse full-attention layer for layer ``layer_idx``."""
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.indexer = (
            BlockTopKIndexer(
                hidden_size=config.hidden_size,
                index_n_heads=config.indexer_n_heads,
                index_kv_heads=config.indexer_kv_heads,
                index_head_dim=config.indexer_head_dim,
                indexer_budget=config.indexer_budget,
                indexer_compress_ratio=config.indexer_compress_ratio,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            if config.qsa_enabled
            else None
        )

    def _paged_indexer_select(
        self,
        hidden_states: Array,
        position_ids: Array,
        cache_view: Qwen4ExpPagedQSAView,
        cache_metadata: RaggedPagesMetadata | OperationsMetadata,
    ) -> tuple[Array, Qwen4ExpPagedQSAView]:
        """Update physical-page QSA history and select logical token offsets."""
        meta = getattr(cache_metadata, "ragged", None) or cache_metadata
        ratio = self.indexer.compress_ratio
        page_size = int(meta.page_size)
        if page_size % ratio:
            raise ValueError("Qwen4 paged QSA requires page_size divisible by indexer_compress_ratio")
        total = hidden_states.shape[1]
        num_pages = cache_view.indexer_key_pages.shape[0]
        req, logical, physical, offset, valid = _paged_qsa_token_map(meta, total, num_pages)
        q, raw_k = self.indexer.project(hidden_states)
        raw_k = raw_k.reshape(total, -1)
        rows = position_ids
        if rows.ndim == 2:
            rows = jnp.broadcast_to(rows[None], (3, *rows.shape))
        rows = rows.transpose(1, 2, 0).reshape(total, 3)

        key_pages = jnp.pad(cache_view.indexer_key_pages, ((0, 1), (0, 0), (0, 0)))
        pos_pages = jnp.pad(cache_view.mrope_position_pages, ((0, 1), (0, 0), (0, 0)))
        key_pages = key_pages.at[physical, offset].set(
            jnp.where(valid[:, None], raw_k.astype(key_pages.dtype), 0), mode="drop"
        )
        pos_pages = pos_pages.at[physical, offset].set(jnp.where(valid[:, None], rows.astype(jnp.int32), 0), mode="drop")
        key_pages = key_pages[:-1]
        pos_pages = pos_pages[:-1]
        cache_view = cache_view.replace(indexer_key_pages=key_pages, mrope_position_pages=pos_pages)

        tables = meta.pages_tables.reshape(meta.query_start_loc.shape[0] - 1, -1)
        blocks_per_page = page_size // ratio
        max_blocks = min(tables.shape[1] * blocks_per_page, num_pages * blocks_per_page)
        member = jnp.arange(ratio, dtype=jnp.int32)
        width = self.indexer.token_budget + ratio - 1

        def _select_all(_):
            token = jnp.arange(width, dtype=jnp.int32)[None, :]
            return jnp.where(token <= logical[:, None], token, -1)

        def _rank_blocks(_):
            block = jnp.arange(max_blocks, dtype=jnp.int32)[None, :]
            block_page = block // blocks_per_page
            block_in_page = block % blocks_per_page
            phys_block_page = jnp.clip(tables[req[:, None], block_page], 0, num_pages - 1)
            physical_raw = key_pages.reshape(num_pages, blocks_per_page, ratio, key_pages.shape[-1])
            physical_pooled = self.indexer.k_layernorm(
                jnp.mean(physical_raw.astype(jnp.float32), axis=2).astype(raw_k.dtype)
            ).reshape(num_pages * blocks_per_page, -1)
            physical_rows = pos_pages[:, ::ratio].reshape(num_pages * blocks_per_page, 3)
            block_cos, block_sin = self.rotary.compute_cos_sin(physical_rows.T[:, None, :], dtype=jnp.float32)
            physical_keys = apply_partial_rope(physical_pooled[None], block_cos, block_sin)[0]
            q_cos, q_sin = self.rotary.compute_cos_sin(rows.transpose(1, 0)[:, :, None], dtype=jnp.float32)
            q_flat = apply_partial_rope(q.reshape(total, self.indexer.index_n_heads, -1), q_cos, q_sin)
            q_score = q_flat.astype(jnp.float32)
            k_score = physical_keys.astype(jnp.float32)

            def _score_head(head, accum):
                head_scores = jnp.einsum("td,pd->tp", q_score[:, head], k_score)
                return accum + jax.nn.relu(head_scores)

            physical_scores = jax.lax.fori_loop(
                0,
                self.indexer.index_n_heads,
                _score_head,
                jnp.zeros((total, physical_keys.shape[0]), dtype=jnp.float32),
            )
            physical_block = phys_block_page * blocks_per_page + block_in_page
            scores = jnp.take_along_axis(physical_scores, physical_block, axis=1)
            scores = scores / np.sqrt(self.indexer.index_head_dim)
            complete = (block * ratio + ratio - 1) <= logical[:, None]
            scores = jnp.where(complete & valid[:, None], scores, -jnp.inf)
            k_pick = min(self.indexer.block_topk, max_blocks)
            top_scores, top_blocks = jax.lax.top_k(scores, k_pick)
            picked = jnp.where(top_scores > -jnp.inf, top_blocks, -1)
            selected_blocks = picked[..., None] * ratio + member
            selected_blocks = jnp.where(picked[..., None] >= 0, selected_blocks, -1).reshape(total, -1)
            tail_start = ((logical + 1) // ratio) * ratio
            tail = tail_start[:, None] + jnp.arange(ratio - 1, dtype=jnp.int32)[None, :]
            tail = jnp.where(tail <= logical[:, None], tail, -1)
            selected = jnp.concatenate([selected_blocks, tail], axis=-1)
            return jnp.pad(
                selected,
                ((0, 0), (0, max(0, width - selected.shape[-1]))),
                constant_values=-1,
            )[:, :width]

        selected = jax.lax.cond(
            jnp.all((logical + 1 <= self.indexer.token_budget) | ~valid),
            _select_all,
            _rank_blocks,
            operand=None,
        )
        return selected[None], cache_view

    def _indexer_select(
        self,
        hidden_states: Array,
        mask_info: MaskInfo | None,
        position_ids: Array,
        cache_view: tp.Any | None,
    ) -> tuple[Array | None, Array | None, tp.Any]:
        """Update QSA indexer state and select visible token indices.

        Returns ``(selected, mask_sel, cache_view)`` where ``selected`` holds
        the selected token indices ``[B, Q, budget + ratio - 1]`` (cached path)
        and ``mask_sel`` is a boolean mask ``[B, 1, Q, S]`` (state-free path);
        exactly one of them is ``None``.
        """
        batch, seq_len = hidden_states.shape[:2]
        # ``dynamic_init`` represents ordinary padding masks with derived
        # segment IDs *and* a materialized attention mask. Only a caller-built
        # segment-only MaskInfo denotes packed documents; forwarding derived
        # padding segments would incorrectly route normal training into the
        # intentionally unsupported packed-QSA path.
        explicit_packed_segments = (
            mask_info is not None
            and getattr(mask_info, "_attention_mask", None) is None
            and getattr(mask_info, "_q_segment_ids", None) is not None
        )
        rows_current = position_ids
        if rows_current.ndim == 2:
            rows_current = jnp.broadcast_to(rows_current[None], (3, *rows_current.shape))

        has_state = cache_view is not None and getattr(cache_view, "indexer_key", None) is not None
        if has_state:
            key_buffer = cache_view.indexer_key
            rows_buffer = cache_view.mrope_positions
            visible_buffer = cache_view.indexer_visible
            write_at = cache_view.indexes.astype(jnp.int32)  # next write index per row
        else:
            key_buffer = None
            rows_buffer = None
            visible_buffer = None
            write_at = jnp.zeros((batch,), jnp.int32)

        # Current padding visibility for this forward.
        visible_current = None
        if mask_info is not None:
            kv_mask = getattr(mask_info, "kv_attention_mask", None)
            if kv_mask is not None and kv_mask.ndim == 2:
                visible_current = kv_mask[:, -seq_len:].astype(jnp.bool_)
        if visible_current is None:
            visible_current = jnp.ones((batch, seq_len), jnp.bool_)

        # mRoPE tables over the full position history. The buffer is
        # batch-major [B, 3, S]; the runtime rows are [3, B, T].
        if rows_buffer is not None:
            rows_buffer = _dynamic_update_rows(
                rows_buffer,
                rows_current.transpose(1, 0, 2).astype(rows_buffer.dtype),
                write_at,
                axis=1,
            )
            full_rows = rows_buffer.transpose(1, 0, 2)
        else:
            full_rows = rows_current
        rotary = self.rotary
        cos_q, sin_q = rotary.compute_cos_sin(rows_current, dtype=jnp.float32)

        if visible_buffer is not None:
            visible_buffer = _dynamic_update_rows(visible_buffer, visible_current.astype(jnp.bool_), write_at)
            visible = visible_buffer
        else:
            visible = visible_current

        q_indices = write_at[:, None] + jnp.arange(seq_len, dtype=jnp.int32)[None, :]

        if key_buffer is not None:
            # Pre-sized buffer: write the current raw keys in. Decode then
            # re-pools only the open block and ranks the frozen pooled-key
            # buffer; prefill/mixed steps pool the whole prefix once and seed
            # the block state for the decode steps that follow.
            q, raw_k_new = self.indexer.project(hidden_states)
            key_buffer = _dynamic_update_rows(key_buffer, raw_k_new.astype(key_buffer.dtype), write_at)
            if seq_len == 1 and cache_view.indexer_block_keys is not None:
                ratio = self.indexer.compress_ratio
                first_visible = jnp.argmax(visible.astype(jnp.int32), axis=1)
                first_visible = jnp.where(visible.any(axis=1), first_visible, 0)
                b_open = jnp.maximum(write_at - first_visible, 0) // ratio
                open_start = first_visible + b_open * ratio
                open_rows = jnp.take_along_axis(rows_buffer, open_start[:, None, None], axis=2)
                open_cos, open_sin = rotary.compute_cos_sin(open_rows.transpose(1, 0, 2), dtype=jnp.float32)
                selected, block_keys, blocks_complete = self.indexer.select_step(
                    q,
                    q_cos=cos_q,
                    q_sin=sin_q,
                    key_buffer=key_buffer,
                    block_keys=cache_view.indexer_block_keys,
                    blocks_complete=cache_view.indexer_blocks_complete,
                    visible=visible,
                    open_cos=open_cos,
                    open_sin=open_sin,
                    write_at=write_at,
                )
                cache_view = cache_view.replace(
                    indexer_key=key_buffer,
                    indexer_visible=visible_buffer,
                    mrope_positions=rows_buffer,
                    indexer_block_keys=block_keys,
                    indexer_blocks_complete=blocks_complete,
                )
                return selected, None, None, cache_view
            cos_full, sin_full = rotary.compute_cos_sin(full_rows, dtype=jnp.float32)
            selected, seed_keys, seed_complete = self.indexer.select(
                q,
                key_buffer,
                q_cos=cos_q,
                q_sin=sin_q,
                k_cos=cos_full,
                k_sin=sin_full,
                visible=visible,
                q_indices=q_indices,
                return_blocks=True,
            )
            cache_view = cache_view.replace(
                indexer_key=key_buffer,
                indexer_visible=visible_buffer,
                mrope_positions=rows_buffer,
                indexer_block_keys=seed_keys.astype(cache_view.indexer_block_keys.dtype),
                indexer_blocks_complete=seed_complete,
            )
            return selected, None, None, cache_view
        cos_full, sin_full = rotary.compute_cos_sin(full_rows, dtype=jnp.float32)
        q_segment_ids = getattr(mask_info, "_q_segment_ids", None) if explicit_packed_segments else None
        kv_segment_ids = getattr(mask_info, "_kv_segment_ids", None) if explicit_packed_segments else None
        if q_segment_ids is not None and q_segment_ids.ndim == 3:
            q_segment_ids = q_segment_ids[:, 0, :]
        if kv_segment_ids is not None and kv_segment_ids.ndim == 3:
            kv_segment_ids = kv_segment_ids[:, 0, :]
        mask_sel, _, score_proxy = self.indexer(
            hidden_states,
            q_cos=cos_q,
            q_sin=sin_q,
            k_cos=cos_full,
            k_sin=sin_full,
            cached_raw_k=None,
            visible=visible,
            q_indices=q_indices,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            return_score_proxy=True,
        )
        return None, mask_sel, score_proxy, cache_view

    def _indexer_bias(
        self,
        hidden_states: Array,
        mask_info: MaskInfo | None,
        position_ids: Array,
        cache_view: tp.Any | None,
    ) -> tuple[Array | None, tp.Any]:
        """Compute the additive QSA selection bias ``[B, 1, Q, K]``.

        Full-width path used by prefill/training: the selected indices are
        scattered into a boolean mask spanning the whole buffer and broadcast
        across heads. Decode takes the gathered path instead (see
        :meth:`_decode_gather_attention`).
        """
        batch = hidden_states.shape[0]
        selected, mask_sel, score_proxy, cache_view = self._indexer_select(
            hidden_states, mask_info, position_ids, cache_view
        )
        if mask_sel is None:
            mask_sel = self.indexer.build_mask(selected, cache_view.indexer_key.shape[1])

        # Broadcast to the full head count: the attention kernel's jaxtyping
        # binds num_heads from the bias, and a singleton head axis would clash
        # with the returned per-head attention weights.
        bias = jnp.where(mask_sel, 0.0, -jnp.inf).astype(jnp.float32)
        if score_proxy is not None:
            # Zero in the primal, identity in the tangent: preserve exact hard
            # QSA attention while allowing LM gradients into indexer scores.
            ste = score_proxy - jax.lax.stop_gradient(score_proxy)
            bias = bias + jnp.where(mask_sel, ste, 0.0)
        bias = jnp.broadcast_to(bias, (batch, self.num_heads, *bias.shape[-2:]))
        return bias, cache_view

    def _write_v3_paged_kv(
        self,
        key_states: Array,
        value_states: Array,
        cache_view: Qwen4ExpPagedQSAView,
        cache_metadata: RaggedPagesMetadata | OperationsMetadata,
    ) -> Qwen4ExpPagedQSAView:
        """Write current Qwen KV tokens before the gathered v3 attention path."""
        meta = getattr(cache_metadata, "ragged", None) or cache_metadata
        if meta.version != "v3":
            return cache_view
        if isinstance(cache_view.kv_pages, ImplicitArray):
            raise NotImplementedError(
                "Qwen4 paged-QSA v3 cannot update an implicit quantized KV cache "
                "without preserving its scales; use a concrete narrow dtype or v2"
            )
        total = key_states.shape[0] * key_states.shape[1]
        _, _, physical, offset, valid = _paged_qsa_token_map(meta, total, cache_view.indexer_key_pages.shape[0])
        key_pages = cache_view.key_pages
        value_pages = cache_view.value_pages
        drop_page = key_pages.shape[0]
        target_page = jnp.where(valid, physical, drop_page)
        key_flat = key_states.reshape(total, key_states.shape[2], key_states.shape[3])
        value_flat = value_states.reshape(total, value_states.shape[2], value_states.shape[3])
        if key_flat.shape[1] != key_pages.shape[2]:
            if key_pages.shape[2] % key_flat.shape[1] != 0:
                raise ValueError("Qwen4 v3 cache KV-head width must be a multiple of the logical width")
            repeats = key_pages.shape[2] // key_flat.shape[1]
            key_flat = jnp.repeat(key_flat, repeats, axis=1)
            value_flat = jnp.repeat(value_flat, repeats, axis=1)
        key_pages = key_pages.at[target_page, offset].set(key_flat.astype(key_pages.dtype), mode="drop")
        value_pages = value_pages.at[target_page, offset].set(value_flat.astype(value_pages.dtype), mode="drop")
        if kv_pair_shares_head_dim_axis(getattr(getattr(cache_view, "metadata", None), "k_headdim", -1)):
            combined = jnp.concatenate((key_pages, value_pages), axis=-1)
        else:
            combined = jnp.stack((key_pages, value_pages), axis=3)
        return cache_view.replace(kv_pages=combined.reshape(cache_view.kv_pages.shape))

    def _paged_gather_attention(
        self,
        query_states: Array,
        cache_view: Qwen4ExpPagedQSAView,
        cache_metadata: RaggedPagesMetadata | OperationsMetadata,
        selected: Array,
    ) -> Array:
        """Gather selected logical positions from v2 physical KV pages in bounded chunks."""
        meta = getattr(cache_metadata, "ragged", None) or cache_metadata
        total = query_states.shape[1]
        req, _, _, _, current_valid = _paged_qsa_token_map(meta, total, cache_view.indexer_key_pages.shape[0])
        tables = meta.pages_tables.reshape(meta.query_start_loc.shape[0] - 1, -1)
        sel = selected[0]
        width = sel.shape[1]
        # Gather interleaved K/V once, then split the selected tensor. Two
        # independent gathers of even/odd cache views cost ~25% more on TPU.
        kv_pages = cache_view.flattened_kv_pages()
        pair_in_head_dim = kv_pair_shares_head_dim_axis(getattr(getattr(cache_view, "metadata", None), "k_headdim", -1))
        kv_heads = kv_pages.shape[2] if pair_in_head_dim else kv_pages.shape[2] // 2
        groups = query_states.shape[2] // kv_heads
        chunk = min(64, total)
        padded = ((total + chunk - 1) // chunk) * chunk
        q = query_states[0].reshape(total, kv_heads, groups, self.head_dim)
        q = jnp.pad(q, ((0, padded - total), (0, 0), (0, 0), (0, 0)))
        req = jnp.pad(req, (0, padded - total))
        current_valid = jnp.pad(current_valid, (0, padded - total), constant_values=False)
        sel = jnp.pad(sel, ((0, padded - total), (0, 0)), constant_values=-1)
        output = jnp.zeros((padded, kv_heads, groups, self.head_dim), dtype=kv_pages.dtype)

        def _attend_chunk(index, output):
            start = index * chunk
            q_chunk = jax.lax.dynamic_slice(q, (start, 0, 0, 0), (chunk, kv_heads, groups, self.head_dim))
            req_chunk = jax.lax.dynamic_slice(req, (start,), (chunk,))
            valid_current = jax.lax.dynamic_slice(current_valid, (start,), (chunk,))
            sel_chunk = jax.lax.dynamic_slice(sel, (start, 0), (chunk, width))
            valid = (sel_chunk >= 0) & valid_current[:, None]
            safe = jnp.maximum(sel_chunk, 0)
            logical_page = safe // int(meta.page_size)
            page_offset = safe % int(meta.page_size)
            physical = tables[req_chunk[:, None], jnp.clip(logical_page, 0, tables.shape[1] - 1)]
            physical = jnp.clip(physical, 0, kv_pages.shape[0] - 1)
            kv_sel = kv_pages[physical, page_offset]
            if pair_in_head_dim:
                key_sel = kv_sel[..., : self.head_dim]
                value_sel = kv_sel[..., self.head_dim : self.head_dim * 2]
            else:
                key_sel = kv_sel[:, :, 0::2, :]
                value_sel = kv_sel[:, :, 1::2, :]
            scores = jnp.einsum(
                "thgd,twhd->thgw",
                q_chunk.astype(jnp.float32),
                key_sel.astype(jnp.float32),
            )
            scores = scores * (self.head_dim**-0.5)
            scores = jnp.where(valid[:, None, None, :], scores, -1e30)
            probs = jax.nn.softmax(scores, axis=-1)
            out = jnp.einsum("thgw,twhd->thgd", probs.astype(value_sel.dtype), value_sel)
            return jax.lax.dynamic_update_slice(output, out, (start, 0, 0, 0))

        output = jax.lax.fori_loop(0, padded // chunk, _attend_chunk, output)
        return output[:total].reshape(1, total, -1, self.head_dim)

    def _decode_gather_attention(
        self,
        query_states: Array,
        key_states: Array,
        value_states: Array,
        selected: Array,
    ) -> Array:
        """Attend only the QSA-selected tokens for a single-token decode step.

        Gathering the ``token_budget`` selected positions (2,048 + tail)
        replaces the full-axis pass over the dense cache: at a 262,144-token
        buffer the kernel otherwise reads ~8.6 GiB of K/V per layer per step
        and consumes a ``[B, heads, 1, S]`` fp32 bias that costs another GiB —
        to attend to at most 2,051 positions. The selected set is
        causality-closed by construction (block ends and the tail are all
        ``<= q_idx``), so no causal mask is needed; ``-1`` pads are masked by
        the bias. Identical attended set as the masked full-width path.

        Args:
            query_states: Post-rope queries ``[B, 1, H, D]``.
            key_states: Full dense key buffer (post-concat) ``[B, S, H_kv, D]``.
            value_states: Full dense value buffer (post-concat) ``[B, S, H_kv, D]``.
            selected: Selected token indices ``[B, 1, W]``, ``-1``-padded.

        Returns:
            Attention output ``[B, 1, H, D]``.
        """
        batch = query_states.shape[0]
        sel = selected[:, 0, :]  # [B, W] — decode carries one query per row
        valid = sel >= 0
        safe = jnp.where(valid, sel, 0)
        rows = jnp.arange(batch, dtype=jnp.int32)[:, None]
        key_sel = key_states[rows, safe]  # [B, W, H_kv, D]
        value_sel = value_states[rows, safe]

        kv_heads = key_sel.shape[2]
        groups = query_states.shape[2] // kv_heads
        q = query_states[:, 0].reshape(batch, kv_heads, groups, self.head_dim)
        scores = jnp.einsum("bhgd,bwhd->bhgw", q.astype(jnp.float32), key_sel.astype(jnp.float32))
        scores = scores * (self.head_dim**-0.5)
        scores = jnp.where(valid[:, None, None, :], scores, -1e30)
        probs = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhgw,bwhd->bhgd", probs.astype(value_sel.dtype), value_sel)
        return out.reshape(batch, 1, -1, self.head_dim)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> AttentionLayerOutput:
        """Qwen4-Exp sparse attention forward.

        Same flow as :class:`Qwen3NextFullAttention` (fused q+gate projection,
        QK norm, rotary, cache concat, kernel, sigmoid output gate, o_proj),
        with the QSA indexer's additive selection bias folded into the kernel
        call. The bias is added to whatever the cache path's lazy bias carries
        (zero in the common case); causal/padding masking still comes from
        ``mask_info``.
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        if mode == common_types.MODE_TRAIN and cache_view is not None and self.indexer is not None:
            raise ValueError(
                "Qwen4 QSA training with a cache is unsupported because hard cached selection "
                "does not carry the indexer score-gradient surrogate; use stateless training"
            )
        if output_attentions and isinstance(cache_view, Qwen4ExpPagedQSAView):
            raise NotImplementedError("output_attentions=True is not supported by paged QSA; use a dense cache/backend")

        selected = None
        paged_selected = None
        indexer_bias = None
        if self.indexer is not None:
            if isinstance(cache_view, Qwen4ExpPagedQSAView):
                if cache_metadata is None:
                    raise ValueError("Qwen4 paged QSA requires RaggedPagesMetadata")
                paged_selected, cache_view = self._paged_indexer_select(
                    hidden_states, position_ids, cache_view, cache_metadata
                )
            elif (
                sequence_length == 1
                and cache_view is not None
                and getattr(cache_view, "indexer_key", None) is not None
                and getattr(cache_view, "indexer_block_keys", None) is not None
                and not output_attentions
            ):
                # Pure decode step: select indices only; the gathered path
                # below attends them directly without the full-width bias.
                selected, _, _, cache_view = self._indexer_select(hidden_states, mask_info, position_ids, cache_view)
            else:
                indexer_bias, cache_view = self._indexer_bias(hidden_states, mask_info, position_ids, cache_view)

        q_multiplier = self._q_proj_multiplier()
        qkv_output = checkpoint_name(self.qkv_proj(hidden_states), "attn_qkv")
        q_proj_output, key_states, value_states = self.qkv_proj.split(qkv_output, config=self.config)

        q_proj_output = q_proj_output.reshape(batch_size, sequence_length, -1, self.head_dim * q_multiplier)
        key_states = key_states.reshape(batch_size, sequence_length, -1, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, -1, self.head_dim)

        if self.attn_output_gate:
            query_states, gate = jnp.split(q_proj_output, 2, axis=-1)
        else:
            query_states, gate = q_proj_output, None

        query_states, key_states, value_states = self._postprocess_qkv(query_states, key_states, value_states)
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)
        query_states, key_states = self._apply_rotary(query_states, key_states, position_ids, frequencies)

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
            sliding_window=self.sliding_window,
        )

        softmax_aux = self._softmax_aux()
        attention_weights = None
        if paged_selected is not None and softmax_aux is None:
            getattr(cache_metadata, "ragged", None) or cache_metadata
            cache_view = self._write_v3_paged_kv(key_states, value_states, cache_view, cache_metadata)

            # The selected set is complete under budget and QSA-ranked above
            # it. Use the same bounded gather for prefill and decode so long
            # paged prefills cannot silently fall back to dense attention.
            attn_output = self._paged_gather_attention(query_states, cache_view, cache_metadata, paged_selected)
        elif selected is not None and softmax_aux is None:
            attn_output = self._decode_gather_attention(query_states, key_states, value_states, selected)
        else:
            bias = indexer_bias
            if init_attention_bias is not None:
                base = init_attention_bias()
                if base.shape[1] == 1 and self.num_heads != 1:
                    base = jnp.broadcast_to(base, (base.shape[0], self.num_heads, *base.shape[2:]))
                bias = base if bias is None else bias + base.astype(bias.dtype)

            attentions: AttentionLayerOutput = self.attention_performer.forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                mode=mode,
                bias=bias,
                cache_metadata=cache_metadata,
                cache_view=cache_view,
                init_bias=None,
                mask_info=mask_info,
                causal=self.causal,
                sliding_window=self.sliding_window,
                softmax_aux=softmax_aux,
            )
            if attentions.cache_view is not None:
                cache_view = attentions.cache_view
            attn_output = attentions.attention_outputs
            attention_weights = attentions.attention_weights
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        if gate is not None:
            if attn_output.dtype in lowfloats or gate.dtype in lowfloats:
                attn_output_dtype = attn_output.dtype
                attn_output = attn_output.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
                attn_output = attn_output.astype(attn_output_dtype)
            else:
                attn_output = attn_output * jax.nn.sigmoid(gate)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.shard_attention_prod(attn_output)
        attn_output = self.o_proj(attn_output)
        attn_output = self.shard_attention_prod(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class Qwen4ExpMLP(spx.Module):
    """Dense SwiGLU MLP (used for the shared expert)."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        intermediate_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize with separate gate/up/down projections (checkpoint layout)."""
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        linear = dict(
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.gate_proj = ColumnParallelLinear(config.hidden_size, intermediate_size, **linear)
        self.up_proj = ColumnParallelLinear(config.hidden_size, intermediate_size, **linear)
        self.down_proj = RowParallelLinear(intermediate_size, config.hidden_size, **linear)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Float[Array, "... hidden"]) -> Float[Array, "... hidden"]:
        """``down(silu(gate(x)) * up(x))``."""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen4ExpMLPStack(spx.Module):
    """Fused SwiGLU expert stack (per-expert checkpoint tensors fused at load)."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the fused expert stack."""
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate_up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=2 * config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            use_bias=False,
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        """Stacked HF ``experts.down_proj`` -> the 3-D runtime kernel."""
        return {
            **moe_down_projection_reform_param(),
        }

    def forward(self, hidden_states: Array, group_sizes: Array, sorted_experts: Array | None = None) -> Array:
        """Apply the grouped per-expert SwiGLU; see BaseMoeModule dispatch."""
        gate_up = self.gate_up_proj(hidden_states, group_sizes, sorted_experts)
        gate, up = split_fused_gate_up_projection(gate_up, config=self.config)
        return self.down_proj(self.act_fn(gate) * up, group_sizes, sorted_experts)


class Qwen4ExpSparseMoeBlock(BaseMoeModule):
    """Qwen4-Exp MoE: softmax top-k router + fused experts + gated shared expert.

    Matches the reference: softmax router with ``norm_topk_prob``
    renormalization; expert outputs summed with
    ``sigmoid(shared_expert_gate(x)) * shared_expert(x)``.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the MoE block."""
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=(MoeRoutingStrategy.TOP_K if config.norm_topk_prob else MoeRoutingStrategy.TOP_K_NDIV),
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        self.experts = Qwen4ExpMLPStack(
            config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )
        self.shared_expert = Qwen4ExpMLP(
            config=config,
            intermediate_size=config.shared_expert_intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.shared_expert_gate = ColumnParallelLinear(
            config.hidden_size,
            1,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )

    @property
    def reform_param(self) -> dict[str, dict[str, tp.Any]]:
        """Fuse the release checkpoint's per-expert gate/up into the 3-D kernel.

        The published ``Qwen3.8-Flash-Next`` shards store every expert's gate
        and up projections separately
        (``experts.{i}.gate_proj.weight`` / ``experts.{i}.up_proj.weight``);
        the runtime consumes one ``experts.gate_up_proj`` kernel of shape
        ``[experts, hidden, 2 * intermediate]`` with the two halves
        TP-interleaved. The rule lives on the block (not the stack) so the
        collector-prefixed sources line up with the checkpoint paths and the
        fused tensor lands under the stack's ``experts.gate_up_proj``.
        """
        return FusedExpertLayout(
            target_prefix="experts.gate_up_proj",
            source_prefix="experts",
            gate_prefix="gate_proj",
            up_prefix="up_proj",
            per_expert_template="{prefix}.{index}.{name}.weight",
            source_per_expert=self.config.num_experts,
        ).reform_param(config=self.config)

    def forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        """Route + combine; returns ``(output, router_logits)``."""
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            gate_up_kernel=self.experts.gate_up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
        )
        stage_mesh = resolve_stage_mesh(self.config.mesh, arr=hidden_states)
        with spx.use_mesh(stage_mesh):
            shared_out = self.shared_expert(hidden_states)
            gate_input = self.shared_expert_gate(hidden_states)
        gate = jax.nn.sigmoid(gate_input.astype(jnp.float32))
        shared_out = shared_out.astype(jnp.float32) * gate
        out = (out.astype(jnp.float32) + shared_out).astype(self.dtype)
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


# ---------------------------------------------------------------------------
# PLE (per-layer n-gram embedding)
# ---------------------------------------------------------------------------


class Qwen4ExpPLELayer(spx.Module):
    """Inject hashed n-gram features into every hyper-connection stream.

    Reference math (HF ``Qwen4ExpTextPLELayer``): the token n-gram embedding is
    projected to one key per residual stream and a shared value; the normed
    stream activations gate the values through a signed-square-root score; a
    dilated depthwise convolution over the normed gated values adds local
    lexical context. Output width is ``hc_count * hidden`` — added to all
    streams by the caller.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_idx: int,
        ple_layer_index: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the PLE block for (0-indexed) ``layer_idx``."""
        self.config = config
        self.layer_idx = layer_idx
        self.ple_layer_index = ple_layer_index
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.dtype = dtype
        self.param_dtype = param_dtype
        hc_hidden = self.hidden_size * self.hc_count
        self.conv_kernel_size = config.ple_conv_kernel_size
        self.conv_dilation = config.ngram_size
        self.short_conv_state_len = (self.conv_kernel_size - 1) * self.conv_dilation

        table_dtype = getattr(config, "ngram_table_dtype", None)
        if table_dtype:
            # fp8 checkpoint tables stay fp8 in HBM (51B params); the scale is
            # applied at gather time (see ``NGramEmbed.lookup``). Note the
            # checkpoint stores float8_e4m3fn, so the config value must name
            # that grid explicitly ("fp8" would resolve to e5m2).
            from eformer.mpric.dtypes.precision_types import STRING_TO_DTYPE_MAP

            table_param_dtype = STRING_TO_DTYPE_MAP[str(table_dtype)]
        else:
            table_param_dtype = param_dtype
        self.ple_embedding = NGramEmbed(
            config,
            embedding_dim=config.ple_embed_dim,
            ple_layer_index=ple_layer_index,
            dtype=dtype,
            param_dtype=table_param_dtype,
            rngs=rngs,
        )
        linear = dict(
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.key_proj = ColumnParallelLinear(config.ple_embed_dim, hc_hidden, **linear)
        self.value_proj = ColumnParallelLinear(config.ple_embed_dim, self.hidden_size, **linear)
        norm = partial(Qwen4ExpRMSNorm, eps=config.rms_norm_eps, param_dtype=param_dtype, rngs=rngs)
        self.norm_key = norm(hc_hidden, group_size=self.hidden_size)
        self.norm_query = norm(hc_hidden, group_size=self.hidden_size)
        self.norm_conv = norm(hc_hidden, group_size=self.hidden_size)

        # Depthwise dilated causal conv; HIO layout so the checkpoint's
        # [C, 1, K] maps with a (2, 1, 0) transpose. Zero-init per reference.
        self.conv1d = ArrayParam.bound(
            shape=(self.conv_kernel_size, 1, hc_hidden),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )

    @property
    def reform_param(self) -> dict[str, dict[str, tp.Any]]:
        """Checkpoint reform rule for the PLE depthwise conv kernel.

        The checkpoint stores ``ple.conv1d.weight`` in torch's ``[C, 1, K]``
        layout while the EasyDeL kernel is HIO ``[K, 1, C]``; the rule
        transposes on load and back on export.
        """

        def _permute(tensor: tp.Any) -> tp.Any:
            return tensor.permute(2, 1, 0).contiguous()

        def _inverse(kernel: tp.Any) -> tp.Any:
            return kernel.permute(2, 1, 0).contiguous()

        return {
            "conv1d.weight$": {
                "splits": [{"name": "conv1d", "spliter": _permute}],
                "inverse_spliter": _inverse,
            },
        }

    def _short_conv(
        self,
        hidden_states: Float[Array, "batch seq channels"],
        conv_state: Float[Array, "batch channels state"] | None = None,
        segment_ids: Int[Array, "batch seq"] | None = None,
        context_segment_ids: Int[Array, "batch ctx"] | None = None,
    ) -> tuple[Array, Array]:
        """Dilated causal depthwise conv + silu, with optional carried state.

        Args:
            hidden_states: ``[batch, seq, channels]`` input.
            conv_state: Previous ``[batch, channels, state_len]`` window
                (decode), or ``None`` (prefill pads with zeros). With segments,
                inputs belonging to older documents must already be zeroed.
            context_segment_ids: Carried n-gram segment context, whose last id
                labels the retained conv inputs. Missing context is invalid
                history when segment masking is requested.

        Returns:
            ``(out, new_state)``: the conv output ``[batch, seq, channels]``
            and the trailing state window to cache.
        """
        batch, seq_len, channels = hidden_states.shape
        x = hidden_states.transpose(0, 2, 1).astype(jnp.float32)  # [B, C, S]
        state_len = self.short_conv_state_len
        if conv_state is not None:
            x_full = jnp.concatenate([conv_state.astype(jnp.float32), x], axis=-1)
        else:
            x_full = jnp.pad(x, ((0, 0), (0, 0), (state_len, 0)))
        new_state = jax.lax.dynamic_slice(x_full, (0, 0, x_full.shape[-1] - state_len), (batch, channels, state_len))
        kernel = self.conv1d.value.astype(jnp.float32)  # [K, 1, C]
        if segment_ids is not None:
            # Cached windows retain only the latest document's inputs. Its id
            # is already carried by the n-gram context, so no wider cache or
            # new cache leaf is needed to mask taps across chunk boundaries.
            previous_segment = (
                context_segment_ids[:, -1:]
                if context_segment_ids is not None and context_segment_ids.shape[1]
                else jnp.full((batch, 1), -1, segment_ids.dtype)
            )
            full_segments = jnp.concatenate(
                [jnp.broadcast_to(previous_segment, (batch, state_len)), segment_ids], axis=1
            )
            # Compute only new-token outputs, not the carried history's
            # outputs. Static tap slices keep one-token decode work bounded.
            out = jnp.zeros((batch, seq_len, channels), jnp.float32)
            for kernel_idx in range(self.conv_kernel_size):
                start = kernel_idx * self.conv_dilation
                shifted = x_full[:, :, start : start + seq_len].transpose(0, 2, 1)
                source_segments = full_segments[:, start : start + seq_len]
                valid = (segment_ids >= 0) & (source_segments == segment_ids)
                out = out + jnp.where(valid[..., None], shifted, 0.0) * kernel[kernel_idx, 0]
            state_segments = full_segments[:, -state_len:] if state_len else full_segments[:, :0]
            new_state = jnp.where(
                ((state_segments == segment_ids[:, -1:]) & (segment_ids[:, -1:] >= 0))[:, None, :],
                new_state,
                0.0,
            )
        else:
            out = jax.lax.conv_general_dilated(
                lhs=x_full,
                rhs=kernel,
                window_strides=(1,),
                padding=[(0, 0)],  # history already prepended on the left
                rhs_dilation=(self.conv_dilation,),
                dimension_numbers=("NCH", "HIO", "NCH"),
                feature_group_count=channels,
            )
            out = out[:, :, -seq_len:].transpose(0, 2, 1)
        return jax.nn.silu(out).astype(hidden_states.dtype), new_state

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hc*hidden"],
        input_ids: Int[Array, "batch seq"],
        conv_mask: Bool[Array, "batch seq"] | None = None,
        segment_ids: Int[Array, "batch seq"] | None = None,
        ple_token_context: Int[Array, "batch ctx"] | None = None,
        ple_segment_context: Int[Array, "batch ctx"] | None = None,
        ple_conv_state: Float[Array, "batch channels state"] | None = None,
        packed_query_start_loc: Int[Array, "rows_plus_one"] | None = None,
    ) -> tuple[Array, Array, Array, Array | None]:
        """Compute the PLE residual update.

        Args:
            hidden_states: Flattened residual streams ``[B, T, hc * hidden]``.
            input_ids: Token ids for the n-gram hash (padding already replaced
                by EOS by the caller).
            conv_mask: Padding mask; padded positions are zeroed before the
                conv so padding does not leak into the dilated window.
            ple_token_context: Carried ``ngram_size - 1`` tokens (decode).
            ple_conv_state: Carried conv window (decode).

        Returns:
            ``(output, new_token_context, new_conv_state)`` — the additive
            stream update ``[B, T, hc * hidden]`` and the states to cache.
        """
        compute_token_context = ple_token_context
        compute_segment_context = ple_segment_context
        compute_conv_state = ple_conv_state
        batch, seq = input_ids.shape
        if packed_query_start_loc is None:
            embeddings = self.ple_embedding(
                input_ids,
                context=compute_token_context,
                segment_ids=segment_ids,
                context_segment_ids=compute_segment_context,
            )
        else:
            # eSurge packs multiple request chunks into batch row zero. Build
            # one short chronological n-gram window per token, seeding tokens
            # before each chunk start from that request's recurrent context.
            starts = packed_query_start_loc[:-1].astype(jnp.int32)
            token_index = jnp.arange(seq, dtype=jnp.int32)
            request_rows = jnp.clip(
                jnp.searchsorted(packed_query_start_loc, token_index, side="right") - 1,
                0,
                starts.shape[0] - 1,
            )
            local_index = token_index - starts[request_rows]
            offsets = (
                local_index[:, None]
                - self.ple_embedding.context_len
                + jnp.arange(self.ple_embedding.ngram_size, dtype=jnp.int32)[None, :]
            )
            current_index = starts[request_rows, None] + jnp.maximum(offsets, 0)
            current_tokens = jnp.take(input_ids[0], jnp.clip(current_index, 0, seq - 1), axis=0)
            if compute_token_context is None:
                compute_token_context = jnp.full(
                    (starts.shape[0], self.ple_embedding.context_len),
                    self.ple_embedding.eos_token_id,
                    input_ids.dtype,
                )
            prior_index = jnp.clip(self.ple_embedding.context_len + offsets, 0, self.ple_embedding.context_len - 1)
            prior_tokens = jnp.take_along_axis(compute_token_context[request_rows], prior_index, axis=1)
            if compute_segment_context is not None:
                prior_segments = jnp.take_along_axis(compute_segment_context[request_rows], prior_index, axis=1)
                # Cleared cache slots contain zero token storage, not a real
                # token-zero prefix. Invalid history uses the same EOS padding
                # as an uncached n-gram lookup (EOS need not be zero).
                prior_tokens = jnp.where(prior_segments >= 0, prior_tokens, self.ple_embedding.eos_token_id)
            token_windows = jnp.where(offsets >= 0, current_tokens, prior_tokens)
            window_segments = jnp.broadcast_to(request_rows[:, None], token_windows.shape).astype(jnp.int32)
            rows = self.ple_embedding.hash_ids(token_windows, segment_ids=window_segments)[:, -1:]
            embeddings = self.ple_embedding.lookup(rows).reshape(1, seq, -1)

        ctx = self.ple_embedding.context_len
        history = (
            jnp.concatenate([compute_token_context, input_ids], axis=1)
            if compute_token_context is not None and packed_query_start_loc is None
            else input_ids
        )
        if history.shape[1] < ctx:
            history = jnp.pad(
                history, ((0, 0), (ctx - history.shape[1], 0)), constant_values=self.ple_embedding.eos_token_id
            )
        new_context = history[:, -ctx:] if ctx else history[:, :0]
        last_live = None
        if conv_mask is not None and packed_query_start_loc is None:
            token_index = jnp.arange(seq, dtype=jnp.int32)[None, :]
            last_live = jnp.max(jnp.where(conv_mask, token_index, -1), axis=1)
            if ctx:
                history_prefix = history.shape[1] - seq
                tail_index = history_prefix + last_live[:, None] + 1 - ctx + jnp.arange(ctx, dtype=jnp.int32)[None, :]
                gathered = jnp.take_along_axis(history, jnp.clip(tail_index, 0, history.shape[1] - 1), axis=1)
                gathered = jnp.where(tail_index >= 0, gathered, self.ple_embedding.eos_token_id)
                fallback = (
                    ple_token_context
                    if ple_token_context is not None
                    else jnp.full_like(gathered, self.ple_embedding.eos_token_id)
                )
                new_context = jnp.where((last_live >= 0)[:, None], gathered, fallback)
        if segment_ids is None:
            new_segment_context = None
        else:
            segment_history = (
                jnp.concatenate([compute_segment_context, segment_ids], axis=1)
                if compute_segment_context is not None and packed_query_start_loc is None
                else segment_ids
            )
            if segment_history.shape[1] < ctx:
                segment_history = jnp.pad(
                    segment_history, ((0, 0), (ctx - segment_history.shape[1], 0)), constant_values=-1
                )
            new_segment_context = segment_history[:, -ctx:] if ctx else segment_history[:, :0]
            if last_live is not None and ctx:
                segment_prefix = segment_history.shape[1] - seq
                tail_index = segment_prefix + last_live[:, None] + 1 - ctx + jnp.arange(ctx, dtype=jnp.int32)[None, :]
                gathered_segments = jnp.take_along_axis(
                    segment_history, jnp.clip(tail_index, 0, segment_history.shape[1] - 1), axis=1
                )
                gathered_segments = jnp.where(tail_index >= 0, gathered_segments, -1)
                fallback_segments = (
                    ple_segment_context if ple_segment_context is not None else jnp.full_like(gathered_segments, -1)
                )
                new_segment_context = jnp.where((last_live >= 0)[:, None], gathered_segments, fallback_segments)

        key_normed = self.norm_key(self.key_proj(embeddings)).reshape(batch, seq, self.hc_count, self.hidden_size)
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).reshape(batch, seq, self.hc_count, self.hidden_size)

        gate = jnp.sum(key_normed * query_normed, axis=-1, keepdims=True) / np.sqrt(self.hidden_size)
        gate = jnp.sqrt(jnp.abs(gate) + 1e-6) * jnp.sign(gate)
        gated_value = jax.nn.sigmoid(gate) * value[:, :, None, :]
        gated_flat = gated_value.reshape(batch, seq, -1)
        gated_normed = self.norm_conv(gated_flat)

        if conv_mask is not None:
            keep = conv_mask[:, :, None].astype(gated_flat.dtype)
            gated_flat = gated_flat * keep
            gated_normed = gated_normed * keep

        if packed_query_start_loc is None:
            conv_out, new_conv_state = self._short_conv(
                gated_normed,
                compute_conv_state,
                segment_ids=segment_ids,
                context_segment_ids=compute_segment_context,
            )
        else:
            # Same continuation rule for the dilated PLE depthwise convolution:
            # taps before a request chunk come from that request's saved state,
            # never from the preceding packed request.
            state_len = self.short_conv_state_len
            kernel_taps = jnp.arange(self.conv_kernel_size, dtype=jnp.int32) * self.conv_dilation
            conv_offsets = local_index[:, None] - state_len + kernel_taps[None, :]
            current_index = starts[request_rows, None] + jnp.maximum(conv_offsets, 0)
            current = jnp.take(gated_normed[0], jnp.clip(current_index, 0, seq - 1), axis=0)
            if compute_conv_state is None:
                compute_conv_state = jnp.zeros((starts.shape[0], gated_normed.shape[-1], state_len), jnp.float32)
            prior_index = jnp.clip(state_len + conv_offsets, 0, state_len - 1)
            prior = jnp.take_along_axis(
                compute_conv_state[request_rows].transpose(0, 2, 1),
                prior_index[..., None],
                axis=1,
            )
            windows = jnp.where(conv_offsets[..., None] >= 0, current, prior).astype(jnp.float32)
            kernel = self.conv1d.value[:, 0, :].astype(jnp.float32)
            conv_out = jax.nn.silu(jnp.einsum("tkc,kc->tc", windows, kernel))[None].astype(hidden_states.dtype)
            new_conv_state = compute_conv_state
        if last_live is not None:
            state_len = self.short_conv_state_len
            old_state = (
                ple_conv_state.astype(jnp.float32)
                if ple_conv_state is not None
                else jnp.zeros((batch, gated_normed.shape[-1], state_len), jnp.float32)
            )
            state_source = jnp.concatenate([old_state, gated_normed.transpose(0, 2, 1).astype(jnp.float32)], axis=-1)
            state_index = (
                state_len + last_live[:, None] - state_len + 1 + jnp.arange(state_len, dtype=jnp.int32)[None, :]
            )
            gathered_state = jnp.take_along_axis(
                state_source, jnp.clip(state_index, 0, state_source.shape[-1] - 1)[:, None, :], axis=2
            )
            if segment_ids is not None:
                previous_segment = (
                    compute_segment_context[:, -1:]
                    if compute_segment_context is not None and compute_segment_context.shape[1]
                    else jnp.full((batch, 1), -1, segment_ids.dtype)
                )
                state_segments = jnp.concatenate(
                    [jnp.broadcast_to(previous_segment, (batch, state_len)), segment_ids], axis=1
                )
                gathered_segments = jnp.take_along_axis(state_segments, state_index, axis=1)
                final_segment = jnp.take_along_axis(segment_ids, jnp.maximum(last_live, 0)[:, None], axis=1)
                gathered_state = jnp.where(
                    ((gathered_segments == final_segment) & (final_segment >= 0))[:, None, :],
                    gathered_state,
                    0.0,
                )
            new_conv_state = jnp.where((last_live >= 0)[:, None, None], gathered_state, old_state)
        if packed_query_start_loc is not None:
            # eSurge packs request tokens on the sequence axis (batch=1).
            # The PLE output above is segment-aware; gather each request's
            # finite token/conv tail into its persistent recurrent-cache row.
            starts = packed_query_start_loc[:-1].astype(jnp.int32)
            ends = packed_query_start_loc[1:].astype(jnp.int32)
            lengths = ends - starts

            def _packed_tail(source, old, width, fill):
                offsets = lengths[:, None] - width + jnp.arange(width, dtype=jnp.int32)[None, :]
                current_index = starts[:, None] + jnp.maximum(offsets, 0)
                current = jnp.take(source[0], jnp.clip(current_index, 0, source.shape[1] - 1), axis=0)
                if old is None:
                    old = jnp.full((starts.shape[0], width, *source.shape[2:]), fill, source.dtype)
                old_index = jnp.clip(width + offsets, 0, width - 1)
                previous = jnp.take_along_axis(old, old_index[(...,) + (None,) * (old.ndim - 2)], axis=1)
                return jnp.where((offsets >= 0)[(...,) + (None,) * (current.ndim - 2)], current, previous)

            if ctx:
                new_context = _packed_tail(input_ids, ple_token_context, ctx, self.ple_embedding.eos_token_id)
                if segment_ids is not None:
                    new_segment_context = _packed_tail(segment_ids, ple_segment_context, ctx, -1)
            state_len = self.short_conv_state_len
            source = gated_normed.transpose(0, 2, 1)
            old = ple_conv_state
            offsets = lengths[:, None] - state_len + jnp.arange(state_len, dtype=jnp.int32)[None, :]
            current_index = starts[:, None] + jnp.maximum(offsets, 0)
            current = jnp.take(source[0], jnp.clip(current_index, 0, source.shape[-1] - 1), axis=1).transpose(1, 0, 2)
            if old is None:
                old = jnp.zeros((starts.shape[0], source.shape[1], state_len), source.dtype)
            old_index = jnp.clip(state_len + offsets, 0, state_len - 1)
            previous = jnp.take_along_axis(old, old_index[:, None, :], axis=2)
            new_conv_state = jnp.where((offsets >= 0)[:, None, :], current, previous)

        output = gated_flat + conv_out
        return output, new_context, new_conv_state, new_segment_context


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Qwen4ExpDecoderLayer(spx.Module):
    """One Qwen4-Exp decoder block over the flattened hyper-connection streams.

    Flow (streams stay flat ``[B, T, hc * hidden]``): optional PLE injection,
    attention-side gated-residual read → GDN or QSA attention → gated write,
    then the same for the MoE MLP.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize the decoder layer."""
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.is_full_attention = config.is_full_attention_layer(layer_idx)
        if self.is_full_attention:
            self.self_attn = Qwen4ExpAttention(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=layer_idx,
            )
        else:
            self.linear_attn = Qwen4ExpGatedDeltaNet(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=layer_idx,
            )
        self.mlp = Qwen4ExpSparseMoeBlock(
            config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )

        ple_layer_index = config.ple_layer_indices_0based.get(layer_idx)
        self.ple = (
            Qwen4ExpPLELayer(
                config,
                layer_idx,
                ple_layer_index,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            if ple_layer_index is not None
            else None
        )

        hc_kwargs = dict(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            eps=config.rms_norm_eps,
            use_combine=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.attn_hyper_connection = GatedResidual(**hc_kwargs)
        self.mlp_hyper_connection = GatedResidual(**hc_kwargs)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hc*hidden"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: tp.Any | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        ple_input_ids: Int[Array, "batch seq"] | None = None,
        conv_mask: Bool[Array, "batch seq"] | None = None,
        segment_ids: Int[Array, "batch seq"] | None = None,
    ) -> DecoderLayerOutput:
        """Run the block over the residual streams."""
        new_cache_view = cache_view
        if self.ple is not None:
            meta = getattr(cache_metadata, "ragged", None) or cache_metadata
            packed_qsl = (
                getattr(meta, "query_start_loc", None) if isinstance(cache_view, Qwen4ExpOperationsLinearView) else None
            )
            packed_segments = segment_ids
            token_context = getattr(cache_view, "ple_token_context", None)
            segment_context = getattr(cache_view, "ple_segment_context", None)
            conv_state = getattr(cache_view, "ple_conv_state", None)
            if packed_qsl is not None:
                token_index = jnp.arange(ple_input_ids.shape[1], dtype=jnp.int32)
                live_rows = jnp.clip(meta.num_seqs.reshape(-1)[0].astype(jnp.int32), 0, packed_qsl.shape[0] - 1)
                packed_valid = token_index < packed_qsl[live_rows]
                request_rows = jnp.clip(
                    jnp.searchsorted(packed_qsl, token_index, side="right") - 1,
                    0,
                    packed_qsl.shape[0] - 2,
                )
                packed_segments = jnp.where(packed_valid, request_rows, -1)[None, :]
            if packed_qsl is not None and mode == common_types.MODE_DECODE:
                # Decode carries one packed token per live request. Rebatch by
                # request row so each token consumes its own persistent PLE
                # n-gram and convolution state, then scatter the updates back.
                slots = getattr(meta, "recurrent_state_indices", None)
                slots = request_rows if slots is None else slots[request_rows]
                token_batch = ple_input_ids.reshape(-1, 1)
                hidden_batch = hidden_states.reshape(-1, 1, hidden_states.shape[-1])
                segment_batch = request_rows[:, None]
                mask_batch = conv_mask.reshape(-1, 1) if conv_mask is not None else None
                ple_out, row_context, row_conv, row_segments = self.ple(
                    hidden_batch,
                    token_batch,
                    conv_mask=mask_batch,
                    segment_ids=segment_batch,
                    ple_token_context=token_context[slots],
                    # Packed request-row ids are transient scheduler metadata;
                    # a recycled request can move rows while retaining its
                    # physical recurrent slot. Its carried history still
                    # belongs to the current request.
                    ple_segment_context=jnp.where(segment_context[slots] >= 0, request_rows[:, None], -1),
                    ple_conv_state=conv_state[slots],
                )
                ple_out = jnp.where(packed_valid[None, :, None], ple_out.reshape(hidden_states.shape), 0)

                def _scatter_valid(pool, values):
                    target = jnp.where(packed_valid, slots, pool.shape[0])
                    return pool.at[target].set(values, mode="drop")

                new_context = _scatter_valid(token_context, row_context)
                new_ple_conv = _scatter_valid(conv_state, row_conv)
                new_segment_context = _scatter_valid(segment_context, row_segments)
            else:
                packed_slots = None
                if packed_qsl is not None:
                    num_requests = packed_qsl.shape[0] - 1
                    packed_slots = getattr(meta, "recurrent_state_indices", None)
                    packed_slots = (
                        jnp.arange(num_requests, dtype=jnp.int32)
                        if packed_slots is None
                        else packed_slots[:num_requests]
                    )
                    token_context_for_compute = token_context[packed_slots]
                    segment_context_for_compute = segment_context[packed_slots]
                    conv_state_for_compute = conv_state[packed_slots]
                else:
                    token_context_for_compute = token_context
                    segment_context_for_compute = segment_context
                    conv_state_for_compute = conv_state
                ple_out, row_context, row_ple_conv, row_segment_context = self.ple(
                    hidden_states,
                    ple_input_ids,
                    conv_mask=conv_mask,
                    segment_ids=packed_segments,
                    ple_token_context=token_context_for_compute,
                    ple_segment_context=segment_context_for_compute,
                    ple_conv_state=conv_state_for_compute,
                    packed_query_start_loc=packed_qsl,
                )
                if packed_slots is None:
                    new_context, new_ple_conv, new_segment_context = (row_context, row_ple_conv, row_segment_context)
                else:
                    new_context = token_context.at[packed_slots].set(row_context)
                    new_ple_conv = conv_state.at[packed_slots].set(row_ple_conv)
                    new_segment_context = segment_context.at[packed_slots].set(row_segment_context)
            hidden_states = hidden_states + ple_out
            if new_cache_view is not None and getattr(new_cache_view, "ple_conv_state", None) is not None:
                new_cache_view = new_cache_view.replace(
                    ple_token_context=new_context,
                    ple_segment_context=new_segment_context,
                    ple_conv_state=new_ple_conv,
                )

        mixed, hyper_input, injection = self.attn_hyper_connection(hidden_states)
        if self.is_full_attention:
            attn_outputs = self.self_attn(
                mixed,
                mask_info,
                position_ids,
                mode,
                new_cache_view,
                cache_metadata,
                output_attentions,
                frequencies,
            )
        else:
            attn_outputs = self.linear_attn(
                mixed,
                mask_info,
                new_cache_view,
                cache_metadata,
            )
        new_cache_view = attn_outputs.cache_view
        hidden_states = inject_streams(hyper_input, attn_outputs.attention_output, injection)

        mixed, hyper_input, injection = self.mlp_hyper_connection(hidden_states)
        mlp_output, router_logits = self.mlp(mixed)
        hidden_states = inject_streams(hyper_input, mlp_output, injection)

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight if output_attentions else None,
            router_logits=router_logits if output_router_logits else None,
            cache_view=new_cache_view,
        )


# ---------------------------------------------------------------------------
# MTP head (mtp.* in the checkpoint; HF ignores it at load)
# ---------------------------------------------------------------------------


def _is_qwen4_training_call(mode, past_key_values, cache_metadata) -> bool:
    """True only for stateless/explicit training, never eSurge cached calls."""
    return mode == common_types.MODE_TRAIN or (mode is None and past_key_values is None and cache_metadata is None)


def _resolve_qwen4_runtime_mode(mode, sequence_length: int, has_cache: bool):
    """Resolve train, prefill, or decode without confusing eSurge metadata caches."""
    if mode is not None:
        return mode
    if not has_cache:
        return common_types.MODE_TRAIN
    return common_types.MODE_DECODE if sequence_length == 1 else common_types.MODE_PREFILL


def _resolve_qwen4_mtp_context(
    input_ids: Array | None,
    inputs_embeds: Array | None,
    attention_mask: Array | None,
    mask_info: MaskInfo | None,
    position_ids: Array | None,
) -> tuple[MaskInfo, Array]:
    """Reuse the main pass's padding/segment semantics for the MTP draft head."""
    resolved_mask = MaskInfo.dynamic_init(
        mask_info=mask_info,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )
    if position_ids is None:
        positions = resolved_mask.q_position_ids
        position_ids = jnp.broadcast_to(positions[None], (3, *positions.shape))
    elif position_ids.ndim == 2:
        position_ids = jnp.broadcast_to(position_ids[None], (3, *position_ids.shape))
    elif position_ids.shape[0] == 4:
        position_ids = position_ids[1:]
    return resolved_mask, position_ids


def _packed_mtp_next_ids(input_ids: Array, mask_info: MaskInfo) -> tuple[Array, Array | None]:
    """Shift MTP inputs by one without borrowing from the next packed document."""
    next_ids = jnp.concatenate([input_ids[:, 1:], jnp.zeros((input_ids.shape[0], 1), input_ids.dtype)], axis=-1)
    segments = getattr(mask_info, "_q_segment_ids", None)
    if segments is None:
        return next_ids, None
    if segments.ndim == 3:
        segments = segments[:, 0, :]
    next_segments = jnp.concatenate([segments[:, 1:], jnp.full((segments.shape[0], 1), -1, segments.dtype)], axis=-1)
    next_ids = jnp.where((segments >= 0) & (next_segments == segments), next_ids, 0)
    return next_ids, segments


@auto_pytree
class Qwen4ExpMTPOutput:
    """Output of the Qwen4-Exp MTP head."""

    last_hidden_state: Float[Array, "batch seq hidden"]
    past_key_values: Qwen4ExpCache | None = None


class Qwen4ExpMTPLayer(spx.Module):
    """Single MTP block: QSA attention + MoE wrapped in gated residuals."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the MTP block (full-attention hybrid layer, no PLE)."""
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.self_attn = Qwen4ExpAttention(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,  # MTP-local cache namespace
        )
        self.mlp = Qwen4ExpSparseMoeBlock(
            config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )
        hc_kwargs = dict(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            eps=config.rms_norm_eps,
            use_combine=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.attn_hyper_connection = GatedResidual(**hc_kwargs)
        self.mlp_hyper_connection = GatedResidual(**hc_kwargs)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hc*hidden"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
        cache_view: tp.Any | None = None,
        cache_metadata=None,
        frequencies: Array | None = None,
    ) -> tuple[Array, tp.Any]:
        """Run the MTP block over the residual streams."""
        mixed, hyper_input, injection = self.attn_hyper_connection(hidden_states)
        attn_outputs = self.self_attn(
            mixed, mask_info, position_ids, mode, cache_view, cache_metadata, False, frequencies
        )
        hidden_states = inject_streams(hyper_input, attn_outputs.attention_output, injection)
        mixed, hyper_input, injection = self.mlp_hyper_connection(hidden_states)
        mlp_output, _router = self.mlp(mixed)
        hidden_states = inject_streams(hyper_input, mlp_output, injection)
        return hidden_states, attn_outputs.cache_view


class Qwen4ExpMTPHead(spx.Module):
    """Qwen4-Exp multi-token-prediction head.

    Reconstructed from the checkpoint layout (HF ships the weights but no
    reference forward): the main model's *pre-collapse* streams are
    group-normed (``pre_fc_norm_hidden`` is ``hc * hidden`` wide), collapsed by
    a mean over streams, fused with the next token's embedding via two square
    projections, widened to ``hc_count`` streams, run through the hybrid MTP
    layer(s), and collapsed by the head's own ``hyper_connection_mixer``. The
    shared LM head is applied by the caller.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the MTP head."""
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        num_layers = int(getattr(config, "mtp_num_hidden_layers", 0))
        if num_layers < 1:
            raise ValueError("Qwen4ExpMTPHead requires mtp_num_hidden_layers >= 1.")
        self.num_mtp_layers = num_layers
        self.embed_tokens = (
            Embed(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                rngs=rngs,
            )
            if config.mtp_use_dedicated_embeddings
            else None
        )

        linear = dict(
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.fc_embedding = ColumnParallelLinear(config.hidden_size, config.hidden_size, **linear)
        self.fc_hidden = ColumnParallelLinear(config.hidden_size, config.hidden_size, **linear)
        self.pre_fc_norm_embedding = Qwen4ExpRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, param_dtype=param_dtype, rngs=rngs
        )
        self.pre_fc_norm_hidden = Qwen4ExpRMSNorm(
            config.hidden_size * config.hc_count,
            eps=config.rms_norm_eps,
            param_dtype=param_dtype,
            group_size=config.hidden_size,
            rngs=rngs,
        )
        self.hyper_connection_mixer = GatedResidual(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            eps=config.rms_norm_eps,
            use_combine=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        remat = auto_remat(
            Qwen4ExpMTPLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList(
            [
                remat(
                    config=config,
                    layer_idx=i,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for i in range(num_layers)
            ]
        )

    def init_cache(self, batch_size: int, max_length: int) -> Qwen4ExpCache:
        """Allocate an MTP-local all-QSA cache, independent of main layer types."""
        cfg = TransformerCacheConfig.create(
            batch_size=batch_size,
            sequence_length=max_length,
            num_hidden_layers=self.num_mtp_layers,
            num_heads=self.config.num_attention_heads,
            head_dim=self.config.head_dim,
            key_heads=self.config.num_key_value_heads,
            value_heads=self.config.num_key_value_heads,
            key_dim=self.config.head_dim,
            value_dim=self.config.head_dim,
        )
        max_blocks = (
            (max_length + self.config.indexer_compress_ratio - 1) // self.config.indexer_compress_ratio
            if self.config.qsa_enabled
            else 0
        )
        views = [
            Qwen4ExpQSAView.init(
                cfg,
                self.config.indexer_head_dim or 0,
                layer_index=i,
                indexer_max_blocks=max_blocks,
                dtype=self.dtype,
                mesh=self.config.mesh,
                runtime_sharding_resolver=self.config.runtime_sharding_resolver,
            )
            for i in range(self.num_mtp_layers)
        ]
        return Qwen4ExpCache(views=views)

    def forward(
        self,
        prev_stream_state: Float[Array, "batch seq hc*hidden"],
        next_token_embeds: Float[Array, "batch seq hidden"],
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: Qwen4ExpCache | None = None,
        cache_metadata=None,
        frequencies: Array | None = None,
    ) -> Qwen4ExpMTPOutput:
        """Fuse stream state + next-token embedding and run the MTP layers.

        Args:
            prev_stream_state: The main model's *pre-collapse* residual
                streams ``[B, S, hc * hidden]``.
            next_token_embeds: Embeddings of the ground-truth next tokens
                (caller shifts: ``embed(roll(input_ids, -1))``).
            mask_info: Reused causal mask info from the main pass.
            position_ids: mRoPE position rows from the main pass.
            mode: Runtime mode.
            past_key_values: MTP-local cache (separate namespace).
            cache_metadata: Cache metadata.
            frequencies: RoPE cache from the main model.

        Returns:
            :class:`Qwen4ExpMTPOutput` with the ``[B, S, hidden]`` head state.
        """
        h = self.pre_fc_norm_hidden(prev_stream_state)
        h = h.reshape(*h.shape[:-1], self.config.hc_count, self.config.hidden_size).mean(axis=-2)
        h = self.fc_hidden(h)
        e = self.fc_embedding(self.pre_fc_norm_embedding(next_token_embeds))
        hidden = checkpoint_name(h + e, "mtp_fused")

        streams = expand_streams(hidden, self.config.hc_count)
        views = past_key_values.views if past_key_values is not None else None
        new_views = []
        for i, layer in enumerate(self.layers):
            cv_in = views[i] if views is not None and i < len(views) else None
            streams, cv_out = layer(streams, mask_info, position_ids, mode, cv_in, cache_metadata, frequencies)
            new_views.append(cv_out)
        collapsed = self.hyper_connection_mixer(streams)
        return Qwen4ExpMTPOutput(
            last_hidden_state=checkpoint_name(collapsed, "mtp_output"),
            past_key_values=Qwen4ExpCache(views=new_views) if new_views else None,
        )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@auto_pytree
class Qwen4ExpTextModelOutputWithPast(ModelOutput):
    """Text-model output carrying the pre-collapse hyper-connection streams.

    Attributes:
        last_hidden_state: Collapsed hidden states ``[B, S, hidden]``.
        last_stream_state: Pre-collapse residual streams
            ``[B, S, hc * hidden]`` — the MTP head's input.
        rope_deltas: mRoPE position deltas for generation, ``[B, 1]``.
    """

    last_hidden_state: Float[Array, "batch seq hidden"] | None = None
    last_stream_state: Float[Array, "batch seq hc*hidden"] | None = None
    hidden_states: tuple | None = None
    attentions: tuple | None = None
    past_key_values: tp.Any = None
    router_logits: tuple | None = None
    rope_deltas: Int[Array, "batch 1"] | None = None


@auto_pytree
class Qwen4ExpCausalLMOutputWithPast(ModelOutput):
    """Causal-LM output with MoE router logits, MTP logits, and rope deltas."""

    logits: Float[Array, "batch seq vocab"] | None = None
    last_hidden_state: Float[Array, "batch seq hidden"] | None = None
    last_stream_state: Float[Array, "batch seq hc*hidden"] | None = None
    mtp_logits: Float[Array, "batch seq vocab"] | None = None
    mtp_loss: Array | None = None
    aux_loss: Array | None = None
    hidden_states: tuple | None = None
    attentions: tuple | None = None
    past_key_values: tp.Any = None
    router_logits: tuple | None = None
    rope_deltas: Int[Array, "batch 1"] | None = None
    loss: Array | None = None


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------


@register_module(TaskType.BASE_MODULE, config=Qwen4ExpTextConfig, model_type="qwen4_exp_text")
class Qwen4ExpTextModel(EasyDeLBaseModule):
    """Qwen4-Exp text decoder: hybrid GDN/QSA MoE over hyper-connection streams.

    There is no final norm: ``hyper_connection_mixer`` (a combine-less
    :class:`GatedResidual`) collapses the ``hc_count`` streams and *is* the
    output normalization.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the text model."""
        assert isinstance(config, Qwen4ExpTextConfig), (
            f"expected config to be of type Qwen4ExpTextConfig but got {type(config)}"
        )
        super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)

        with self.assign_layer_stage(0, total_layers=config.num_hidden_layers):
            self.embed_tokens = Embed(
                num_embeddings=config.vocab_size,
                features=config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
                rngs=rngs,
            )

        remat_layer = auto_remat(
            Qwen4ExpDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            with self.assign_layer_stage(i, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer(
                        config=config,
                        layer_idx=i,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.hyper_connection_mixer = GatedResidual(
                hidden_size=config.hidden_size,
                hc_count=config.hc_count,
                hc_lowrank=config.hc_lowrank,
                eps=config.rms_norm_eps,
                use_combine=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

    @cached_property
    def frequencies(self):
        """Precomputed base RoPE ``[cos | sin]`` cache over the partial rotary width.

        ``MultiModalRotaryEmbedding`` consumes this cache with shape
        ``[max_position, rotary_dim]`` holding the un-interleaved per-position
        cos/sin halves (it gathers one row per mrope axis and re-interleaves
        them itself), so the cache is built directly from
        ``compute_basic_inv_frequencies`` rather than through
        ``get_basic_frequencies`` whose width convention differs.
        """
        from easydel.layers.rotary._compute_fns import compute_basic_inv_frequencies

        config = self.config
        rotary_dim = config.rotary_dim
        max_pos = getattr(config, "granted_freq_max_position_embedding", None) or config.max_position_embeddings
        inv_freq = compute_basic_inv_frequencies(config.rope_theta, rotary_dim)
        positions = jnp.arange(max_pos, dtype=jnp.float32)
        angles = positions[:, None] * inv_freq[None, :]
        return jnp.concatenate((jnp.cos(angles), jnp.sin(angles)), axis=-1)

    def build_cache_configs(
        self,
        batch_size: int,
        max_length: int,
    ) -> tuple[TransformerCacheConfig, HybridCacheConfig, dict]:
        """Assemble the per-kind cache metadata for generation.

        Returns:
            ``(transformer_config, hybrid_config, extras)``: the QSA-layer KV
            cache config, the linear-layer GDN state config (whose
            ``head_dim``/``num_attention_heads``/``d_state`` carry the *linear*
            dims — the linear view init reads them), and the QSA/PLE extras
            for :meth:`Qwen4ExpCache.init_cache`.
        """
        config = self.config
        transformer_config = TransformerCacheConfig.create(
            batch_size=batch_size,
            sequence_length=max_length,
            num_hidden_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            key_heads=config.num_key_value_heads,
            value_heads=config.num_key_value_heads,
            key_dim=config.head_dim,
            value_dim=config.head_dim,
        )
        mapped_types = tuple("full_attention" if t == QWEN4_FULL else "linear_attention" for t in config.layer_types)
        hybrid_config = HybridCacheConfig.create(
            num_hidden_layers=config.num_hidden_layers,
            partition_axis=config.partition_axis,
            batch_size=batch_size,
            sequence_length=max_length,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.linear_key_head_dim,
            d_inner=config.linear_d_inner,
            d_conv=config.linear_conv_kernel_dim,
            d_state=config.linear_d_state,
            num_attention_heads=config.linear_num_value_heads,
            layer_types=mapped_types,
        )
        extras = dict(
            qsa_layers=tuple(i for i, t in enumerate(config.layer_types) if t == QWEN4_FULL),
            indexer_head_dim=config.indexer_head_dim if config.qsa_enabled else config.indexer_head_dim or 0,
            indexer_max_blocks=(
                (max_length + config.indexer_compress_ratio - 1) // config.indexer_compress_ratio
                if config.qsa_enabled
                else 0
            ),
            ple_layers=tuple(config.ple_layer_indices_0based),
            ple_conv_dim=config.hidden_size * config.hc_count,
            ple_conv_state_len=(config.ple_conv_kernel_size - 1) * config.ngram_size,
            ple_context_len=config.ngram_size - 1,
        )
        return transformer_config, hybrid_config, extras

    def init_cache(self, batch_size: int, max_length: int, **kwargs) -> "Qwen4ExpCache":
        """Allocate the generation cache (KV + GDN + indexer + PLE state)."""
        transformer_config, hybrid_config, extras = self.build_cache_configs(batch_size, max_length)
        return Qwen4ExpCache.init_cache(
            transformer_config=transformer_config,
            hybrid_config=hybrid_config,
            dtype=self.dtype,
            recurrent_dtype=jnp.dtype(self.config.mamba_ssm_dtype),
            mesh=self.config.mesh,
            runtime_sharding_resolver=self.config.runtime_sharding_resolver,
            **extras,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: Qwen4ExpCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        ple_input_ids: Int[Array, "batch seq_len"] | None = None,
    ) -> Qwen4ExpTextModelOutputWithPast:
        """Run the decoder stack over hyper-connection streams.

        Args:
            input_ids: Token ids ``[B, T]``.
            inputs_embeds: Precomputed embeddings (mutually exclusive with
                ``input_ids``); when PLE is active, token ids are still
                required via ``ple_input_ids``.
            attention_mask: Padding mask ``[B, T]``.
            mask_info: Advanced mask container.
            position_ids: ``[B, T]`` text positions or ``[3, B, T]`` mRoPE
                rows (from the VLM ``get_rope_index``).
            mode: Runtime mode; auto-detected when None.
            past_key_values: Hybrid cache (empty-initialized when None).
            cache_metadata: Cache metadata.
            output_attentions / output_hidden_states / output_router_logits:
                Output collection flags.
            ple_input_ids: Token ids for the PLE hash when ``inputs_embeds``
                are passed (the reference reverse-embeds; we require ids).

        Returns:
            :class:`Qwen4ExpTextModelOutputWithPast` with the collapsed hidden
            state and the pre-collapse streams.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify either input_ids or inputs_embeds, but not both.")

        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")

        batch_size, sequence_length = inputs_embeds.shape[:2]

        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached! (<= {self.config.max_position_embeddings}, got {sequence_length})"
        )

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # mRoPE rows for rotary/indexer: [3, B, T].
        if position_ids is None:
            pos_2d = mask_info.q_position_ids
            mrope_rows = jnp.broadcast_to(pos_2d[None, :, :], (3, batch_size, sequence_length))
        elif position_ids.ndim == 2:
            mrope_rows = jnp.broadcast_to(position_ids[None, :, :], (3, batch_size, sequence_length))
        elif position_ids.shape[0] == 4:
            mrope_rows = position_ids[1:]
        else:
            mrope_rows = position_ids

        mode = _resolve_qwen4_runtime_mode(
            mode,
            sequence_length,
            past_key_values is not None or cache_metadata is not None,
        )

        if past_key_values is None:
            past_key_values = Qwen4ExpCache.init_empty(len(self.layers))

        # PLE token ids: padding positions are replaced by EOS so n-grams
        # never read padding (reference behavior).
        ple_ids = None
        conv_mask = None
        ple_segment_ids = getattr(mask_info, "_q_segment_ids", None)
        if ple_segment_ids is not None and ple_segment_ids.ndim == 3:
            ple_segment_ids = ple_segment_ids[:, 0, :]
        if self.config.ple_layer_ids:
            ple_ids = ple_input_ids if ple_input_ids is not None else input_ids
            if ple_ids is None:
                raise ValueError(
                    "Qwen4-Exp PLE needs token ids: pass input_ids or ple_input_ids "
                    "(inputs_embeds-only forward is unsupported when ple_layer_ids is set)."
                )
            q_mask = getattr(mask_info, "q_attention_mask", None)
            if q_mask is not None and q_mask.ndim == 2:
                conv_mask = q_mask.astype(jnp.bool_)
                eos = self.config.eos_token_id
                eos = eos[0] if isinstance(eos, (list, tuple)) else eos
                ple_ids = jnp.where(conv_mask, ple_ids, jnp.full_like(ple_ids, eos))

        hidden_states = expand_streams(inputs_embeds, self.config.hc_count)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        views = past_key_values.views if past_key_values is not None else None
        frequencies = self.frequencies

        new_views: list = []
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=mrope_rows,
                    mode=mode,
                    cache_view=views[idx] if views is not None else None,
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    frequencies=frequencies,
                    ple_input_ids=ple_ids,
                    conv_mask=conv_mask,
                    segment_ids=ple_segment_ids,
                )
            hidden_states = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)
            if output_attentions:
                # Preserve one slot per decoder layer. Linear-attention layers
                # do not materialize an O(S^2) matrix and therefore report None.
                all_attentions += (layer_outputs.attention_weight,)
            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)
            if views is not None:
                new_views.append(layer_outputs.cache_view)

        if views is not None and any(v is not None for v in views):
            past_key_values = past_key_values.replace(views=new_views)

        collapsed = self.hyper_connection_mixer(hidden_states)
        collapsed = checkpoint_name(collapsed, "model_output")
        if output_hidden_states:
            all_hidden_states += (collapsed,)

        return Qwen4ExpTextModelOutputWithPast(
            last_hidden_state=collapsed,
            last_stream_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits if output_router_logits else None,
        )

    def get_encoder(self):
        """Decoder-only; raises."""
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self) -> "Qwen4ExpTextModel":
        """Return self (decoder-only)."""
        return self

    def get_lm_head(self):
        """Base model has no LM head; raises."""
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self) -> Embed:
        """Return the token embedding layer."""
        return self.embed_tokens


# ---------------------------------------------------------------------------
# Causal LM (text-only)
# ---------------------------------------------------------------------------


@register_module(TaskType.CAUSAL_LM, config=Qwen4ExpTextConfig, model_type="qwen4_exp_text")
class Qwen4ExpForCausalLM(BaseCausalLMModule[Qwen4ExpTextModel, Qwen4ExpTextConfig]):  # type: ignore
    """Qwen4-Exp text model with LM head (and the optional MTP head)."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "qwen4_exp_text"
    _config_class = Qwen4ExpTextConfig

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.bfloat16 | jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the causal LM wrapper."""
        super().__init__(
            config=config,
            base_model_class=Qwen4ExpTextModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )
        self.mtp = (
            Qwen4ExpMTPHead(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
            if int(getattr(config, "mtp_num_hidden_layers", 0)) > 0
            else None
        )

    def init_cache(self, batch_size: int, max_length: int, **kwargs):
        """Allocate the Qwen4-Exp hybrid cache via the text model."""
        return self.model.init_cache(batch_size, max_length, **kwargs)

    def get_operations_cache_view(self) -> dict[int, type]:
        views = super().get_operations_cache_view()
        for idx, layer_type in enumerate(self.config.layer_types):
            view = views.get(idx)
            if (
                self.config.qsa_enabled
                and layer_type == QWEN4_FULL
                and isinstance(view, type)
                and issubclass(view, RaggedPagesCacheView)
            ):
                views[idx] = Qwen4ExpPagedQSAView
            elif (
                self.config.qsa_enabled
                and layer_type == QWEN4_FULL
                and isinstance(view, type)
                and issubclass(view, UnifiedAttentionCacheView)
            ):
                raise NotImplementedError(
                    "Qwen4 QSA serving currently requires ragged-page attention; "
                    "unified/GPU paged caches do not preserve QSA indexer history"
                )
            elif idx in self.config.ple_layer_indices_0based:
                views[idx] = Qwen4ExpOperationsLinearView
        return views

    def init_operations_cache_config(self, *args, **kwargs):
        configs = super().init_operations_cache_config(*args, **kwargs)
        for idx, layer_type in enumerate(self.config.layer_types):
            if self.config.qsa_enabled and layer_type == QWEN4_FULL and idx < len(configs) and configs[idx] is not None:
                object.__setattr__(configs[idx], "qwen4_indexer_head_dim", self.config.indexer_head_dim)
            if idx in self.config.ple_layer_indices_0based and idx < len(configs) and configs[idx] is not None:
                object.__setattr__(configs[idx], "qwen4_with_ple", True)
                object.__setattr__(configs[idx], "qwen4_ple_conv_dim", self.config.hidden_size * self.config.hc_count)
                object.__setattr__(
                    configs[idx],
                    "qwen4_ple_conv_state_len",
                    (self.config.ple_conv_kernel_size - 1) * self.config.ngram_size,
                )
                object.__setattr__(configs[idx], "qwen4_ple_context_len", self.config.ngram_size - 1)
        return configs

    def compute_mtp_outputs(
        self,
        last_stream_state: Array,
        next_token_ids: Array,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        mode=None,
        past_key_values: Qwen4ExpCache | None = None,
    ) -> Qwen4ExpMTPOutput | None:
        """Run the MTP head on the main model's pre-collapse streams.

        Args:
            last_stream_state: ``Qwen4ExpTextModelOutputWithPast.last_stream_state``.
            next_token_ids: Ground-truth next tokens (``roll(input_ids, -1)``).
            mask_info: Reused mask info from the main pass.
            position_ids: Reused mRoPE rows from the main pass.
            mode: Runtime mode.

        Returns:
            The MTP output, or ``None`` when the head is disabled.
        """
        if self.mtp is None:
            return None
        embed_layer = self.mtp.embed_tokens or self.model.embed_tokens
        embeds = embed_layer(next_token_ids.astype("i4"))
        if mask_info is None:
            mask_info = MaskInfo.dynamic_init(mask_info=None, input_ids=next_token_ids.astype("i4"))
        if position_ids is None:
            # Default to contiguous text-only mRoPE rows so a bare MTP call
            # (no reused main-pass rows) still ropes at absolute positions.
            batch, seq_len = next_token_ids.shape
            rows = jnp.arange(seq_len, dtype="i4")[None, None, :]
            position_ids = jnp.broadcast_to(rows, (3, batch, seq_len))
        return self.mtp(
            last_stream_state,
            embeds,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            frequencies=self.model.frequencies,
            past_key_values=past_key_values,
        )

    def apply_mtp_lm_head(self, mtp_output: Qwen4ExpMTPOutput) -> Array:
        """Project the MTP hidden state with the shared LM head."""
        return self.apply_lm_head(mtp_output.last_hidden_state)

    @staticmethod
    def compute_mtp_loss(
        mtp_logits: Array,
        labels: Array,
        attention_mask: Array | None = None,
        segment_ids: Array | None = None,
    ) -> Array:
        """Cross entropy for ``labels[t + 2]``, without crossing packed documents."""
        batch = labels.shape[0]
        shifted = jnp.concatenate([labels[:, 2:], jnp.full((batch, 2), -100, labels.dtype)], axis=-1)
        if segment_ids is not None:
            target_segments = jnp.concatenate([segment_ids[:, 2:], jnp.full((batch, 2), -1, segment_ids.dtype)], axis=-1)
            shifted = jnp.where((segment_ids >= 0) & (target_segments == segment_ids), shifted, -100)
        if attention_mask is not None:
            target_mask = jnp.concatenate([attention_mask[:, 2:], jnp.zeros((batch, 2), attention_mask.dtype)], axis=-1)
            shifted = jnp.where(target_mask.astype(jnp.bool_), shifted, -100)
        log_probs = jax.nn.log_softmax(mtp_logits.astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(log_probs, jnp.maximum(shifted, 0)[..., None], axis=-1).squeeze(-1)
        valid = (shifted != -100).astype(jnp.float32)
        return jnp.sum(nll * valid) / jnp.maximum(jnp.sum(valid), 1.0)

    def forward(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,
        past_key_values=None,
        cache_metadata=None,
        labels: Array | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> Qwen4ExpCausalLMOutputWithPast:
        """Causal-LM forward with router and MTP training losses exposed."""
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        base = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
        )
        logits = None
        if apply_lm_head:
            logits = self.compute_lm_logits(self.prepare_lm_head_inputs(base.last_hidden_state))
        aux_loss = self.compute_router_aux_loss(base)
        mtp_logits = None
        mtp_loss = None
        training = _is_qwen4_training_call(mode, past_key_values, cache_metadata)
        coef = float(getattr(self.config, "mtp_loss_coef", 0.0))
        if self.mtp is not None and input_ids is not None and training and coef > 0:
            mtp_mask_info, mtp_position_ids = _resolve_qwen4_mtp_context(
                input_ids, inputs_embeds, attention_mask, mask_info, position_ids
            )
            next_ids, mtp_segments = _packed_mtp_next_ids(input_ids, mtp_mask_info)
            mtp_out = self.compute_mtp_outputs(
                base.last_stream_state,
                next_ids,
                mask_info=mtp_mask_info,
                position_ids=mtp_position_ids,
                mode=mode,
            )
            mtp_logits = self.apply_mtp_lm_head(mtp_out)
            mtp_targets = input_ids if labels is None else labels
            mtp_loss = self.compute_mtp_loss(mtp_logits, mtp_targets, attention_mask, mtp_segments) * coef
            aux_loss = mtp_loss if aux_loss is None else aux_loss + mtp_loss
        return Qwen4ExpCausalLMOutputWithPast(
            logits=logits,
            last_hidden_state=base.last_hidden_state,
            last_stream_state=base.last_stream_state,
            mtp_logits=mtp_logits,
            mtp_loss=mtp_loss,
            aux_loss=aux_loss,
            hidden_states=base.hidden_states,
            attentions=base.attentions,
            past_key_values=base.past_key_values,
            router_logits=base.router_logits,
            rope_deltas=base.rope_deltas,
        )


# ---------------------------------------------------------------------------
# Multimodal model
# ---------------------------------------------------------------------------


class Qwen4ExpVisionTransformer(Qwen3VisionTransformerPretrainedModel):
    """Qwen4-Exp vision tower — the shared Qwen3-VL tower with the Qwen4 config."""

    config_class = Qwen4ExpVisionConfig


@register_module(TaskType.BASE_MODULE, config=Qwen4ExpConfig, model_type="qwen4_exp")
@register_module(TaskType.VISION_LM, config=Qwen4ExpConfig, model_type="qwen4_exp")
class Qwen4ExpModel(EasyDeLBaseModule):
    """Qwen4-Exp multimodal model: vision tower + hybrid text decoder.

    Checkpoint layout: ``model.visual.*`` / ``model.language_model.*``.
    """

    def __init__(
        self,
        config: Qwen4ExpConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the multimodal model."""
        super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.visual = (
            None
            if config.language_model_only
            else Qwen4ExpVisionTransformer(
                config.vision_config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
            )
        )
        self.language_model = Qwen4ExpTextModel(
            config.text_config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )

    def get_input_embeddings(self):
        """Return the token embedding layer."""
        return self.language_model.get_embedding()

    def set_input_embeddings(self, value):
        """Set the token embedding layer."""
        self.language_model.embed_tokens = value

    def set_decoder(self, decoder):
        """Set the text decoder."""
        self.language_model = decoder

    def get_decoder(self):
        """Return the text decoder."""
        return self.language_model

    def get_encoder(self):
        """Return the vision encoder."""
        return self.visual

    def get_rope_index(
        self,
        input_ids: Array,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        attention_mask: Array | None = None,
        mm_token_type_ids: Array | None = None,
    ) -> tuple[Array, Array]:
        """Compute the 3-row mRoPE position ids for mixed-modality input.

        Reference semantics (Qwen4-Exp): videos carry per-frame timestamp
        separators, so each video grid is first split into per-frame grids;
        grouping is by modality token type (0=text, 1=image, 2=video).

        Args:
            input_ids: Token ids ``[B, T]``.
            image_grid_thw: Image grids ``[n_images, 3]``.
            video_grid_thw: Video grids ``[n_videos, 3]``.
            attention_mask: Padding mask.
            mm_token_type_ids: Per-token modality ids; derived from the
                placeholder tokens when not given.

        Returns:
            ``(position_ids, mrope_position_deltas)`` — ``[3, B, T]`` and
            ``[B, 1]``.
        """
        if image_grid_thw is None and video_grid_thw is None:
            if attention_mask is None:
                pos = jnp.broadcast_to(jnp.arange(input_ids.shape[1], dtype=jnp.int32)[None, :], input_ids.shape)
            else:
                valid = attention_mask.astype(jnp.bool_)
                pos = jnp.maximum(jnp.cumsum(valid, axis=-1, dtype=jnp.int32) - 1, 0)
                pos = jnp.where(valid, pos, 0)
            return jnp.broadcast_to(pos[None, :, :], (3, *pos.shape)), jnp.zeros((input_ids.shape[0], 1), jnp.int32)

        video_grid = None
        if video_grid_thw is not None:
            video_grid = np.asarray(video_grid_thw)
            # Timestamps separate frames: split the video into per-frame grids.
            video_grid = np.repeat(video_grid, video_grid[:, 0], axis=0)
            video_grid[:, 0] = 1
        if mm_token_type_ids is None:
            ids = np.asarray(input_ids)
            mm_token_type_ids = np.zeros_like(ids)
            mm_token_type_ids[ids == self.config.image_token_id] = 1
            mm_token_type_ids[ids == self.config.video_token_id] = 2
        return _get_rope_index_from_mm_token_types(
            input_ids=np.asarray(input_ids),
            mm_token_type_ids=np.asarray(mm_token_type_ids),
            image_grid_thw=np.asarray(image_grid_thw) if image_grid_thw is not None else None,
            video_grid_thw=video_grid,
            attention_mask=np.asarray(attention_mask) if attention_mask is not None else None,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
        )

    def get_image_features(self, pixel_values, image_grid_thw=None, image_max_grid_size=None):
        """Encode images through the vision tower.

        Args:
            pixel_values: Flattened image patches.
            image_grid_thw: Temporal/height/width grid for each image.
            image_max_grid_size: Optional compile-time grid bound.

        Returns:
            Pooled image features.
        """
        if self.visual is None:
            raise ValueError("Image inputs are unavailable when language_model_only=True.")
        # The shared Qwen3 vision tower returns (merged_features, deepstack).
        # Qwen4 uses the merged features; its vision config has no DeepStack.
        features, _deepstack = self.visual(pixel_values, grid_thw=image_grid_thw, max_grid_size=image_max_grid_size)
        return features

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, video_max_grid_size=None):
        """Encode videos through the vision tower.

        Args:
            pixel_values_videos: Flattened video-frame patches.
            video_grid_thw: Temporal/height/width grid for each video.
            video_max_grid_size: Optional compile-time grid bound.

        Returns:
            Pooled video features.
        """
        if self.visual is None:
            raise ValueError("Video inputs are unavailable when language_model_only=True.")
        features, _deepstack = self.visual(
            pixel_values_videos, grid_thw=video_grid_thw, max_grid_size=video_max_grid_size
        )
        return features

    def get_placeholder_mask(self, input_ids, inputs_embeds=None, image_features=None, video_features=None):
        """Boolean placeholder mask for image/video token positions."""
        mask = jnp.zeros(input_ids.shape, jnp.bool_)
        if image_features is not None:
            mask |= input_ids == self.config.image_token_id
        if video_features is not None:
            mask |= input_ids == self.config.video_token_id
        return mask

    def compute_embedding(
        self,
        input_ids,
        *,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        image_max_grid_size=None,
        video_max_grid_size=None,
        image_embeds=None,
        video_embeds=None,
        **kwargs,
    ):
        """Compute text embeddings with visual features merged at placeholders."""
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")
            inputs_embeds = self.language_model.embed_tokens(input_ids.astype("i4"))

        if input_ids is None and (image_embeds is not None or video_embeds is not None):
            raise ValueError("`input_ids` must be provided to merge multimodal embeddings.")

        if image_embeds is None and pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, image_max_grid_size)
            if isinstance(image_embeds, tuple):
                image_embeds = jnp.concatenate(list(image_embeds), axis=0)
        if image_embeds is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=image_embeds.astype(inputs_embeds.dtype),
                placeholder_token_id=self.config.image_token_id,
            )

        if video_embeds is None and pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, video_max_grid_size)
            if isinstance(video_embeds, tuple):
                video_embeds = jnp.concatenate(list(video_embeds), axis=0)
        if video_embeds is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=video_embeds.astype(inputs_embeds.dtype),
                placeholder_token_id=self.config.video_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        past_key_values: Qwen4ExpCache | None = None,
        inputs_embeds: Array | None = None,
        pixel_values: Array | None = None,
        pixel_values_videos: Array | None = None,
        image_embeds: Array | None = None,
        video_embeds: Array | None = None,
        image_grid_thw: Array | None = None,
        video_grid_thw: Array | None = None,
        rope_deltas: Array | None = None,
        mask_info: MaskInfo | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        cache_metadata=None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        **kwargs,
    ) -> Qwen4ExpTextModelOutputWithPast:
        """Multimodal forward: embed (with vision merge) → text decoder.

        Position ids follow the reference: when multimodal grids are present,
        the 3-row mRoPE ids from :meth:`get_rope_index` are used and the
        deltas are tracked for generation.
        """
        has_multimodal_input = any(
            value is not None
            for value in (
                pixel_values,
                pixel_values_videos,
                image_embeds,
                video_embeds,
                image_grid_thw,
                video_grid_thw,
            )
        )
        decoder_input_ids = input_ids
        ple_input_ids = None
        if has_multimodal_input:
            if input_ids is None:
                raise ValueError("input_ids are required for multimodal placeholder merging and PLE.")
            inputs_embeds = self.compute_embedding(
                input_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_embeds=image_embeds,
                video_embeds=video_embeds,
                image_max_grid_size=kwargs.pop("image_max_grid_size", None),
                video_max_grid_size=kwargs.pop("video_max_grid_size", None),
            )
            # The decoder's exclusive input contract requires embeddings only;
            # PLE still consumes the original token IDs through its dedicated path.
            decoder_input_ids = None
            ple_input_ids = input_ids

        if position_ids is None and (image_grid_thw is not None or video_grid_thw is not None):
            if input_ids is None:
                raise ValueError("input_ids are required to derive multimodal RoPE positions.")
            if isinstance(input_ids, jax.core.Tracer):
                raise ValueError(
                    "Compiled multimodal Qwen4-Exp forward requires host-precomputed position_ids; "
                    "call get_rope_index before jax.jit and pass the returned rows."
                )
            position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)

        outputs = self.language_model(
            input_ids=decoder_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            ple_input_ids=ple_input_ids,
        )
        # RoPE deltas are explicit call/output state; mutating the module leaks
        # one request's multimodal positions into unrelated batches under JIT.
        outputs.rope_deltas = rope_deltas
        return outputs


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Qwen4ExpConfig, model_type="qwen4_exp")
class Qwen4ExpForConditionalGeneration(BaseVisionLanguageModule[Qwen4ExpModel, Qwen4ExpConfig]):
    """Qwen4-Exp for conditional generation (vision + text + LM head + MTP).

    Checkpoint layout: ``model.{visual,language_model}.*``, ``lm_head.*``,
    ``mtp.*``.
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "qwen4_exp"
    _config_class = Qwen4ExpConfig
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    _vision_tower_name = "visual"
    _projector_name = "merger"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Qwen4ExpConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the conditional-generation wrapper."""
        super().__init__(
            config=config,
            base_model_class=Qwen4ExpModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
            image_token_index=config.image_token_id,
            video_token_index=config.video_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
            router_aux_loss_coef=getattr(config.text_config, "router_aux_loss_coef", 0.001),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )
        self.vocab_size = config.text_config.vocab_size
        self.mtp = (
            Qwen4ExpMTPHead(config.text_config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
            if int(getattr(config.text_config, "mtp_num_hidden_layers", 0)) > 0
            else None
        )

    def get_input_embeddings(self):
        """Return the token embedding layer."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set the token embedding layer."""
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        """Set the text decoder."""
        self.model.set_decoder(decoder)

    def get_decoder(self):
        """Return the text decoder."""
        return self.model.get_decoder()

    @property
    def visual(self):
        """Return the vision tower."""
        return self.model.visual

    @property
    def language_model(self):
        """Return the text decoder."""
        return self.model.language_model

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, video_max_grid_size=None):
        """Encode videos via the model.

        Args:
            pixel_values_videos: Flattened video-frame patches.
            video_grid_thw: Temporal/height/width grid for each video.
            video_max_grid_size: Optional compile-time grid bound.

        Returns:
            Pooled video features.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, video_max_grid_size)

    def get_image_features(self, pixel_values, image_grid_thw=None, image_max_grid_size=None):
        """Encode images via the model.

        Args:
            pixel_values: Flattened image patches.
            image_grid_thw: Temporal/height/width grid for each image.
            image_max_grid_size: Optional compile-time grid bound.

        Returns:
            Pooled image features.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, image_max_grid_size)

    def compute_embedding(self, input_ids, *args, **kwargs):
        """Compute embeddings with multimodal fusion; delegates to the model."""
        return self.model.compute_embedding(input_ids, *args, **kwargs)

    def init_cache(self, batch_size: int, max_length: int, **kwargs):
        """Allocate the Qwen4-Exp hybrid cache via the text model."""
        return self.model.language_model.init_cache(batch_size, max_length, **kwargs)

    def get_operations_cache_view(self) -> dict[int, type]:
        views = super().get_operations_cache_view()
        for idx, layer_type in enumerate(self.config.text_config.layer_types):
            view = views.get(idx)
            if (
                self.config.text_config.qsa_enabled
                and layer_type == QWEN4_FULL
                and isinstance(view, type)
                and issubclass(view, RaggedPagesCacheView)
            ):
                views[idx] = Qwen4ExpPagedQSAView
            elif (
                self.config.text_config.qsa_enabled
                and layer_type == QWEN4_FULL
                and isinstance(view, type)
                and issubclass(view, UnifiedAttentionCacheView)
            ):
                raise NotImplementedError(
                    "Qwen4 QSA serving currently requires ragged-page attention; "
                    "unified/GPU paged caches do not preserve QSA indexer history"
                )
            elif idx in self.config.text_config.ple_layer_indices_0based:
                views[idx] = Qwen4ExpOperationsLinearView
        return views

    def init_operations_cache_config(self, *args, **kwargs):
        configs = super().init_operations_cache_config(*args, **kwargs)
        text = self.config.text_config
        for idx, layer_type in enumerate(text.layer_types):
            if text.qsa_enabled and layer_type == QWEN4_FULL and idx < len(configs) and configs[idx] is not None:
                object.__setattr__(configs[idx], "qwen4_indexer_head_dim", text.indexer_head_dim)
            if idx in text.ple_layer_indices_0based and idx < len(configs) and configs[idx] is not None:
                object.__setattr__(configs[idx], "qwen4_with_ple", True)
                object.__setattr__(configs[idx], "qwen4_ple_conv_dim", text.hidden_size * text.hc_count)
                object.__setattr__(
                    configs[idx], "qwen4_ple_conv_state_len", (text.ple_conv_kernel_size - 1) * text.ngram_size
                )
                object.__setattr__(configs[idx], "qwen4_ple_context_len", text.ngram_size - 1)
        return configs

    def compute_mtp_outputs(
        self,
        last_stream_state: Array,
        next_token_ids: Array,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        mode=None,
        past_key_values: Qwen4ExpCache | None = None,
    ) -> Qwen4ExpMTPOutput | None:
        """Run the MTP head on the main model's pre-collapse streams.

        Args:
            last_stream_state: Pre-collapse streams from the main forward.
            next_token_ids: Ground-truth next tokens (caller shifts).
            mask_info: Reused mask info.
            position_ids: Reused mRoPE rows.
            mode: Runtime mode.

        Returns:
            The MTP output, or ``None`` when the head is disabled.
        """
        if self.mtp is None:
            return None
        embed_layer = self.mtp.embed_tokens or self.model.language_model.embed_tokens
        embeds = embed_layer(next_token_ids.astype("i4"))
        if mask_info is None:
            mask_info = MaskInfo.dynamic_init(mask_info=None, input_ids=next_token_ids.astype("i4"))
        if position_ids is None:
            batch, seq_len = next_token_ids.shape
            rows = jnp.arange(seq_len, dtype="i4")[None, None, :]
            position_ids = jnp.broadcast_to(rows, (3, batch, seq_len))
        return self.mtp(
            last_stream_state,
            embeds,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            frequencies=self.model.language_model.frequencies,
            past_key_values=past_key_values,
        )

    def compute_mtp_logits(self, mtp_output: Qwen4ExpMTPOutput) -> Array:
        """Project the MTP hidden state with the shared LM head."""
        return self.apply_lm_head(mtp_output.last_hidden_state)

    def forward(self, *args, **kwargs):
        """Multimodal forward with MTP loss folded into the trainer aux channel."""
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask")
        labels = kwargs.pop("labels", None)
        mode = kwargs.get("mode")
        if (
            kwargs.get("position_ids") is None
            and input_ids is not None
            and (kwargs.get("image_grid_thw") is not None or kwargs.get("video_grid_thw") is not None)
            and not isinstance(input_ids, jax.core.Tracer)
        ):
            kwargs["position_ids"], kwargs["rope_deltas"] = self.model.get_rope_index(
                input_ids,
                kwargs.get("image_grid_thw"),
                kwargs.get("video_grid_thw"),
                attention_mask,
            )
        outputs = super().forward(*args, **kwargs)
        training = _is_qwen4_training_call(mode, kwargs.get("past_key_values"), kwargs.get("cache_metadata"))
        coef = float(getattr(self.config.text_config, "mtp_loss_coef", 0.0))
        if self.mtp is None or input_ids is None or not training or coef <= 0:
            return outputs
        if outputs.last_stream_state is None:
            return outputs
        mtp_mask_info, mtp_position_ids = _resolve_qwen4_mtp_context(
            input_ids,
            kwargs.get("inputs_embeds"),
            attention_mask,
            kwargs.get("mask_info"),
            kwargs.get("position_ids"),
        )
        next_ids, mtp_segments = _packed_mtp_next_ids(input_ids, mtp_mask_info)
        mtp_out = self.compute_mtp_outputs(
            outputs.last_stream_state,
            next_ids,
            mask_info=mtp_mask_info,
            position_ids=mtp_position_ids,
            mode=mode,
        )
        mtp_logits = self.compute_mtp_logits(mtp_out)
        mtp_targets = input_ids if labels is None else labels
        mtp_loss = Qwen4ExpForCausalLM.compute_mtp_loss(mtp_logits, mtp_targets, attention_mask, mtp_segments) * coef
        aux_loss = mtp_loss if outputs.aux_loss is None else outputs.aux_loss + mtp_loss
        return outputs.replace(aux_loss=aux_loss, mtp_logits=mtp_logits, mtp_loss=mtp_loss)


__all__ = [
    "Qwen4ExpAttention",
    "Qwen4ExpCache",
    "Qwen4ExpCausalLMOutputWithPast",
    "Qwen4ExpConfig",
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpGatedDeltaNet",
    "Qwen4ExpLinearView",
    "Qwen4ExpMLP",
    "Qwen4ExpMLPStack",
    "Qwen4ExpMTPHead",
    "Qwen4ExpMTPLayer",
    "Qwen4ExpMTPOutput",
    "Qwen4ExpModel",
    "Qwen4ExpPLELayer",
    "Qwen4ExpQSAView",
    "Qwen4ExpRMSNorm",
    "Qwen4ExpSparseMoeBlock",
    "Qwen4ExpTextConfig",
    "Qwen4ExpTextModel",
    "Qwen4ExpTextModelOutputWithPast",
    "Qwen4ExpVisionConfig",
    "Qwen4ExpVisionTransformer",
]
