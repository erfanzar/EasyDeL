# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""vLLM-style Unified (paged) Attention operation.

This operation wraps ejkernel's Triton implementation of vLLM's unified
attention kernel (paged KV cache + ragged queries).

Notes:
    - Inference-oriented: the underlying kernel is causal-only.
    - Cache *updates* are handled by the cache view's `concatenate_to_cache(...)`
      (called from `AttentionModule.concatenate(...)`). This op assumes the KV
      cache already contains the current step's K/V for the query tokens.
"""

from __future__ import annotations

import jax
from eformer import common_types as ct
from ejkernel.modules import UnifiedAttentionConfig, unified_attention
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Float

from easydel.layers.caching import OperationsMetadata, RaggedPagesMetadata, unwrap_metadata
from easydel.layers.caching.unified_attention import UnifiedAttentionCacheView

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import CacheType, ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder


@OperationRegistry.register
class UnifiedAttn(OperationImpl):
    """
    Attention implementation using vLLM-style Unified (Paged) Attention mechanism with Triton kernels.

    This class provides a GPU-optimized attention mechanism for scenarios where the
    Key-Value cache is managed in non-contiguous pages (Paged KV Cache). It leverages
    Triton kernels for efficient execution on GPUs, handling ragged queries with paged KV cache.

    The implementation is inference-oriented and causal-only. Cache updates are handled
    separately by the cache view's `concatenate_to_cache()` method (called from
    `AttentionModule.concatenate()`). This operation assumes the KV cache already
    contains the current step's K/V for the query tokens.

    Attributes:
        metadata (OperationMetadata): Configuration metadata for the attention mechanism.
            While this class uses `OperationMetadata`, it primarily relies on the
            additional `RaggedPagesMetadata` passed during the forward call for
            paged-specific information.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name for this attention implementation.

        Returns:
            str | tuple[str]: The name "unified_attention".
        """
        return "unified_attention"

    def get_impl_metadata(self) -> OperationMetadata:
        """
        Retrieves the metadata associated with this attention implementation instance.

        Returns:
            OperationMetadata: The metadata object provided during initialization.
        """
        return self.metadata

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """
        Returns the operation requirements for UnifiedAttention.

        Specifies the required and optional metadata fields, supported cache types,
        and cache view types needed for this operation to function correctly.

        Args:
            mode: The execution mode (PREFILL, DECODE, or MIXED). Defaults to MIXED.

        Returns:
            OperationRequirements: Requirements specification including:
                - Required metadata: SEQ_LENS, CONTEXT_LENS, POSITIONS, QUERY_START_LOC,
                  PAGES_TABLES, SLOT_MAPPING
                - Optional metadata: LOGITS_INDICES
                - Supported cache type: RAGGED_PAGES
                - Cache view type: UnifiedAttentionCacheView
        """
        return (
            RequirementsBuilder("unified_attention")
            .require_metadata(
                MetadataField.SEQ_LENS
                | MetadataField.CONTEXT_LENS
                | MetadataField.POSITIONS
                | MetadataField.QUERY_START_LOC
                | MetadataField.PAGES_TABLES
                | MetadataField.SLOT_MAPPING
            )
            .optional_metadata(MetadataField.LOGITS_INDICES)
            .support_cache(CacheType.RAGGED_PAGES)
            .use_cache_view(UnifiedAttentionCacheView)
            .build()
        )

    def forward_native(
        self,
        query: Float[Array, "... num_q_heads head_dim"],
        cache_view: UnifiedAttentionCacheView,
        cache_metadata: RaggedPagesMetadata | OperationsMetadata,
        softmax_scale: float | None = None,
        causal: bool = True,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        **ignore,
    ) -> AttentionOutput:
        """
        Native forward pass for unified paged attention using Triton kernels.

        This implementation handles attention with a paged KV cache stored in non-contiguous
        memory pages. It uses the `unified_attention` Triton kernel which efficiently processes
        variable-length sequences with page table lookups in a ragged format.

        The method automatically handles query tensor reshaping from 4D [batch, seq_len, num_heads, head_dim]
        to ragged 3D format [total_tokens, num_heads, head_dim] for kernel processing, then
        restores the original shape in the output.

        Args:
            query: Query tensor [..., num_q_heads, head_dim]. Can be 3D [total_tokens, num_q_heads, head_dim]
                in ragged format, or 4D [batch, seq_len, num_q_heads, head_dim].
                If 4D, will be reshaped to ragged format internally.
            cache_view: Unified attention cache view containing:
                - key_cache: Paged key tensors [num_pages, page_size, num_kv_heads, head_dim].
                - value_cache: Paged value tensors [num_pages, page_size, num_kv_heads, head_dim].
            cache_metadata: Metadata for paged cache, either RaggedPagesMetadata or OperationsMetadata
                (will be unwrapped to RaggedPagesMetadata). Includes:
                - context_lens: Length of each sequence [num_seqs].
                - pages_tables: Page table for cache access [num_seqs, max_pages].
                - query_start_loc: Starting index for each sequence [num_seqs + 1].
            softmax_scale: Scaling factor for attention logits. Defaults to 1/sqrt(head_dim) if None.
            causal: Whether to apply causal masking. Defaults to True.
            sliding_window: Sliding window size for local attention. If None, uses full attention.
            logits_soft_cap: Soft capping value for attention logits to prevent extreme values. Optional.
            softmax_aux: Auxiliary softmax tensor for sink tokens. Can be:
                - [num_kv_heads, num_sinks]: Per-head sink tokens.
                - [num_sinks]: Shared sink tokens across heads.
                - None: No sink tokens.
            **ignore: Additional ignored keyword arguments.

        Returns:
            AttentionOutput: Contains:
                - attention_outputs: Attention output tensor matching input query shape.
                - attention_weights: None (not computed).
                - cache_view: Updated cache view (unchanged for this operation).

        Raises:
            NotImplementedError: If the JAX backend is not GPU (Triton kernels require GPU).
            ValueError: If cache_view is None.

        Note:
            - This operation requires a GPU backend with Triton support.
            - The KV cache must already contain the current step's K/V values for query tokens.
            - Cache updates are handled separately via `cache_view.concatenate_to_cache()`.
        """
        if jax.default_backend() != "gpu":
            raise NotImplementedError("UnifiedAttn currently requires a GPU backend (Triton).")

        if cache_view is None:
            raise ValueError("UnifiedAttn requires `cache_view` (paged KV cache).")

        # Unwrap OperationsMetadata -> RaggedPagesMetadata (reuses the same runtime fields).
        cache_metadata = unwrap_metadata(cache_metadata, "ragged")
        kv_pages = cache_view.key_cache
        manager = self.metadata.partition_manager
        resolve = manager.resolve

        sinks_axis = None

        if softmax_aux is not None:
            sinks_axis = resolve(axes=[ct.HEAD], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)
            softmax_aux = softmax_aux.astype("f4")

        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=query.shape)

        kv_pages_spec = resolve(
            axes=[ct.EMPTY, ct.EMPTY, ct.KV_HEAD, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )

        query_in = query
        orig_is_4d = query_in.ndim == 4
        if orig_is_4d:
            batch, seqlen, num_heads, head_dim = query_in.shape
        query_ragged = query_in.reshape(-1, *query_in.shape[-2:])

        cfg = self.metadata.get_operation_config("unified_attention")
        if cfg is None:
            cfg = UnifiedAttentionConfig(
                num_warps=8,
                num_stages=2,
                num_par_softmax_segments=4,
                seq_threshold_3d=64,
            )
        extras = dict(
            softmax_scale=softmax_scale,
            causal=causal,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            in_specs=(qaxes, kv_pages_spec, kv_pages_spec, Ps(), Ps(), Ps(), None, None, sinks_axis),
            out_specs=qaxes,
            mesh=self.metadata.mesh,
            platform="triton",
            cfg=cfg,
        )
        out = unified_attention(
            query_ragged.astype(self.metadata.runtime_dtype),
            cache_view.key_cache,
            cache_view.value_cache,
            cache_metadata.context_lens.astype(jnp.int32),
            cache_metadata.pages_tables.astype(jnp.int32),
            cache_metadata.query_start_loc.astype(jnp.int32),
            None,
            None,
            softmax_aux,
            **extras,
        )

        if orig_is_4d:
            out = out.reshape(batch, seqlen, num_heads, head_dim)

        return AttentionOutput(attention_weights=None, attention_outputs=out, cache_view=cache_view)
