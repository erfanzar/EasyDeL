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
from ejkernel.modules import unified_attention
from jax import numpy as jnp
from jaxtyping import Array, Float

from easydel.layers.caching import OperationsMetadata, RaggedPagesMetadata, unwrap_metadata
from easydel.layers.caching.unified_attention import UnifiedAttentionCacheView

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import CacheType, ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder


@OperationRegistry.register
class UnifiedAttn(OperationImpl):
    """UnifiedAttention (paged) operation backed by Triton."""

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        return "unified_attention"

    def get_impl_metadata(self) -> OperationMetadata:
        return self.metadata

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
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
        **ignore,
    ) -> AttentionOutput:
        if jax.default_backend() != "gpu":
            raise NotImplementedError("UnifiedAttn currently requires a GPU backend (Triton).")

        if cache_view is None:
            raise ValueError("UnifiedAttn requires `cache_view` (paged KV cache).")

        # Unwrap OperationsMetadata -> RaggedPagesMetadata (reuses the same runtime fields).
        cache_metadata = unwrap_metadata(cache_metadata, "ragged")

        # Flatten to ragged `[total_tokens, heads, dim]` (works for `[B,T,H,D]` and `[B,H,D]`).
        query_in = query
        orig_is_4d = query_in.ndim == 4
        if orig_is_4d:
            batch, seqlen, num_heads, head_dim = query_in.shape
        query_ragged = query_in.reshape(-1, *query_in.shape[-2:])

        cfg = self.metadata.get_operation_config("unified_attention")
        out = unified_attention(
            query_ragged.astype(self.metadata.runtime_dtype),
            cache_view.key_cache,
            cache_view.value_cache,
            cache_metadata.context_lens.astype(jnp.int32),
            cache_metadata.pages_tables.astype(jnp.int32),
            cache_metadata.query_start_loc.astype(jnp.int32),
            softmax_scale=softmax_scale,
            causal=causal,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            platform="triton",
            cfg=cfg,
        )

        if orig_is_4d:
            out = out.reshape(batch, seqlen, num_heads, head_dim)

        return AttentionOutput(attention_weights=None, attention_outputs=out, cache_view=cache_view)
