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

"""Paged Flash Attention for CUDA with paged KV cache support.

This implementation uses ejkernel's Flash Attention with block tables to
address paged KV caches. It is CUDA-only and intended for fixed-length
batches (no varlen cu_seqlens). The KV cache layout matches
UnifiedAttentionCacheView (separate K/V pages).
"""

from __future__ import annotations

import jax
from eformer import common_types as ct
from eformer.escale import with_sharding_constraint
from ejkernel.modules import flash_attention
from ejkernel.types import MaskInfo
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Float

from easydel.axis import ATTN_DP
from easydel.caching import OperationsMetadata, RaggedPagesMetadata, UnifiedAttentionCacheView, unwrap_metadata
from easydel.utils.helpers import check_bool_flag

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import CacheType, ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder

ENABLE_DP_LOCAL_PAGE_PATH = check_bool_flag("EASURGE_ENABLE_DP_LOCAL_PAGE_PATH", default=True)


def _dp_page_axis(cache_view: UnifiedAttentionCacheView):
    """Resolve the logical page axis for the active cache view."""
    dp_size = max(1, int(getattr(cache_view.metadata, "data_parallel_size", 1)))
    return ATTN_DP if dp_size > 1 else ct.EMPTY


def _localize_block_tables_for_dp_pages(
    block_tables: Array,
    *,
    num_pages: int,
    dp_size: int,
) -> Array:
    """Translate global page IDs into DP-local page IDs per request row.

    Assumes rows are assigned contiguously per DP shard and page IDs are from
    a globally indexed page pool partitioned evenly over DP shards.
    """
    if not ENABLE_DP_LOCAL_PAGE_PATH or dp_size <= 1:
        return block_tables
    rows = int(block_tables.shape[0])
    if rows <= 0 or rows % dp_size != 0 or num_pages <= 0 or num_pages % dp_size != 0:
        return block_tables
    rows_per_shard = rows // dp_size
    pages_per_shard = num_pages // dp_size
    row_ids = jnp.arange(rows, dtype=jnp.int32)[:, None]
    row_shards = row_ids // jnp.int32(rows_per_shard)
    offsets = row_shards * jnp.int32(pages_per_shard)
    # Keep explicit padding entries unchanged (current page-table padding is 0).
    return jnp.where(block_tables == 0, block_tables, block_tables - offsets)


@OperationRegistry.register
class PagedFlashAttn(OperationImpl):
    """Paged Flash Attention using CUDA flash_attention with block tables."""

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        return "paged_flash_attention"

    def get_impl_metadata(self) -> OperationMetadata:
        return self.metadata

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        return (
            RequirementsBuilder("paged_flash_attention")
            .require_metadata(MetadataField.paged_v2())
            .support_cache(CacheType.RAGGED_PAGES)
            .use_cache_view(UnifiedAttentionCacheView)
            .build()
        )

    def forward_native(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        cache_view: UnifiedAttentionCacheView,
        cache_metadata: RaggedPagesMetadata | OperationsMetadata,
        mask_info: MaskInfo | None = None,
        bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
        softmax_scale: float | None = None,
        dropout_prob: float = 0.0,
        causal: bool = False,
        dropout_seed: int | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        normalize_output: bool = True,
        precision: lax.PrecisionLike = lax.Precision.DEFAULT,
        **ignore,
    ) -> AttentionOutput:
        if jax.default_backend() != "gpu":
            raise NotImplementedError("PagedFlashAttn requires a GPU backend (CUDA).")

        if cache_view is None:
            raise ValueError("PagedFlashAttn requires `cache_view` (paged KV cache).")
        if cache_metadata is None:
            raise ValueError("PagedFlashAttn requires `cache_metadata` with pages_tables.")

        if query.ndim != 4:
            raise ValueError(f"PagedFlashAttn expects query in BTHD layout; got ndim={query.ndim}.")

        cache_metadata = unwrap_metadata(cache_metadata, "ragged")
        block_tables = cache_metadata.pages_tables
        if block_tables is None:
            raise ValueError("PagedFlashAttn requires `pages_tables` (block_tables) in cache_metadata.")
        if block_tables.ndim != 2:
            raise ValueError("PagedFlashAttn expects block_tables with shape [batch, max_blocks].")
        if int(block_tables.shape[0]) != int(query.shape[0]):
            from .unified_attention import UnifiedAttn

            unified = UnifiedAttn(self.metadata)
            return unified.forward_native(
                query=query,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
                softmax_scale=softmax_scale,
                causal=causal,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                softmax_aux=softmax_aux,
                **ignore,
            )

        if mask_info is not None and (
            getattr(mask_info, "_q_segment_ids", None) is not None
            or getattr(mask_info, "_kv_segment_ids", None) is not None
        ):
            raise ValueError("PagedFlashAttn does not support segment IDs with paged KV (CUDA).")

        dtype: jnp.dtype = self.metadata.runtime_dtype
        query = query.astype(dtype)
        key_cache = cache_view.key_cache.astype(dtype)
        value_cache = cache_view.value_cache.astype(dtype)
        dp_size = max(1, int(getattr(cache_view.metadata, "data_parallel_size", 1)))
        page_axis = _dp_page_axis(cache_view)
        block_tables = _localize_block_tables_for_dp_pages(
            block_tables.astype(jnp.int32),
            num_pages=int(key_cache.shape[0]),
            dp_size=dp_size,
        )
        if bias is not None:
            bias = bias.astype(dtype)

        model_mode = self.get_mode(query=query, BTHD=True)  # type: ignore
        is_decode_mode = model_mode == ct.MODE_DECODE
        causal_computed = causal if not is_decode_mode else False

        shardings = self.metadata.get_shardings(model_mode, layout="bthd")
        query_sharding = self.create_stable_sharding(shardings.query, tensor=query, preserved_indices=[0, 2])
        bias_sharding = self.create_stable_sharding(
            shardings.bias,
            dep=bias,
            tensor=bias,
            preserved_indices=[0, 1],
        )
        softmax_aux_sharding = self.create_stable_sharding(
            shardings.softmax_aux,
            dep=softmax_aux,
            tensor=softmax_aux,
        )

        resolve = self.metadata.partition_manager.resolve
        key_sharding = resolve(
            axes=[page_axis, ct.EMPTY, ct.KV_HEAD, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=key_cache.shape,
        )
        value_sharding = resolve(
            axes=[page_axis, ct.EMPTY, ct.KV_HEAD, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=value_cache.shape,
        )
        output_sharding = self.create_stable_sharding(shardings.output, tensor=query, preserved_indices=[0, 2])

        attn: Float[Array, "batch seq_len_q num_heads head_dim"] = flash_attention(
            query,
            key_cache,
            value_cache,
            bias,
            None,
            None,
            softmax_aux,
            block_tables,
            mask_info=mask_info,
            softmax_scale=softmax_scale,
            dropout_prob=dropout_prob,
            causal=causal_computed,
            dropout_seed=dropout_seed,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            normalize_output=normalize_output,
            precision=precision,
            logits_dtype=jnp.bfloat16,
            cfg=self.metadata.get_operation_config("paged_flash_attention"),
            mesh=self.metadata.mesh,
            in_specs=(
                query_sharding,
                key_sharding,
                value_sharding,
                bias_sharding,
                None,
                None,
                softmax_aux_sharding,
            ),
            out_specs=output_sharding,
        )

        attn_sharded = with_sharding_constraint(arr=attn, sharding=shardings.output)
        return AttentionOutput(attention_weights=None, attention_outputs=attn_sharded)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_gpu(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_native(*args, **kwargs)
