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

"""Ragged Paged Attention implementations for continuous batching.

This module provides paged attention implementations optimized for serving
scenarios with continuous batching, where requests of varying lengths are
processed together efficiently using paged KV-cache management.

Key features:
- Non-contiguous paged KV-cache for memory-efficient serving
- Support for variable-length sequences in the same batch
- TPU-optimized Pallas kernels with GPU/CPU fallbacks
- Ragged tensor format for handling batches without padding waste
- Integration with vLLM-style paged memory management

Two versions are provided:

RaggedPageAttnV2:
    Uses slot mapping for cache updates. Each token is mapped to a specific
    slot in the paged cache via a precomputed slot_mapping tensor.

    Required metadata:
    - SEQ_LENS, CONTEXT_LENS, POSITIONS
    - QUERY_START_LOC, PAGES_TABLES, SLOT_MAPPING

RaggedPageAttnV3:
    Uses request distribution for more efficient batch processing. Instead
    of per-token slot mapping, uses request-level distribution information
    for cache updates.

    Required metadata:
    - SEQ_LENS, CONTEXT_LENS, POSITIONS
    - QUERY_START_LOC, PAGES_TABLES, REQUEST_DISTRIBUTION

Implementation details:
- Queries are expected in ragged format: [total_tokens, num_heads, head_dim]
- Page tables map logical positions to physical pages in the KV cache
- Context lengths track the total KV cache length per sequence
- Query start locations mark sequence boundaries in the ragged tensor

Example:
    >>> from easydel.operations import OperationMetadata
    >>> from easydel.operations.kernels import RaggedPageAttnV3
    >>>
    >>> metadata = OperationMetadata(runtime_dtype=jnp.bfloat16)
    >>> attn = RaggedPageAttnV3(metadata)
    >>>
    >>> # Queries in ragged format (no padding)
    >>> output = attn(
    ...     query=query,  # [total_tokens, num_heads, head_dim]
    ...     key=key,
    ...     value=value,
    ...     cache_view=paged_cache,
    ...     cache_metadata=ragged_metadata,
    ... )

Note:
    These implementations are designed for inference serving with continuous
    batching. For training or single-request inference, consider using
    FlashAttn or VanillaAttn instead.

References:
    - vLLM PagedAttention: https://arxiv.org/abs/2309.06180
    - EasyDeL serving documentation
"""

from functools import partial

import jax
from eformer import common_types as ct
from ejkernel.modules import ragged_page_attention_v2, ragged_page_attention_v3
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, DTypeLike, Float

from easydel.axis import ATTN_DP
from easydel.caching import RaggedPagesCacheView, RaggedPagesMetadata

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)

USE_SHARDMAP = True


def _normalize_axis_names(axis: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Normalize a partition axis spec into a tuple of concrete mesh axis names."""
    if axis is None:
        return ()
    if isinstance(axis, (tuple, list)):
        return tuple(str(a) for a in axis if a)
    return (str(axis),)


def _mesh_axis_size(mesh, axis_names: tuple[str, ...]) -> int:
    """Compute the product of mesh axis sizes for the provided axis names."""
    size = 1
    if mesh is None:
        return size
    for axis_name in axis_names:
        size *= int(mesh.shape.get(axis_name, 1))
    return int(size)


def _axis_index(axis_names: tuple[str, ...]) -> jax.Array:
    """Return a linearized axis index over one or more mesh axes."""
    if not axis_names:
        return jnp.int32(0)
    idx = jax.lax.axis_index(axis_names[0]).astype(jnp.int32)
    for axis_name in axis_names[1:]:
        axis_size = jax.lax.psum(jnp.int32(1), axis_name)
        idx = idx * axis_size + jax.lax.axis_index(axis_name).astype(jnp.int32)
    return idx


def _dp_page_axis(cache_view: RaggedPagesCacheView):
    """Resolve the logical page axis for the active cache view."""
    dp_size = max(1, int(getattr(cache_view.metadata, "data_parallel_size", 1)))
    return ATTN_DP if dp_size > 1 else ct.EMPTY


class _RaggedPageAttn(OperationImpl):
    """
    Attention implementation using the Paged Attention mechanism with TPU Pallas kernels.

    This class provides an attention mechanism suitable for scenarios where the
    Key-Value cache is managed in non-contiguous pages (Paged KV Cache). It leverages
    specialized kernels
    for efficient execution on TPUs, handling prefill and decode phases separately
    or in a mixed mode.

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
            tp.Union[str, tp.Tuple[str]]: The name "ragged_page_attention_v3" or "ragged_page_attention_v2".
        """
        raise NotImplementedError()

    def get_impl_metadata(self) -> OperationMetadata:
        """
        Retrieves the metadata associated with this attention implementation instance.

        Returns:
            OperationMetadata: The metadata object provided during initialization.
        """
        return self.metadata

    def forward_v2(
        self,
        query: Float[Array, "total_tokens num_q_heads head_dim"],
        cache_view: RaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        compute_dtype: DTypeLike = jnp.bfloat16,
        optimized: bool = False,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        mask_value: float | None = None,
        vmem_limit_bytes: int | None = None,
        **ignore,
    ) -> AttentionOutput:
        kv_pages: Float[Array, "num_pages page_size num_kv_heads head_dim"] = cache_view.kv_pages
        manager = self.metadata.partition_manager
        resolve = manager.resolve
        num_seqs_flat: Array = cache_metadata.num_seqs.reshape(-1)
        page_axis = _dp_page_axis(cache_view)
        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=query.shape)

        aux_spec = PartitionSpec(None)
        if softmax_aux is not None:
            num_aux_dims: int = softmax_aux.ndim
            if num_aux_dims == 2:
                aux_spec = resolve(axes=[ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)
            elif num_aux_dims == 1:
                aux_spec = resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)

        if compute_dtype is None:
            dtype_for_compute = jnp.bfloat16
        else:
            dtype_for_compute = compute_dtype
        platform = "pallas" if jax.default_backend() == "tpu" else "auto"
        cfg = self.metadata.get_operation_config("ragged_page_attention_v2")

        if platform == "pallas":
            if query.shape[-1] not in [128, 256]:
                platform = "xla"

        output = ragged_page_attention_v2(
            query,
            kv_pages,
            cache_metadata.context_lens,
            cache_metadata.pages_tables,
            cache_metadata.query_start_loc,
            num_seqs_flat,
            softmax_aux,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            vmem_limit_bytes=vmem_limit_bytes,
            optimized=optimized,
            compute_dtype=dtype_for_compute,
            mask_value=mask_value,
            sliding_window=sliding_window,
            cfg=cfg,
            platform=platform,
            in_specs=(
                qaxes,
                resolve(
                    axes=[page_axis, ct.EMPTY, ct.HEAD, ct.EMPTY],
                    mode=ct.MODE_PREFILL,
                    shape=kv_pages.shape,
                ),
                Ps(),
                Ps(),
                Ps(),
                Ps(),
                aux_spec,
            ),
            out_specs=qaxes,
            mesh=self.metadata.mesh,
        )

        return AttentionOutput(attention_weights=None, attention_outputs=output)

    def forward_v3(
        self,
        query: Float[Array, "total_tokens num_q_heads head_dim"],
        key: Float[Array, "total_tokens num_kv_heads head_dim"],
        value: Float[Array, "total_tokens num_kv_heads head_dim"],
        cache_view: RaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_sinks"] | None = None,  # noqa
        vmem_limit_bytes: int | None = None,
        **ignore,
    ) -> AttentionOutput:
        kv_pages = cache_view.kv_pages
        manager = self.metadata.partition_manager
        resolve = manager.resolve
        request_distribution = cache_metadata.request_distribution
        page_axis = _dp_page_axis(cache_view)

        sinks_axis = None

        if softmax_aux is not None:
            sinks_axis = resolve(axes=[ct.HEAD], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)
            softmax_aux = softmax_aux.astype("f4")

        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=query.shape)
        kvaxes = resolve(axes=[ct.EMPTY, ct.KV_HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=key.shape)

        kv_pages_spec = resolve(
            axes=[page_axis, ct.EMPTY, ct.HEAD, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )
        page_axis_names = _normalize_axis_names(kv_pages_spec[0] if len(kv_pages_spec) > 0 else None)
        page_axis_size = _mesh_axis_size(self.metadata.mesh, page_axis_names)
        kv_pages_spec_replicated = resolve(
            axes=[ct.EMPTY, ct.EMPTY, ct.HEAD, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )

        platform = "pallas" if jax.default_backend() == "tpu" else "auto"
        cfg = self.metadata.get_operation_config("ragged_page_attention_v3")
        common_call_kwargs = dict(
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            vmem_limit_bytes=vmem_limit_bytes,
            sliding_window=sliding_window,
            cfg=cfg,
            platform=platform,
        )
        # For DP-sharded page buffers, consume per-DP-local block tables and sequence metadata
        # inside the shard_map body. This avoids materializing global page all-gathers.
        can_use_dp_local = (
            page_axis == ATTN_DP
            and page_axis_size > 1
            and len(page_axis_names) > 0
            and int(cache_metadata.context_lens.shape[0]) % page_axis_size == 0
        )
        if can_use_dp_local:
            rows_per_shard = int(cache_metadata.context_lens.shape[0]) // page_axis_size
            max_pages_per_req = int(cache_metadata.pages_tables.shape[1])

            @partial(
                jax.shard_map,
                mesh=self.metadata.mesh,
                in_specs=(qaxes, kvaxes, kvaxes, kv_pages_spec, Ps(), Ps(), Ps(), Ps(), sinks_axis),
                out_specs=(qaxes, kv_pages_spec),
                check_vma=False,
            )
            def _mapped(
                local_query,
                local_key,
                local_value,
                local_kv_pages,
                full_context_lens,
                full_pages_tables,
                full_query_start_loc,
                full_distribution,
                local_softmax_aux,
            ):
                shard_idx = _axis_index(page_axis_names)
                row_start = shard_idx * jnp.int32(rows_per_shard)

                local_context_lens = jax.lax.dynamic_slice_in_dim(
                    full_context_lens,
                    start_index=row_start,
                    slice_size=rows_per_shard,
                    axis=0,
                )
                local_pages_rows = jax.lax.dynamic_slice_in_dim(
                    full_pages_tables,
                    start_index=row_start,
                    slice_size=rows_per_shard,
                    axis=0,
                )
                local_block_tables = local_pages_rows.reshape((rows_per_shard * max_pages_per_req,))
                local_query_start_loc = jax.lax.dynamic_slice_in_dim(
                    full_query_start_loc,
                    start_index=row_start,
                    slice_size=rows_per_shard + 1,
                    axis=0,
                )

                row_start_i32 = row_start.astype(jnp.int32)
                local_total = jnp.clip(full_distribution[2] - row_start_i32, 0, rows_per_shard).astype(jnp.int32)
                local_decode = jnp.clip(full_distribution[0] - row_start_i32, 0, local_total).astype(jnp.int32)
                local_prefill = jnp.clip(full_distribution[1] - row_start_i32, 0, local_total).astype(jnp.int32)
                local_distribution = jnp.stack((local_decode, local_prefill, local_total)).astype(jnp.int32)

                local_num_pages = jnp.int32(local_kv_pages.shape[0])
                page_offset = shard_idx * local_num_pages
                local_block_tables = local_block_tables - page_offset

                local_output, local_kv_pages = ragged_page_attention_v3(
                    local_query,
                    local_key,
                    local_value,
                    local_kv_pages,
                    local_context_lens,
                    local_block_tables,
                    local_query_start_loc,
                    local_distribution,
                    local_softmax_aux,
                    **common_call_kwargs,
                )

                # Keep only this shard's request spans, then reduce over dp to reconstruct
                # the full token output while KV pages stay sharded.
                row_ids = jnp.arange(rows_per_shard, dtype=jnp.int32)[:, None]
                token_ids = jnp.arange(local_query.shape[0], dtype=jnp.int32)[None, :]
                starts = local_query_start_loc[:-1, None]
                ends = local_query_start_loc[1:, None]
                active_rows = row_ids < local_total
                local_token_mask = jnp.any(active_rows & (token_ids >= starts) & (token_ids < ends), axis=0)
                local_output = jnp.where(
                    local_token_mask[:, None, None],
                    local_output,
                    jnp.zeros_like(local_output),
                )
                if len(page_axis_names) == 1:
                    output = jax.lax.psum(local_output.astype(jnp.float32), page_axis_names[0]).astype(
                        local_output.dtype
                    )
                else:
                    output = jax.lax.psum(local_output.astype(jnp.float32), tuple(page_axis_names)).astype(
                        local_output.dtype
                    )
                return output, local_kv_pages

            output, kv_pages = _mapped(
                query,
                key,
                value,
                kv_pages,
                cache_metadata.context_lens,
                cache_metadata.pages_tables,
                cache_metadata.query_start_loc,
                request_distribution,
                softmax_aux,
            )
        else:
            output, kv_pages = ragged_page_attention_v3(
                query,
                key,
                value,
                kv_pages,
                cache_metadata.context_lens,
                cache_metadata.pages_tables.reshape(-1),
                cache_metadata.query_start_loc,
                request_distribution,
                softmax_aux,
                in_specs=(qaxes, kvaxes, kvaxes, kv_pages_spec_replicated, Ps(), Ps(), Ps(), Ps(), sinks_axis),
                out_specs=(qaxes, kv_pages_spec_replicated),
                mesh=self.metadata.mesh,
                **common_call_kwargs,
            )
        cache_view.kv_pages = kv_pages
        return AttentionOutput(attention_weights=None, attention_outputs=output, cache_view=cache_view)

    def forward_native(
        self,
        query: Float[Array, "total_tokens num_q_heads head_dim"],
        key: Float[Array, "total_tokens num_kv_heads head_dim"],
        value: Float[Array, "total_tokens num_kv_heads head_dim"],
        cache_view: RaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        vmem_limit_bytes: int | None = None,
        mask_value: float | None = None,
        compute_dtype: DTypeLike = jnp.bfloat16,
        optimized: bool = False,
        **ignore,
    ):
        """
        Native forward pass for paged attention using ragged format.

        This implementation handles attention with a paged KV cache stored in non-contiguous
        memory pages. It uses the `ragged_page_attention_v2` kernel which efficiently processes
        variable-length sequences with page table lookups.

        Args:
            query: Query tensor [total_tokens, num_q_heads, head_dim] in ragged format.
                Total_tokens is the sum of all sequence lengths in the batch.
            cache_view: Paged KV cache view containing:
                - kv_pages: Paged key/value tensors [num_pages, page_size, num_kv_heads, head_dim].
            cache_metadata: Metadata for paged cache including:
                - context_lens: Length of each sequence [num_seqs].
                - pages_tables: Page table for cache access [num_seqs, max_pages].
                - query_start_loc: Starting index for each sequence [num_seqs + 1].
                - num_seqs: Number of sequences in the batch.
            softmax_scale: Scaling factor for attention logits. Defaults to 1/sqrt(head_dim).
            logits_soft_cap: Soft capping value for attention logits. Optional.
            compute_dtype: Data type for kernel computation. Defaults to bfloat16.
            optimized: Use optimized kernel variant if available. Defaults to False.
            sliding_window: Sliding window size for local attention. Optional.
            softmax_aux: Auxiliary softmax tensor for sink tokens. Optional.
            mask_value: Value for masked positions. Optional.
            vmem_limit_bytes: VMEM limit for TPU memory management. Optional.
            **ignore: Additional ignored arguments.

        Returns:
            AttentionOutput: Contains attention outputs [total_tokens, num_q_heads, head_dim].
                Attention weights are not computed.
        """
        fn = self.forward_v3 if self.get_impl_name() == "ragged_page_attention_v3" else self.forward_v2
        return fn(
            query=query,
            key=key,
            value=value,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            vmem_limit_bytes=vmem_limit_bytes,
            optimized=optimized,
            compute_dtype=compute_dtype,
            softmax_aux=softmax_aux,
            mask_value=mask_value,
            sliding_window=sliding_window,
            **ignore,
        )

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm GPU forward pass. Not implemented for Paged Attention."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch tokens num_heads head_dim"],
        key: Float[Array, "batch tokens num_kv_heads head_dim"],
        value: Float[Array, "batch tokens num_kv_heads head_dim"],
        cache_view: RaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        compute_dtype: DTypeLike = jnp.bfloat16,
        optimized: bool = False,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        mask_value: float | None = None,
        vmem_limit_bytes: int | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Executes paged attention by dispatching to the appropriate backend implementation.

        This method handles attention with non-contiguous paged KV cache, preprocessing
        the query tensor by reshaping if needed, then restoring the original shape in the output.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim] or [batch, num_heads, head_dim].
            cache_view: Contains the paged KV cache tensors with page table information.
            cache_metadata: Metadata describing batch state including:
                - context_lens: Length of each sequence in the batch.
                - pages_tables: Page table mapping for cache access.
                - query_start_loc: Starting locations for queries in ragged format.
                - num_seqs: Number of sequences in the batch.
            softmax_scale: Scaling factor for attention logits. Defaults to 1/sqrt(head_dim).
            logits_soft_cap: Soft capping value for attention logits. Optional.
            compute_dtype: Data type for computation (e.g., bfloat16, float32).
            optimized: Use optimized kernel variant if available.
            sliding_window: Sliding window size for local attention. Optional.
            softmax_aux: Auxiliary softmax tensor for sink tokens. Optional.
            mask_value: Value to use for masked positions. Optional.
            vmem_limit_bytes: VMEM limit in bytes for TPU memory management. Optional.
            **ignore: Additional ignored keyword arguments.

        Returns:
            AttentionOutput: Contains attention outputs with shape matching input query.
                Attention weights are not computed.
        """
        num_query_dims: int = query.ndim
        is_4d: bool = num_query_dims == 4

        batch: int
        sequence: int
        head: int
        dim: int
        if is_4d:
            batch, sequence, head, dim = query.shape

        # Reshape query to ragged format [total_tokens, num_heads, head_dim]
        query_reshaped = query.reshape(-1, *query.shape[-2:])
        key_reshaped = key.reshape(-1, *key.shape[-2:])
        value_reshaped = value.reshape(-1, *value.shape[-2:])

        output: AttentionOutput = super().__call__(
            query=query_reshaped,
            key=key_reshaped,
            value=value_reshaped,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            vmem_limit_bytes=vmem_limit_bytes,
            optimized=optimized,
            compute_dtype=compute_dtype,
            softmax_aux=softmax_aux,
            mask_value=mask_value,
            sliding_window=sliding_window,
            **ignore,
        )

        # Restore original shape if input was 4D
        if is_4d:
            outputs_reshaped = output.attention_outputs.reshape(batch, sequence, head, dim)
            output.attention_outputs = outputs_reshaped

        return output


@OperationRegistry.register
class RaggedPageAttnV2(_RaggedPageAttn):
    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name for this attention implementation.

        Returns:
            tp.Union[str, tp.Tuple[str]]: The name "ragged_page_attention_v2".
        """
        return "ragged_page_attention_v2"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for RaggedPageAttnV2 (slot mapping based)."""
        return (
            RequirementsBuilder("ragged_page_attention_v2")
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
            .use_cache_view(RaggedPagesCacheView)
            .build()
        )


@OperationRegistry.register
class RaggedPageAttnV3(_RaggedPageAttn):
    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name for this attention implementation.

        Returns:
            tp.Union[str, tp.Tuple[str]]: The name "ragged_page_attention_v3".
        """
        return "ragged_page_attention_v3"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for RaggedPageAttnV3 (request distribution based)."""
        return (
            RequirementsBuilder("ragged_page_attention_v3")
            .require_metadata(
                MetadataField.SEQ_LENS
                | MetadataField.CONTEXT_LENS
                | MetadataField.POSITIONS
                | MetadataField.QUERY_START_LOC
                | MetadataField.PAGES_TABLES
                | MetadataField.REQUEST_DISTRIBUTION
            )
            .optional_metadata(MetadataField.LOGITS_INDICES)
            .support_cache(CacheType.RAGGED_PAGES)
            .use_cache_view(RaggedPagesCacheView)
            .build()
        )
