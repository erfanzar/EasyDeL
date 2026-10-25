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

"""Paged Flash Attention V2 backed by ejkernel's CUDA kernel with block tables.

This module implements :class:`PagedFlashAttn`, an :class:`OperationImpl`
that drives ejkernel's :func:`flash_attention` over a paged KV buffer
addressed via per-request block tables (the same layout used by
:class:`UnifiedAttentionCacheView`, i.e. separate ``key_cache`` and
``value_cache`` page pools).

Capabilities and constraints:

* CUDA-only: the underlying kernel is a Triton/CUDA Flash Attention V2
  variant; on non-GPU backends ``forward_native`` raises
  ``NotImplementedError``.
* Fixed-length batches in BTHD layout. Variable-length packing
  (``cum_seqlens``) is not supported by this op; the falsfback to
  :class:`UnifiedAttn` covers query/page-table batch mismatches.
* Optional data-parallel page sharding is honored: the ``ATTN_DP`` axis
  enters the in_specs only when the cache reports
  ``data_parallel_size > 1`` and the
  ``EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH`` flag is on, in which case
  block-table page IDs are translated into per-shard local IDs by
  :func:`_localize_block_tables_for_dp_pages`.
* Segment IDs in the ``mask_info`` are not supported (paged KV makes them
  ambiguous) and raise ``ValueError`` if present.
"""

from __future__ import annotations

import jax
from ejkernel.modules import flash_attention  # pyright: ignore[reportMissingTypeStubs]
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float
from spectrax import common_types as ct
from spectrax import with_sharding_constraint

from easydel.caching import OperationsMetadata, RaggedPagesMetadata, UnifiedAttentionCacheView, unwrap_metadata

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationRegistry
from .._operation_meta import normalize_head_sinks
from ..requirements import CacheType, ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder
from ._paged_common import ENABLE_DP_LOCAL_PAGE_PATH, dp_page_axis

# Backward-compatible alias for callers importing the former adapter helper.
_dp_page_axis = dp_page_axis


def _localize_block_tables_for_dp_pages(
    block_tables: Array,
    *,
    num_pages: int,
    dp_size: int,
) -> Array:
    """Translate global page IDs into DP-local page IDs per request row.

    Assumes rows are assigned contiguously per DP shard and page IDs are
    from a globally indexed page pool partitioned evenly over DP shards.

    Args:
        block_tables: ``int[rows, max_blocks]`` table of page IDs per
            request.
        num_pages: Total number of pages in the global pool.
        dp_size: Data-parallel mesh size.

    Returns:
        Array: ``block_tables`` with the per-shard page-ID offset removed.
        Returns the input unchanged when DP localization is disabled or when
        ``rows`` and ``num_pages`` cannot be evenly partitioned. Padding
        entries (``0``) are preserved.
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
    result: Array = jnp.where(block_tables == 0, block_tables, block_tables - offsets)
    return result


@OperationRegistry.register
class PagedFlashAttn(OperationImpl):
    """Flash Attention V2 over a paged KV cache addressed by per-request block tables.

    Bridges the BTHD query convention used by EasyDeL training/serving paths
    to ejkernel's CUDA Flash Attention V2 kernel, while reading keys and
    values from a paged buffer. The expected cache layout is the same as
    :class:`UnifiedAttentionCacheView`: separate ``key_cache`` /
    ``value_cache`` tensors of shape
    ``[num_pages, page_size, num_kv_heads, head_dim]`` and a
    ``[batch, max_blocks]`` ``pages_tables`` mapping logical positions to
    physical page IDs.

    The operator is registered under ``"paged_flash_attention"`` and is
    primarily used in continuous-batching inference where queries come in
    fixed-length batches but the underlying KV grows page-by-page. It does
    not own cache updates; callers must populate the pages before invoking
    this op (see ``UnifiedAttentionCacheView.concatenate_to_cache``).

    Notes:
        - GPU/CUDA-only. Other backends fall through to ``forward_native``
          which raises ``NotImplementedError``.
        - Falls back to :class:`UnifiedAttn` when the query batch dimension
          and ``pages_tables`` batch dimension disagree (e.g. mid-step
          rebatching scenarios).
        - Honors the ``EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH`` flag for
          DP-sharded page pools; block tables are translated via
          :func:`_localize_block_tables_for_dp_pages` so each DP shard sees
          local page IDs only.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the registered name of this operation.

        Returns:
            str: ``"paged_flash_attention"``.
        """
        return "paged_flash_attention"

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Declare metadata and cache requirements.

        Args:
            mode: Execution mode (currently unused; requirements are mode
                independent).

        Returns:
            OperationRequirements: Requires the paged-V2 metadata bundle and
            uses :class:`UnifiedAttentionCacheView` over ragged-page caches.
        """
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
        cache_view: UnifiedAttentionCacheView | None,
        cache_metadata: RaggedPagesMetadata | OperationsMetadata | None,
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
        """Forward pass for paged flash attention on GPU.

        Reads key/value data from a paged cache via block tables and computes
        flash attention. Falls back to ``UnifiedAttn`` when batch dimensions
        of query and block tables disagree.

        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim].
            key: Key tensor (unused directly; keys come from cache_view).
            value: Value tensor (unused directly; values come from cache_view).
            cache_view: Paged KV cache view containing key/value page pools.
            cache_metadata: Metadata with ``pages_tables`` (block tables).
            mask_info: Optional mask information (segment IDs not supported).
            bias: Optional attention bias [batch, num_heads, seq_q, seq_k].
            softmax_scale: Scaling factor for logits. Defaults to 1/sqrt(head_dim).
            dropout_prob: Attention dropout probability.
            causal: Whether to apply causal masking.
            dropout_seed: Seed for dropout RNG.
            sliding_window: Window size for local/sliding-window attention.
            logits_soft_cap: Soft capping value for attention logits.
            softmax_aux: Auxiliary tensor for sink-token attention.
            normalize_output: Whether to normalize the attention output.
            precision: JAX precision setting for matmuls.
            **ignore: Additional ignored keyword arguments.

        Returns:
            AttentionOutput with attended values and no explicit weights.

        Raises:
            NotImplementedError: If backend is not GPU.
            ValueError: If cache_view, cache_metadata, or pages_tables is missing.
        """
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
        rows, num_pages = int(block_tables.shape[0]), int(key_cache.shape[0])
        use_local_pages = (
            ENABLE_DP_LOCAL_PAGE_PATH
            and dp_size > 1
            and rows > 0
            and rows % dp_size == 0
            and num_pages > 0
            and num_pages % dp_size == 0
        )
        # Global IDs require a replicated pool. Decide localization and page
        # sharding together, including when row/page geometry is incompatible.
        page_axis = dp_page_axis(cache_view) if use_local_pages else ct.EMPTY
        block_tables = block_tables.astype(jnp.int32)
        if use_local_pages:
            block_tables = _localize_block_tables_for_dp_pages(
                block_tables,
                num_pages=num_pages,
                dp_size=dp_size,
            )
        if bias is not None:
            bias = bias.astype(dtype)

        model_mode = self.get_mode(query=query, BTHD=True)
        context_lens = getattr(cache_metadata, "context_lens", None)
        if context_lens is None:
            raise ValueError("PagedFlashAttn requires `context_lens` to mask unused cache slots.")
        batch, q_len = query.shape[:2]
        kv_len = block_tables.shape[1] * key_cache.shape[1]
        lengths = jnp.asarray(context_lens, jnp.int32).reshape(batch, 1)
        kv_positions = jnp.arange(kv_len, dtype=jnp.int32)[None, None, :]
        q_positions = lengths - q_len + jnp.arange(q_len, dtype=jnp.int32)[None, :]
        if mask_info is not None and mask_info.q_positions is not None:
            q_positions = jnp.broadcast_to(jnp.asarray(mask_info.q_positions, jnp.int32), (batch, q_len))
        visible = (kv_positions < lengths[:, :, None]) & (q_positions[:, :, None] >= 0)
        if causal:
            visible = visible & (kv_positions <= q_positions[:, :, None])
        if sliding_window is not None:
            if isinstance(sliding_window, int):
                left = right = max(sliding_window - 1, 0)
            else:
                left, right = sliding_window
            visible = visible & (kv_positions >= q_positions[:, :, None] - left)
            visible = visible & (kv_positions <= q_positions[:, :, None] + right)
        visible = visible[:, None, :, :]
        if mask_info is not None:
            existing = jnp.asarray(mask_info.get_or_compute_attention_mask(dtype=jnp.bool_), jnp.bool_)
            if existing.ndim == 2:
                existing = existing[:, None, None, :]
            elif existing.ndim == 3:
                existing = existing[:, None, :, :]
            if existing.shape[-1] < kv_len:
                existing = jnp.pad(
                    existing, ((0, 0), (0, 0), (0, 0), (0, kv_len - existing.shape[-1])), constant_values=True
                )
            visible = visible & existing[..., :q_len, :kv_len]
        # The kernel sees padded page-table width, not each request's length.
        # Bake causal/window terms in absolute coordinates rather than letting
        # dense bottom-right alignment use that padded width as the KV length.
        mask_info = MaskInfo(_attention_mask=visible)
        causal_computed = False
        sliding_window = None

        softmax_aux, _ = normalize_head_sinks(
            softmax_aux,
            num_q_heads=query.shape[2],
            num_kv_heads=key_cache.shape[2],
            per_head_2d=False,
            head_shard_factor=self.metadata.head_shard_factor(model_mode, query.shape[2]),
        )
        shardings = self.metadata.get_shardings(
            model_mode,
            layout="bthd",
            softmax_aux=softmax_aux,
            num_q_heads=query.shape[2],
            num_kv_heads=key_cache.shape[2],
        )
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

        resolve = self.metadata.runtime_sharding_resolver.with_mesh(self.metadata.mesh).resolve
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
        # Block-table rows follow the query batch partition (block axis replicated). Without this eighth
        # spec ejkernel replicated the whole table while the query batch was sharded, so each shard saw
        # every request's rows and the paged kernel rejected the batch mismatch (r14 S07-2).
        batch_axis = query_sharding[0] if query_sharding is not None and len(query_sharding) else None
        block_tables_sharding = PartitionSpec(batch_axis, None)
        head_axis = query_sharding[2] if query_sharding is not None and len(query_sharding) > 2 else None
        mask_info = mask_info.replace(
            batch_axis_name=batch_axis,
            qheads_axis_name=head_axis,
            sequence_axis_name=None,
        )

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
            # The ejkernel CUDA / CuTe / Triton flash kernels accumulate logits in fp32 only; follow the
            # configured softmax dtype (fp32 by default) instead of forcing a lower precision.
            logits_dtype=self.metadata.runtime_softmax_dtype or jnp.float32,
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
                block_tables_sharding,
            ),
            out_specs=output_sharding,
        )

        attn_sharded = with_sharding_constraint(arr=attn, sharding=shardings.output, mesh=self.metadata.mesh)
        return AttentionOutput(attention_weights=None, attention_outputs=attn_sharded)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA forward pass. Delegates to ``forward_gpu``."""
        return self.forward_gpu(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU forward pass. Delegates to ``forward_native``."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass. Delegates to ``forward_native``."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Delegates to ``forward_native``."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm forward pass. Delegates to ``forward_native``."""
        return self.forward_native(*args, **kwargs)
