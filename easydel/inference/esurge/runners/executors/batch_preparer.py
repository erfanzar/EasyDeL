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

"""CPU-first batch metadata preparation for eSurge execution.

This module extracts the host-side batch preparation logic from the runner's
execution manager. It performs fast NumPy computations on CPU and moves a single
consolidated payload to device via `jax.device_put`.

The CPU-first approach offers several performance benefits:
    - Avoids repeated small device transfers (high latency overhead)
    - Enables NumPy vectorization for metadata computation
    - Consolidates all data into a single device_put call
    - Supports async/double-buffered transfers for pipelining

The BatchMetadataPreparer handles:
    - Token packing: Flattening multiple requests into a contiguous batch
    - Position computation: Computing absolute positions for each token
    - Page table preparation: Setting up KV cache page mappings
    - Slot mapping (v2): Computing per-slice KV cache slot assignments
    - Request distribution (v3): Computing decode/prefill request splits
    - Sampling parameter extraction: Temperature, top_k, top_p, min_p
    - VLM data preparation: mRoPE positions, prefill embeddings, visual data

Classes:
    BatchMetadataPreparer: Prepares and transfers per-step metadata for
        model execution using CPU-first computation strategy.

Example:
    >>> preparer = BatchMetadataPreparer(
    ...     metadata=cache_config,
    ...     empty_sharding=sharding,
    ...     max_num_tokens=4096,
    ...     max_num_reqs=32,
    ...     max_model_len=8192,
    ...     min_input_pad=8,
    ... )
    >>>
    >>> batch_metadata, *buffers = preparer.prepare_batch_metadata(
    ...     num_tokens_static=256,
    ...     scheduled_full_cpu=scheduled,
    ...     active_mask_full_cpu=active_mask,
    ...     input_ids_buf=input_ids_buf,
    ...     position_ids_buf=position_ids_buf,
    ...     token_ids_cpu=token_ids,
    ...     num_computed_tokens_cpu=num_computed,
    ...     temperature_cpu=temperatures,
    ...     top_p_cpu=top_ps,
    ...     top_k_cpu=top_ks,
    ...     min_p_cpu=min_ps,
    ...     page_table_cpu=page_table,
    ...     padded_num_reqs_in=16,
    ... )
"""

from __future__ import annotations

import time

import jax
import numpy as np

from easydel.caching import RaggedPagesCacheConfig, UnifiedAttentionCacheConfig
from easydel.caching._metadatabuilder import AttentionMetadataBuilder
from easydel.utils.helpers import check_bool_flag

from ...core.dp_sharding import dp_shard_page_bounds, pages_per_dp_shard
from ...page_table import PAGE_TABLE_PADDING_VAL, SLOT_MAPPING_PADDING_VAL
from ..execution_types import BatchMetadata


class BatchMetadataPreparer:
    """Prepare and transfer per-step metadata for model execution.

    The BatchMetadataPreparer handles the CPU-side preparation of batch
    metadata required for each inference step. It uses preallocated NumPy
    buffers to minimize per-step allocation overhead and consolidates all
    data into efficient device transfers.

    The preparer supports two operation modes:
        - Synchronous: prepare_batch_metadata() computes and transfers immediately
        - Async (double-buffered): start_async_prep() initiates transfer,
          get_async_prep_result() retrieves results

    Attributes:
        metadata (RaggedPagesCacheConfig | UnifiedAttentionCacheConfig): KV cache
            configuration containing page size and attention version.
        max_num_tokens (int): Maximum tokens per batch.
        max_num_reqs (int): Maximum concurrent requests.
        max_model_len (int): Maximum sequence length.
        min_input_pad (int): Minimum padding for request count bucketing.
        last_prep_stats (dict[str, float]): Timing statistics from the last
            prepare_batch_metadata() call.

    Example:
        >>> preparer = BatchMetadataPreparer(
        ...     metadata=cache_config,
        ...     empty_sharding=sharding,
        ...     max_num_tokens=4096,
        ...     max_num_reqs=32,
        ...     max_model_len=8192,
        ...     min_input_pad=8,
        ... )
        >>> batch_metadata, *_ = preparer.prepare_batch_metadata(...)
    """

    def __init__(
        self,
        *,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig | None,
        empty_sharding: jax.sharding.Sharding,
        max_num_tokens: int,
        max_num_reqs: int,
        max_model_len: int,
        min_input_pad: int,
    ) -> None:
        """Initialize the BatchMetadataPreparer.

        Args:
            metadata: KV cache configuration. Determines which attention
                version (v2 with slot_mapping or v3 with request_distribution)
                is used for metadata preparation.
            empty_sharding: Default JAX sharding for replicated arrays.
                Applied to all transferred tensors.
            max_num_tokens: Maximum tokens per batch. Determines buffer sizes
                for token-level arrays (input_ids, positions).
            max_num_reqs: Maximum concurrent requests. Determines buffer sizes
                for request-level arrays (temperatures, page tables).
            max_model_len: Maximum sequence length. Used for validation and
                some buffer sizing.
            min_input_pad: Minimum padding for request count bucketing.
                Request counts are padded to powers of 2 starting from this
                minimum.

        Note:
            This constructor preallocates all CPU buffers to avoid per-step
            memory allocation. The memory footprint scales with max_num_tokens
            and max_num_reqs.
        """
        self.metadata = metadata
        self._empty_sharding = empty_sharding

        self.max_num_tokens = int(max_num_tokens)
        self.max_num_reqs = int(max_num_reqs)
        self.max_model_len = int(max_model_len)
        self.min_input_pad = int(min_input_pad)

        self._metadata_version = metadata.version
        self._use_slot_mapping = self._metadata_version == "v2"
        self._use_request_distribution = not self._use_slot_mapping
        self._enable_dp_local_page_path = check_bool_flag("EASURGE_ENABLE_DP_LOCAL_PAGE_PATH", default=True)

        # Ragged paging shapes (compile-stable).
        num_reqs_max_model_len = (
            min(int(metadata.get_max_num_seqs()), self.max_num_reqs) if metadata is not None else self.max_num_reqs
        )
        self._num_reqs_max_model_len = max(int(num_reqs_max_model_len), 1)

        max_pages_per_req = int(getattr(metadata, "max_num_pages_per_req", 0)) if metadata is not None else 0
        self._max_pages_per_req = max(int(max_pages_per_req), 1)

        # Pre-allocate CPU buffers for fast batch metadata preparation
        self._input_ids_cpu = np.zeros((self.max_num_tokens,), dtype=np.int32)
        self._positions_cpu = np.zeros((self.max_num_tokens,), dtype=np.int32)
        self._query_start_loc_cpu = np.zeros((self.max_num_reqs + 1,), dtype=np.int32)
        self._seq_lens_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._logits_indices_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._scheduled_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._packed_qsl_seqlens_cpu = np.zeros((2, self.max_num_reqs + 1), dtype=np.int32)
        self._packed_i32_padded_cpu = np.zeros((3, self.max_num_reqs), dtype=np.int32)
        self._packed_f32_padded_cpu = np.zeros((3, self.max_num_reqs), dtype=np.float32)
        self._packed_misc_i32_cpu = np.zeros((5,), dtype=np.int32)
        self._arange_cpu = np.arange(self.max_num_tokens, dtype=np.int32)
        self._pages_tables_cpu = np.full(
            (self._num_reqs_max_model_len, self._max_pages_per_req),
            PAGE_TABLE_PADDING_VAL,
            dtype=np.int32,
        )
        self._request_distribution_placeholder = np.zeros((3,), dtype=np.int32)
        self._slot_mapping_placeholder = np.zeros((3, 1), dtype=np.int32)
        self._num_kv_update_placeholder = np.zeros((1,), dtype=np.int32)
        self._async_slot_mapping_placeholder = np.zeros((3, 1), dtype=np.int32)

        # Async staging buffers (avoid per-call allocations in overlap mode).
        self._async_input_ids_cpu = np.zeros((self.max_num_tokens,), dtype=np.int32)
        self._async_positions_cpu = np.zeros((self.max_num_tokens,), dtype=np.int32)
        self._async_query_start_loc_cpu = np.zeros((self.max_num_reqs + 1,), dtype=np.int32)
        self._async_seq_lens_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_logits_indices_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_scheduled_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_packed_qsl_seqlens_cpu = np.zeros((2, self.max_num_reqs + 1), dtype=np.int32)
        self._async_packed_i32_padded_cpu = np.zeros((3, self.max_num_reqs), dtype=np.int32)
        self._async_packed_f32_padded_cpu = np.zeros((3, self.max_num_reqs), dtype=np.float32)
        self._async_packed_misc_i32_cpu = np.zeros((5,), dtype=np.int32)
        self._async_pages_tables_cpu = np.full(
            (self._num_reqs_max_model_len, self._max_pages_per_req),
            PAGE_TABLE_PADDING_VAL,
            dtype=np.int32,
        )
        self._async_request_distribution_cpu = np.zeros((3,), dtype=np.int32)
        self._async_num_kv_update_cpu = np.zeros((1,), dtype=np.int32)

        self._async_num_computed_tokens_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_scheduled_full_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_active_mask_full_cpu = np.zeros((self.max_num_reqs,), dtype=bool)
        self._async_temperature_cpu = np.zeros((self.max_num_reqs,), dtype=np.float32)
        self._async_top_p_cpu = np.zeros((self.max_num_reqs,), dtype=np.float32)
        self._async_top_k_cpu = np.zeros((self.max_num_reqs,), dtype=np.int32)
        self._async_min_p_cpu = np.zeros((self.max_num_reqs,), dtype=np.float32)
        self._async_page_table_cpu = np.zeros((self.max_num_reqs, self._max_pages_per_req), dtype=np.int32)

        # Device cache for `pages_tables`. This table is derived from the CPU page
        # table and is only required to change when pages are allocated/replaced
        # or request rows are moved/swapped/condensed.
        self._cached_pages_tables_dev: jax.Array | None = None
        self._cached_page_table_version: int | None = None
        self._cached_rows_to_copy: int | None = None

        if self._use_slot_mapping and metadata is not None:
            self._slices_per_page = metadata.num_slices_per_kv_cache_update_page
            self._max_padded_slices = int(metadata.get_padded_num_slices(self.max_num_tokens, self.max_num_reqs))
            self._slot_mapping_cpu = np.full(
                (3, self._max_padded_slices),
                SLOT_MAPPING_PADDING_VAL,
                dtype=np.int32,
            )
            self._async_slot_mapping_cpu = np.full(
                (3, self._max_padded_slices),
                SLOT_MAPPING_PADDING_VAL,
                dtype=np.int32,
            )
            self._slot_mapping_indices = np.arange(self._max_padded_slices, dtype=np.int32)
        else:
            self._max_padded_slices = None
            self._slot_mapping_cpu = None
            self._slices_per_page = None
            self._slot_mapping_indices = None
            self._async_slot_mapping_cpu = None

        # Double buffering: store pending async device transfer
        self._pending_transfer: tuple | None = None
        self._pending_transfer_metadata: dict | None = None

        # Last-prep timing breakdown (seconds).
        self.last_prep_stats: dict[str, float] = {}

        # Cache device-side zero buffers for optional VLM inputs. This lets us keep
        # the input PyTree stable (non-None fields) without paying host→device cost
        # on steps where those buffers are effectively unused (e.g., all-false masks).
        self._zero_dev_cache: dict[tuple[str, tuple[int, ...], str], jax.Array] = {}

    def _enforce_dp_local_page_tables(
        self,
        *,
        num_requests: int,
        scheduled: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
    ) -> None:
        """Validate that active rows only use page IDs local to their DP shard.

        The current DP-sharded page-cache path requires request rows and block-table
        IDs to be aligned by shard:
            - row shard: ``req_idx // rows_per_dp``
            - page shard: ``page_id // pages_per_dp``

        If this invariant is violated, TPU kernels may trigger page all-gathers or
        access non-local page IDs. We fail early with a clear error so allocator/
        scheduler plumbing can be corrected.
        """
        if not self._enable_dp_local_page_path:
            return
        dp_size = max(1, int(getattr(self.metadata, "data_parallel_size", 1)))
        if dp_size <= 1 or num_requests <= 0:
            return

        total_rows = int(self._num_reqs_max_model_len)
        if total_rows % dp_size != 0:
            raise ValueError(
                "DP-local page-table invariant requires rows divisible by data-parallel size: "
                f"rows={total_rows}, dp_size={dp_size}."
            )

        total_pages = int(getattr(self.metadata, "num_pages", 0) or 0)
        pages_per_shard = pages_per_dp_shard(total_pages, dp_size)
        if pages_per_shard is None:
            # DP-local page partitioning is only valid when usable pages
            # (excluding null page 0) split evenly across DP shards.
            return

        rows_per_shard = total_rows // dp_size
        page_size = max(1, int(getattr(self.metadata, "page_size", 1)))
        max_pages_per_req = int(page_table_cpu.shape[1])

        for req_idx in range(int(num_requests)):
            seq_len = int(num_computed_tokens_cpu[req_idx]) + int(scheduled[req_idx])
            if seq_len <= 0:
                continue

            req_shard = min(req_idx // rows_per_shard, dp_size - 1)
            page_lo, page_hi = dp_shard_page_bounds(req_shard, pages_per_shard)
            page_cnt = min((seq_len + page_size - 1) // page_size, max_pages_per_req)

            row = np.asarray(page_table_cpu[req_idx, :page_cnt], dtype=np.int32)
            # Ignore explicit padding/null entries.
            row = row[row != int(PAGE_TABLE_PADDING_VAL)]
            if row.size == 0:
                continue

            invalid = row[(row < page_lo) | (row >= page_hi)]
            if invalid.size:
                raise ValueError(
                    "Non-DP-local page IDs detected for active request row. "
                    f"req_idx={req_idx} req_shard={req_shard} "
                    f"valid_range=[{page_lo}, {page_hi}) sample_bad_page={int(invalid[0])}. "
                    "Ensure scheduler/allocator produce per-DP-local block IDs "
                    "(or run with data-parallel page sharding disabled)."
                )

    def _get_zero_dev(self, *, namespace: str, shape: tuple[int, ...], dtype: np.dtype) -> jax.Array:
        """Get or create a cached zero-filled device array.

        Retrieves a cached device array of zeros with the specified shape and
        dtype. If no cached array exists, creates one and caches it for reuse.
        This avoids repeated device_put calls for zero arrays.

        Args:
            namespace: Identifier for the array purpose (e.g., "prefill_embeds").
            shape: Shape of the zero array.
            dtype: NumPy dtype of the array.

        Returns:
            Device-resident JAX array of zeros with the specified shape/dtype.
        """
        key = (str(namespace), tuple(int(x) for x in shape), str(np.dtype(dtype)))
        cached = self._zero_dev_cache.get(key)
        if cached is None:
            cached = jax.device_put(np.zeros(shape, dtype=dtype), self._empty_sharding)
            self._zero_dev_cache[key] = cached
        return cached

    def _get_zero_dev_like(self, *, namespace: str, arr: np.ndarray) -> jax.Array:
        """Get a cached zero device array with shape/dtype matching an array.

        Args:
            namespace: Identifier for the array purpose.
            arr: NumPy array whose shape/dtype should be matched.

        Returns:
            Device-resident JAX array of zeros matching arr's shape/dtype.
        """
        return self._get_zero_dev(namespace=namespace, shape=tuple(arr.shape), dtype=arr.dtype)

    @property
    def uses_slot_mapping(self) -> bool:
        """Check if this preparer uses v2-style slot mapping.

        Returns:
            True if using ragged page attention v2 with slot mapping.
        """
        return self._use_slot_mapping

    @property
    def uses_request_distribution(self) -> bool:
        """Check if this preparer uses v3-style request distribution.

        Returns:
            True if using ragged page attention v3 with request distribution.
        """
        return self._use_request_distribution

    def _compute_slot_mapping_v2(
        self,
        *,
        num_requests: int,
        scheduled: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        slot_mapping_out: np.ndarray | None = None,
        copy_out: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Compute slot mapping tensor for ragged-page attention v2.

        Builds the slot_mapping tensor that maps logical token positions to
        physical KV cache locations. The v2 attention kernel uses this mapping
        to efficiently scatter/gather KV values during attention computation.

        The slot mapping has shape [3, max_padded_slices] where each column
        represents a "slice" (a contiguous range of tokens within a page):
            - Row 0: KV cache start index (page_number * page_size + offset)
            - Row 1: New KV start index (cumulative position in new KV buffer)
            - Row 2: Slice length (number of tokens in this slice)

        Args:
            num_requests: Number of active requests.
            scheduled: Number of tokens scheduled per request.
            num_computed_tokens_cpu: Tokens already computed per request.
            page_table_cpu: Page table mapping request/page to physical page.
            slot_mapping_out: Optional output buffer. If None, uses internal
                buffer.
            copy_out: If True, returns a copy of the result. If False, returns
                the buffer directly (caller must not modify).

        Returns:
            Tuple of (slot_mapping, total_pages) where slot_mapping has shape
            [3, padded_num_slices] and total_pages is the number of pages
            touched by this batch.

        Note:
            This is called only for v2 attention (self._use_slot_mapping=True).
            For v3 attention, request_distribution is used instead.
        """

        slot_mapping = self._slot_mapping_cpu if slot_mapping_out is None else slot_mapping_out
        if slot_mapping is None:
            return np.zeros((3, 1), dtype=np.int32), 0

        slot_mapping.fill(SLOT_MAPPING_PADDING_VAL)

        if num_requests <= 0:
            return (slot_mapping.copy(), 0) if copy_out else (slot_mapping, 0)

        scheduled_active = np.asarray(scheduled[:num_requests], dtype=np.int32)
        num_computed_active = np.asarray(num_computed_tokens_cpu[:num_requests], dtype=np.int32)
        if not np.any(scheduled_active):
            return (slot_mapping.copy(), 0) if copy_out else (slot_mapping, 0)

        page_size = int(self.metadata.page_size)
        max_pages_per_req = int(self.metadata.max_num_pages_per_req)
        slices_per_page = int(self._slices_per_page or 1)

        start_tokens = num_computed_active
        end_tokens = num_computed_active + scheduled_active
        lps = start_tokens // page_size
        lpe = (np.maximum(end_tokens, 1) - 1) // page_size
        page_lens = np.where(scheduled_active > 0, lpe - lps + 1, 0).astype(np.int32)

        if not np.any(page_lens):
            return (slot_mapping.copy(), 0) if copy_out else (slot_mapping, 0)

        page_cum = np.cumsum(page_lens, dtype=np.int32)
        total_pages = int(page_cum[-1])

        max_padded_slices = int(self._max_padded_slices or 1)
        padded_num_slices = min(
            ((total_pages + slices_per_page - 1) // slices_per_page) * slices_per_page,
            max_padded_slices,
        )

        indices = self._slot_mapping_indices
        if indices is None:
            return slot_mapping.copy(), total_pages

        slice_active_mask = (indices < total_pages) & (indices < padded_num_slices)
        active_positions = indices[slice_active_mask]
        if active_positions.size == 0:
            return (slot_mapping.copy(), total_pages) if copy_out else (slot_mapping, total_pages)

        page_cum_prev = np.concatenate(([0], page_cum[:-1]))
        req_for_slice = np.searchsorted(page_cum, active_positions, side="right").astype(np.int32)
        local_off = active_positions - page_cum_prev[req_for_slice]

        pt = np.asarray(page_table_cpu[:num_requests, :max_pages_per_req], dtype=np.int32)
        pt_flat = pt.reshape(-1)
        gather_idx = req_for_slice * max_pages_per_req + lps[req_for_slice] + local_off
        np.clip(gather_idx, 0, pt_flat.size - 1, out=gather_idx)
        page_numbers = pt_flat[gather_idx]

        s_mod = start_tokens % page_size
        e_mod = ((np.maximum(end_tokens, 1) - 1) % page_size) + 1
        lens_rep = page_lens

        is_first = local_off == 0
        is_last = local_off == (lens_rep[req_for_slice] - 1)

        kv_local_st = np.where(is_first, s_mod[req_for_slice], 0)
        kv_local_en = np.where(is_last, e_mod[req_for_slice], page_size)
        slice_lens = np.maximum(kv_local_en - kv_local_st, 0).astype(np.int32)
        kv_cache_start = kv_local_st + page_numbers * page_size

        new_kv_start = np.cumsum(slice_lens, dtype=np.int32)
        if new_kv_start.size:
            new_kv_start = np.roll(new_kv_start, 1)
            new_kv_start[0] = 0

        slot_mapping[0, active_positions] = kv_cache_start
        slot_mapping[1, active_positions] = new_kv_start
        slot_mapping[2, active_positions] = slice_lens

        return (slot_mapping.copy(), total_pages) if copy_out else (slot_mapping, total_pages)

    def _build_host_payload(
        self,
        *,
        num_tokens_static: int,
        scheduled_full_cpu: np.ndarray,
        active_mask_full_cpu: np.ndarray,
        token_ids_cpu: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        temperature_cpu: np.ndarray,
        top_p_cpu: np.ndarray,
        top_k_cpu: np.ndarray,
        min_p_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        page_table_version: int | None,
        padded_num_reqs_in: int,
        copy_slot_mapping: bool,
        use_async_buffers: bool = False,
    ) -> tuple[tuple, int, int, int]:
        """Build the host-side payload for device transfer.

        This is the hot-path function that computes all batch metadata using
        preallocated CPU buffers. It packs tokens, computes positions,
        prepares page tables, and extracts sampling parameters.

        Args:
            num_tokens_static: Static token count for this batch (bucket size).
            scheduled_full_cpu: Tokens scheduled per request.
            active_mask_full_cpu: Boolean mask for active requests.
            token_ids_cpu: All token IDs for all requests [max_reqs, max_len].
            num_computed_tokens_cpu: Tokens already computed per request.
            temperature_cpu: Temperature sampling parameters per request.
            top_p_cpu: Top-p sampling parameters per request.
            top_k_cpu: Top-k sampling parameters per request.
            min_p_cpu: Min-p sampling parameters per request.
            page_table_cpu: Page table [max_reqs, max_pages_per_req].
            page_table_version: Optional version for cache invalidation.
            padded_num_reqs_in: Requested padding for request count.
            copy_slot_mapping: If True, copy slot mapping to avoid aliasing.
            use_async_buffers: If True, use async staging buffers instead
                of primary buffers.

        Returns:
            Tuple of (host_payload, padded_num_reqs, num_requests, rows_to_copy)
            where host_payload is a tuple ready for device_put.

        Raises:
            ValueError: If scheduled tokens exceed num_tokens_static or if
                token_ids_cpu doesn't have enough columns for a request.

        Note:
            This function modifies preallocated buffers in place for performance.
            The returned payload references these buffers directly (unless
            copy_slot_mapping=True for slot_mapping).
        """

        max_num_reqs = int(self.max_num_reqs)
        active_rows = np.flatnonzero(np.asarray(active_mask_full_cpu[:max_num_reqs], dtype=np.bool_))
        # Preserve highest live row index (+1) so sparse row layouts remain aligned
        # with page-table row IDs under DP-local page sharding.
        num_requests = int(active_rows[-1]) + 1 if active_rows.size > 0 else 0
        num_requests = min(num_requests, max_num_reqs)

        padded_num_reqs = AttentionMetadataBuilder.compute_padded_num_reqs(
            num_requests=num_requests,
            max_num_reqs=max_num_reqs,
            min_input_pad=int(self.min_input_pad),
            padded_num_reqs_in=int(padded_num_reqs_in),
        )

        if use_async_buffers:
            scheduled = self._async_scheduled_cpu
            qsl = self._async_query_start_loc_cpu
            input_ids = self._async_input_ids_cpu[: int(num_tokens_static)]
            positions = self._async_positions_cpu[: int(num_tokens_static)]
            seq_lens = self._async_seq_lens_cpu
            logits_indices = self._async_logits_indices_cpu
            pages_tables = self._async_pages_tables_cpu
            packed_qsl_seqlens = self._async_packed_qsl_seqlens_cpu
            packed_i32_padded = self._async_packed_i32_padded_cpu
            packed_f32_padded = self._async_packed_f32_padded_cpu
            packed_misc_i32 = self._async_packed_misc_i32_cpu
            request_distribution = self._async_request_distribution_cpu
            num_kv_update_cpu = self._async_num_kv_update_cpu
            slot_mapping_buf = self._async_slot_mapping_cpu
        else:
            scheduled = self._scheduled_cpu
            qsl = self._query_start_loc_cpu
            input_ids = self._input_ids_cpu[: int(num_tokens_static)]
            positions = self._positions_cpu[: int(num_tokens_static)]
            seq_lens = self._seq_lens_cpu
            logits_indices = self._logits_indices_cpu
            pages_tables = self._pages_tables_cpu
            packed_qsl_seqlens = self._packed_qsl_seqlens_cpu
            packed_i32_padded = self._packed_i32_padded_cpu
            packed_f32_padded = self._packed_f32_padded_cpu
            packed_misc_i32 = self._packed_misc_i32_cpu
            request_distribution = self._request_distribution_placeholder
            num_kv_update_cpu = self._num_kv_update_placeholder
            slot_mapping_buf = self._slot_mapping_cpu

        # scheduled: only active prefix is meaningful.
        scheduled.fill(0)
        if num_requests > 0:
            scheduled[:num_requests] = scheduled_full_cpu[:num_requests]

        # query_start_loc: prefix sums of scheduled tokens.
        qsl.fill(0)
        if num_requests > 0:
            np.cumsum(scheduled[:num_requests], out=qsl[1 : num_requests + 1])
            qsl[num_requests + 1 :] = qsl[num_requests]

        actual_num_tokens = int(qsl[num_requests]) if num_requests > 0 else 0
        if actual_num_tokens > int(num_tokens_static):
            raise ValueError(
                f"Scheduled {actual_num_tokens} tokens but `num_tokens_static`={num_tokens_static}; "
                "select a larger token bucket."
            )

        # Contiguous token batch: fill only the current bucket size.
        input_ids.fill(0)
        positions.fill(0)

        off = 0
        for req_idx in range(num_requests):
            n = int(scheduled[req_idx])
            if n <= 0:
                continue
            start = int(num_computed_tokens_cpu[req_idx])
            end = start + n
            if end > token_ids_cpu.shape[1]:
                raise ValueError(
                    f"Request {req_idx} scheduled [{start}:{end}] exceeds token_ids width {token_ids_cpu.shape[1]}."
                )
            positions[off : off + n] = start + self._arange_cpu[:n]
            input_ids[off : off + n] = token_ids_cpu[req_idx, start:end]
            off += n

        # seq_lens: computed tokens after this forward (start + scheduled).
        seq_lens.fill(0)
        if num_requests > 0:
            np.add(
                num_computed_tokens_cpu[:num_requests],
                scheduled[:num_requests],
                out=seq_lens[:num_requests],
                dtype=np.int32,
            )
            self._enforce_dp_local_page_tables(
                num_requests=num_requests,
                scheduled=scheduled,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                page_table_cpu=page_table_cpu,
            )

        # logits_indices: last token index per request in the packed batch.
        logits_indices.fill(0)
        if num_requests > 0:
            logits_indices[:num_requests] = qsl[1 : num_requests + 1] - 1

        # pages_tables: copy active rows, pad inactive.
        rows_to_copy = min(num_requests, int(self._num_reqs_max_model_len), int(page_table_cpu.shape[0]))
        reuse_pages_tables = (
            page_table_version is not None
            and self._cached_pages_tables_dev is not None
            and self._cached_page_table_version == int(page_table_version)
            and self._cached_rows_to_copy == int(rows_to_copy)
        )
        if reuse_pages_tables:
            pages_tables_payload = self._cached_pages_tables_dev
        else:
            pages_tables.fill(int(PAGE_TABLE_PADDING_VAL))
            if rows_to_copy > 0:
                pages_tables[:rows_to_copy, :] = page_table_cpu[:rows_to_copy, : pages_tables.shape[1]]
            pages_tables_payload = pages_tables

        # request_distribution (v3) / slot_mapping (v2)
        request_distribution.fill(0)
        slot_mapping_cpu = None
        num_kv_update_cpu.fill(0)

        if self._use_request_distribution:
            if num_requests > 0:
                is_decode = (scheduled[:num_requests] == 1) & (num_computed_tokens_cpu[:num_requests] > 0)
                decode_count = int(np.sum(is_decode))
            else:
                decode_count = 0
            request_distribution[0] = decode_count
            request_distribution[1] = decode_count
            request_distribution[2] = num_requests

        if self._use_slot_mapping:
            slot_mapping_cpu, total_pages = self._compute_slot_mapping_v2(
                num_requests=num_requests,
                scheduled=scheduled,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                page_table_cpu=page_table_cpu,
                slot_mapping_out=slot_mapping_buf,
                copy_out=copy_slot_mapping,
            )
            num_kv_update_cpu[0] = int(total_pages)

        slot_mapping_placeholder = (
            self._async_slot_mapping_placeholder if use_async_buffers else self._slot_mapping_placeholder
        )

        # Pack small vectors/scalars into dense buffers to reduce PyTree leaf count.
        packed_qsl_seqlens[0] = qsl
        packed_qsl_seqlens[1].fill(0)
        packed_qsl_seqlens[1, :-1] = seq_lens

        packed_i32_padded[:, :padded_num_reqs].fill(0)
        packed_i32_padded[0, :padded_num_reqs] = scheduled[:padded_num_reqs]
        packed_i32_padded[1, :padded_num_reqs] = logits_indices[:padded_num_reqs]
        packed_i32_padded[2, :padded_num_reqs] = top_k_cpu[:padded_num_reqs]

        packed_f32_padded[:, :padded_num_reqs].fill(0)
        packed_f32_padded[0, :padded_num_reqs] = temperature_cpu[:padded_num_reqs]
        packed_f32_padded[1, :padded_num_reqs] = top_p_cpu[:padded_num_reqs]
        packed_f32_padded[2, :padded_num_reqs] = min_p_cpu[:padded_num_reqs]

        packed_misc_i32.fill(0)
        packed_misc_i32[0] = np.int32(num_requests)
        packed_misc_i32[1] = np.int32(padded_num_reqs)
        packed_misc_i32[2:5] = request_distribution

        if self._use_slot_mapping:
            host_payload = (
                input_ids,
                positions,
                packed_qsl_seqlens,
                pages_tables_payload,
                packed_i32_padded[:, :padded_num_reqs],
                packed_f32_padded[:, :padded_num_reqs],
                packed_misc_i32,
                slot_mapping_cpu if slot_mapping_cpu is not None else slot_mapping_placeholder,
                num_kv_update_cpu,
                scheduled_full_cpu,
                active_mask_full_cpu,
            )
        else:
            host_payload = (
                input_ids,
                positions,
                packed_qsl_seqlens,
                pages_tables_payload,
                packed_i32_padded[:, :padded_num_reqs],
                packed_f32_padded[:, :padded_num_reqs],
                packed_misc_i32,
                scheduled_full_cpu,
                active_mask_full_cpu,
            )

        return host_payload, padded_num_reqs, num_requests, rows_to_copy

    def prepare_batch_metadata(
        self,
        *,
        num_tokens_static: int,
        scheduled_full_cpu: np.ndarray,
        active_mask_full_cpu: np.ndarray,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        token_ids_cpu: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        temperature_cpu: np.ndarray,
        top_p_cpu: np.ndarray,
        top_k_cpu: np.ndarray,
        min_p_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        padded_num_reqs_in: int,
        page_table_version: int | None = None,
        # VLM prefill helpers (optional)
        mrope_position_ids_cpu: np.ndarray | None = None,
        prefill_embeds_cpu: np.ndarray | None = None,
        prefill_embeds_mask_cpu: np.ndarray | None = None,
        # DeepStack-style visual injection (optional)
        visual_pos_masks_cpu: np.ndarray | None = None,
        deepstack_visual_embeds_cpu: list[np.ndarray] | None = None,
        # Vision-language model data (optional)
        pixel_values: np.ndarray | None = None,
        image_grid_thw: np.ndarray | None = None,
        pixel_values_videos: np.ndarray | None = None,
        video_grid_thw: np.ndarray | None = None,
    ) -> tuple[BatchMetadata, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Prepare batch metadata using CPU-first computation strategy.

        This is the main entry point for synchronous batch preparation. It
        builds all metadata on CPU using preallocated buffers, then transfers
        everything to device in a single consolidated device_put call.

        Args:
            num_tokens_static: Static token count for bucket selection.
            scheduled_full_cpu: Tokens scheduled per request [max_num_reqs].
            active_mask_full_cpu: Boolean mask for active requests [max_num_reqs].
            input_ids_buf: Device buffer for input token IDs (reused).
            position_ids_buf: Device buffer for position IDs (reused).
            token_ids_cpu: All token IDs [max_num_reqs, max_model_len].
            num_computed_tokens_cpu: Computed tokens per request [max_num_reqs].
            temperature_cpu: Temperature per request [max_num_reqs].
            top_p_cpu: Top-p per request [max_num_reqs].
            top_k_cpu: Top-k per request [max_num_reqs].
            min_p_cpu: Min-p per request [max_num_reqs].
            page_table_cpu: Page table [max_num_reqs, max_pages_per_req].
            padded_num_reqs_in: Requested padding for request count bucketing.
            page_table_version: Optional version for page table caching.
            mrope_position_ids_cpu: Optional mRoPE positions [3, num_tokens].
            prefill_embeds_cpu: Optional prefill embeddings [num_tokens, hidden].
            prefill_embeds_mask_cpu: Optional mask for prefill [num_tokens].
            visual_pos_masks_cpu: Optional visual position masks [num_tokens].
            deepstack_visual_embeds_cpu: Optional DeepStack embeddings list.
            pixel_values: Optional raw image pixel values for VLMs.
            image_grid_thw: Optional image grid shape for VLMs.
            pixel_values_videos: Optional raw video pixel values for VLMs.
            video_grid_thw: Optional video grid shape for VLMs.

        Returns:
            Tuple of (batch_metadata, input_ids_buf, position_ids_buf,
            scheduled_full_dev, active_mask_full_dev) with all tensors
            device-resident and ready for model execution.

        Note:
            Updates self.last_prep_stats with timing information:
            - prep_host_time: CPU computation time
            - prep_put_time: Main device_put time
            - prep_extra_put_time: VLM data transfer time
        """
        host_build_start = time.time()
        host_payload, _padded_num_reqs, _num_requests, rows_to_copy = self._build_host_payload(
            num_tokens_static=int(num_tokens_static),
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            token_ids_cpu=token_ids_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            temperature_cpu=temperature_cpu,
            top_p_cpu=top_p_cpu,
            top_k_cpu=top_k_cpu,
            min_p_cpu=min_p_cpu,
            page_table_cpu=page_table_cpu,
            page_table_version=page_table_version,
            padded_num_reqs_in=int(padded_num_reqs_in),
            copy_slot_mapping=False,
        )
        host_build_took = time.time() - host_build_start

        device_put_start = time.time()

        slot_mapping_dev = None
        num_kv_update_dev = None
        if self._use_slot_mapping:
            (
                input_ids_buf,
                position_ids_buf,
                packed_qsl_seqlens_dev,
                pages_tables_dev,
                packed_i32_padded_dev,
                packed_f32_padded_dev,
                packed_misc_i32_dev,
                slot_mapping_dev,
                num_kv_update_dev,
                scheduled_full_dev,
                active_mask_full_dev,
            ) = jax.device_put(host_payload, self._empty_sharding)
        else:
            (
                input_ids_buf,
                position_ids_buf,
                packed_qsl_seqlens_dev,
                pages_tables_dev,
                packed_i32_padded_dev,
                packed_f32_padded_dev,
                packed_misc_i32_dev,
                scheduled_full_dev,
                active_mask_full_dev,
            ) = jax.device_put(host_payload, self._empty_sharding)
        device_put_took = time.time() - device_put_start

        # Cache device `pages_tables` when the caller provides a version counter.
        if page_table_version is not None:
            self._cached_pages_tables_dev = pages_tables_dev
            self._cached_page_table_version = int(page_table_version)
            self._cached_rows_to_copy = int(rows_to_copy)

        mrope_position_ids_dev = None
        prefill_embeds_dev = None
        prefill_embeds_mask_dev = None
        extra_put_start = time.time()
        if mrope_position_ids_cpu is not None:
            mrope_position_ids_dev = jax.device_put(mrope_position_ids_cpu, self._empty_sharding)

        # Prefill embedding overrides are rare outside of multimodal prompt regions.
        # Keep device buffers stable, but only transfer when the mask selects any rows.
        if prefill_embeds_cpu is not None:
            prefill_embeds_dev = self._get_zero_dev_like(namespace="prefill_embeds", arr=prefill_embeds_cpu)
            if prefill_embeds_mask_cpu is not None:
                prefill_embeds_mask_dev = self._get_zero_dev_like(
                    namespace="prefill_embeds_mask", arr=prefill_embeds_mask_cpu
                )
                if bool(prefill_embeds_mask_cpu.any()):
                    prefill_embeds_dev = jax.device_put(prefill_embeds_cpu, self._empty_sharding)
                    prefill_embeds_mask_dev = jax.device_put(prefill_embeds_mask_cpu, self._empty_sharding)
            else:
                prefill_embeds_mask_dev = self._get_zero_dev(
                    namespace="prefill_embeds_mask",
                    shape=(prefill_embeds_cpu.shape[0],),
                    dtype=np.bool_,
                )

        visual_pos_masks_dev = None
        deepstack_visual_embeds_dev = None
        if visual_pos_masks_cpu is not None:
            visual_pos_masks_dev = self._get_zero_dev_like(namespace="visual_pos_masks", arr=visual_pos_masks_cpu)
            if bool(visual_pos_masks_cpu.any()):
                visual_pos_masks_dev = jax.device_put(visual_pos_masks_cpu, self._empty_sharding)
        if deepstack_visual_embeds_cpu is not None:
            deepstack_visual_embeds_dev = tuple(
                self._get_zero_dev_like(namespace=f"deepstack_visual_embeds:{idx}", arr=arr)
                for idx, arr in enumerate(deepstack_visual_embeds_cpu)
            )
            if visual_pos_masks_cpu is not None and bool(visual_pos_masks_cpu.any()):
                deepstack_visual_embeds_dev = tuple(
                    jax.device_put(arr, self._empty_sharding) for arr in deepstack_visual_embeds_cpu
                )

        pixel_values_dev = None
        image_grid_thw_dev = None
        pixel_values_videos_dev = None
        video_grid_thw_dev = None
        if pixel_values is not None:
            pixel_values_dev = jax.device_put(pixel_values, self._empty_sharding)
        if image_grid_thw is not None:
            image_grid_thw_dev = jax.device_put(image_grid_thw, self._empty_sharding)
        if pixel_values_videos is not None:
            pixel_values_videos_dev = jax.device_put(pixel_values_videos, self._empty_sharding)
        if video_grid_thw is not None:
            video_grid_thw_dev = jax.device_put(video_grid_thw, self._empty_sharding)
        extra_put_took = time.time() - extra_put_start

        # Expose a timing breakdown to the execution manager (for perf logs).
        # Keep in seconds to match other internal metrics.
        self.last_prep_stats = {
            "prep_host_time": float(host_build_took),
            "prep_put_time": float(device_put_took),
            "prep_extra_put_time": float(extra_put_took),
        }

        metadata = BatchMetadata(
            packed_qsl_seqlens=packed_qsl_seqlens_dev,
            packed_i32_padded=packed_i32_padded_dev,
            packed_f32_padded=packed_f32_padded_dev,
            packed_misc_i32=packed_misc_i32_dev,
            pages_tables=pages_tables_dev,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            slot_mapping=slot_mapping_dev if self._use_slot_mapping else None,
            num_kv_update_slices=num_kv_update_dev if self._use_slot_mapping else None,
            pixel_values=pixel_values_dev,
            image_grid_thw=image_grid_thw_dev,
            pixel_values_videos=pixel_values_videos_dev,
            video_grid_thw=video_grid_thw_dev,
            mrope_position_ids=mrope_position_ids_dev,
            prefill_embeds=prefill_embeds_dev,
            prefill_embeds_mask=prefill_embeds_mask_dev,
            visual_pos_masks=visual_pos_masks_dev,
            deepstack_visual_embeds=deepstack_visual_embeds_dev,
        )

        return (
            metadata,
            input_ids_buf,
            position_ids_buf,
            scheduled_full_dev,
            active_mask_full_dev,
        )

    def start_async_prep(
        self,
        *,
        num_tokens_static: int,
        scheduled_full_cpu: np.ndarray,
        active_mask_full_cpu: np.ndarray,
        input_ids_buf: jax.Array,
        position_ids_buf: jax.Array,
        token_ids_cpu: np.ndarray,
        num_computed_tokens_cpu: np.ndarray,
        temperature_cpu: np.ndarray,
        top_p_cpu: np.ndarray,
        top_k_cpu: np.ndarray,
        min_p_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        padded_num_reqs_in: int,
        page_table_version: int | None = None,
    ) -> None:
        """Start async device transfer for double-buffered batch preparation.

        Initiates an asynchronous device transfer for the next batch's metadata
        while the current batch is being processed. This enables pipelining
        of CPU preparation with device computation.

        The async workflow is:
            1. Call start_async_prep() with next batch's data
            2. Continue processing current batch
            3. Call get_async_prep_result() to retrieve the prepared metadata

        Args:
            num_tokens_static: Static token count for the next batch.
            scheduled_full_cpu: Tokens scheduled per request for next batch.
            active_mask_full_cpu: Active request mask for next batch.
            input_ids_buf: Device buffer for input IDs (unused, for API compat).
            position_ids_buf: Device buffer for positions (unused, for API compat).
            token_ids_cpu: Token IDs for all requests.
            num_computed_tokens_cpu: Computed tokens per request.
            temperature_cpu: Temperature per request.
            top_p_cpu: Top-p per request.
            top_k_cpu: Top-k per request.
            min_p_cpu: Min-p per request.
            page_table_cpu: Page table for all requests.
            padded_num_reqs_in: Requested padding for request count.
            page_table_version: Optional version for page table caching.

        Raises:
            RuntimeError: If a previous async prep is still pending. Must call
                get_async_prep_result() before starting another.

        Note:
            This method snapshots mutable CPU arrays into async staging buffers
            to avoid data races. The actual device_put is non-blocking.
        """
        if self._pending_transfer is not None:
            raise RuntimeError("Async prep already in-flight; call get_async_prep_result() before starting another.")

        # Snapshot mutable CPU state into staging buffers (avoid per-call allocations).
        np.copyto(self._async_num_computed_tokens_cpu, num_computed_tokens_cpu)
        np.copyto(self._async_scheduled_full_cpu, scheduled_full_cpu)
        np.copyto(self._async_active_mask_full_cpu, active_mask_full_cpu)
        np.copyto(self._async_temperature_cpu, temperature_cpu)
        np.copyto(self._async_top_p_cpu, top_p_cpu)
        np.copyto(self._async_top_k_cpu, top_k_cpu)
        np.copyto(self._async_min_p_cpu, min_p_cpu)
        np.copyto(self._async_page_table_cpu, page_table_cpu)

        host_build_start = time.time()
        host_payload, padded_num_reqs, _num_requests, rows_to_copy = self._build_host_payload(
            num_tokens_static=int(num_tokens_static),
            scheduled_full_cpu=self._async_scheduled_full_cpu,
            active_mask_full_cpu=self._async_active_mask_full_cpu,
            token_ids_cpu=token_ids_cpu,
            num_computed_tokens_cpu=self._async_num_computed_tokens_cpu,
            temperature_cpu=self._async_temperature_cpu,
            top_p_cpu=self._async_top_p_cpu,
            top_k_cpu=self._async_top_k_cpu,
            min_p_cpu=self._async_min_p_cpu,
            page_table_cpu=self._async_page_table_cpu,
            page_table_version=page_table_version,
            padded_num_reqs_in=int(padded_num_reqs_in),
            copy_slot_mapping=False,
            use_async_buffers=True,
        )
        host_build_took = time.time() - host_build_start

        device_put_start = time.time()
        self._pending_transfer = jax.device_put(host_payload, self._empty_sharding)
        device_put_took = time.time() - device_put_start
        self._pending_transfer_metadata = {
            "num_tokens_static": num_tokens_static,
            "padded_num_reqs": padded_num_reqs,
            "page_table_version": page_table_version,
            "rows_to_copy": rows_to_copy,
            "host_build_time": float(host_build_took),
            "device_put_time": float(device_put_took),
        }

    def get_async_prep_result(
        self,
    ) -> (
        tuple[
            tuple[BatchMetadata, jax.Array, jax.Array, jax.Array, jax.Array],
            dict,
        ]
        | None
    ):
        """Retrieve results from a previously started async batch preparation.

        Completes an async prep operation started by start_async_prep(),
        returning the prepared batch metadata and associated buffers. This
        may block briefly if the device transfer is not yet complete.

        Returns:
            If an async prep was pending, returns a tuple of:
            - (batch_metadata, input_ids_buf, position_ids_buf,
               scheduled_full_dev, active_mask_full_dev)
            - Metadata dict with timing and configuration information

            If no async prep was pending, returns None.

        Note:
            After calling this method, the async prep state is cleared and
            start_async_prep() can be called again. The returned buffers
            are device-resident and ready for model execution.
        """

        if self._pending_transfer is None:
            return None

        device_put_took = 0.0
        host_build_took = 0.0
        transfer_meta = self._pending_transfer_metadata or {}
        try:
            device_put_took = float(transfer_meta.get("device_put_time", 0.0))
        except Exception:
            device_put_took = 0.0
        try:
            host_build_took = float(transfer_meta.get("host_build_time", 0.0))
        except Exception:
            host_build_took = 0.0

        slot_mapping_dev = None
        num_kv_update_dev = None
        if self._use_slot_mapping:
            (
                input_ids_buf,
                position_ids_buf,
                packed_qsl_seqlens_dev,
                pages_tables_dev,
                packed_i32_padded_dev,
                packed_f32_padded_dev,
                packed_misc_i32_dev,
                slot_mapping_dev,
                num_kv_update_dev,
                scheduled_full_dev,
                active_mask_full_dev,
            ) = self._pending_transfer
        else:
            (
                input_ids_buf,
                position_ids_buf,
                packed_qsl_seqlens_dev,
                pages_tables_dev,
                packed_i32_padded_dev,
                packed_f32_padded_dev,
                packed_misc_i32_dev,
                scheduled_full_dev,
                active_mask_full_dev,
            ) = self._pending_transfer

        # Update `pages_tables` cache when async prep is used.
        pt_ver = transfer_meta.get("page_table_version")
        rows_to_copy = transfer_meta.get("rows_to_copy")
        if pt_ver is not None and rows_to_copy is not None:
            try:
                self._cached_pages_tables_dev = pages_tables_dev
                self._cached_page_table_version = int(pt_ver)
                self._cached_rows_to_copy = int(rows_to_copy)
            except Exception:
                pass

        # For async prep, host build happened in `start_async_prep`.
        self.last_prep_stats = {
            "prep_host_time": float(host_build_took),
            "prep_put_time": float(device_put_took),
            "prep_extra_put_time": 0.0,
        }

        metadata = BatchMetadata(
            packed_qsl_seqlens=packed_qsl_seqlens_dev,
            packed_i32_padded=packed_i32_padded_dev,
            packed_f32_padded=packed_f32_padded_dev,
            packed_misc_i32=packed_misc_i32_dev,
            pages_tables=pages_tables_dev,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            slot_mapping=slot_mapping_dev if self._use_slot_mapping else None,
            num_kv_update_slices=num_kv_update_dev if self._use_slot_mapping else None,
        )

        self._pending_transfer = None
        self._pending_transfer_metadata = None

        return (
            (
                metadata,
                input_ids_buf,
                position_ids_buf,
                scheduled_full_dev,
                active_mask_full_dev,
            ),
            transfer_meta,
        )
