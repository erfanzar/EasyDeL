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
execution manager. The step inputs are built on the host by surgecore's compiled
``HostPayloadBuilder`` (EasyDeL's native host library, GIL released) into
preallocated NumPy staging buffers, and the consolidated payload moves to the
device via `jax.device_put`.

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

import os
import time
import typing as tp

import jax
import numpy as np
from jax import numpy as jnp

from easydel.caching import RaggedPagesCacheConfig, RaggedPagesMetadata, UnifiedAttentionCacheConfig
from easydel.inference.esurge.runners.host_sync import host_payload_broadcast_needed, put_host_payload
from easydel.utils import flags

from ...page_table import PAGE_TABLE_PADDING_VAL, SLOT_MAPPING_PADDING_VAL
from ..async_types import DeviceInputTokenHandoff, DeviceSpecWindowHandoff
from ..execution_types import BatchMetadata, CacheGroupInputs

#: Host-payload slots at or below this size are content-hashed and their
#: device copies reused when unchanged (steady decode keeps most slots
#: byte-identical). Larger slots always transfer: hashing them would cost
#: more than the dispatch it saves.
_PAYLOAD_PUT_CACHE_LIMIT_BYTES = 1 << 16

#: The page-table slot keeps its own version-driven device cache upstream;
#: when it arrives here as a host array that cache already decided a fresh
#: transfer is required (bumped or unknown version), and the content cache
#: must not alias it back to a stale device array.
_PAYLOAD_PAGE_TABLE_INDEX = 3


_batch_metadata_dump_dir: str | None = None


def set_batch_metadata_dump_dir(path: str | None) -> None:
    """Enable (or disable with ``None``) the per-step batch-metadata ring dump.

    Debug API: when set, every ``prepare_batch_metadata`` call writes an
    8-slot ring of ``batchmeta_{step%%8}.npz`` files (scheduled/computed/
    seq_lens/qsl/page_table snapshots) to *path*. This is the capture that
    root-caused the RPA v3 empty-row device hang; pair it with a standalone
    kernel replay to reproduce device faults from live engine metadata.

    Args:
        path: Target directory (created on first write), or ``None`` to
            disable dumping.
    """
    global _batch_metadata_dump_dir
    _batch_metadata_dump_dir = path


def get_batch_metadata_dump_dir() -> str | None:
    """Return the active batch-metadata dump directory (``None`` = off)."""
    return _batch_metadata_dump_dir


def _host_payload_builder(buffers: dict[str, np.ndarray], **config: tp.Any) -> tp.Any:
    """Construct surgecore's step-input builder over one set of staging buffers.

    surgecore is EasyDeL's compiled host library. It is imported here, when a
    preparer (or its async buffer set) is constructed, so ``import easydel`` and
    training without generation never load it.

    Raises:
        ImportError: surgecore or its compiled module is not available.
    """
    try:
        from surgecore import HostPayloadBuilder
    except ImportError as exc:
        raise ImportError(
            "eSurge's batch preparer builds its step inputs with surgecore, EasyDeL's compiled host library "
            "(libs/surgecore). Install the surgecore wheel, or build it with maturin as libs/surgecore/README.md "
            "describes."
        ) from exc
    return HostPayloadBuilder(buffers=buffers, **config)


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
        enable_spec_recurrent_commit: bool = False,
        input_sharding: jax.sharding.Sharding | None = None,
        slot_indexed_state: bool = False,
        recurrent_slot_group_size: int = 1,
        cache_group_configs: tuple[RaggedPagesCacheConfig | UnifiedAttentionCacheConfig, ...] = (),
        layer_to_cache_group: tuple[int, ...] = (),
        host_payloads_replicated: bool = False,
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
            enable_spec_recurrent_commit: If True, the host payload reserves a
                slot for the speculative recurrent commit metadata. Required
                when running the recurrent-drafter speculative decoding path.
            slot_indexed_state: If True (requests own pooled state slots: compressed-window caches and recurrent
                layers that declare the eSurge slot contract), the per-row physical state-slot ids ride the payload as
                ``dp_recurrent_state_indices`` (shape ``[1, rows]``) even
                without SPMD DP, so the model can address per-request slot
                state that survives SequenceBuffer row moves.
            recurrent_slot_group_size: Recurrent slots per request, ``W``. Under rank-major DP every request's group
                lies in its rank's block of ``rows_per_rank * W`` slots and the payload carries rank-local slots.
            cache_group_configs: Independent pool configurations in runtime order.
                Empty preserves the single-table preparation API.
            layer_to_cache_group: Static model-layer to pool-index ownership mapping.
            host_payloads_replicated: This runner's host payloads are replicated
                by its coordinator, so redundant broadcasts can be omitted.

        Note:
            This constructor preallocates all CPU buffers to avoid per-step
            memory allocation. The memory footprint scales with max_num_tokens
            and max_num_reqs.
        """
        self.metadata = metadata
        self.host_payloads_replicated = bool(host_payloads_replicated)
        self.cache_group_configs = tuple(cache_group_configs)
        self.layer_to_cache_group = tuple(layer_to_cache_group)
        if bool(self.cache_group_configs) != bool(self.layer_to_cache_group) or any(
            index < 0 or index >= len(self.cache_group_configs) for index in self.layer_to_cache_group
        ):
            raise ValueError("Grouped cache configs require a valid layer_to_cache_group mapping")
        for config in self.cache_group_configs:
            if int(getattr(config, "data_parallel_size", 1)) != int(getattr(metadata, "data_parallel_size", 1)):
                raise ValueError("Every cache group must use the model batch's data-parallel layout")
            if min(int(config.get_max_num_seqs()), int(max_num_reqs)) != min(
                int(metadata.get_max_num_seqs()), int(max_num_reqs)
            ):
                raise ValueError("Every cache group must use the same request-row capacity")
        self._cache_group_preparers: dict[int, tuple[tuple[int, ...], BatchMetadataPreparer]] = {}
        self._pending_group_result = None
        self._empty_sharding = empty_sharding
        self._input_sharding = input_sharding if input_sharding is not None else empty_sharding

        self.max_num_tokens = int(max_num_tokens)
        self.max_num_reqs = int(max_num_reqs)
        self.max_model_len = int(max_model_len)
        self.min_input_pad = int(min_input_pad)
        self._enable_spec_recurrent_commit = bool(enable_spec_recurrent_commit)
        self._slot_indexed_state = bool(slot_indexed_state)
        self._recurrent_slot_group_size = max(1, int(recurrent_slot_group_size))
        # Non-DP steps carry the per-row physical state slots as ``dp_recurrent_state_indices`` [1, rows].
        self._plain_recurrent_slots = self._slot_indexed_state

        self._metadata_version = metadata.version
        self._use_slot_mapping = self._metadata_version == "v2"
        self._use_request_distribution = not self._use_slot_mapping
        self._enable_dp_local_page_path = flags.get_bool(flags.EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH)
        self._data_parallel_size = max(1, int(getattr(metadata, "data_parallel_size", 1) or 1))

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
        self._packed_f32_padded_cpu = np.zeros((6, self.max_num_reqs), dtype=np.float32)
        self._packed_misc_i32_cpu = np.zeros((5,), dtype=np.int32)
        self._spec_recurrent_commit_cpu = np.zeros((2, self.max_num_reqs), dtype=np.int32)
        self._pages_tables_cpu = np.full(
            (self._num_reqs_max_model_len, self._max_pages_per_req),
            PAGE_TABLE_PADDING_VAL,
            dtype=np.int32,
        )
        self._request_distribution_cpu = np.zeros((3,), dtype=np.int32)
        dp_query_rows = (
            (self._num_reqs_max_model_len // self._data_parallel_size) + 1
            if self._data_parallel_size > 0 and self._num_reqs_max_model_len % self._data_parallel_size == 0
            else 1
        )
        self._dp_query_start_loc_cpu = np.zeros((self._data_parallel_size, dp_query_rows), dtype=np.int32)
        self._dp_request_distribution_cpu = np.zeros((self._data_parallel_size, 3), dtype=np.int32)
        dp_rows_per_rank = max(1, dp_query_rows - 1)
        self._dp_context_lens_cpu = np.zeros((self._data_parallel_size, dp_rows_per_rank), dtype=np.int32)
        self._dp_recurrent_state_indices_cpu = np.zeros_like(self._dp_context_lens_cpu)
        self._dp_model_row_for_request_cpu = np.full((self.max_num_reqs,), -1, dtype=np.int32)
        self._num_kv_update_cpu = np.zeros((1,), dtype=np.int32)
        self._scheduled_full_placeholder_dev = jax.device_put(
            np.zeros((self.max_num_reqs,), dtype=np.int32),
            self._empty_sharding,
        )
        self._active_mask_full_placeholder_dev = jax.device_put(
            np.zeros((self.max_num_reqs,), dtype=np.bool_),
            self._empty_sharding,
        )

        # Async staging buffers are allocated lazily by ``_ensure_async_buffers`` on the first
        # ``start_async_prep`` call, so synchronous-only preparers do not pay for a second
        # full set of CPU buffers.
        self._async_buffers_ready = False

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
        else:
            self._max_padded_slices = None
            self._slot_mapping_cpu = None
            self._slices_per_page = None

        # Double buffering: store pending async device transfer
        self._pending_transfer: tuple | None = None
        self._pending_transfer_metadata: dict | None = None

        # Last-prep timing breakdown (seconds).
        self.last_prep_stats: dict[str, float] = {}

        # Cache device-side zero buffers for optional VLM inputs. This lets us keep
        # the input PyTree stable (non-None fields) without paying host→device cost
        # on steps where those buffers are effectively unused (e.g., all-false masks).
        self._zero_dev_cache: dict[tuple[str, tuple[int, ...], str], jax.Array] = {}
        # Content-keyed device copies of small host-payload slots. In steady
        # decode most metadata slots (sampling params, misc counters, slot
        # indices) are byte-identical step over step; reusing the previous
        # device array skips their per-step device_put dispatch. Metadata is
        # never donated by the compiled step (only kv_pages is), so cached
        # arrays stay valid across calls.
        self._payload_put_cache: dict[int, tuple[bytes, tuple[int, ...], str, jax.Array]] = {}

        # surgecore's step-input builder fills the staging buffers above in place, with the GIL released.
        staging = self._sync_staging_buffers()
        self._builder = _host_payload_builder(staging, **self._host_payload_builder_config())
        self._payload_layout = self._payload_layout_of(staging)

    @property
    def host_payloads_replicated(self) -> bool:
        """Return this preparer's owning runner's host-replication guarantee."""
        return self._host_payloads_replicated

    @host_payloads_replicated.setter
    def host_payloads_replicated(self, enabled: bool) -> None:
        """Update the policy, including already-created grouped/async preparers.

        Engine coordination may be initialized after preparation has warmed up.
        Assigning the engine-owned policy must also update retained children;
        copying it only in child constructors leaves their broadcasts enabled.
        Call between steps, on the same scheduler thread as preparation.
        """
        self._host_payloads_replicated = bool(enabled)
        for _, child in getattr(self, "_cache_group_preparers", {}).values():
            child.host_payloads_replicated = enabled
        async_preparer = getattr(self, "_async_group_preparer", None)
        if async_preparer is not None:
            async_preparer.host_payloads_replicated = enabled

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

    def _device_put_host_payload(self, host_payload: tuple) -> tuple:
        """Transfer only host leaves to device, preserving already-device leaves.

        Iterates over the input tuple and partitions entries into existing
        ``jax.Array`` leaves (preserved as-is) and host-side values (collected
        into a single ``jax.device_put`` call). This avoids redundant transfers
        for cached device buffers like the page-table cache.

        Args:
            host_payload: Tuple of mixed host arrays and device arrays.

        Returns:
            Tuple of the same length where each element is device-resident.
        """
        outputs: list[object | None] = [None] * len(host_payload)
        put_indices: list[int] = []
        put_values: list[object] = []
        put_shardings: list[jax.sharding.Sharding] = []
        pending_cache_keys: dict[int, tuple[bytes, tuple[int, ...], str]] = {}
        for idx, value in enumerate(host_payload):
            if isinstance(value, jax.Array):
                outputs[idx] = value
                continue
            arr = np.asarray(value)
            cache_entry = None
            if idx != _PAYLOAD_PAGE_TABLE_INDEX and arr.nbytes <= _PAYLOAD_PUT_CACHE_LIMIT_BYTES:
                shape = tuple(int(dim) for dim in arr.shape)
                dtype = str(arr.dtype)
                key = arr.tobytes()
                cached = self._payload_put_cache.get(idx)
                if cached is not None and cached[0] == key and cached[1] == shape and cached[2] == dtype:
                    outputs[idx] = cached[3]
                    continue
                cache_entry = (key, shape, dtype)
                pending_cache_keys[idx] = cache_entry
            put_indices.append(idx)
            put_values.append(value)
            put_shardings.append(
                self._input_sharding if self._uses_spmd_dp() and idx in (0, 1) else self._empty_sharding
            )

        if put_values:
            put_results = put_host_payload(
                tuple(put_values),
                tuple(put_shardings),
                host_payloads_replicated=getattr(self, "host_payloads_replicated", None),
            )
            for idx, value in zip(put_indices, put_results, strict=True):
                outputs[idx] = value
                cache_entry = pending_cache_keys.get(idx)
                if cache_entry is not None:
                    self._payload_put_cache[idx] = (*cache_entry, value)

        return tuple(outputs)

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

    def _uses_spmd_dp(self) -> bool:
        """Return whether this step should build rank-major DP model inputs."""
        return self._data_parallel_size > 1

    def _sync_staging_buffers(self) -> dict[str, np.ndarray]:
        """The synchronous buffer set's staging arrays, keyed by surgecore's builder names."""
        buffers = {
            "scheduled": self._scheduled_cpu,
            "query_start_loc": self._query_start_loc_cpu,
            "input_ids": self._input_ids_cpu,
            "positions": self._positions_cpu,
            "seq_lens": self._seq_lens_cpu,
            "logits_indices": self._logits_indices_cpu,
            "pages_tables": self._pages_tables_cpu,
            "packed_qsl_seqlens": self._packed_qsl_seqlens_cpu,
            "packed_i32_padded": self._packed_i32_padded_cpu,
            "packed_f32_padded": self._packed_f32_padded_cpu,
            "packed_misc_i32": self._packed_misc_i32_cpu,
            "spec_recurrent_commit": self._spec_recurrent_commit_cpu,
            "request_distribution": self._request_distribution_cpu,
            "dp_query_start_loc": self._dp_query_start_loc_cpu,
            "dp_request_distribution": self._dp_request_distribution_cpu,
            "dp_context_lens": self._dp_context_lens_cpu,
            "dp_recurrent_state_indices": self._dp_recurrent_state_indices_cpu,
            "dp_model_row_for_request": self._dp_model_row_for_request_cpu,
            "num_kv_update": self._num_kv_update_cpu,
        }
        if self._slot_mapping_cpu is not None:
            buffers["slot_mapping"] = self._slot_mapping_cpu
        return buffers

    def _host_payload_builder_config(self) -> dict[str, tp.Any]:
        """The static layout surgecore's builder takes (shared by the synchronous and async buffer sets)."""
        return {
            "max_num_reqs": self.max_num_reqs,
            "num_reqs_max_model_len": self._num_reqs_max_model_len,
            "data_parallel_size": self._data_parallel_size,
            "use_slot_mapping": self._use_slot_mapping,
            "use_request_distribution": self._use_request_distribution,
            "plain_recurrent_slots": self._plain_recurrent_slots,
            "recurrent_slot_group_size": self._recurrent_slot_group_size,
            "enable_spec_recurrent_commit": self._enable_spec_recurrent_commit,
            "enable_dp_local_page_path": self._enable_dp_local_page_path,
            "page_size": int(getattr(self.metadata, "page_size", 1)),
            "max_num_pages_per_req": int(getattr(self.metadata, "max_num_pages_per_req", 0)),
            "slices_per_page": int(self._slices_per_page or 1),
            "max_padded_slices": int(self._max_padded_slices or 1),
        }

    def _payload_layout_of(self, buffers: dict[str, np.ndarray]) -> tuple:
        """``(input_ids, positions, packed_qsl_seqlens, pages_tables, tail, buffers)`` of one buffer set.

        A step's host payload is ``(input_ids[:T], positions[:T], packed_qsl_seqlens, pages_tables, *tail)``. The
        tail depends only on the preparer's static layout: rank-major DP vectors (or the plain per-row state slots),
        the speculative recurrent commit, then the v2 KV slot mapping and its slice count.
        """
        tail = (buffers["packed_i32_padded"], buffers["packed_f32_padded"], buffers["packed_misc_i32"])
        if self._uses_spmd_dp():
            tail += (
                buffers["dp_query_start_loc"],
                buffers["dp_request_distribution"],
                buffers["dp_context_lens"],
                buffers["dp_recurrent_state_indices"],
            )
        elif self._plain_recurrent_slots:
            tail += (buffers["dp_recurrent_state_indices"],)
        if self._enable_spec_recurrent_commit:
            tail += (buffers["spec_recurrent_commit"],)
        if self._use_slot_mapping:
            tail += (buffers["slot_mapping"], buffers["num_kv_update"])
        return (
            buffers["input_ids"],
            buffers["positions"],
            buffers["packed_qsl_seqlens"],
            buffers["pages_tables"],
            tail,
            buffers,
        )

    def _fill_step_inputs(
        self,
        builder: tp.Any,
        layout: tuple,
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
        frequency_penalties_cpu: np.ndarray,
        presence_penalties_cpu: np.ndarray,
        repetition_penalties_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        page_table_version: int | None,
        padded_num_reqs_in: int,
        window_row_indices_cpu: np.ndarray | None,
        recurrent_slot_indices_cpu: np.ndarray | None,
        spec_recurrent_commit_cpu: np.ndarray | None,
    ) -> tuple[tuple, int, int, int]:
        """Fill one buffer set's staging arrays for a forward window and return its host payload.

        surgecore's builder runs with the GIL released over the preallocated staging arrays. It packs the tokens and
        positions (in request order, or rank-major under SPMD DP), the query start locations and sequence lengths
        (zero for budget-skipped rows), the logits indices, the page-table rows, the positional request distribution
        (v3) or the KV slot mapping (v2), the per-row recurrent state slots and the padded sampling vectors, and
        checks that DP rows only use their shard's pages. It leaves the page-table rows untouched when the device copy
        cached for the same ``page_table_version`` and row count is reused.

        Args:
            builder: The buffer set's ``surgecore.HostPayloadBuilder``.
            layout: The buffer set's :meth:`_payload_layout_of`.
            num_tokens_static: Token bucket of the step.
            scheduled_full_cpu: Scheduled tokens per window row.
            active_mask_full_cpu: Live window rows.
            token_ids_cpu: Token history of the window rows.
            num_computed_tokens_cpu: Computed tokens per window row.
            temperature_cpu: Temperature per row.
            top_p_cpu: Top-p per row.
            top_k_cpu: Top-k per row.
            min_p_cpu: Min-p per row.
            frequency_penalties_cpu: Frequency penalty per row.
            presence_penalties_cpu: Presence penalty per row.
            repetition_penalties_cpu: Repetition penalty per row.
            page_table_cpu: Page table of the window rows.
            page_table_version: Version of ``page_table_cpu`` for the device page-table cache, or ``None``.
            padded_num_reqs_in: Requested request-count bucket.
            window_row_indices_cpu: Global row of each window row (rank-major DP and the DP-local page check).
            recurrent_slot_indices_cpu: Physical recurrent state slot of each window row.
            spec_recurrent_commit_cpu: Optional speculative recurrent commit metadata ``[2, rows]``.

        Returns:
            ``(host_payload, padded_num_reqs, num_requests, rows_to_copy)``. ``host_payload`` references the staging
            arrays, and the cached device page table when it is reused.

        Raises:
            ValueError: The window does not fit the token bucket, the DP layout, the token history or its pages.
        """
        num_tokens_static = int(num_tokens_static)
        page_table_cached = self._cached_pages_tables_dev is not None
        num_requests, padded_num_reqs, rows_to_copy, reuse_pages_tables, _total_pages = builder.build(
            num_tokens_static=num_tokens_static,
            scheduled_full=scheduled_full_cpu,
            active_mask_full=active_mask_full_cpu,
            token_ids=token_ids_cpu,
            num_computed_tokens=num_computed_tokens_cpu,
            temperature=temperature_cpu,
            top_p=top_p_cpu,
            top_k=top_k_cpu,
            min_p=min_p_cpu,
            frequency_penalties=frequency_penalties_cpu,
            presence_penalties=presence_penalties_cpu,
            repetition_penalties=repetition_penalties_cpu,
            page_table=page_table_cpu,
            padded_num_reqs_in=int(padded_num_reqs_in),
            window_row_indices=window_row_indices_cpu,
            recurrent_slot_indices=recurrent_slot_indices_cpu,
            spec_recurrent_commit=spec_recurrent_commit_cpu,
            page_table_version=None if page_table_version is None else int(page_table_version),
            cached_page_table_version=self._cached_page_table_version if page_table_cached else None,
            cached_rows_to_copy=self._cached_rows_to_copy,
            num_pages=int(getattr(self.metadata, "num_pages", 0) or 0),
            metadata_data_parallel_size=int(getattr(self.metadata, "data_parallel_size", 1) or 1),
        )
        input_ids, positions, packed_qsl_seqlens, pages_tables, tail, buffers = layout
        dump_dir = get_batch_metadata_dump_dir()
        if dump_dir:
            self._dump_batch_metadata(
                dump_dir,
                buffers,
                num_requests=num_requests,
                num_tokens_static=num_tokens_static,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                page_table_cpu=page_table_cpu,
            )
        host_payload = (
            input_ids[:num_tokens_static],
            positions[:num_tokens_static],
            packed_qsl_seqlens,
            self._cached_pages_tables_dev if reuse_pages_tables else pages_tables,
            *tail,
        )
        return host_payload, padded_num_reqs, num_requests, rows_to_copy

    def _dump_batch_metadata(
        self,
        dump_dir: str,
        buffers: dict[str, np.ndarray],
        *,
        num_requests: int,
        num_tokens_static: int,
        num_computed_tokens_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
    ) -> None:
        """Write one slot of the ``batchmeta_{step % 8}.npz`` ring (see :func:`set_batch_metadata_dump_dir`)."""
        os.makedirs(dump_dir, exist_ok=True)
        self._meta_dump_step = getattr(self, "_meta_dump_step", -1) + 1
        rows = max(num_requests, 1)
        np.savez(
            f"{dump_dir}/batchmeta_{self._meta_dump_step % 8}.npz",
            step=np.int64(self._meta_dump_step),
            num_requests=np.int64(num_requests),
            num_tokens_static=np.int64(num_tokens_static),
            scheduled=buffers["scheduled"][:rows].copy(),
            computed=num_computed_tokens_cpu[:rows].copy(),
            seq_lens=buffers["seq_lens"].copy(),
            qsl=buffers["query_start_loc"].copy(),
            page_table=page_table_cpu[:rows, :64].copy(),
        )

    def _prepare_cache_groups(
        self,
        result: tuple,
        cache_group_inputs: tuple[CacheGroupInputs, ...],
        **kwargs: tp.Any,
    ) -> tuple:
        """Attach independently prepared read/write payloads in static group order.

        Each pool owns its transfer/version cache. Window identity is part of
        that cache's identity, so equal allocator versions from different row
        packings cannot alias. Child preparers perform the same inactive-row
        padding, rank-major DP packing and v2 slot construction as a single pool.
        """
        if len(cache_group_inputs) != len(self.cache_group_configs):
            raise ValueError("cache_group_inputs must contain one payload per configured cache group")
        metadata, *rest = result
        stats = dict(self.last_prep_stats)
        groups = []
        row_count = int(kwargs["token_ids_cpu"].shape[0])
        for index, (config, group) in enumerate(zip(self.cache_group_configs, cache_group_inputs, strict=True)):
            if group.page_table_cpu.ndim != 2 or group.page_table_cpu.shape[0] != row_count:
                raise ValueError("Every group page table must align with the compact token window rows")
            identity = group.window_identity
            if identity is None:
                rows = kwargs.get("window_row_indices_cpu")
                identity = tuple(range(row_count)) if rows is None else tuple(int(row) for row in rows[:row_count])
            if len(identity) != row_count:
                raise ValueError("Cache group window identity must identify every compact token window row")
            if group.num_computed_tokens_cpu is not None and group.num_computed_tokens_cpu.shape != (row_count,):
                raise ValueError("Cache group computed progress must align with the compact token window rows")
            cached = self._cache_group_preparers.get(index)
            if cached is None or cached[0] != identity:
                child = BatchMetadataPreparer(
                    metadata=config,
                    empty_sharding=self._empty_sharding,
                    max_num_tokens=self.max_num_tokens,
                    max_num_reqs=self.max_num_reqs,
                    max_model_len=self.max_model_len,
                    min_input_pad=self.min_input_pad,
                    enable_spec_recurrent_commit=self._enable_spec_recurrent_commit,
                    input_sharding=self._input_sharding,
                    slot_indexed_state=self._slot_indexed_state,
                    recurrent_slot_group_size=self._recurrent_slot_group_size,
                    host_payloads_replicated=self.host_payloads_replicated,
                )
                self._cache_group_preparers[index] = (identity, child)
            else:
                child = cached[1]
            group_kwargs = dict(kwargs, page_table_cpu=group.page_table_cpu, page_table_version=group.page_table_version)
            if group.num_computed_tokens_cpu is not None:
                group_kwargs["num_computed_tokens_cpu"] = group.num_computed_tokens_cpu
            prepared, *_ = child.prepare_batch_metadata(**group_kwargs)
            for key, value in child.last_prep_stats.items():
                stats[key] = stats.get(key, 0.0) + value
            use_dp = child._uses_spmd_dp()
            cap = min(int(config.get_max_num_seqs()), self.max_num_reqs)
            groups.append(
                RaggedPagesMetadata(
                    pages_tables=prepared.pages_tables,
                    context_lens=prepared.rank_context_lens[:cap] if use_dp else prepared.seq_lens[:cap],
                    query_start_loc=prepared.rank_query_start_loc if use_dp else prepared.query_start_loc[: cap + 1],
                    num_seqs=prepared.packed_misc_i32[:1],
                    position_ids=prepared.position_ids_buf,
                    request_distribution=prepared.rank_request_distribution,
                    slot_mapping=prepared.slot_mapping,
                    num_kv_update_slices=prepared.num_kv_update_slices,
                    spec_recurrent_commit=prepared.spec_recurrent_commit,
                    recurrent_state_indices=(
                        prepared.rank_recurrent_state_indices if (use_dp or self._plain_recurrent_slots) else None
                    ),
                    version=config.version,
                    page_size=config.page_size,
                    num_slices_per_kv_cache_update_page=config.num_slices_per_kv_cache_update_page,
                )
            )
        metadata = metadata.replace(cache_groups=tuple(groups), layer_to_cache_group=self.layer_to_cache_group)
        self.last_prep_stats = stats
        return (metadata, *rest)

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
        frequency_penalties_cpu: np.ndarray,
        presence_penalties_cpu: np.ndarray,
        repetition_penalties_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        padded_num_reqs_in: int,
        window_row_indices_cpu: np.ndarray | None = None,
        recurrent_slot_indices_cpu: np.ndarray | None = None,
        spec_recurrent_commit_cpu: np.ndarray | None = None,
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
        model_side_inputs_cpu: dict[str, np.ndarray] | None = None,
        cache_group_inputs: tuple[CacheGroupInputs, ...] = (),
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
            frequency_penalties_cpu: Frequency penalty per request [max_num_reqs].
            presence_penalties_cpu: Presence penalty per request [max_num_reqs].
            repetition_penalties_cpu: Repetition penalty per request [max_num_reqs].
            page_table_cpu: Page table [max_num_reqs, max_pages_per_req].
            padded_num_reqs_in: Requested padding for request count bucketing.
            spec_recurrent_commit_cpu: Optional speculative recurrent commit
                metadata [2, max_num_reqs]. Row 0 is the active mask, row 1 is
                the real prefix length before drafts.
            page_table_version: Optional version for page table caching.
            cache_group_inputs: Complete per-pool inputs in configured group order,
                with independent versions, row identities and optional rebased progress.
            mrope_position_ids_cpu: Optional mRoPE positions [3, num_tokens].
            prefill_embeds_cpu: Optional prefill embeddings [num_tokens, hidden].
            prefill_embeds_mask_cpu: Optional mask for prefill [num_tokens].
            visual_pos_masks_cpu: Optional visual position masks [num_tokens].
            deepstack_visual_embeds_cpu: Optional DeepStack embeddings list.
            pixel_values: Optional raw image pixel values for VLMs.
            image_grid_thw: Optional image grid shape for VLMs.
            pixel_values_videos: Optional raw video pixel values for VLMs.
            video_grid_thw: Optional video grid shape for VLMs.
            model_side_inputs_cpu: Optional model-declared per-token side inputs ``{name: [num_tokens, ...]}`` aligned
                with the flat token stream (see ``multimodal/side_inputs.py``).

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
        if len(cache_group_inputs) != len(self.cache_group_configs):
            raise ValueError("cache_group_inputs must contain one payload per configured cache group")
        # Capture only public inputs, before local staging variables exist.
        group_kwargs = None
        if cache_group_inputs:
            group_kwargs = {
                "num_tokens_static": num_tokens_static,
                "scheduled_full_cpu": scheduled_full_cpu,
                "active_mask_full_cpu": active_mask_full_cpu,
                "input_ids_buf": input_ids_buf,
                "position_ids_buf": position_ids_buf,
                "token_ids_cpu": token_ids_cpu,
                "num_computed_tokens_cpu": num_computed_tokens_cpu,
                "temperature_cpu": temperature_cpu,
                "top_p_cpu": top_p_cpu,
                "top_k_cpu": top_k_cpu,
                "min_p_cpu": min_p_cpu,
                "frequency_penalties_cpu": frequency_penalties_cpu,
                "presence_penalties_cpu": presence_penalties_cpu,
                "repetition_penalties_cpu": repetition_penalties_cpu,
                "page_table_cpu": page_table_cpu,
                "page_table_version": page_table_version,
                "padded_num_reqs_in": padded_num_reqs_in,
                "window_row_indices_cpu": window_row_indices_cpu,
                "recurrent_slot_indices_cpu": recurrent_slot_indices_cpu,
                "spec_recurrent_commit_cpu": spec_recurrent_commit_cpu,
            }
        host_build_start = time.time()
        host_payload, _padded_num_reqs, _num_requests, rows_to_copy = self._fill_step_inputs(
            self._builder,
            self._payload_layout,
            num_tokens_static=num_tokens_static,
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            token_ids_cpu=token_ids_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            temperature_cpu=temperature_cpu,
            top_p_cpu=top_p_cpu,
            top_k_cpu=top_k_cpu,
            min_p_cpu=min_p_cpu,
            frequency_penalties_cpu=frequency_penalties_cpu,
            presence_penalties_cpu=presence_penalties_cpu,
            repetition_penalties_cpu=repetition_penalties_cpu,
            page_table_cpu=page_table_cpu,
            page_table_version=page_table_version,
            padded_num_reqs_in=padded_num_reqs_in,
            window_row_indices_cpu=window_row_indices_cpu,
            recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
            spec_recurrent_commit_cpu=spec_recurrent_commit_cpu,
        )
        host_build_took = time.time() - host_build_start

        device_put_start = time.time()

        # Replicated-driver multi-host: broadcast host-side arrays from
        # process 0 so all hosts pass identical data to `device_put` with
        # replicated sharding. Skipped under the ZMQ step-replication plane,
        # where payloads are identical by construction (see host_sync).
        if host_payload_broadcast_needed(self.host_payloads_replicated):
            from jax.experimental import multihost_utils

            host_payload = multihost_utils.broadcast_one_to_all(host_payload)

        device_payload = self._device_put_host_payload(host_payload)

        payload_idx = 0
        input_ids_buf = device_payload[payload_idx]
        payload_idx += 1
        position_ids_buf = device_payload[payload_idx]
        payload_idx += 1
        packed_qsl_seqlens_dev = device_payload[payload_idx]
        payload_idx += 1
        pages_tables_dev = device_payload[payload_idx]
        payload_idx += 1
        packed_i32_padded_dev = device_payload[payload_idx]
        payload_idx += 1
        packed_f32_padded_dev = device_payload[payload_idx]
        payload_idx += 1
        packed_misc_i32_dev = device_payload[payload_idx]
        payload_idx += 1
        dp_query_start_loc_dev = None
        dp_request_distribution_dev = None
        dp_context_lens_dev = None
        dp_recurrent_state_indices_dev = None
        if self._uses_spmd_dp():
            dp_query_start_loc_dev = device_payload[payload_idx]
            payload_idx += 1
            dp_request_distribution_dev = device_payload[payload_idx]
            payload_idx += 1
            dp_context_lens_dev = device_payload[payload_idx]
            payload_idx += 1
            dp_recurrent_state_indices_dev = device_payload[payload_idx]
            payload_idx += 1
        elif self._plain_recurrent_slots:
            dp_recurrent_state_indices_dev = device_payload[payload_idx]
            payload_idx += 1
        spec_recurrent_commit_dev = None
        if self._enable_spec_recurrent_commit:
            spec_recurrent_commit_dev = device_payload[payload_idx]
            payload_idx += 1
        slot_mapping_dev = None
        num_kv_update_dev = None
        if self._use_slot_mapping:
            slot_mapping_dev = device_payload[payload_idx]
            payload_idx += 1
            num_kv_update_dev = device_payload[payload_idx]
            payload_idx += 1
        scheduled_full_dev = self._scheduled_full_placeholder_dev
        active_mask_full_dev = self._active_mask_full_placeholder_dev
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
        model_side_inputs_dev = None
        if model_side_inputs_cpu:
            model_side_inputs_dev = {
                name: jax.device_put(value, self._empty_sharding) for name, value in model_side_inputs_cpu.items()
            }
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
            dp_query_start_loc=dp_query_start_loc_dev,
            dp_request_distribution=dp_request_distribution_dev,
            dp_context_lens=dp_context_lens_dev,
            dp_recurrent_state_indices=dp_recurrent_state_indices_dev,
            input_token_handoff_positions=self._get_zero_dev(
                namespace="input_token_handoff_positions",
                shape=(self.max_num_reqs,),
                dtype=np.int32,
            ),
            input_token_handoff_ids=self._get_zero_dev(
                namespace="input_token_handoff_ids",
                shape=(self.max_num_reqs,),
                dtype=np.int32,
            ),
            input_token_handoff_count=self._get_zero_dev(
                namespace="input_token_handoff_count",
                shape=(),
                dtype=np.int32,
            ),
            input_token_handoff_offset=self._get_zero_dev(
                namespace="input_token_handoff_offset",
                shape=(),
                dtype=np.int32,
            ),
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
            model_side_inputs=model_side_inputs_dev,
            spec_recurrent_commit=spec_recurrent_commit_dev,
        )

        result = (metadata, input_ids_buf, position_ids_buf, scheduled_full_dev, active_mask_full_dev)
        if group_kwargs is not None:
            return self._prepare_cache_groups(result, cache_group_inputs, **group_kwargs)
        return result

    def _ensure_async_buffers(self) -> None:
        """Allocate the async (double-buffer) CPU staging buffers on first use.

        Synchronous-only preparers never call :meth:`start_async_prep`, so the
        second full set of CPU buffers is only materialized when the async path
        is actually exercised. Idempotent.
        """
        if self._async_buffers_ready:
            return
        max_reqs = self.max_num_reqs
        self._async_input_ids_cpu = np.zeros((self.max_num_tokens,), dtype=np.int32)
        self._async_positions_cpu = np.zeros((self.max_num_tokens,), dtype=np.int32)
        self._async_query_start_loc_cpu = np.zeros((max_reqs + 1,), dtype=np.int32)
        self._async_seq_lens_cpu = np.zeros((max_reqs,), dtype=np.int32)
        self._async_logits_indices_cpu = np.zeros((max_reqs,), dtype=np.int32)
        self._async_scheduled_cpu = np.zeros((max_reqs,), dtype=np.int32)
        self._async_packed_qsl_seqlens_cpu = np.zeros((2, max_reqs + 1), dtype=np.int32)
        self._async_packed_i32_padded_cpu = np.zeros((3, max_reqs), dtype=np.int32)
        self._async_packed_f32_padded_cpu = np.zeros((6, max_reqs), dtype=np.float32)
        self._async_packed_misc_i32_cpu = np.zeros((5,), dtype=np.int32)
        self._async_spec_recurrent_commit_cpu = np.zeros((2, max_reqs), dtype=np.int32)
        self._async_pages_tables_cpu = np.full(
            (self._num_reqs_max_model_len, self._max_pages_per_req),
            PAGE_TABLE_PADDING_VAL,
            dtype=np.int32,
        )
        self._async_request_distribution_cpu = np.zeros((3,), dtype=np.int32)
        self._async_dp_query_start_loc_cpu = np.zeros_like(self._dp_query_start_loc_cpu)
        self._async_dp_request_distribution_cpu = np.zeros_like(self._dp_request_distribution_cpu)
        self._async_dp_context_lens_cpu = np.zeros_like(self._dp_context_lens_cpu)
        self._async_dp_recurrent_state_indices_cpu = np.zeros_like(self._dp_recurrent_state_indices_cpu)
        self._async_dp_model_row_for_request_cpu = np.full_like(self._dp_model_row_for_request_cpu, -1)
        self._async_num_kv_update_cpu = np.zeros((1,), dtype=np.int32)

        self._async_num_computed_tokens_cpu = np.zeros((max_reqs,), dtype=np.int32)
        self._async_scheduled_full_cpu = np.zeros((max_reqs,), dtype=np.int32)
        self._async_active_mask_full_cpu = np.zeros((max_reqs,), dtype=bool)
        self._async_window_row_indices_cpu = np.arange(max_reqs, dtype=np.int32)
        self._async_recurrent_slot_indices_cpu = np.arange(max_reqs, dtype=np.int32)
        self._async_temperature_cpu = np.zeros((max_reqs,), dtype=np.float32)
        self._async_top_p_cpu = np.zeros((max_reqs,), dtype=np.float32)
        self._async_top_k_cpu = np.zeros((max_reqs,), dtype=np.int32)
        self._async_min_p_cpu = np.zeros((max_reqs,), dtype=np.float32)
        self._async_frequency_penalties_cpu = np.zeros((max_reqs,), dtype=np.float32)
        self._async_presence_penalties_cpu = np.zeros((max_reqs,), dtype=np.float32)
        self._async_repetition_penalties_cpu = np.ones((max_reqs,), dtype=np.float32)
        self._async_page_table_cpu = np.zeros((max_reqs, self._max_pages_per_req), dtype=np.int32)
        if self._slot_mapping_cpu is not None:
            self._async_slot_mapping_cpu = np.full(
                (3, self._max_padded_slices),
                SLOT_MAPPING_PADDING_VAL,
                dtype=np.int32,
            )
        else:
            self._async_slot_mapping_cpu = None
        staging = {
            "scheduled": self._async_scheduled_cpu,
            "query_start_loc": self._async_query_start_loc_cpu,
            "input_ids": self._async_input_ids_cpu,
            "positions": self._async_positions_cpu,
            "seq_lens": self._async_seq_lens_cpu,
            "logits_indices": self._async_logits_indices_cpu,
            "pages_tables": self._async_pages_tables_cpu,
            "packed_qsl_seqlens": self._async_packed_qsl_seqlens_cpu,
            "packed_i32_padded": self._async_packed_i32_padded_cpu,
            "packed_f32_padded": self._async_packed_f32_padded_cpu,
            "packed_misc_i32": self._async_packed_misc_i32_cpu,
            "spec_recurrent_commit": self._async_spec_recurrent_commit_cpu,
            "request_distribution": self._async_request_distribution_cpu,
            "dp_query_start_loc": self._async_dp_query_start_loc_cpu,
            "dp_request_distribution": self._async_dp_request_distribution_cpu,
            "dp_context_lens": self._async_dp_context_lens_cpu,
            "dp_recurrent_state_indices": self._async_dp_recurrent_state_indices_cpu,
            "dp_model_row_for_request": self._async_dp_model_row_for_request_cpu,
            "num_kv_update": self._async_num_kv_update_cpu,
        }
        if self._async_slot_mapping_cpu is not None:
            staging["slot_mapping"] = self._async_slot_mapping_cpu
        # The async set has its own builder, so a pending async payload is never overwritten by a synchronous build.
        self._async_builder = _host_payload_builder(staging, **self._host_payload_builder_config())
        self._async_payload_layout = self._payload_layout_of(staging)
        self._async_buffers_ready = True

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
        frequency_penalties_cpu: np.ndarray,
        presence_penalties_cpu: np.ndarray,
        repetition_penalties_cpu: np.ndarray,
        page_table_cpu: np.ndarray,
        padded_num_reqs_in: int,
        window_row_indices_cpu: np.ndarray | None = None,
        recurrent_slot_indices_cpu: np.ndarray | None = None,
        spec_recurrent_commit_cpu: np.ndarray | None = None,
        page_table_version: int | None = None,
        cache_group_inputs: tuple[CacheGroupInputs, ...] = (),
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
            frequency_penalties_cpu: Frequency penalty per request.
            presence_penalties_cpu: Presence penalty per request.
            repetition_penalties_cpu: Repetition penalty per request.
            page_table_cpu: Page table for all requests.
            padded_num_reqs_in: Requested padding for request count.
            spec_recurrent_commit_cpu: Optional speculative recurrent commit
                metadata [2, max_num_reqs].
            page_table_version: Optional version for page table caching.
            cache_group_inputs: Complete per-pool inputs in configured group order,
                with independent versions, row identities and optional rebased progress.

        Raises:
            RuntimeError: If a previous async prep is still pending. Must call
                get_async_prep_result() before starting another.

        Note:
            This method snapshots mutable CPU arrays into async staging buffers
            to avoid data races. The actual device_put is non-blocking.
        """
        if self._pending_transfer is not None:
            raise RuntimeError("Async prep already in-flight; call get_async_prep_result() before starting another.")

        if len(cache_group_inputs) != len(self.cache_group_configs):
            raise ValueError("cache_group_inputs must contain one payload per configured cache group")
        if cache_group_inputs:
            # Dispatch every pool now, before the caller can mutate its host
            # inputs. Device puts are asynchronous; completed metadata is kept
            # separate from the single-pool double-buffer unpacking ABI.
            snapshot = tuple(
                CacheGroupInputs(
                    page_table_cpu=group.page_table_cpu.copy(),
                    page_table_version=group.page_table_version,
                    window_identity=group.window_identity,
                    num_computed_tokens_cpu=(
                        None if group.num_computed_tokens_cpu is None else group.num_computed_tokens_cpu.copy()
                    ),
                )
                for group in cache_group_inputs
            )
            # Preserve true double buffering: synchronous preparation while
            # this result is pending must never overwrite its staging arrays.
            async_preparer = getattr(self, "_async_group_preparer", None)
            if async_preparer is None:
                async_preparer = BatchMetadataPreparer(
                    metadata=self.metadata,
                    empty_sharding=self._empty_sharding,
                    max_num_tokens=self.max_num_tokens,
                    max_num_reqs=self.max_num_reqs,
                    max_model_len=self.max_model_len,
                    min_input_pad=self.min_input_pad,
                    enable_spec_recurrent_commit=self._enable_spec_recurrent_commit,
                    input_sharding=self._input_sharding,
                    slot_indexed_state=self._slot_indexed_state,
                    recurrent_slot_group_size=self._recurrent_slot_group_size,
                    host_payloads_replicated=self.host_payloads_replicated,
                    cache_group_configs=self.cache_group_configs,
                    layer_to_cache_group=self.layer_to_cache_group,
                )
                self._async_group_preparer = async_preparer
            self._pending_group_result = async_preparer.prepare_batch_metadata(
                num_tokens_static=num_tokens_static,
                scheduled_full_cpu=scheduled_full_cpu,
                active_mask_full_cpu=active_mask_full_cpu,
                input_ids_buf=input_ids_buf,
                position_ids_buf=position_ids_buf,
                token_ids_cpu=token_ids_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                temperature_cpu=temperature_cpu,
                top_p_cpu=top_p_cpu,
                top_k_cpu=top_k_cpu,
                min_p_cpu=min_p_cpu,
                frequency_penalties_cpu=frequency_penalties_cpu,
                presence_penalties_cpu=presence_penalties_cpu,
                repetition_penalties_cpu=repetition_penalties_cpu,
                page_table_cpu=page_table_cpu,
                padded_num_reqs_in=padded_num_reqs_in,
                window_row_indices_cpu=window_row_indices_cpu,
                recurrent_slot_indices_cpu=recurrent_slot_indices_cpu,
                spec_recurrent_commit_cpu=spec_recurrent_commit_cpu,
                page_table_version=page_table_version,
                cache_group_inputs=snapshot,
            )
            self._pending_transfer = ()
            self._pending_transfer_metadata = dict(
                async_preparer.last_prep_stats,
                num_tokens_static=num_tokens_static,
                padded_num_reqs=padded_num_reqs_in,
                page_table_version=page_table_version,
                rows_to_copy=async_preparer._cached_rows_to_copy,
                host_build_time=async_preparer.last_prep_stats.get("prep_host_time", 0.0),
                device_put_time=async_preparer.last_prep_stats.get("prep_put_time", 0.0),
            )
            return

        self._ensure_async_buffers()

        # Snapshot mutable CPU state into staging buffers (avoid per-call allocations).
        np.copyto(self._async_num_computed_tokens_cpu, num_computed_tokens_cpu)
        np.copyto(self._async_scheduled_full_cpu, scheduled_full_cpu)
        np.copyto(self._async_active_mask_full_cpu, active_mask_full_cpu)
        if window_row_indices_cpu is None:
            self._async_window_row_indices_cpu[:] = np.arange(self.max_num_reqs, dtype=np.int32)
        else:
            np.copyto(self._async_window_row_indices_cpu, window_row_indices_cpu)
        if recurrent_slot_indices_cpu is None:
            self._async_recurrent_slot_indices_cpu[:] = self._async_window_row_indices_cpu
        else:
            np.copyto(self._async_recurrent_slot_indices_cpu, recurrent_slot_indices_cpu)
        np.copyto(self._async_temperature_cpu, temperature_cpu)
        np.copyto(self._async_top_p_cpu, top_p_cpu)
        np.copyto(self._async_top_k_cpu, top_k_cpu)
        np.copyto(self._async_min_p_cpu, min_p_cpu)
        np.copyto(self._async_frequency_penalties_cpu, frequency_penalties_cpu)
        np.copyto(self._async_presence_penalties_cpu, presence_penalties_cpu)
        np.copyto(self._async_repetition_penalties_cpu, repetition_penalties_cpu)
        np.copyto(self._async_page_table_cpu, page_table_cpu)

        host_build_start = time.time()
        host_payload, padded_num_reqs, _num_requests, rows_to_copy = self._fill_step_inputs(
            self._async_builder,
            self._async_payload_layout,
            num_tokens_static=num_tokens_static,
            scheduled_full_cpu=self._async_scheduled_full_cpu,
            active_mask_full_cpu=self._async_active_mask_full_cpu,
            token_ids_cpu=token_ids_cpu,
            num_computed_tokens_cpu=self._async_num_computed_tokens_cpu,
            temperature_cpu=self._async_temperature_cpu,
            top_p_cpu=self._async_top_p_cpu,
            top_k_cpu=self._async_top_k_cpu,
            min_p_cpu=self._async_min_p_cpu,
            frequency_penalties_cpu=self._async_frequency_penalties_cpu,
            presence_penalties_cpu=self._async_presence_penalties_cpu,
            repetition_penalties_cpu=self._async_repetition_penalties_cpu,
            page_table_cpu=self._async_page_table_cpu,
            page_table_version=page_table_version,
            padded_num_reqs_in=padded_num_reqs_in,
            window_row_indices_cpu=self._async_window_row_indices_cpu,
            recurrent_slot_indices_cpu=self._async_recurrent_slot_indices_cpu,
            spec_recurrent_commit_cpu=spec_recurrent_commit_cpu,
        )
        host_build_took = time.time() - host_build_start

        device_put_start = time.time()
        if host_payload_broadcast_needed(self.host_payloads_replicated):
            from jax.experimental import multihost_utils

            host_payload = multihost_utils.broadcast_one_to_all(host_payload)
        self._pending_transfer = self._device_put_host_payload(host_payload)
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
        if self._pending_group_result is not None:
            result = self._pending_group_result
            timings = self._pending_transfer_metadata or {}
            self._pending_group_result = None
            self._pending_transfer = None
            self._pending_transfer_metadata = None
            return result, timings

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

        payload_idx = 0
        input_ids_buf = self._pending_transfer[payload_idx]
        payload_idx += 1
        position_ids_buf = self._pending_transfer[payload_idx]
        payload_idx += 1
        packed_qsl_seqlens_dev = self._pending_transfer[payload_idx]
        payload_idx += 1
        pages_tables_dev = self._pending_transfer[payload_idx]
        payload_idx += 1
        packed_i32_padded_dev = self._pending_transfer[payload_idx]
        payload_idx += 1
        packed_f32_padded_dev = self._pending_transfer[payload_idx]
        payload_idx += 1
        packed_misc_i32_dev = self._pending_transfer[payload_idx]
        payload_idx += 1
        dp_query_start_loc_dev = None
        dp_request_distribution_dev = None
        dp_context_lens_dev = None
        dp_recurrent_state_indices_dev = None
        if self._uses_spmd_dp():
            dp_query_start_loc_dev = self._pending_transfer[payload_idx]
            payload_idx += 1
            dp_request_distribution_dev = self._pending_transfer[payload_idx]
            payload_idx += 1
            dp_context_lens_dev = self._pending_transfer[payload_idx]
            payload_idx += 1
            dp_recurrent_state_indices_dev = self._pending_transfer[payload_idx]
            payload_idx += 1
        elif self._plain_recurrent_slots:
            dp_recurrent_state_indices_dev = self._pending_transfer[payload_idx]
            payload_idx += 1
        spec_recurrent_commit_dev = None
        if self._enable_spec_recurrent_commit:
            spec_recurrent_commit_dev = self._pending_transfer[payload_idx]
            payload_idx += 1
        slot_mapping_dev = None
        num_kv_update_dev = None
        if self._use_slot_mapping:
            slot_mapping_dev = self._pending_transfer[payload_idx]
            payload_idx += 1
            num_kv_update_dev = self._pending_transfer[payload_idx]
            payload_idx += 1
        scheduled_full_dev = self._scheduled_full_placeholder_dev
        active_mask_full_dev = self._active_mask_full_placeholder_dev

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
            dp_query_start_loc=dp_query_start_loc_dev,
            dp_request_distribution=dp_request_distribution_dev,
            dp_context_lens=dp_context_lens_dev,
            dp_recurrent_state_indices=dp_recurrent_state_indices_dev,
            input_token_handoff_positions=self._get_zero_dev(
                namespace="input_token_handoff_positions",
                shape=(self.max_num_reqs,),
                dtype=np.int32,
            ),
            input_token_handoff_ids=self._get_zero_dev(
                namespace="input_token_handoff_ids",
                shape=(self.max_num_reqs,),
                dtype=np.int32,
            ),
            input_token_handoff_count=self._get_zero_dev(
                namespace="input_token_handoff_count",
                shape=(),
                dtype=np.int32,
            ),
            input_token_handoff_offset=self._get_zero_dev(
                namespace="input_token_handoff_offset",
                shape=(),
                dtype=np.int32,
            ),
            slot_mapping=slot_mapping_dev if self._use_slot_mapping else None,
            num_kv_update_slices=num_kv_update_dev if self._use_slot_mapping else None,
            spec_recurrent_commit=spec_recurrent_commit_dev,
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

    @staticmethod
    def _apply_device_token_handoff(
        batch_metadata: BatchMetadata,
        handoff: DeviceInputTokenHandoff,
    ) -> BatchMetadata:
        """Attach a compiled-step token handoff to batch metadata.

        PP async decode has a one-token loop-carried dependency: token ``N``
        sampled by the final stage is token ``N+1``'s input on stage 0. The old
        path resolved that dependency by materializing token ``N`` on the host
        and repairing the CPU sequence buffer before the next launch. This
        helper keeps the dependency on device by carrying the sampled-token
        arrays into :class:`BatchMetadata`; the compiled model step resolves
        placeholders via ``BatchMetadata.model_input_ids`` inside the JIT region.

        The returned :class:`BatchMetadata` is a fresh PyTree with all metadata
        leaves preserved except the handoff buffers. CPU request state is
        repaired separately after dispatch, which keeps scheduler/accounting
        correctness while removing host token materialization from the
        stage-launch critical path.
        """
        return batch_metadata.replace(
            input_token_handoff_positions=handoff.input_positions,
            input_token_handoff_ids=handoff.token_ids,
            input_token_handoff_count=handoff.count,
            input_token_handoff_offset=(
                handoff.offset if handoff.offset is not None else jnp.zeros((), dtype=jnp.int32)
            ),
        )

    def apply_spec_window_handoff(
        self,
        batch_metadata: BatchMetadata,
        handoff: DeviceSpecWindowHandoff,
    ) -> tuple[BatchMetadata, jax.Array]:
        """Rewrite step ``N+1``'s optimistic verify windows from step ``N``'s device results.

        The host built ``batch_metadata`` as if every draft step ``N`` verified was accepted. One compiled program
        replaces each fed window's input ids with ``[last emitted token, next drafts]`` and moves its positions,
        sequence length (and every cache group's) back by ``delta = d - a``; with slot-group recurrent state it also
        rebases the slot read so the row reads slot ``g + a``. Nothing is read back to the host, so every host of a
        multi-host run builds the same inputs from the same plan.

        Args:
            batch_metadata: Step ``N+1``'s metadata as the host built it.
            handoff: Step ``N``'s device results and the window-row map.

        Returns:
            ``(metadata, delta)``: the corrected metadata, and int32 ``[max_num_reqs]`` per window row the tokens its
            host-built start ran ahead (``d - a``, 0 for rows without a source). The verify program subtracts it
            from its host-built seed positions and drafter-cache bases.

        Raises:
            ValueError: For layouts whose host-built KV write map depends on the position (the v2 slot mapping, in
                the main pool or a cache group) or that pack rows rank-major (SPMD DP).
        """
        groups = tuple(batch_metadata.cache_groups)
        if self._use_slot_mapping or any(group.slot_mapping is not None for group in groups):
            raise ValueError("Speculative window handoff needs request-distribution (v3) metadata, not a slot mapping.")
        if self._uses_spmd_dp():
            raise ValueError("Speculative window handoff does not serve rank-major SPMD DP batches.")
        sources = handoff.sources
        if not isinstance(sources, jax.Array):
            sources = put_host_payload(
                np.asarray(sources, dtype=np.int32),
                self._empty_sharding,
                host_payloads_replicated=self.host_payloads_replicated,
            )
        # The verify program may run on another placement (the drafter's mesh); move its outputs device-to-device
        # onto the step's input placement (no copy when it already matches, never a host read).
        last_token, accepted, drafts = (
            jax.device_put(x, self._empty_sharding) for x in (handoff.last_token, handoff.accepted, handoff.drafts)
        )
        recurrent_commit = batch_metadata.spec_recurrent_commit if handoff.rewrite_recurrent_base else None
        group_commits = tuple(
            group.spec_recurrent_commit if handoff.rewrite_recurrent_base else None for group in groups
        )
        delta, ids, positions, packed, commit, group_lens, group_positions, group_commits = _rewrite_spec_windows(
            batch_metadata.input_ids_buf,
            batch_metadata.position_ids_buf,
            batch_metadata.packed_qsl_seqlens,
            batch_metadata.packed_misc_i32[0],
            sources,
            last_token,
            accepted,
            drafts,
            recurrent_commit,
            tuple(group.context_lens for group in groups),
            tuple(group.position_ids for group in groups),
            group_commits,
        )
        new_groups = tuple(
            group.replace(
                context_lens=lens,
                position_ids=pos,
                spec_recurrent_commit=group.spec_recurrent_commit if group_commit is None else group_commit,
            )
            for group, lens, pos, group_commit in zip(groups, group_lens, group_positions, group_commits, strict=True)
        )
        fixed = batch_metadata.replace(
            input_ids_buf=ids,
            position_ids_buf=positions,
            packed_qsl_seqlens=packed,
            spec_recurrent_commit=batch_metadata.spec_recurrent_commit if commit is None else commit,
            cache_groups=new_groups,
        )
        return fixed, delta


@jax.jit
def _rewrite_spec_windows(
    input_ids: jax.Array,
    positions: jax.Array,
    packed_qsl_seqlens: jax.Array,
    num_requests: jax.Array,
    sources: jax.Array,
    last_token: jax.Array,
    accepted: jax.Array,
    drafts: jax.Array,
    recurrent_commit: jax.Array | None,
    group_context_lens: tuple[jax.Array, ...],
    group_positions: tuple[jax.Array | None, ...],
    group_commits: tuple[jax.Array | None, ...],
):
    """The device half of :meth:`BatchMetadataPreparer.apply_spec_window_handoff` (see there)."""
    qsl = packed_qsl_seqlens[0]
    rows = qsl.shape[0] - 1
    row_ids = jnp.arange(rows, dtype=jnp.int32)
    src = sources[0, :rows]
    fed = (row_ids < num_requests) & (src >= 0)
    src = jnp.clip(src, 0, last_token.shape[0] - 1)
    acc = accepted.astype(jnp.int32)[src]
    delta = jnp.where(fed, sources[1, :rows] - acc, 0)

    tok = jnp.arange(input_ids.shape[0], dtype=jnp.int32)
    tok_row = jnp.clip(jnp.searchsorted(qsl[1:], tok, side="right"), 0, rows - 1).astype(jnp.int32)
    live_end = qsl[jnp.clip(num_requests, 0, rows)]
    tok_fed = (tok < live_end) & fed[tok_row]
    j = tok - qsl[tok_row]
    tok_src = src[tok_row]
    window_token = jnp.where(
        j == 0,
        last_token.astype(jnp.int32)[tok_src],
        drafts.astype(jnp.int32)[tok_src, jnp.clip(j - 1, 0, drafts.shape[1] - 1)],
    )
    ids = jnp.where(tok_fed & (j <= drafts.shape[1]), window_token.astype(input_ids.dtype), input_ids)
    tok_delta = jnp.where(tok_fed, delta[tok_row], 0)
    new_positions = positions - tok_delta.astype(positions.dtype)

    seq_lens = packed_qsl_seqlens[1, :rows] - delta
    packed = packed_qsl_seqlens.at[1, :rows].set(seq_lens)

    query_lens = qsl[1:] - qsl[:-1]

    def rebase(commit):
        # Slot g covers the request's tokens through its last step's first token: corrected context before this
        # step minus the drafts that step accepted. Bases count absolute tokens, so every pool uses the main lengths.
        if commit is None:
            return None
        n = min(rows, commit.shape[1])
        before = seq_lens[:n] - query_lens[:n]
        return commit.at[0, :n].set(jnp.where(fed[:n], before - acc[:n], commit[0, :n]))

    new_lens = tuple(lens - delta[: lens.shape[0]] for lens in group_context_lens)
    new_group_positions = tuple(
        None if pos is None else pos - tok_delta[: pos.shape[0]].astype(pos.dtype) for pos in group_positions
    )
    return (
        delta,
        ids,
        new_positions,
        packed,
        rebase(recurrent_commit),
        new_lens,
        new_group_positions,
        tuple(rebase(commit) for commit in group_commits),
    )
