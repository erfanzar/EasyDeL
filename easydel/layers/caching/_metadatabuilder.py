"""Unified builder for attention/cache runtime metadata.

EasyDeL supports multiple attention mechanisms and cache formats:
- Standard transformer KV-cache (`TransformerMetadata`)
- Paged/ragged KV-cache (`RaggedPagesMetadata`, v2 or v3)
- Hybrid and recurrent caches (use transformer-style metadata or none)

Different attention operations expect different metadata objects. This builder
accepts a superset of possible inputs and can *compute* missing ragged/paged
fields from raw scheduler state (similar to eSurge's `prepare_batch_metadata`):

- If you pass precomputed ragged fields (`pages_tables`, `context_lens`,
  `query_start_loc`, etc.) it will just wrap them.
- If you pass raw batch inputs (`scheduled_full`, `active_mask_full`,
  `num_computed_tokens`, `page_table`) it will derive the ragged fields,
  including:
    - `query_start_loc`
    - `context_lens`
    - masked `pages_tables`
    - v2 `slot_mapping` + `num_kv_update_slices`
    - v3 `request_distribution`

For non-ragged mechanisms it returns transformer-style metadata.

Note: the "compute from batch" path is CPU-first (NumPy) because the reference
implementation is CPU metadata prep; do not call that path inside `jit`.

This module can also compute additional *batch-level* fields commonly needed by
high-performance inference runners (e.g. eSurge), such as contiguous `input_ids`,
`position_ids`, and `logits_indices` for a scheduled batch. These helpers are
also CPU-first and return NumPy arrays suitable for a single `jax.device_put`.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from .ragged_page import RaggedPagesCacheConfig, RaggedPagesMetadata
from .recurrent import RecurrentMetadata
from .transformer import TransformerMetadata

JaxArray = jax.Array
NumpyArray = np.ndarray

# Accept either JAX or NumPy arrays, and allow basic Python sequences for convenience.
ArrayLike = JaxArray | NumpyArray
IntVectorLike = ArrayLike | tp.Sequence[int]
IntMatrixLike = ArrayLike | tp.Sequence[tp.Sequence[int]]
BoolVectorLike = ArrayLike | tp.Sequence[bool] | tp.Sequence[int]


class SupportsStartsIndexs(tp.Protocol):
    """Protocol for cache views that carry transformer-style starts/indexs.

    This protocol defines the interface for cache views that support
    sequence position tracking through starts and indexs arrays.
    Used by the metadata builder to extract position information
    from existing cache views.

    Attributes:
        starts: Array-like containing starting positions for each sequence.
            Shape: [batch_size]
        indexs: Array-like containing current position indices for each sequence.
            Shape: [batch_size]
    """

    starts: ArrayLike
    indexs: ArrayLike


class _HasCpuTensor(tp.Protocol):
    """Protocol for objects exposing a CPU-side tensor.

    Used by the metadata builder to extract CPU-resident arrays from
    wrapper objects. This enables compatibility with tensor wrappers
    that maintain separate CPU and device tensors.

    Methods:
        get_cpu_tensor: Return the CPU-resident NumPy array.
    """

    def get_cpu_tensor(self) -> np.ndarray:
        """Return the CPU-resident NumPy array.

        Returns:
            np.ndarray: The tensor data as a NumPy array on CPU.
        """
        ...


class _HasDeviceTensor(tp.Protocol):
    """Protocol for objects exposing a device-side tensor.

    Used by the metadata builder to extract device-resident arrays from
    wrapper objects. The tensor will be transferred to CPU for metadata
    computation.

    Methods:
        get_device_tensor: Return the device-resident JAX array.
    """

    def get_device_tensor(self) -> jax.Array:
        """Return the device-resident JAX array.

        Returns:
            jax.Array: The tensor data as a JAX array on device.
        """
        ...


PageTableLike = IntMatrixLike | _HasCpuTensor | _HasDeviceTensor


class _RaggedComputed(tp.TypedDict):
    """TypedDict for computed ragged/paged attention metadata fields.

    Contains the CPU-computed fields needed for ragged page attention.
    These fields are derived from raw batch scheduling information.

    Attributes:
        pages_tables: Page table mapping requests to physical pages.
            Shape: [max_num_reqs, max_pages_per_req]
        context_lens: Context length (total tokens) per request.
            Shape: [max_num_reqs]
        query_start_loc: Cumulative query start positions.
            Shape: [max_num_reqs + 1]
        num_seqs: Number of active sequences.
            Shape: [1]
        request_distribution: Distribution counts for v3 (decode, prefill, total).
            Shape: [3]. Optional, only present for v3.
        slot_mapping: Slot mapping for v2-style cache updates.
            Shape: [3, num_slices]. Optional, only present for v2.
        num_kv_update_slices: Number of KV update slices for v2.
            Shape: [1]. Optional, only present for v2.
    """

    pages_tables: np.ndarray
    context_lens: np.ndarray
    query_start_loc: np.ndarray
    num_seqs: np.ndarray
    request_distribution: tp.NotRequired[np.ndarray]
    slot_mapping: tp.NotRequired[np.ndarray]
    num_kv_update_slices: tp.NotRequired[np.ndarray]


class _PagedBatchComputed(tp.TypedDict):
    """CPU-computed fields for a paged/ragged attention batch.

    These fields are intended for high-performance inference runners that:
    - Build a contiguous token batch on CPU
    - Compute ragged/page metadata fields on CPU
    - Transfer everything to device in a single `jax.device_put` call

    This TypedDict defines the complete payload structure for paged
    attention batch processing.

    Attributes:
        input_ids: Contiguous token IDs for the batch.
            Shape: [num_tokens_static]
        position_ids: Position indices for each token.
            Shape: [num_tokens_static]
        query_start_loc: Cumulative query start positions.
            Shape: [max_num_reqs + 1]
        seq_lens: Context length per request.
            Shape: [max_num_reqs]
        logits_indices: Indices for extracting logits per request.
            Shape: [padded_num_reqs]
        pages_tables: Page table mapping.
            Shape: [num_reqs, max_pages_per_req]
        scheduled: Scheduled token counts per request.
            Shape: [padded_num_reqs]
        num_requests: Scalar count of active requests.
        padded_num_reqs: Scalar padded request count.
        temperature: Optional sampling temperature per request.
            Shape: [padded_num_reqs]
        top_p: Optional top-p sampling parameter.
            Shape: [padded_num_reqs]
        top_k: Optional top-k sampling parameter.
            Shape: [padded_num_reqs]
        min_p: Optional min-p sampling parameter.
            Shape: [padded_num_reqs]
        request_distribution: v3-style distribution [decode, prefill, total].
            Shape: [3]. Optional, only present for v3.
        slot_mapping: v2-style slot mapping.
            Shape: [3, num_slices]. Optional, only present for v2.
        num_kv_update_slices: v2 update slice count.
            Shape: [1]. Optional, only present for v2.
        actual_num_tokens: Total number of scheduled tokens.
            Shape: scalar. Optional.
    """

    input_ids: np.ndarray
    position_ids: np.ndarray
    query_start_loc: np.ndarray
    seq_lens: np.ndarray
    logits_indices: np.ndarray
    pages_tables: np.ndarray
    scheduled: np.ndarray
    num_requests: np.ndarray
    padded_num_reqs: np.ndarray
    temperature: tp.NotRequired[np.ndarray]
    top_p: tp.NotRequired[np.ndarray]
    top_k: tp.NotRequired[np.ndarray]
    min_p: tp.NotRequired[np.ndarray]
    request_distribution: tp.NotRequired[np.ndarray]
    slot_mapping: tp.NotRequired[np.ndarray]
    num_kv_update_slices: tp.NotRequired[np.ndarray]
    actual_num_tokens: tp.NotRequired[np.ndarray]


class AttentionMetadataBuilder:
    """Factory for runtime attention metadata across cache types.

    This builder provides a unified interface for creating runtime metadata
    objects for different attention mechanisms. It supports:
    - Standard transformer KV-cache (TransformerMetadata)
    - Paged/ragged attention (RaggedPagesMetadata, v2 or v3)
    - Recurrent/SSM models (RecurrentMetadata)

    The builder can either wrap precomputed metadata fields or derive
    them from raw batch scheduling state (CPU-first computation).

    Key Features:
        - Automatic metadata type selection based on attention mechanism
        - CPU-first computation for ragged fields (avoids JIT issues)
        - Support for both v2 (slot_mapping) and v3 (request_distribution) formats
        - Synonym normalization (pages_tables/block_tables, context_lens/seq_lens)

    Class Attributes:
        _RAGGED_MECH_PREFIXES: Attention mechanism name prefixes that use ragged metadata.
        _RAGGED_MECH_EXACT: Exact attention mechanism names that use ragged metadata.

    Example:
        >>> # Build transformer metadata
        >>> meta = AttentionMetadataBuilder.build_transformer_metadata(
        ...     postpadded=False,
        ...     starts=jnp.zeros((batch_size,), dtype=jnp.int32)
        ... )
        >>>
        >>> # Build ragged metadata from precomputed fields
        >>> meta = AttentionMetadataBuilder.build_ragged_page_metadata(
        ...     pages_tables=block_tables,
        ...     context_lens=seq_lens,
        ...     query_start_loc=query_locs,
        ...     num_seqs=num_active
        ... )
        >>>
        >>> # Auto-select metadata type
        >>> meta = AttentionMetadataBuilder.build(
        ...     attention_mechanism="ragged_page_attention_v3",
        ...     pages_tables=tables,
        ...     context_lens=lens,
        ...     query_start_loc=locs
        ... )
    """

    _RAGGED_MECH_PREFIXES: tp.ClassVar[tuple[str, ...]] = ("ragged_page_attention",)
    _RAGGED_MECH_EXACT: tp.ClassVar[set[str]] = {
        "ragged_page_attention_v2",
        "ragged_page_attention_v3",
        "page_attention",
        "paged_attention",
        "unified_attention",
    }

    @classmethod
    def build_transformer_metadata(
        cls,
        *,
        postpadded: bool = False,
        starts: IntVectorLike | None = None,
        indexs: IntVectorLike | None = None,
        cache_view: SupportsStartsIndexs | None = None,
    ) -> TransformerMetadata:
        """Build transformer-style runtime metadata.

        Creates a TransformerMetadata object for standard KV-cache attention.
        If `starts`/`indexs` are not provided but a `cache_view` is, values
        are taken from the view.

        Args:
            postpadded: Whether sequences are post-padded (padding at end).
                Affects how attention masks are computed.
            starts: Starting positions for each sequence in the batch.
                Shape: [batch_size]. Optional.
            indexs: Current position indices for each sequence.
                Shape: [batch_size]. Optional.
            cache_view: Optional cache view implementing SupportsStartsIndexs
                protocol to infer starts/indexs if not provided.

        Returns:
            TransformerMetadata: Runtime metadata for transformer attention.

        Example:
            >>> meta = AttentionMetadataBuilder.build_transformer_metadata(
            ...     postpadded=False,
            ...     starts=jnp.zeros((batch_size,), dtype=jnp.int32)
            ... )
        """
        if cache_view is not None:
            if starts is None:
                starts = cache_view.starts
            if indexs is None:
                indexs = cache_view.indexs

        starts_arr = None if starts is None else jnp.asarray(starts, dtype=jnp.int32)
        indexs_arr = None if indexs is None else jnp.asarray(indexs, dtype=jnp.int32)

        return TransformerMetadata(postpadded=postpadded, starts=starts_arr, indexs=indexs_arr)

    @classmethod
    def build_recurrent_metadata(cls) -> RecurrentMetadata:
        """Build recurrent/SSM runtime metadata.

        Creates a RecurrentMetadata object for recurrent state-space models
        like Mamba, Mamba2, or GatedDeltaNet. Currently returns an empty
        metadata placeholder as these models typically don't require
        runtime metadata beyond the cache state itself.

        Returns:
            RecurrentMetadata: Runtime metadata for recurrent models.

        Example:
            >>> meta = AttentionMetadataBuilder.build_recurrent_metadata()
        """
        return RecurrentMetadata()

    @classmethod
    def build_ragged_page_metadata(
        cls,
        *,
        pages_tables: IntMatrixLike | None = None,
        block_tables: IntMatrixLike | None = None,
        context_lens: IntVectorLike | None = None,
        seq_lens: IntVectorLike | None = None,
        query_start_loc: IntVectorLike | None = None,
        num_seqs: int | IntVectorLike | None = None,
        slot_mapping: IntVectorLike | None = None,
        position_ids: IntVectorLike | None = None,
        request_distribution: IntVectorLike | None = None,
        num_kv_update_slices: IntVectorLike | None = None,
        version: tp.Literal["v2", "v3"] = "v3",
        page_size: int = 128,
        prefill_chunk_size: int = 512,
        num_slices_per_kv_cache_update_page: int | None = None,
        # Raw batch inputs (optional, for deriving ragged fields)
        scheduled_full: IntVectorLike | None = None,
        active_mask_full: BoolVectorLike | None = None,
        num_computed_tokens: IntVectorLike | None = None,
        page_table: PageTableLike | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        ragged_config: RaggedPagesCacheConfig | None = None,
    ) -> RaggedPagesMetadata:
        """Build ragged/paged attention runtime metadata (v2 or v3).

        Creates a RaggedPagesMetadata object for paged attention mechanisms.
        If required ragged fields are missing but raw batch inputs are provided,
        they are computed on CPU (NumPy) similarly to eSurge's metadata prep.

        Synonym normalization:
            - `block_tables` is treated as synonym for `pages_tables`
            - `seq_lens` is treated as synonym for `context_lens`

        Args:
            pages_tables: Page table mapping requests to physical pages.
                Shape: [max_num_reqs, max_pages_per_req]. Optional.
            block_tables: Alias for pages_tables. Optional.
            context_lens: Context length (total tokens) per request.
                Shape: [max_num_reqs]. Optional.
            seq_lens: Alias for context_lens. Optional.
            query_start_loc: Cumulative query start positions.
                Shape: [max_num_reqs + 1]. Optional.
            num_seqs: Number of active sequences. Can be int or array.
            slot_mapping: v2-style slot mapping for cache updates.
                Shape: [3, num_slices]. Required for v2.
            position_ids: Position IDs for each token. Optional.
            request_distribution: v3-style distribution [decode, prefill, total].
                Shape: [3]. Auto-computed for v3 if not provided.
            num_kv_update_slices: Number of update slices for v2.
                Shape: [1]. Auto-computed for v2.
            version: Ragged attention version ("v2" or "v3").
            page_size: Number of tokens per page.
            prefill_chunk_size: Chunk size for prefill processing.
            num_slices_per_kv_cache_update_page: Slices per update page (v2).
            scheduled_full: Raw scheduled tokens per request for deriving fields.
            active_mask_full: Raw active mask for deriving fields.
            num_computed_tokens: Raw computed token counts for deriving fields.
            page_table: Raw page table for deriving fields.
            max_num_reqs: Maximum number of requests (for padding).
            max_num_tokens: Maximum number of tokens (for padding).
            ragged_config: RaggedPagesCacheConfig for additional parameters.

        Returns:
            RaggedPagesMetadata: Runtime metadata for paged attention.

        Raises:
            ValueError: If required fields cannot be derived or are missing.

        Example:
            >>> meta = AttentionMetadataBuilder.build_ragged_page_metadata(
            ...     pages_tables=block_tables,
            ...     context_lens=seq_lens,
            ...     query_start_loc=query_locs,
            ...     num_seqs=num_active,
            ...     version="v3"
            ... )
        """
        # Normalize synonyms.
        if pages_tables is None and block_tables is not None:
            pages_tables = block_tables
        if context_lens is None and seq_lens is not None:
            context_lens = seq_lens

        needs_compute = pages_tables is None or context_lens is None or query_start_loc is None
        has_raw_batch = (
            scheduled_full is not None
            and active_mask_full is not None
            and num_computed_tokens is not None
            and page_table is not None
        )
        if needs_compute and has_raw_batch:
            computed = cls._compute_ragged_from_batch_cpu(
                scheduled_full=scheduled_full,
                active_mask_full=active_mask_full,
                num_computed_tokens=num_computed_tokens,
                page_table=page_table,
                version=version,
                page_size=page_size,
                num_slices_per_kv_cache_update_page=num_slices_per_kv_cache_update_page,
                max_num_reqs=max_num_reqs,
                max_num_tokens=max_num_tokens,
                ragged_config=ragged_config,
            )
            pages_tables = computed["pages_tables"]
            context_lens = computed["context_lens"]
            query_start_loc = computed["query_start_loc"]
            num_seqs = computed["num_seqs"]
            request_distribution = computed.get("request_distribution", request_distribution)
            slot_mapping = computed.get("slot_mapping", slot_mapping)
            num_kv_update_slices = computed.get("num_kv_update_slices", num_kv_update_slices)

        if pages_tables is None or context_lens is None or query_start_loc is None:
            raise ValueError(
                "Ragged/paged attention metadata requires `pages_tables` (or `block_tables`), "
                "`context_lens` (or `seq_lens`), and `query_start_loc`."
            )

        pages_tables_arr = jnp.asarray(pages_tables, dtype=jnp.int32)
        context_lens_arr = jnp.asarray(context_lens, dtype=jnp.int32)
        query_start_loc_arr = jnp.asarray(query_start_loc, dtype=jnp.int32)

        if num_seqs is None:
            num_seqs_arr = jnp.array([context_lens_arr.shape[0]], dtype=jnp.int32)
        else:
            num_seqs_arr = jnp.asarray(num_seqs, dtype=jnp.int32)
            if num_seqs_arr.ndim == 0:
                num_seqs_arr = num_seqs_arr.reshape(1)

        if version == "v3" and request_distribution is None:
            request_distribution = jnp.zeros((3,), dtype=jnp.int32)
        if version == "v2":
            if slot_mapping is None:
                raise ValueError("version='v2' requires `slot_mapping`.")
            if num_kv_update_slices is None:
                num_kv_update_slices = jnp.zeros((1,), dtype=jnp.int32)

        return RaggedPagesMetadata(
            pages_tables=pages_tables_arr,
            context_lens=context_lens_arr,
            query_start_loc=query_start_loc_arr,
            num_seqs=num_seqs_arr,
            slot_mapping=None if slot_mapping is None else jnp.asarray(slot_mapping, dtype=jnp.int32),
            position_ids=None if position_ids is None else jnp.asarray(position_ids, dtype=jnp.int32),
            request_distribution=None
            if request_distribution is None
            else jnp.asarray(request_distribution, dtype=jnp.int32),
            num_kv_update_slices=None
            if num_kv_update_slices is None
            else jnp.asarray(num_kv_update_slices, dtype=jnp.int32),
            version=version,
            num_slices_per_kv_cache_update_page=num_slices_per_kv_cache_update_page,
            page_size=page_size,
            prefill_chunk_size=prefill_chunk_size,
        )

    @classmethod
    def build_page_metadata(
        cls,
        *,
        pages_tables: IntMatrixLike | None = None,
        block_tables: IntMatrixLike | None = None,
        context_lens: IntVectorLike | None = None,
        seq_lens: IntVectorLike | None = None,
        query_start_loc: IntVectorLike | None = None,
        num_seqs: int | IntVectorLike | None = None,
        slot_mapping: IntVectorLike | None = None,
        position_ids: IntVectorLike | None = None,
        request_distribution: IntVectorLike | None = None,
        num_kv_update_slices: IntVectorLike | None = None,
        version: tp.Literal["v2", "v3"] = "v3",
        page_size: int = 128,
        prefill_chunk_size: int = 512,
        num_slices_per_kv_cache_update_page: int | None = None,
        scheduled_full: IntVectorLike | None = None,
        active_mask_full: BoolVectorLike | None = None,
        num_computed_tokens: IntVectorLike | None = None,
        page_table: PageTableLike | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        ragged_config: RaggedPagesCacheConfig | None = None,
    ) -> RaggedPagesMetadata:
        """Alias of `build_ragged_page_metadata`.

        This method provides an alternative name for build_ragged_page_metadata
        to accommodate different naming conventions.

        See build_ragged_page_metadata for full documentation.

        Returns:
            RaggedPagesMetadata: Runtime metadata for paged attention.
        """
        return cls.build_ragged_page_metadata(
            pages_tables=pages_tables,
            block_tables=block_tables,
            context_lens=context_lens,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
            slot_mapping=slot_mapping,
            position_ids=position_ids,
            request_distribution=request_distribution,
            num_kv_update_slices=num_kv_update_slices,
            version=version,
            page_size=page_size,
            prefill_chunk_size=prefill_chunk_size,
            num_slices_per_kv_cache_update_page=num_slices_per_kv_cache_update_page,
            scheduled_full=scheduled_full,
            active_mask_full=active_mask_full,
            num_computed_tokens=num_computed_tokens,
            page_table=page_table,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            ragged_config=ragged_config,
        )

    @classmethod
    def build_paged_metadata(
        cls,
        *,
        pages_tables: IntMatrixLike | None = None,
        block_tables: IntMatrixLike | None = None,
        context_lens: IntVectorLike | None = None,
        seq_lens: IntVectorLike | None = None,
        query_start_loc: IntVectorLike | None = None,
        num_seqs: int | IntVectorLike | None = None,
        slot_mapping: IntVectorLike | None = None,
        position_ids: IntVectorLike | None = None,
        request_distribution: IntVectorLike | None = None,
        num_kv_update_slices: IntVectorLike | None = None,
        version: tp.Literal["v2", "v3"] = "v3",
        page_size: int = 128,
        prefill_chunk_size: int = 512,
        num_slices_per_kv_cache_update_page: int | None = None,
        scheduled_full: IntVectorLike | None = None,
        active_mask_full: BoolVectorLike | None = None,
        num_computed_tokens: IntVectorLike | None = None,
        page_table: PageTableLike | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        ragged_config: RaggedPagesCacheConfig | None = None,
    ) -> RaggedPagesMetadata:
        """Alias of `build_ragged_page_metadata`.

        This method provides an alternative name for build_ragged_page_metadata
        to accommodate paged attention terminology.

        See build_ragged_page_metadata for full documentation.

        Returns:
            RaggedPagesMetadata: Runtime metadata for paged attention.
        """
        return cls.build_page_metadata(
            pages_tables=pages_tables,
            block_tables=block_tables,
            context_lens=context_lens,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
            slot_mapping=slot_mapping,
            position_ids=position_ids,
            request_distribution=request_distribution,
            num_kv_update_slices=num_kv_update_slices,
            version=version,
            page_size=page_size,
            prefill_chunk_size=prefill_chunk_size,
            num_slices_per_kv_cache_update_page=num_slices_per_kv_cache_update_page,
            scheduled_full=scheduled_full,
            active_mask_full=active_mask_full,
            num_computed_tokens=num_computed_tokens,
            page_table=page_table,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            ragged_config=ragged_config,
        )

    @classmethod
    def build(
        cls,
        *,
        attention_mechanism: str | None = None,
        expected_cache_type: tp.Literal["auto", "transformer", "ragged", "recurrent"] = "auto",
        # --- Transformer / hybrid fields ---
        postpadded: bool = False,
        starts: IntVectorLike | None = None,
        indexs: IntVectorLike | None = None,
        # Optional cache view to infer starts/indexs when absent.
        cache_view: SupportsStartsIndexs | None = None,
        # --- Ragged/paged fields ---
        pages_tables: IntMatrixLike | None = None,
        block_tables: IntMatrixLike | None = None,
        context_lens: IntVectorLike | None = None,
        seq_lens: IntVectorLike | None = None,
        query_start_loc: IntVectorLike | None = None,
        num_seqs: int | IntVectorLike | None = None,
        slot_mapping: IntVectorLike | None = None,
        position_ids: IntVectorLike | None = None,
        request_distribution: IntVectorLike | None = None,
        num_kv_update_slices: IntVectorLike | None = None,
        version: tp.Literal["v2", "v3"] = "v3",
        page_size: int = 128,
        prefill_chunk_size: int = 512,
        num_slices_per_kv_cache_update_page: int | None = None,
        # --- Raw batch inputs for computing ragged fields ---
        scheduled_full: IntVectorLike | None = None,
        active_mask_full: BoolVectorLike | None = None,
        num_computed_tokens: IntVectorLike | None = None,
        page_table: PageTableLike | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        ragged_config: RaggedPagesCacheConfig | None = None,
    ) -> TransformerMetadata | RaggedPagesMetadata | RecurrentMetadata:
        """Build the appropriate runtime metadata object.

        Args:
            attention_mechanism: Name of the attention mechanism (string form of
                `AttentionMechanisms`). Used only for disambiguation when
                `expected_cache_type="auto"`.
            expected_cache_type: Force the type to build. "auto" infers based on
                mechanism name and provided fields.
            postpadded, starts, indexs: Transformer-style runtime fields.
            cache_view: Optional cache view to infer `starts`/`indexs` if missing.
            pages_tables / block_tables: Page/block tables for paged attention.
            context_lens / seq_lens: Per-request context lengths.
            query_start_loc: Cumulative query start offsets.
            num_seqs: Number of active sequences. Can be int or array-like.
            slot_mapping, num_kv_update_slices: v2-only paged attention fields.
            request_distribution: v3-only paged attention field.
            version: "v2" or "v3" for ragged page attention.
            page_size, prefill_chunk_size, num_slices_per_kv_cache_update_page:
                Paged attention configuration values.
            scheduled_full, active_mask_full, num_computed_tokens, page_table:
                If precomputed ragged fields are not provided, these raw batch
                inputs are used to derive them.
            max_num_reqs, max_num_tokens, ragged_config:
                Optional caps/config used for padding and slot-mapping sizing.

        Returns:
            A `TransformerMetadata`, `RaggedPagesMetadata`, or `RecurrentMetadata`
            instance depending on inputs.
        """

        mech = (attention_mechanism or "").lower()
        is_ragged_mech = mech in cls._RAGGED_MECH_EXACT or mech.startswith(cls._RAGGED_MECH_PREFIXES)
        has_raw_batch = (
            scheduled_full is not None
            and active_mask_full is not None
            and num_computed_tokens is not None
            and page_table is not None
        )

        if expected_cache_type == "ragged" or (
            expected_cache_type == "auto"
            and (pages_tables is not None or block_tables is not None or has_raw_batch or is_ragged_mech)
        ):
            return cls.build_ragged_page_metadata(
                pages_tables=pages_tables,
                block_tables=block_tables,
                context_lens=context_lens,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                num_seqs=num_seqs,
                slot_mapping=slot_mapping,
                position_ids=position_ids,
                request_distribution=request_distribution,
                num_kv_update_slices=num_kv_update_slices,
                version=version,
                page_size=page_size,
                prefill_chunk_size=prefill_chunk_size,
                num_slices_per_kv_cache_update_page=num_slices_per_kv_cache_update_page,
                scheduled_full=scheduled_full,
                active_mask_full=active_mask_full,
                num_computed_tokens=num_computed_tokens,
                page_table=page_table,
                max_num_reqs=max_num_reqs,
                max_num_tokens=max_num_tokens,
                ragged_config=ragged_config,
            )

        if expected_cache_type == "recurrent":
            return cls.build_recurrent_metadata()

        return cls.build_transformer_metadata(
            postpadded=postpadded,
            starts=starts,
            indexs=indexs,
            cache_view=cache_view,
        )

    @staticmethod
    def _ensure_cpu_array(
        x: ArrayLike
        | _HasCpuTensor
        | _HasDeviceTensor
        | tp.Sequence[int]
        | tp.Sequence[bool]
        | tp.Sequence[tp.Sequence[int]],
    ) -> np.ndarray:
        """Convert various array-like inputs to a NumPy array on CPU.

        Handles multiple input types including JAX arrays, NumPy arrays,
        Python sequences, and wrapper objects with CPU/device tensor accessors.

        Args:
            x: Input array-like value to convert. Can be:
                - JAX array (will be transferred from device)
                - NumPy array (returned as-is after asarray)
                - Python sequence of ints/bools
                - Object with get_cpu_tensor() method
                - Object with get_device_tensor() method

        Returns:
            np.ndarray: The input converted to a NumPy array.

        Raises:
            ValueError: If x is None.
        """
        if x is None:
            raise ValueError("Expected an array-like value, got None.")
        if hasattr(x, "get_cpu_tensor"):
            return np.asarray(x.get_cpu_tensor())
        if hasattr(x, "get_device_tensor"):
            return np.asarray(jax.device_get(x.get_device_tensor()))
        if isinstance(x, jax.Array):
            return np.asarray(jax.device_get(x))
        return np.asarray(x)

    @classmethod
    def _compute_slot_mapping_v2_cpu(
        cls,
        *,
        num_requests: int,
        scheduled: np.ndarray,
        num_computed_tokens: np.ndarray,
        page_table: np.ndarray,
        page_size: int,
        max_pages_per_req: int,
        slices_per_page: int,
        max_padded_slices: int | None,
    ) -> tuple[np.ndarray, int]:
        """Compute v2 slot_mapping and total updated pages on CPU.

        Generates the slot mapping array for v2-style paged cache updates.
        The slot mapping describes how to copy new KV tokens into the
        paged cache, organized as slices with (cache_start, new_kv_start, length).

        Args:
            num_requests: Number of active requests in the batch.
            scheduled: Scheduled token counts per request.
                Shape: [max_num_reqs]
            num_computed_tokens: Already computed token counts per request.
                Shape: [max_num_reqs]
            page_table: Page table mapping requests to physical pages.
                Shape: [max_num_reqs, max_pages_per_req]
            page_size: Number of tokens per page.
            max_pages_per_req: Maximum pages allocated per request.
            slices_per_page: Slices per processing page (for padding alignment).
            max_padded_slices: Optional maximum padded slice count.

        Returns:
            tuple: Pair of (slot_mapping, total_pages) where:
                - slot_mapping: Array of shape [3, padded_num_slices] containing
                  (cache_start, new_kv_start, length) per slice
                - total_pages: Total number of pages being updated
        """
        if num_requests <= 0:
            return np.zeros((3, 1), dtype=np.int32), 0

        scheduled_active = scheduled[:num_requests].astype(np.int32)
        num_computed_active = num_computed_tokens[:num_requests].astype(np.int32)
        if not np.any(scheduled_active):
            return np.zeros((3, 1), dtype=np.int32), 0

        start_tokens = num_computed_active
        end_tokens = num_computed_active + scheduled_active

        lps = start_tokens // page_size
        lpe = (np.maximum(end_tokens, 1) - 1) // page_size
        page_lens = np.where(scheduled_active > 0, lpe - lps + 1, 0).astype(np.int32)
        if not np.any(page_lens):
            return np.zeros((3, 1), dtype=np.int32), 0

        page_cum = np.cumsum(page_lens, dtype=np.int32)
        total_pages = int(page_cum[-1])

        padded_num_slices = ((total_pages + slices_per_page - 1) // slices_per_page) * slices_per_page
        if max_padded_slices is not None:
            padded_num_slices = min(int(max_padded_slices), int(padded_num_slices))

        padded_num_slices = max(int(padded_num_slices), 1)
        indices = np.arange(padded_num_slices, dtype=np.int32)
        active_positions = indices[indices < total_pages]

        slot_mapping = np.zeros((3, padded_num_slices), dtype=np.int32)
        if active_positions.size == 0:
            return slot_mapping, total_pages

        page_cum_prev = np.concatenate(([0], page_cum[:-1]))
        req_for_slice = np.searchsorted(page_cum, active_positions, side="right").astype(np.int32)
        local_off = active_positions - page_cum_prev[req_for_slice]

        pt = page_table[:num_requests, :max_pages_per_req].astype(np.int32)
        pt_flat = pt.reshape(-1)
        gather_idx = req_for_slice * max_pages_per_req + lps[req_for_slice] + local_off
        gather_idx = np.clip(gather_idx, 0, pt_flat.size - 1)
        page_numbers = pt_flat[gather_idx]

        s_mod = start_tokens % page_size
        e_mod = ((np.maximum(end_tokens, 1) - 1) % page_size) + 1

        is_first = local_off == 0
        is_last = local_off == (page_lens[req_for_slice] - 1)

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

        return slot_mapping, total_pages

    @classmethod
    def _compute_ragged_from_batch_cpu(
        cls,
        *,
        scheduled_full: IntVectorLike,
        active_mask_full: BoolVectorLike,
        num_computed_tokens: IntVectorLike,
        page_table: PageTableLike,
        version: tp.Literal["v2", "v3"],
        page_size: int,
        num_slices_per_kv_cache_update_page: int | None,
        max_num_reqs: int | None,
        max_num_tokens: int | None,
        ragged_config: RaggedPagesCacheConfig | None,
    ) -> _RaggedComputed:
        """Compute ragged metadata fields from raw batch inputs on CPU.

        Derives all necessary ragged/paged attention metadata from raw
        scheduler state. This is the CPU-first computation path that
        mirrors eSurge's metadata preparation logic.

        Args:
            scheduled_full: Scheduled token counts per request slot.
                Shape: [max_num_reqs]
            active_mask_full: Boolean mask indicating active requests.
                Shape: [max_num_reqs]
            num_computed_tokens: Already computed token counts per request.
                Shape: [max_num_reqs]
            page_table: Page table mapping requests to physical pages.
                Shape: [max_num_reqs, max_pages_per_req]
            version: Ragged attention version ("v2" or "v3").
            page_size: Number of tokens per page.
            num_slices_per_kv_cache_update_page: Slices per update page (v2).
            max_num_reqs: Maximum number of requests for padding.
            max_num_tokens: Maximum number of tokens for padding.
            ragged_config: Optional RaggedPagesCacheConfig for parameters.

        Returns:
            _RaggedComputed: Dictionary containing:
                - pages_tables: Masked page table
                - context_lens: Per-request context lengths
                - query_start_loc: Cumulative query positions
                - num_seqs: Active sequence count
                - request_distribution (v3): [decode, prefill, total]
                - slot_mapping (v2): Update slice mapping
                - num_kv_update_slices (v2): Total update slices

        Raises:
            ValueError: If v2 is requested but num_slices_per_kv_cache_update_page
                is not provided.
        """
        scheduled_full_np = cls._ensure_cpu_array(scheduled_full).astype(np.int32)
        active_mask_np = cls._ensure_cpu_array(active_mask_full).astype(bool)
        num_computed_np = cls._ensure_cpu_array(num_computed_tokens).astype(np.int32)
        page_table_np = cls._ensure_cpu_array(page_table).astype(np.int32)

        max_num_reqs_cap = int(max_num_reqs or scheduled_full_np.shape[0])
        max_num_reqs_cap = min(max_num_reqs_cap, scheduled_full_np.shape[0])

        num_requests = min(int(np.sum(active_mask_np[:max_num_reqs_cap])), max_num_reqs_cap)
        mask_reqs = np.arange(max_num_reqs_cap) < num_requests

        scheduled = np.where(mask_reqs, scheduled_full_np[:max_num_reqs_cap], 0).astype(np.int32)

        # query_start_loc: length max_num_reqs_cap+1 (padded)
        query_start_loc_np = np.zeros((max_num_reqs_cap + 1,), dtype=np.int32)
        if num_requests > 0:
            query_start_loc_np[1 : num_requests + 1] = np.cumsum(scheduled[:num_requests], dtype=np.int32)
            query_start_loc_np[num_requests + 1 :] = query_start_loc_np[num_requests]

        # context/seq lens per request (padded)
        context_lens_np = np.zeros((max_num_reqs_cap,), dtype=np.int32)
        if num_requests > 0:
            context_lens_np[:num_requests] = num_computed_np[:num_requests] + scheduled[:num_requests]

        # pages_tables masked for active requests
        if ragged_config is not None and hasattr(ragged_config, "max_num_pages_per_req"):
            max_pages_per_req = int(ragged_config.max_num_pages_per_req)
        else:
            max_pages_per_req = int(page_table_np.shape[1])

        num_reqs_max_model_len = num_requests
        if ragged_config is not None and hasattr(ragged_config, "get_max_num_seqs"):
            try:
                num_reqs_max_model_len = min(int(ragged_config.get_max_num_seqs()), max_num_reqs_cap)
            except Exception:
                num_reqs_max_model_len = num_requests

        pt_src = page_table_np[: min(page_table_np.shape[0], num_reqs_max_model_len), :max_pages_per_req]
        mask_rows = np.arange(num_reqs_max_model_len) < min(num_requests, num_reqs_max_model_len)
        pages_tables_np = np.where(mask_rows[:, None], pt_src, 0).astype(np.int32)

        out: dict[str, np.ndarray] = dict(
            pages_tables=pages_tables_np,
            context_lens=context_lens_np[:num_reqs_max_model_len],
            query_start_loc=query_start_loc_np[: num_reqs_max_model_len + 1],
            num_seqs=np.array([num_requests], dtype=np.int32),
        )

        if version == "v3":
            active_num_computed = np.where(mask_reqs, num_computed_np[:max_num_reqs_cap], 0)
            is_decode = (scheduled == 1) & (active_num_computed > 0)
            decode_count = int(np.sum(is_decode))
            out["request_distribution"] = np.array([decode_count, decode_count, num_requests], dtype=np.int32)

        elif version == "v2":
            if num_slices_per_kv_cache_update_page is None:
                raise ValueError("version='v2' requires `num_slices_per_kv_cache_update_page` or ragged_config.")

            slices_per_page = int(num_slices_per_kv_cache_update_page)
            max_padded_slices = None
            if ragged_config is not None and hasattr(ragged_config, "get_padded_num_slices"):
                try:
                    max_padded_slices = int(ragged_config.get_padded_num_slices(max_num_tokens, max_num_reqs_cap))
                except Exception:
                    max_padded_slices = None

            slot_mapping_np, total_pages = cls._compute_slot_mapping_v2_cpu(
                num_requests=num_requests,
                scheduled=scheduled,
                num_computed_tokens=num_computed_np,
                page_table=page_table_np,
                page_size=int(page_size),
                max_pages_per_req=int(max_pages_per_req),
                slices_per_page=slices_per_page,
                max_padded_slices=max_padded_slices,
            )
            out["slot_mapping"] = slot_mapping_np
            out["num_kv_update_slices"] = np.array([total_pages], dtype=np.int32)
        else:
            raise ValueError(f"Unknown ragged attention version: {version!r}.")

        return out

    @classmethod
    def compute_ragged_batch_fields_cpu(
        cls,
        *,
        scheduled_full: IntVectorLike,
        active_mask_full: BoolVectorLike,
        num_computed_tokens: IntVectorLike,
        page_table: PageTableLike,
        version: tp.Literal["v2", "v3"],
        page_size: int = 128,
        num_slices_per_kv_cache_update_page: int | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        ragged_config: RaggedPagesCacheConfig | None = None,
    ) -> _RaggedComputed:
        """Compute ragged/paged batch fields on CPU.

        Public wrapper around the CPU-first computation used by
        `build_ragged_page_metadata`. Returns NumPy arrays suitable for
        host payload assembly (e.g., ExecutionManager in eSurge).

        This is intended for use cases where you need the raw computed
        fields before constructing the final metadata object.

        Args:
            scheduled_full: Scheduled token counts per request.
                Shape: [max_num_reqs]
            active_mask_full: Boolean mask indicating active requests.
                Shape: [max_num_reqs]
            num_computed_tokens: Already computed token counts.
                Shape: [max_num_reqs]
            page_table: Page table mapping requests to physical pages.
                Shape: [max_num_reqs, max_pages_per_req]
            version: Ragged attention version ("v2" or "v3").
            page_size: Number of tokens per page. Default: 128.
            num_slices_per_kv_cache_update_page: Slices per update page (v2).
            max_num_reqs: Maximum requests for padding.
            max_num_tokens: Maximum tokens for padding.
            ragged_config: Optional configuration for additional parameters.

        Returns:
            _RaggedComputed: Dictionary of NumPy arrays with computed fields.

        Example:
            >>> fields = AttentionMetadataBuilder.compute_ragged_batch_fields_cpu(
            ...     scheduled_full=scheduled,
            ...     active_mask_full=active_mask,
            ...     num_computed_tokens=computed,
            ...     page_table=pages,
            ...     version="v3"
            ... )
            >>> pages_tables = fields["pages_tables"]
        """
        return cls._compute_ragged_from_batch_cpu(
            scheduled_full=scheduled_full,
            active_mask_full=active_mask_full,
            num_computed_tokens=num_computed_tokens,
            page_table=page_table,
            version=version,
            page_size=page_size,
            num_slices_per_kv_cache_update_page=num_slices_per_kv_cache_update_page,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            ragged_config=ragged_config,
        )

    @staticmethod
    def compute_padded_num_reqs(
        *,
        num_requests: int,
        max_num_reqs: int,
        min_input_pad: int,
        padded_num_reqs_in: int | None,
    ) -> int:
        """Compute a compilation-friendly padded request count.

        Uses a bucketing strategy to minimize JIT recompilation:
        - Pad small batches up to `min_input_pad`
        - Otherwise pad to the next power of 2
        - Honor caller-requested bucket when larger
        - Cap at `max_num_reqs`

        This strategy ensures consistent tensor shapes across similar
        batch sizes, reducing compilation overhead in production.

        Args:
            num_requests: Actual number of active requests.
            max_num_reqs: Maximum allowed requests (hard cap).
            min_input_pad: Minimum padding threshold. Small batches
                are padded to this value.
            padded_num_reqs_in: Optional explicit bucket size from caller.
                If provided and larger than num_requests, used directly.

        Returns:
            int: Padded request count for tensor allocation.

        Example:
            >>> # 3 requests with min_pad=8, max=1024
            >>> padded = AttentionMetadataBuilder.compute_padded_num_reqs(
            ...     num_requests=3,
            ...     max_num_reqs=1024,
            ...     min_input_pad=8,
            ...     padded_num_reqs_in=None
            ... )
            >>> padded
            8
        """
        max_num_reqs_i = max(int(max_num_reqs), 1)
        min_input_pad_i = max(int(min_input_pad), 1)

        nr_safe = max(int(num_requests), 1)
        if padded_num_reqs_in is not None:
            # When the caller provides an explicit bucket (e.g. precompiled buckets),
            # honor it directly (as long as it can fit the active requests).
            return min(max(int(padded_num_reqs_in), nr_safe), max_num_reqs_i)

        next_pow2 = 1 << (nr_safe - 1).bit_length()
        fallback_bucket = min_input_pad_i if nr_safe <= min_input_pad_i else next_pow2
        return min(fallback_bucket, max_num_reqs_i)

    @classmethod
    def compute_paged_attention_batch_fields_cpu(
        cls,
        *,
        num_tokens_static: int,
        scheduled_full: IntVectorLike,
        active_mask_full: BoolVectorLike,
        token_ids: IntMatrixLike,
        num_computed_tokens: IntVectorLike,
        page_table: PageTableLike,
        padded_num_reqs_in: int | None,
        min_input_pad: int,
        version: tp.Literal["v2", "v3"] = "v3",
        ragged_config: RaggedPagesCacheConfig | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        page_size: int | None = None,
        max_pages_per_req: int | None = None,
        num_slices_per_kv_cache_update_page: int | None = None,
        temperature: IntVectorLike | None = None,
        top_p: IntVectorLike | None = None,
        top_k: IntVectorLike | None = None,
        min_p: IntVectorLike | None = None,
        page_table_padding_val: int = 0,
        slot_mapping_padding_val: int = 0,
    ) -> _PagedBatchComputed:
        """Compute CPU batch fields for paged/ragged attention runners.

        This is a higher-level helper than `compute_ragged_batch_fields_cpu`.
        In addition to ragged fields, it also:
        - Gathers a contiguous `input_ids` batch from `token_ids`
        - Computes `position_ids` for the gathered tokens
        - Computes `logits_indices` for per-request sampling
        - Slices sampling parameters to `padded_num_reqs`

        This function is designed for high-performance inference runners
        that need to prepare a complete batch payload on CPU before
        a single device_put transfer.

        Important:
            This function assumes active requests are packed in the first
            `num_requests` slots (i.e., no holes in `active_mask_full` for
            the active prefix). This matches the typical scheduling contract
            used by eSurge and similar runners.

        Args:
            num_tokens_static: Fixed token bucket size for compilation.
            scheduled_full: Scheduled token counts per request.
                Shape: [max_num_reqs]
            active_mask_full: Boolean mask indicating active requests.
                Shape: [max_num_reqs]
            token_ids: Full token ID buffer for all requests.
                Shape: [max_num_reqs, max_seq_len]
            num_computed_tokens: Already computed token counts.
                Shape: [max_num_reqs]
            page_table: Page table mapping requests to physical pages.
                Shape: [max_num_reqs, max_pages_per_req]
            padded_num_reqs_in: Optional explicit padded request count.
            min_input_pad: Minimum padding for request count bucketing.
            version: Ragged attention version ("v2" or "v3").
            ragged_config: Optional configuration for additional parameters.
            max_num_reqs: Maximum requests cap.
            max_num_tokens: Maximum tokens cap.
            page_size: Tokens per page (for v2).
            max_pages_per_req: Maximum pages per request.
            num_slices_per_kv_cache_update_page: Slices per page (for v2).
            temperature: Sampling temperature per request. Optional.
            top_p: Top-p sampling parameter per request. Optional.
            top_k: Top-k sampling parameter per request. Optional.
            min_p: Min-p sampling parameter per request. Optional.
            page_table_padding_val: Padding value for page tables.
            slot_mapping_padding_val: Padding value for slot mapping (v2).

        Returns:
            _PagedBatchComputed: Dictionary containing:
                - input_ids: Contiguous token IDs [num_tokens_static]
                - position_ids: Position IDs [num_tokens_static]
                - query_start_loc: Cumulative positions [max_num_reqs + 1]
                - seq_lens: Context lengths [max_num_reqs]
                - logits_indices: Sampling indices [padded_num_reqs]
                - pages_tables: Page table [num_reqs, max_pages_per_req]
                - scheduled: Scheduled counts [padded_num_reqs]
                - num_requests: Active request count
                - padded_num_reqs: Padded request count
                - actual_num_tokens: Total scheduled tokens
                - Plus version-specific fields (v2 or v3)

        Raises:
            ValueError: If token_ids is not 2D or if scheduled tokens
                exceed num_tokens_static.

        Example:
            >>> batch = AttentionMetadataBuilder.compute_paged_attention_batch_fields_cpu(
            ...     num_tokens_static=2048,
            ...     scheduled_full=scheduled,
            ...     active_mask_full=active_mask,
            ...     token_ids=all_tokens,
            ...     num_computed_tokens=computed,
            ...     page_table=pages,
            ...     padded_num_reqs_in=None,
            ...     min_input_pad=8,
            ...     version="v3"
            ... )
        """

        scheduled_full_np = cls._ensure_cpu_array(scheduled_full).astype(np.int32, copy=False)
        active_mask_np = cls._ensure_cpu_array(active_mask_full).astype(bool, copy=False)
        token_ids_np = cls._ensure_cpu_array(token_ids).astype(np.int32, copy=False)
        num_computed_np = cls._ensure_cpu_array(num_computed_tokens).astype(np.int32, copy=False)
        page_table_np = cls._ensure_cpu_array(page_table).astype(np.int32, copy=False)

        if token_ids_np.ndim != 2:
            raise ValueError(f"`token_ids` must be a 2D array, got shape={token_ids_np.shape}")

        max_num_reqs_cap = int(max_num_reqs or scheduled_full_np.shape[0])
        max_num_reqs_cap = max(1, min(max_num_reqs_cap, scheduled_full_np.shape[0]))
        max_num_reqs_cap = min(
            max_num_reqs_cap, active_mask_np.shape[0], num_computed_np.shape[0], token_ids_np.shape[0]
        )

        max_num_tokens_cap = int(max_num_tokens or num_tokens_static)
        if num_tokens_static > max_num_tokens_cap:
            raise ValueError(
                f"`num_tokens_static` ({num_tokens_static}) exceeds `max_num_tokens` ({max_num_tokens_cap})"
            )

        num_requests = min(int(np.sum(active_mask_np[:max_num_reqs_cap])), max_num_reqs_cap)
        mask_reqs = np.arange(max_num_reqs_cap) < num_requests

        scheduled = np.where(mask_reqs, scheduled_full_np[:max_num_reqs_cap], 0).astype(np.int32, copy=False)

        padded_num_reqs = cls.compute_padded_num_reqs(
            num_requests=num_requests,
            max_num_reqs=max_num_reqs_cap,
            min_input_pad=min_input_pad,
            padded_num_reqs_in=padded_num_reqs_in,
        )

        query_start_loc_np = np.zeros((max_num_reqs_cap + 1,), dtype=np.int32)
        if num_requests > 0:
            query_start_loc_np[1 : num_requests + 1] = np.cumsum(scheduled[:num_requests], dtype=np.int32)
            query_start_loc_np[num_requests + 1 :] = query_start_loc_np[num_requests]

        seq_lens_np = np.zeros((max_num_reqs_cap,), dtype=np.int32)
        if num_requests > 0:
            seq_lens_np[:num_requests] = num_computed_np[:num_requests] + scheduled[:num_requests]

        logits_indices_full = np.zeros((max_num_reqs_cap,), dtype=np.int32)
        if num_requests > 0:
            logits_indices_full[:num_requests] = query_start_loc_np[1 : num_requests + 1] - 1

        # Build contiguous token batch (input_ids, position_ids).
        # Avoid `repeat/concatenate` which allocate large temporary arrays; loop only over requests.
        input_ids_np = np.zeros((num_tokens_static,), dtype=np.int32)
        position_ids_np = np.zeros((num_tokens_static,), dtype=np.int32)

        actual_num_tokens = int(query_start_loc_np[num_requests]) if num_requests > 0 else 0
        if actual_num_tokens > num_tokens_static:
            raise ValueError(
                f"Scheduled {actual_num_tokens} tokens but `num_tokens_static`={num_tokens_static}; "
                "select a larger token bucket."
            )

        if actual_num_tokens > 0:
            max_sched = int(np.max(scheduled[:num_requests])) if num_requests > 0 else 0
            token_arange = np.arange(max(max_sched, 1), dtype=np.int32)
            off = 0
            for req_idx in range(num_requests):
                n = int(scheduled[req_idx])
                if n <= 0:
                    continue
                start = int(num_computed_np[req_idx])
                end = start + n
                if end > token_ids_np.shape[1]:
                    raise ValueError(
                        f"Request {req_idx} scheduled [{start}:{end}] exceeds token_ids width {token_ids_np.shape[1]}."
                    )
                position_ids_np[off : off + n] = start + token_arange[:n]
                input_ids_np[off : off + n] = token_ids_np[req_idx, start:end]
                off += n

        # pages_tables: fixed shape based on config/caps, masked for inactive rows.
        if max_pages_per_req is None:
            if ragged_config is not None and hasattr(ragged_config, "max_num_pages_per_req"):
                max_pages_per_req = int(ragged_config.max_num_pages_per_req)
            else:
                max_pages_per_req = int(page_table_np.shape[1])

        if ragged_config is not None and hasattr(ragged_config, "get_max_num_seqs"):
            try:
                num_reqs_max_model_len = min(int(ragged_config.get_max_num_seqs()), max_num_reqs_cap)
            except Exception:
                num_reqs_max_model_len = max_num_reqs_cap
        else:
            num_reqs_max_model_len = max_num_reqs_cap

        pages_tables_np = np.full(
            (num_reqs_max_model_len, int(max_pages_per_req)),
            int(page_table_padding_val),
            dtype=np.int32,
        )
        rows_to_copy = min(int(page_table_np.shape[0]), num_reqs_max_model_len)
        if rows_to_copy > 0:
            pages_tables_np[:rows_to_copy, :] = page_table_np[:rows_to_copy, : int(max_pages_per_req)]
        if num_requests < num_reqs_max_model_len:
            pages_tables_np[num_requests:, :] = int(page_table_padding_val)

        out: dict[str, np.ndarray] = dict(
            input_ids=input_ids_np,
            position_ids=position_ids_np,
            query_start_loc=query_start_loc_np,
            seq_lens=seq_lens_np,
            logits_indices=logits_indices_full[:padded_num_reqs],
            pages_tables=pages_tables_np,
            scheduled=scheduled[:padded_num_reqs],
            num_requests=np.int32(num_requests),
            padded_num_reqs=np.int32(padded_num_reqs),
            actual_num_tokens=np.int32(actual_num_tokens),
        )

        # Optional sampling params (sliced to padded_num_reqs).
        if temperature is not None:
            out["temperature"] = cls._ensure_cpu_array(temperature).astype(np.float32, copy=False)[:padded_num_reqs]
        if top_p is not None:
            out["top_p"] = cls._ensure_cpu_array(top_p).astype(np.float32, copy=False)[:padded_num_reqs]
        if top_k is not None:
            out["top_k"] = cls._ensure_cpu_array(top_k).astype(np.int32, copy=False)[:padded_num_reqs]
        if min_p is not None:
            out["min_p"] = cls._ensure_cpu_array(min_p).astype(np.float32, copy=False)[:padded_num_reqs]

        if version == "v3":
            active_num_computed = np.where(mask_reqs, num_computed_np[:max_num_reqs_cap], 0)
            is_decode = (scheduled == 1) & (active_num_computed > 0)
            decode_count = int(np.sum(is_decode))
            out["request_distribution"] = np.array([decode_count, decode_count, num_requests], dtype=np.int32)
        elif version == "v2":
            if page_size is None:
                page_size = int(ragged_config.page_size) if ragged_config is not None else 128
            if num_slices_per_kv_cache_update_page is None:
                num_slices_per_kv_cache_update_page = (
                    int(ragged_config.num_slices_per_kv_cache_update_page) if ragged_config is not None else None
                )
            if num_slices_per_kv_cache_update_page is None:
                raise ValueError("version='v2' requires `num_slices_per_kv_cache_update_page` or ragged_config.")

            slices_per_page = int(num_slices_per_kv_cache_update_page)
            max_padded_slices = None
            if ragged_config is not None and hasattr(ragged_config, "get_padded_num_slices"):
                try:
                    max_padded_slices = int(ragged_config.get_padded_num_slices(max_num_tokens_cap, max_num_reqs_cap))
                except Exception:
                    max_padded_slices = None

            slot_mapping_np, total_pages = cls._compute_slot_mapping_v2_cpu_padded(
                num_requests=num_requests,
                scheduled=scheduled,
                num_computed_tokens=num_computed_np,
                page_table=page_table_np,
                page_size=int(page_size),
                max_pages_per_req=int(max_pages_per_req),
                slices_per_page=slices_per_page,
                max_padded_slices=max_padded_slices,
                pad_value=int(slot_mapping_padding_val),
            )
            out["slot_mapping"] = slot_mapping_np
            out["num_kv_update_slices"] = np.array([total_pages], dtype=np.int32)
        else:
            raise ValueError(f"Unknown ragged attention version: {version!r}.")

        return tp.cast(_PagedBatchComputed, out)

    @classmethod
    def _compute_slot_mapping_v2_cpu_padded(
        cls,
        *,
        num_requests: int,
        scheduled: np.ndarray,
        num_computed_tokens: np.ndarray,
        page_table: np.ndarray,
        page_size: int,
        max_pages_per_req: int,
        slices_per_page: int,
        max_padded_slices: int | None,
        pad_value: int,
    ) -> tuple[np.ndarray, int]:
        """Compute v2 slot_mapping with a fixed padded shape.

        Similar to _compute_slot_mapping_v2_cpu but always returns an array
        of shape [3, max_padded_slices] for compilation stability. Unused
        slots are filled with pad_value.

        Args:
            num_requests: Number of active requests.
            scheduled: Scheduled token counts per request.
            num_computed_tokens: Already computed token counts.
            page_table: Page table mapping requests to pages.
            page_size: Tokens per page.
            max_pages_per_req: Maximum pages per request.
            slices_per_page: Slices per processing page.
            max_padded_slices: Fixed output slice dimension.
            pad_value: Value for padding unused slots.

        Returns:
            tuple: Pair of (slot_mapping, total_pages) where:
                - slot_mapping: [3, max_padded_slices] with padded shape
                - total_pages: Actual number of pages being updated
        """

        if max_padded_slices is None or int(max_padded_slices) <= 0:
            return cls._compute_slot_mapping_v2_cpu(
                num_requests=num_requests,
                scheduled=scheduled,
                num_computed_tokens=num_computed_tokens,
                page_table=page_table,
                page_size=page_size,
                max_pages_per_req=max_pages_per_req,
                slices_per_page=slices_per_page,
                max_padded_slices=max_padded_slices,
            )

        max_padded_slices_i = int(max_padded_slices)
        slot_mapping = np.full((3, max_padded_slices_i), pad_value, dtype=np.int32)
        if num_requests <= 0:
            return slot_mapping, 0

        scheduled_active = scheduled[:num_requests].astype(np.int32, copy=False)
        num_computed_active = num_computed_tokens[:num_requests].astype(np.int32, copy=False)
        if not np.any(scheduled_active):
            return slot_mapping, 0

        start_tokens = num_computed_active
        end_tokens = num_computed_active + scheduled_active

        lps = start_tokens // page_size
        lpe = (np.maximum(end_tokens, 1) - 1) // page_size
        page_lens = np.where(scheduled_active > 0, lpe - lps + 1, 0).astype(np.int32)
        if not np.any(page_lens):
            return slot_mapping, 0

        page_cum = np.cumsum(page_lens, dtype=np.int32)
        total_pages = int(page_cum[-1])

        padded_num_slices = ((total_pages + slices_per_page - 1) // slices_per_page) * slices_per_page
        padded_num_slices = min(max_padded_slices_i, max(int(padded_num_slices), 1))

        indices = np.arange(max_padded_slices_i, dtype=np.int32)
        active_positions = indices[(indices < total_pages) & (indices < padded_num_slices)]
        if active_positions.size == 0:
            return slot_mapping, total_pages

        page_cum_prev = np.concatenate(([0], page_cum[:-1]))
        req_for_slice = np.searchsorted(page_cum, active_positions, side="right").astype(np.int32)
        local_off = active_positions - page_cum_prev[req_for_slice]

        pt = page_table[:num_requests, :max_pages_per_req].astype(np.int32, copy=False)
        pt_flat = pt.reshape(-1)
        gather_idx = req_for_slice * max_pages_per_req + lps[req_for_slice] + local_off
        np.clip(gather_idx, 0, pt_flat.size - 1, out=gather_idx)
        page_numbers = pt_flat[gather_idx]

        s_mod = start_tokens % page_size
        e_mod = ((np.maximum(end_tokens, 1) - 1) % page_size) + 1

        is_first = local_off == 0
        is_last = local_off == (page_lens[req_for_slice] - 1)

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

        return slot_mapping, total_pages
