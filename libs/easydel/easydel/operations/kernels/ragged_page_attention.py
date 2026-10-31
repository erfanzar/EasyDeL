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
- Integration with serving-style paged memory management

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
    - PagedAttention: https://arxiv.org/abs/2309.06180
    - EasyDeL serving documentation
"""

from functools import partial

import jax
from ejkernel.loggings import get_logger
from ejkernel.modules import (  # pyright: ignore[reportMissingTypeStubs]
    ragged_page_attention_v2,
    ragged_page_attention_v3,
    reduce_sinks_to_heads,
)
from ejkernel.modules.operations.configs import RaggedPageAttentionv3Config
from ejkernel.quantization.turboquant import (
    turboquant_compress_keys,
    turboquant_compress_values,
    turboquant_dequantize_values,
    unpack_signs,
)
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, DTypeLike, Float
from spectrax import common_types as ct

from easydel.axis import ATTN_DP
from easydel.caching import RaggedPagesCacheView, RaggedPagesMetadata, UnifiedAttentionCacheView, unwrap_metadata
from easydel.caching.mla_ragged_page import MLARaggedPagesCacheView
from easydel.caching.ragged_page.cache import HEAD_DIM_ALIGNMENT, align_to_multiple, kv_pair_shares_head_dim_axis
from easydel.caching.turboquant_ragged_page import TurboQuantRaggedPagesCacheView
from easydel.infra.sharding import axis_index, mesh_axis_size, normalize_axis_names

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)
from ._paged_common import ENABLE_DP_LOCAL_PAGE_PATH, dp_page_axis, request_distribution_bounds

logger = get_logger(__name__)

# Alias for callers importing the private helper under its adapter-local name.
_request_distribution_bounds = request_distribution_bounds


def _chunk_prefill_size_from_cfg(cfg) -> int | None:
    """Extract ``chunk_prefill_size`` from a dict or dataclass-style config.

    Args:
        cfg: Operation config; may be ``None``, a ``dict``, or a dataclass.

    Returns:
        int | None: The configured chunk prefill size, or ``None`` when
        unset or ``cfg`` is ``None``.
    """
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get("chunk_prefill_size")
    return getattr(cfg, "chunk_prefill_size", None)


def _cfg_value(cfg, name: str):
    """Look up an attribute or key on a dual-shape (dict/dataclass) config.

    Args:
        cfg: Configuration object or dictionary; may be ``None``.
        name: Attribute / key name to fetch.

    Returns:
        Any: The configured value, or ``None`` when missing.
    """
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(name)
    return getattr(cfg, name, None)


def _ceil_div(a: int, b: int) -> int:
    """Return ``ceil(a / b)`` for positive integers.

    Args:
        a: Dividend.
        b: Positive divisor.

    Returns:
        int: The ceiling of the integer division.
    """
    return (a + b - 1) // b


def _align_to(value: int, boundary: int) -> int:
    """Round ``value`` up to the next multiple of ``boundary``.

    Args:
        value: Non-negative integer to align.
        boundary: Positive alignment boundary.

    Returns:
        int: The smallest multiple of ``boundary`` that is at least
        ``value``.
    """
    return _ceil_div(value, boundary) * boundary


def _next_power_of_2(value: int) -> int:
    """Return the smallest power of 2 greater than or equal to ``value``.

    Args:
        value: Non-negative integer.

    Returns:
        int: ``1`` when ``value <= 1``; otherwise the next power of two
        ``>= value``.
    """
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _tpu_version() -> int:
    """Detect the major TPU generation of the local devices.

    Inspects the first JAX device's ``device_kind`` string and maps it to
    a small integer used by :func:`_default_tpu_rpa_v3_cfg` to pick block
    sizes per TPU generation.

    Returns:
        int: ``7`` for TPU v7-class (Ironwood) devices, the trailing
        version number (e.g. ``5`` for TPU v5) for ``TPU v*`` devices,
        and ``-1`` when not running on TPU or the kind cannot be parsed.
    """
    try:
        kind = jax.devices()[0].device_kind
    except Exception:
        return -1
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    if kind.endswith("p") or kind.endswith("e"):
        kind = kind[:-1]
    if kind == "TPU7x":
        return 7
    if not kind.startswith("TPU v"):
        return -1
    return int(kind[-1])


def _default_tpu_rpa_v3_cfg(
    cfg,
    *,
    query: Array,
    key: Array,
    kv_pages: Array,
    context_lens: Array,
    pages_tables: Array,
):
    """Choose serving-style TPU RPA block geometry when no user override exists.

    Inspects the runtime tensor shapes plus the local TPU version to derive
    sensible ``num_kv_pages_per_block`` and ``num_queries_per_block`` values
    that mirror the TPU serving defaults. Returns the input config unchanged if
    the user already supplied either knob, if the backend is not TPU, or
    if the shape introspection fails.

    Args:
        cfg: Existing operation config (dict, dataclass, or ``None``).
        query: Ragged query tensor used to derive ``max_num_tokens`` and
            head counts.
        key: Ragged key tensor used to read the KV-head count.
        kv_pages: KV-page pool used to read ``page_size``.
        context_lens: Per-request context lengths; ``shape[0]`` gives the
            scheduled request count.
        pages_tables: ``[num_seqs, max_pages]`` block table from which the
            per-sequence page count is inferred.

    Returns:
        RaggedPageAttentionv3Config | Any: A new config carrying TPU-tuned
        block sizes, or the original ``cfg`` if no defaults apply.
    """
    if jax.default_backend() != "tpu":
        return cfg
    if _cfg_value(cfg, "num_kv_pages_per_block") is not None or _cfg_value(cfg, "num_queries_per_block") is not None:
        return cfg

    try:
        actual_num_q_heads = int(query.shape[1])
        actual_num_kv_heads = int(key.shape[1])
        page_size = int(kv_pages.shape[1])
        max_num_tokens = int(query.shape[0])
        max_num_seqs = int(context_lens.shape[0])
        block_table_size = int(pages_tables.size)
    except (AttributeError, IndexError, TypeError, ValueError):
        return cfg
    if actual_num_q_heads <= 0 or actual_num_kv_heads <= 0 or page_size <= 0 or max_num_tokens <= 0:
        return cfg
    if max_num_seqs <= 0 or block_table_size % max_num_seqs != 0:
        return cfg

    pages_per_seq = block_table_size // max_num_seqs
    max_q = _next_power_of_2(max_num_tokens)
    max_kv = pages_per_seq * page_size
    q_heads_per_kv = _next_power_of_2(actual_num_q_heads // actual_num_kv_heads)
    # The token/head budgets below were tuned for head_dim ~= 128. Larger head
    # dims (e.g. 256 on Qwen3.5) scale the per-block scratchpad linearly, so
    # shrink the budgets by head_dim/128 to keep the kernel within the 16MB TPU
    # scoped-VMEM limit (otherwise the chosen block_q overflows it).
    try:
        head_dim = int(query.shape[2])
    except (IndexError, TypeError, ValueError):
        head_dim = 128
    head_dim_factor = max(1, head_dim // 128)

    match _tpu_version():
        case 5 | 6:
            block_q = min((1024 // head_dim_factor) // q_heads_per_kv, max_q // 2)
            block_kv_tokens = min(1024 // head_dim_factor, max_kv)
        case 7:
            block_q = min((2048 // head_dim_factor) // q_heads_per_kv, max_q // 2)
            block_kv_tokens = min(2048 // head_dim_factor, max_kv // 2)
        case _:
            return cfg

    return RaggedPageAttentionv3Config(
        chunk_prefill_size=_chunk_prefill_size_from_cfg(cfg),
        num_kv_pages_per_block=max(1, _align_to(max(1, int(block_kv_tokens)), page_size) // page_size),
        num_queries_per_block=max(1, int(block_q)),
        num_warps=_cfg_value(cfg, "num_warps") or 4,
        num_stages=_cfg_value(cfg, "num_stages") or 1,
        platform=_cfg_value(cfg, "platform") or "auto",
        backend=_cfg_value(cfg, "backend") or "any",
    )


def _tpu_mixed_request_distribution(request_distribution: Array, cfg) -> Array:
    """Match the TPU serving path's RPA request-distribution convention.

    The mixed-prefill convention routes every non-decode request through the mixed RPA case unless a
    chunk-prefill size is configured. The native ejkernel TPU path supports the
    same case layout, and using the same distribution keeps the EasyDeL call
    geometry aligned with the rank-major runner.

    Args:
        request_distribution: ``int32[3]`` distribution vector
            ``[decode_end, prefill_end, total]``.
        cfg: Operation config whose ``chunk_prefill_size`` controls whether
            the mixed remapping should be applied.

    Returns:
        Array: The input distribution unchanged for non-TPU backends or
        when ``chunk_prefill_size`` is set; otherwise a remapped vector
        ``[decode_end, decode_end, total]`` that routes prefill traffic
        through the mixed RPA case.
    """
    chunk_prefill_size = _chunk_prefill_size_from_cfg(cfg)
    if jax.default_backend() != "tpu" or chunk_prefill_size is not None:
        return request_distribution
    return jnp.stack((request_distribution[0], request_distribution[0], request_distribution[2])).astype(
        request_distribution.dtype
    )


# Alias for callers importing the private helper under its adapter-local name.
_dp_page_axis = dp_page_axis


def _kv_axis(cache_view: RaggedPagesCacheView, sharded_axis: str):
    """Return ``sharded_axis`` only when the cache keeps a sharded KV-head axis.

    Args:
        cache_view: Ragged-pages cache view.
        sharded_axis: Axis name to use when KV heads are sharded.

    Returns:
        str: ``sharded_axis`` if the cache reports more than one KV-head
        shard, otherwise ``ct.EMPTY`` (replicated).
    """
    kv_head_shards = max(1, int(getattr(cache_view.metadata, "kv_head_shards", 1)))
    return sharded_axis if kv_head_shards > 1 else ct.EMPTY


def _repeat_kv_heads_to_cache_width(
    key: Float[Array, "total_tokens num_kv_heads head_dim"],
    value: Float[Array, "total_tokens num_kv_heads head_dim"],
    cache_view: RaggedPagesCacheView,
):
    """Repeat GQA/MQA KV heads to match a padded TP-sharded cache width.

    When the cache view was allocated with a wider KV-head axis than the
    model produces (e.g. for TP padding), this duplicates each head along
    axis 1 until the source matches the cache width.

    Args:
        key: Ragged keys of shape ``(total_tokens, num_kv_heads, head_dim)``.
        value: Ragged values of shape ``(total_tokens, num_kv_heads,
            head_dim)``.
        cache_view: Cache view whose ``metadata.num_kv_heads`` defines the
            target width.

    Returns:
        tuple: ``(key, value)`` widened to match the cache. Returned
        unchanged when the widths already agree.

    Raises:
        ValueError: When the cache width is narrower than the source or
            not an integer multiple of it.
    """
    target_num_kv_heads = int(getattr(cache_view.metadata, "num_kv_heads", key.shape[1]))
    source_num_kv_heads = int(key.shape[1])
    if target_num_kv_heads == source_num_kv_heads:
        return key, value
    if target_num_kv_heads < source_num_kv_heads or target_num_kv_heads % source_num_kv_heads != 0:
        raise ValueError(
            "Ragged-page KV cache width is incompatible with the model KV heads: "
            f"cache num_kv_heads={target_num_kv_heads}, tensor num_kv_heads={source_num_kv_heads}."
        )
    repeat_factor = target_num_kv_heads // source_num_kv_heads
    return jnp.repeat(key, repeat_factor, axis=1), jnp.repeat(value, repeat_factor, axis=1)


def _runtime_sharding_resolver(metadata, cache_view):
    """Resolve the sharding resolver from op metadata or the active cache view.

    Falls back through ``runtime_sharding_resolver`` on metadata, then on
    cache view, then ``partition_manager`` on either object.

    Args:
        metadata: :class:`OperationMetadata` for the call.
        cache_view: Ragged-pages cache view; checked when metadata does not
            expose a resolver directly.

    Returns:
        Any: A resolver bound to ``metadata.mesh`` if one is available,
        otherwise the raw resolver.

    Raises:
        AttributeError: When neither object exposes a resolver or partition
            manager.
    """
    resolver = getattr(metadata, "runtime_sharding_resolver", None)
    if resolver is None:
        resolver = getattr(cache_view, "runtime_sharding_resolver", None)
    if resolver is None:
        resolver = getattr(metadata, "partition_manager", None)
    if resolver is None:
        resolver = getattr(cache_view, "partition_manager", None)
    if resolver is None:
        raise AttributeError("Ragged page attention requires a runtime sharding resolver or partition manager.")
    mesh = getattr(metadata, "mesh", None)
    if hasattr(resolver, "with_mesh"):
        return resolver.with_mesh(mesh)
    return resolver


@OperationRegistry.register
class PagedReferenceAttn(OperationImpl):
    """Paged-cache adapter for key-based dropout and true ring attention.

    Request-local gathers preserve page-table addressing, ragged query boundaries
    and absolute causal positions. Only one request's dense KV is materialized at
    a time; the attention calculation still goes through the operation registry.
    Global page indexing lets JAX preserve DP/TP ownership through gather/scatter
    instead of assuming that physical page ids are local to a device shard.
    """

    @classmethod
    def get_impl_name(cls):
        return "paged_attention_reference"

    @classmethod
    def get_requirements(cls, mode=ExecutionMode.MIXED):
        return OperationRequirements.create(
            name=cls.get_impl_name(),
            required_metadata=(
                MetadataField.basic()
                | MetadataField.CONTEXT_LENS
                | MetadataField.QUERY_START_LOC
                | MetadataField.PAGES_TABLES
            ),
            optional_metadata=MetadataField.SLOT_MAPPING | MetadataField.REQUEST_DISTRIBUTION,
            supported_cache=CacheType.RAGGED_PAGES,
            cache_view_class=None,  # Both interleaved and split-page views are accepted below.
        )

    def forward_native(
        self,
        query,
        key=None,
        value=None,
        cache_view=None,
        cache_metadata=None,
        softmax_scale=None,
        logits_soft_cap=None,
        sliding_window=None,
        softmax_aux=None,
        deterministic=True,
        dropout_rng=None,
        dropout_prob=0.0,
        normalize_output=True,
        causal=True,
        precision=None,
        compute_dtype=None,
        dense_impl="vanilla",
        chunk_size=None,
        bias=None,
        mask_info=None,
        alibi_slopes=None,
        queries_nope=None,
        queries_pe=None,
        keys_values=None,
        keys_pe=None,
        **ignore,
    ):
        """Update paged KV and attend within each scheduled request.

        Inputs may be BTHD or packed THD. Metadata context lengths include the
        current scheduled tokens; rank-major DP boundaries retain each rank's
        padded token capacity. Omit both key and value to read a populated cache
        without writing it. ``dense_impl='ring'`` invokes the registered ring
        kernel on each request, not an alias to another mechanism. Active dropout
        uses vanilla's key-aware weight dropout; the same key is folded with the
        request row so repeated calls are reproducible. Masks and bias may use
        BTHD batch-local or packed query rows and request-relative KV columns.
        Segment-only masks and ragged-v3 ALiBi slopes are also preserved.
        MLA views accept the same explicit ``queries_nope``, ``queries_pe``,
        ``keys_values`` and ``keys_pe`` operands as the MLA kernels. Their shared
        latent and RoPE components retain independently aligned cache offsets;
        attention returns the latent value width for the caller's absorbed V
        projection. Cache-only calls may supply a concatenated latent/RoPE query.
        """
        mla_pages = isinstance(cache_view, MLARaggedPagesCacheView)
        if mla_pages:
            latent_dim = cache_view.metadata.kv_lora_rank
            rope_dim = cache_view.metadata.qk_rope_head_dim
            rope_offset = align_to_multiple(latent_dim, HEAD_DIM_ALIGNMENT)
            if queries_nope is not None or queries_pe is not None:
                if queries_nope is None or queries_pe is None:
                    raise ValueError("MLA requires both queries_nope and queries_pe.")
                query = jnp.concatenate((queries_nope, queries_pe), axis=-1)
            if query.shape[-1] != latent_dim + rope_dim:
                raise ValueError("MLA queries must contain the latent and RoPE dimensions.")
            if keys_values is not None or keys_pe is not None:
                if keys_values is None or keys_pe is None:
                    raise ValueError("MLA requires both keys_values and keys_pe for cache updates.")
                # MLA pages store one shared latent/position row per token,
                # not the alternating K/V head layout of ordinary ragged pages.
                latent = keys_values.reshape(-1, latent_dim)
                positional = keys_pe.reshape(query.size // (query.shape[-2] * query.shape[-1]), rope_dim)
                key = jnp.concatenate((latent, positional), axis=-1)[:, None, :]
                value = latent[:, None, :]
            elif key is not None and value is not None:
                # The generic cache-concatenation path supplies combined K and
                # V placeholders. Values are the shared latent, not that whole row.
                value = key[..., :latent_dim]
        if not isinstance(cache_view, (RaggedPagesCacheView, UnifiedAttentionCacheView)):
            raise TypeError("PagedReferenceAttn requires a ragged-pages or unified-attention cache view.")
        if not 0.0 <= dropout_prob < 1.0:
            raise ValueError("dropout_prob must be in [0, 1).")
        if not deterministic and dropout_prob > 0 and dropout_rng is None:
            raise ValueError("dropout_rng is required for attention-weight dropout.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        runtime = unwrap_metadata(cache_metadata, "ragged")
        if hasattr(runtime, "for_layer"):
            runtime = runtime.for_layer(cache_view.layer_index)
        if runtime is None:
            raise ValueError("PagedReferenceAttn requires paged runtime metadata.")
        q = query.reshape(-1, *query.shape[-2:])
        if (key is None) != (value is None):
            raise ValueError("Provide both key and value, or neither for a populated-cache read.")
        update_cache = key is not None
        split_pages = isinstance(cache_view, UnifiedAttentionCacheView)
        compressed_pages = isinstance(cache_view, TurboQuantRaggedPagesCacheView)
        if update_cache:
            k = key.reshape(-1, *key.shape[-2:])
            v = value.reshape(-1, *value.shape[-2:])
            if not (q.shape[0] == k.shape[0] == v.shape[0]) or k.shape[1] != v.shape[1]:
                raise ValueError("Paged Q/K/V must describe the same packed tokens and matching KV-head counts.")
            if mla_pages:
                if k.shape[1:] != (1, latent_dim + rope_dim) or v.shape[1:] != (1, latent_dim):
                    raise ValueError("MLA cache updates require one shared latent/RoPE row per token.")
            else:
                k, v = _repeat_kv_heads_to_cache_width(k, v, cache_view)
            num_kv_heads, key_dim, value_dim = k.shape[1], k.shape[-1], v.shape[-1]
        else:
            num_kv_heads = 1 if mla_pages else int(cache_view.metadata.num_kv_heads)
            key_dim = query.shape[-1]
            value_dim = (
                latent_dim
                if mla_pages
                else cache_view.value_cache.shape[-1]
                if split_pages
                else int(cache_view.metadata.v_headdim)
            )
        if compressed_pages and (key_dim != cache_view.metadata.k_headdim or value_dim != cache_view.metadata.v_headdim):
            raise ValueError("TurboQuant KV head dimensions must match the cache configuration.")
        output_dtype = compute_dtype or self.metadata.runtime_dtype or query.dtype
        starts = runtime.query_start_loc
        contexts = runtime.context_lens.reshape(-1)
        num_requests = contexts.shape[0]
        if num_requests == 0 or q.shape[0] == 0:
            return AttentionOutput(
                attention_outputs=jnp.zeros((*query.shape[:-1], value_dim), dtype=output_dtype),
                attention_weights=None,
                cache_view=cache_view,
            )
        tables = runtime.pages_tables.reshape(num_requests, -1)
        if starts.ndim == 2:
            # eSurge packs a fixed token capacity per DP rank, not just the
            # active tokens. Offset rank-local boundaries by that capacity;
            # retaining the gaps is essential for idle ranks and padded rows.
            ranks, boundaries = starts.shape
            if ranks * (boundaries - 1) != num_requests or q.shape[0] % ranks:
                raise ValueError("Rank-major boundaries must match request rows and equal rank token capacities.")
            rank_offsets = jnp.arange(ranks, dtype=starts.dtype)[:, None] * (q.shape[0] // ranks)
            request_starts = (starts[:, :-1] + rank_offsets).reshape(-1)
            request_ends = (starts[:, 1:] + rank_offsets).reshape(-1)
        else:
            starts = starts.reshape(-1)
            if starts.shape[0] != num_requests + 1:
                raise ValueError("query_start_loc must contain one boundary per request plus the final boundary.")
            request_starts, request_ends = starts[:-1], starts[1:]
        scheduled = request_ends - request_starts
        token_ids = jnp.arange(q.shape[0], dtype=jnp.int32)
        rows = jnp.minimum(jnp.searchsorted(request_ends, token_ids, side="right"), num_requests - 1)
        positions = contexts[rows] - scheduled[rows] + token_ids - request_starts[rows]
        active_tokens = (
            (token_ids >= request_starts[rows])
            & (token_ids < request_ends[rows])
            & (scheduled[rows] > 0)
            & (positions >= 0)
        )
        paired_dim = (
            not split_pages
            and not compressed_pages
            and not mla_pages
            and cache_view.metadata.is_v3
            and kv_pair_shares_head_dim_axis(key_dim)
        )
        if compressed_pages:
            # TurboQuant inherits the ragged view but its kv_pages is a scalar
            # placeholder. Never flatten or overwrite it as dense KV storage.
            num_pages, page_size = cache_view.key_indices_pages.shape[:2]
            constants = cache_view.constants
        else:
            if mla_pages:
                pages = cache_view.flattened_kv_pages()
                # Packing groups adjacent tokens, not heads. Preserve any
                # rounded-up physical page rows while exposing logical rows only.
                flat = pages.reshape(pages.shape[0], -1, pages.shape[-1])
                key_pages = flat[:, : cache_view.metadata.page_size, None, :]
                value_pages = key_pages[..., :latent_dim]
            elif split_pages:
                key_pages, value_pages = cache_view.key_cache, cache_view.value_cache
            else:
                flat = cache_view.flattened_kv_pages()
                key_pages = flat[..., :key_dim] if paired_dim else flat[:, :, 0::2, :key_dim]
                value_pages = flat[..., key_dim : key_dim + value_dim] if paired_dim else flat[:, :, 1::2, :value_dim]
            num_pages, page_size = key_pages.shape[:2]
        logical_pages = positions // page_size
        in_table = (logical_pages >= 0) & (logical_pages < tables.shape[1])
        physical_pages = tables[rows, jnp.clip(logical_pages, 0, tables.shape[1] - 1)]
        active_tokens &= in_table & (physical_pages >= 0) & (physical_pages < num_pages)
        # OOB scatter destinations are dropped, not clamped to a real request's page.
        destinations = jnp.where(active_tokens, physical_pages, num_pages)
        offsets = positions % page_size
        if update_cache and compressed_pages:
            compressed = (
                *turboquant_compress_keys(
                    k, constants.rotation_matrix, constants.key_codebook, constants.qjl_projection
                ),
                *turboquant_compress_values(v, constants.rotation_matrix, constants.value_codebook),
            )
            names = (
                "key_indices_pages",
                "key_signs_pages",
                "key_norms_pages",
                "value_indices_pages",
                "value_norms_pages",
            )
            cache_view = cache_view.replace(
                **{
                    name: getattr(cache_view, name)
                    .at[destinations, offsets]
                    .set(data.astype(getattr(cache_view, name).dtype), mode="drop")
                    for name, data in zip(names, compressed, strict=True)
                }
            )
        elif update_cache and mla_pages:
            flat = flat.at[destinations, offsets, :latent_dim].set(v[:, 0].astype(flat.dtype), mode="drop")
            flat = flat.at[destinations, offsets, rope_offset : rope_offset + rope_dim].set(
                k[:, 0, latent_dim:].astype(flat.dtype), mode="drop"
            )
            cache_view = cache_view.replace(kv_pages=flat.reshape(cache_view.kv_pages.shape))
            key_pages = flat[:, :page_size, None, :]
            value_pages = key_pages[..., :latent_dim]
        elif update_cache and split_pages:
            key_pages = key_pages.at[destinations, offsets].set(k.astype(key_pages.dtype), mode="drop")
            value_pages = value_pages.at[destinations, offsets].set(v.astype(value_pages.dtype), mode="drop")
            cache_view = cache_view.replace(key_cache=key_pages, value_cache=value_pages)
        elif update_cache:
            combined = (
                jnp.concatenate((k, v), axis=-1)
                if paired_dim
                else jnp.stack((k, v), axis=2).reshape(k.shape[0], 2 * k.shape[1], k.shape[-1])
            )
            flat = flat.at[destinations, offsets, : combined.shape[1], : combined.shape[2]].set(
                combined.astype(flat.dtype), mode="drop"
            )
            cache_view = cache_view.replace(kv_pages=flat.reshape(cache_view.kv_pages.shape))
            key_pages = flat[..., :key_dim] if paired_dim else flat[:, :, 0::2, :key_dim]
            value_pages = flat[..., key_dim : key_dim + value_dim] if paired_dim else flat[:, :, 1::2, :value_dim]
        # Storage-head alignment can add padding heads; never expose these as GQA heads.
        if not compressed_pages:
            key_pages, value_pages = key_pages[:, :, :num_kv_heads], value_pages[:, :, :num_kv_heads]
        dense_name = "vanilla" if not deterministic and dropout_prob > 0 else dense_impl
        if not normalize_output:
            dense_name = "vanilla"
        dense = OperationRegistry.create(dense_name, self.metadata)
        kv_positions = jnp.arange(tables.shape[1] * page_size, dtype=jnp.int32)
        initial = jnp.zeros((q.shape[0], q.shape[1], value_dim), dtype=output_dtype)
        # Pairwise visibility and document membership are independent constraints.
        # MaskInfo.attention_mask returns the explicit mask preferentially; using
        # that property alone silently drops segment IDs when both are supplied.
        supplied_mask = None if mask_info is None else getattr(mask_info, "_attention_mask", None)
        q_segments = None if mask_info is None else getattr(mask_info, "_q_segment_ids", None)
        kv_segments = None if mask_info is None else getattr(mask_info, "_kv_segment_ids", None)
        if q_segments is not None or kv_segments is not None:
            if q_segments is not None and q_segments.ndim == 3 and q_segments.shape[-1] == 1:
                q_segments = q_segments[..., 0]
            if kv_segments is not None and kv_segments.ndim == 3 and kv_segments.shape[-1] == 1:
                kv_segments = kv_segments[..., 0]
            if q_segments is None:
                q_segments = kv_segments[..., : query.shape[-3]]
            if kv_segments is None:
                kv_segments = q_segments
            segment_mask = (
                (q_segments[:, :, None] >= 0)
                & (kv_segments[:, None, :] >= 0)
                & (q_segments[:, :, None] == kv_segments[:, None, :])
            )[:, None, :, :]
        else:
            segment_mask = None
        if alibi_slopes is not None and alibi_slopes.shape != (q.shape[1],):
            raise ValueError(f"alibi_slopes must have shape ({q.shape[1]},).")

        def request_pairwise(tensor, row, *, is_mask=False):
            """Map BTHD or request-local BHQK features to the packed query axis."""
            if tensor is None:
                return None
            if tensor.ndim == 2:
                tensor = tensor[:, None, None, :] if is_mask else tensor[None, None, :, :]
            elif tensor.ndim == 3:
                tensor = tensor[:, None, :, :] if is_mask else tensor[None, :, :, :]
            if tensor.ndim != 4:
                raise ValueError("Paged mask/bias must be broadcastable BHQK tensors.")
            if (
                query.ndim == 4
                and query.shape[0] > 1
                and tensor.shape[0] in (1, query.shape[0])
                and tensor.shape[-2] in (1, query.shape[1])
            ):
                tensor = jnp.broadcast_to(tensor, (query.shape[0], tensor.shape[1], query.shape[1], tensor.shape[-1]))
                tensor = tensor.transpose(1, 0, 2, 3).reshape(1, tensor.shape[1], q.shape[0], tensor.shape[-1])
            elif tensor.shape[0] != 1:
                if tensor.shape[0] != num_requests:
                    raise ValueError("Paged mask/bias batch must match query batches or request rows.")
                tensor = jax.lax.dynamic_slice_in_dim(tensor, row, 1, axis=0)
                if tensor.shape[-2] not in (1, q.shape[0]):
                    local_rows = jnp.clip(token_ids - request_starts[row], 0, tensor.shape[-2] - 1)
                    tensor = jnp.take(tensor, local_rows, axis=-2)
            tensor = tensor[..., : q.shape[0], : kv_positions.shape[0]]
            if tensor.shape[-1] not in (1, kv_positions.shape[0]) and tensor.shape[-1] < kv_positions.shape[0]:
                tensor = jnp.pad(tensor, ((0, 0), (0, 0), (0, 0), (0, kv_positions.shape[0] - tensor.shape[-1])))
            return tensor

        def attend_request(row, outputs):
            page_ids = tables[row]
            page_valid = (page_ids >= 0) & (page_ids < num_pages)
            ids = jnp.clip(page_ids, 0, num_pages - 1)
            if compressed_pages:
                # Reconstruct only this request's pages and retain the QJL
                # residual estimator: centroid*norm + signs@S*residual_norm*c.
                # Inverse rotation gives dense keys with the same q dot k scores.
                norms = cache_view.key_norms_pages[ids].astype(jnp.float32)
                keys = turboquant_dequantize_values(
                    cache_view.key_indices_pages[ids], norms[..., 0], constants.key_codebook, constants.rotation_matrix
                )
                signs = unpack_signs(cache_view.key_signs_pages[ids])
                factor = jnp.sqrt(jnp.float32(jnp.pi / 2)) / jnp.float32(constants.qjl_dim)
                correction = (signs @ constants.qjl_projection) @ constants.rotation_matrix
                keys = keys + correction * (norms[..., 1] * factor)[..., None]
                values = turboquant_dequantize_values(
                    cache_view.value_indices_pages[ids],
                    cache_view.value_norms_pages[ids],
                    constants.value_codebook,
                    constants.rotation_matrix,
                )
                keys = keys.reshape(1, -1, num_kv_heads, key_dim)
                values = values.reshape(1, -1, num_kv_heads, value_dim)
            elif mla_pages:
                # Remove independent component padding only after the request
                # gather; never concatenate/copy the entire MLA page pool.
                stored = key_pages[ids].reshape(1, -1, 1, flat.shape[-1])
                values = stored[..., :latent_dim]
                keys = jnp.concatenate((values, stored[..., rope_offset : rope_offset + rope_dim]), axis=-1)
            else:
                keys = key_pages[ids].reshape(1, -1, num_kv_heads, key_dim)
                values = value_pages[ids].reshape(1, -1, num_kv_heads, value_dim)
            query_valid = active_tokens & (rows == row)
            valid_kv = (kv_positions < contexts[row]) & jnp.repeat(page_valid, page_size)
            # Unallocated/padded cells may contain arbitrary values. Mask their
            # operands before matmuls so zero attention cannot multiply a NaN.
            keys = jnp.where(valid_kv[None, :, None, None], keys, 0.0)
            values = jnp.where(valid_kv[None, :, None, None], values, 0.0)
            request_queries = jnp.where(query_valid[:, None, None], q, 0.0)[None, ...]
            visible = query_valid[:, None] & valid_kv[None, :]
            if causal:
                visible &= kv_positions[None, :] <= positions[:, None]
            if sliding_window is not None:
                left, right = sliding_window if isinstance(sliding_window, tuple) else (max(sliding_window - 1, 0),) * 2
                visible &= (kv_positions[None, :] >= positions[:, None] - left) & (
                    kv_positions[None, :] <= positions[:, None] + right
                )
            if chunk_size is not None:
                visible &= kv_positions[None, :] // chunk_size == positions[:, None] // chunk_size
            visibility = visible[None, None, :, :]
            extra_mask = request_pairwise(supplied_mask, row, is_mask=True)
            if extra_mask is not None:
                visibility = visibility & extra_mask.astype(jnp.bool_)
            document_mask = request_pairwise(segment_mask, row, is_mask=True)
            if document_mask is not None:
                visibility = visibility & document_mask
            request_bias = request_pairwise(bias, row)
            if alibi_slopes is not None:
                # Match ragged-v3's slope_h * absolute_key_position contract,
                # including its effect on sink normalization.
                alibi_bias = alibi_slopes[None, :, None, None].astype(jnp.float32) * kv_positions[None, None, None, :]
                request_bias = alibi_bias if request_bias is None else request_bias + alibi_bias
            request_mask = MaskInfo(
                _attention_mask=visibility,
                q_positions=positions[None, :],
                kv_positions=kv_positions[None, :],
                causal_mask_baked_in=True,
                sliding_window_baked_in=True,
            )
            request_key = None if dropout_rng is None else jax.random.fold_in(dropout_rng, row)
            result = dense(
                query=request_queries,
                key=keys,
                value=values,
                mask_info=request_mask,
                bias=request_bias,
                causal=False,
                sliding_window=None,
                softmax_scale=softmax_scale,
                softmax_aux=softmax_aux,
                logits_soft_cap=logits_soft_cap,
                deterministic=deterministic,
                dropout_prob=dropout_prob,
                dropout_rng=request_key,
                normalize_output=normalize_output,
                precision=precision,
                compute_dtype=compute_dtype,
            )
            return jnp.where(query_valid[:, None, None], result.attention_outputs[0].astype(output_dtype), outputs)

        def step(row, outputs):
            return jax.lax.cond(
                scheduled[row] > 0, lambda result: attend_request(row, result), lambda result: result, outputs
            )

        outputs = jax.lax.fori_loop(0, num_requests, step, initial)
        return AttentionOutput(
            attention_outputs=outputs.reshape(*query.shape[:-1], value_dim),
            attention_weights=None,
            cache_view=cache_view,
        )

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)


class _RaggedPageAttn(OperationImpl):
    """Common base for ragged paged-attention operators (v2 / v3).

    Concentrates the shared machinery used by both
    :class:`RaggedPageAttnV2` and :class:`RaggedPageAttnV3`:

    * Sharding-spec resolution via
      :func:`_runtime_sharding_resolver`, including DP-page axis selection
      (``ATTN_DP`` vs ``ct.EMPTY``) and KV-head axis fan-out.
    * Two ``shard_map`` paths controlled by
      ``EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH``: a DP-local path that slices
      ``context_lens`` / ``pages_tables`` per DP shard and ``psum``-reduces
      the per-shard outputs, and a globally replicated path that runs the
      kernel against the full batch.
    * Variants for the standard ragged-pages cache view as well as the
      TurboQuant-compressed cache view, with the latter going through a
      separate ejkernel entry point that consumes the index/sign/norm
      pages.
    * BTHD <-> ragged ``[total_tokens, heads, dim]`` reshaping in
      :meth:`__call__` so callers can pass both 4-D batched and 3-D ragged
      tensors interchangeably.

    Subclasses only override :meth:`get_impl_name` and
    :meth:`get_requirements` and select the v2 or v3 ejkernel through
    :meth:`forward_v2` / :meth:`forward_v3` (dispatched by
    :meth:`forward_native`).

    Attributes:
        metadata (OperationMetadata): Operator configuration. Most paged
            specifics (page tables, query start locations, request
            distribution, ...) come in per-call via the ``cache_view`` and
            ``cache_metadata`` arguments rather than living on the
            instance.
        autotune (bool): Leave unspecified v3 block sizes to ejkernel's autotuner.
            Set to False to use the static TPU serving heuristic instead.
    """

    def __init__(self, metadata: OperationMetadata, *, autotune: bool = True) -> None:
        """Configure ragged paged attention.

        Args:
            metadata: Runtime metadata, including explicit ejkernel config overrides.
            autotune: Use ejkernel autotuning for unspecified v3 block sizes. False
                selects the static TPU heuristic; explicit block sizes always win.
        """
        super().__init__(metadata)
        self.autotune = autotune

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """Return the registered name for this attention implementation.

        Subclasses must override.

        Returns:
            str | tuple[str]: ``"ragged_page_attention_v2"`` or
            ``"ragged_page_attention_v3"``.

        Raises:
            NotImplementedError: Always, since the base class has no fixed
                kernel; concrete subclasses provide the name.
        """
        raise NotImplementedError()

    def forward_v2(
        self,
        query: Float[Array, "total_tokens num_q_heads head_dim"],
        cache_view: RaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        compute_dtype: DTypeLike | None = jnp.bfloat16,
        optimized: bool = False,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        mask_value: float | None = None,
        vmem_limit_bytes: int | None = None,
        chunk_size: int | None = None,
        alibi_slopes: Float[Array, "num_q_heads"] | None = None,  # noqa
        **ignore,
    ) -> AttentionOutput:
        """V2 ragged paged attention (read-only over cached KV pages).

        Dispatches to the TurboQuant-aware code path for compressed cache
        views, otherwise calls ``ragged_page_attention_v2`` from ejkernel.

        Args:
            query: Ragged query tensor of shape
                ``(total_tokens, num_q_heads, head_dim)``.
            cache_view: Ragged-pages cache view. May be a
                :class:`TurboQuantRaggedPagesCacheView`.
            cache_metadata: Per-batch metadata describing context lengths,
                page tables and query start locations.
            softmax_scale: Logits scaling factor; defaults to
                ``1 / sqrt(head_dim)``.
            logits_soft_cap: Optional soft-cap for logits.
            compute_dtype: Dtype used inside the kernel; defaults to
                ``bfloat16``.
            optimized: Use the optimized kernel variant when available.
            sliding_window: Optional window size for local attention.
            softmax_aux: Optional sink-token logits.
            mask_value: Value used for masked positions.
            vmem_limit_bytes: VMEM budget hint forwarded to the kernel.
            **ignore: Forward-compatibility kwargs (ignored).

        Returns:
            AttentionOutput: ``attention_outputs`` of shape
            ``(total_tokens, num_q_heads, head_dim)``. ``attention_weights``
            is ``None``.

        Raises:
            NotImplementedError: If ``chunk_size`` or ``alibi_slopes`` is requested;
                only ragged v3 implements them, and silently ignoring them would
                change the model's output.
        """
        if chunk_size is not None or alibi_slopes is not None:
            raise NotImplementedError("chunked local attention and ALiBi require ragged_page_attention_v3.")
        if isinstance(cache_view, TurboQuantRaggedPagesCacheView):
            return self._forward_v2_turboquant(
                query,
                cache_view,
                cache_metadata,
                softmax_scale=softmax_scale,
                logits_soft_cap=logits_soft_cap,
                sliding_window=sliding_window,
                softmax_aux=softmax_aux,
            )
        kv_pages: Float[Array, "num_pages page_size num_kv_heads head_dim"] = cache_view.kv_pages
        resolver = _runtime_sharding_resolver(self.metadata, cache_view)
        resolve = resolver.resolve
        num_seqs_flat: Array = cache_metadata.num_seqs.reshape(-1)
        page_axis = dp_page_axis(cache_view)
        kv_page_axis = _kv_axis(cache_view, ct.HEAD)
        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=query.shape)

        aux_spec = PartitionSpec(None)
        if softmax_aux is not None:
            # Any supported sink layout collapses (logsumexp over the sink axis) to one
            # logit per query head, which shards exactly like the query heads.
            softmax_aux = reduce_sinks_to_heads(softmax_aux, query.shape[1], kv_pages.shape[2] // 2)
            aux_spec = resolve(axes=[ct.HEAD], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)

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
                    axes=[page_axis, ct.EMPTY, kv_page_axis, ct.EMPTY],
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

    def _forward_v2_turboquant(
        self,
        query: Float[Array, "total_tokens num_q_heads head_dim"],
        cache_view: "TurboQuantRaggedPagesCacheView",
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_sinks"] | None = None,  # noqa
        vmem_limit_bytes: int | None = None,
        **ignore,
    ) -> AttentionOutput:
        """Forward pass for V2 (read-only) with TurboQuant-compressed KV cache."""
        from ejkernel.modules.operations.ragged_page_attention_v2_turboquant import (
            ragged_page_attention_v2_turboquant as rpa_v2_tq,
        )

        constants = cache_view.constants
        tq_config = cache_view.metadata.turboquant_config
        qjl_dim = constants.qjl_dim

        platform = "pallas" if jax.default_backend() == "tpu" else "auto"
        num_seqs_flat = cache_metadata.num_seqs.reshape(-1)

        resolver = _runtime_sharding_resolver(self.metadata, cache_view)
        resolve = resolver.resolve
        page_axis = dp_page_axis(cache_view)
        kv_token_axis = _kv_axis(cache_view, ct.KV_HEAD)

        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=query.shape)
        pages_4d_spec = resolve(
            axes=[page_axis, ct.EMPTY, kv_token_axis, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=cache_view.key_indices_pages.shape,
        )
        pages_3d_spec = resolve(
            axes=[page_axis, ct.EMPTY, kv_token_axis],
            mode=ct.MODE_PREFILL,
            shape=cache_view.value_norms_pages.shape,
        )
        sinks_axis = None
        if softmax_aux is not None:
            sinks_axis = resolve(axes=[ct.EMPTY], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)

        output = rpa_v2_tq(
            query,
            cache_view.key_indices_pages,
            cache_view.key_signs_pages,
            cache_view.key_norms_pages,
            cache_view.value_indices_pages,
            cache_view.value_norms_pages,
            cache_metadata.context_lens,
            cache_metadata.pages_tables,
            cache_metadata.query_start_loc,
            num_seqs_flat,
            constants.rotation_matrix,
            constants.qjl_projection,
            constants.key_codebook,
            constants.value_codebook,
            softmax_aux,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            sliding_window=sliding_window,
            bits=tq_config.bits,
            qjl_dim=qjl_dim,
            platform=platform,
            mesh=self.metadata.mesh,
            in_specs=(
                qaxes,
                pages_4d_spec,
                pages_4d_spec,
                pages_4d_spec,  # ki, ks, kn
                pages_4d_spec,  # vi
                pages_3d_spec,  # vn
                Ps(),
                Ps(),
                Ps(),
                Ps(),  # context_lens, block_tables, qsl, num_seqs
                Ps(),
                Ps(),
                Ps(),
                Ps(),  # rotation, projection, key_cb, val_cb
                sinks_axis,
            ),
            out_specs=qaxes,
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
        chunk_size: int | None = None,
        alibi_slopes: Float[Array, "num_q_heads"] | None = None,  # noqa
        **ignore,
    ) -> AttentionOutput:
        """V3 ragged paged attention with cache-update support.

        V3 takes new ``key``/``value`` projections, scatters them into the
        paged KV buffer using the request-distribution metadata and
        produces output for the just-written tokens. Dispatches to the
        TurboQuant variant for compressed cache views.

        Args:
            query: Ragged query tensor of shape
                ``(total_tokens, num_q_heads, head_dim)``.
            key: Ragged keys to be appended to the cache, of shape
                ``(total_tokens, num_kv_heads, head_dim)``.
            value: Ragged values to be appended to the cache, same shape
                convention as ``key``.
            cache_view: Ragged-pages cache view (possibly TurboQuant-
                compressed).
            cache_metadata: Per-batch metadata.
            softmax_scale: Logits scaling factor.
            logits_soft_cap: Optional soft-cap for logits.
            sliding_window: Optional window size for local attention.
            chunk_size: Optional chunked-local-attention size (Llama4).
            alibi_slopes: Optional per-query-head ALiBi slopes (Falcon).
            **ignore: Additional kwargs forwarded to the underlying kernel
                (e.g. ``softmax_aux``, ``vmem_limit_bytes``).

        Returns:
            AttentionOutput: Attention output and the (mutated) cache view.

        Raises:
            NotImplementedError: If ``chunk_size`` or ``alibi_slopes`` is requested
                with a TurboQuant-compressed cache, which does not implement them.
        """
        if isinstance(cache_view, TurboQuantRaggedPagesCacheView):
            if chunk_size is not None or alibi_slopes is not None:
                raise NotImplementedError(
                    "chunked local attention and ALiBi are not supported with a TurboQuant ragged cache."
                )
            return self._forward_v3_turboquant(
                query,
                key,
                value,
                cache_view,
                cache_metadata,
                softmax_scale=softmax_scale,
                logits_soft_cap=logits_soft_cap,
                sliding_window=sliding_window,
                **ignore,
            )
        return self._forward_v3_standard(
            query,
            key,
            value,
            cache_view,
            cache_metadata,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            alibi_slopes=alibi_slopes,
            **ignore,
        )

    def _forward_v3_turboquant(
        self,
        query: Float[Array, "total_tokens num_q_heads head_dim"],
        key: Float[Array, "total_tokens num_kv_heads head_dim"],
        value: Float[Array, "total_tokens num_kv_heads head_dim"],
        cache_view: TurboQuantRaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_sinks"] | None = None,  # noqa
        vmem_limit_bytes: int | None = None,
        **ignore,
    ) -> AttentionOutput:
        """Forward pass using TurboQuant-compressed KV cache via module operation."""
        from ejkernel.modules import ragged_page_attention_v3_turboquant

        constants = cache_view.constants
        tq_config = cache_view.metadata.turboquant_config
        qjl_dim = constants.qjl_dim
        request_distribution = cache_metadata.request_distribution

        resolver = _runtime_sharding_resolver(self.metadata, cache_view)
        resolve = resolver.resolve
        page_axis = dp_page_axis(cache_view)

        sinks_axis = None
        if softmax_aux is not None:
            sinks_axis = resolve(axes=[ct.HEAD], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)
            softmax_aux = softmax_aux.astype("f4")

        qaxes = resolve(axes=[ct.EMPTY, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=query.shape)
        kvaxes = resolve(axes=[ct.EMPTY, ct.KV_HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=key.shape)

        pages_4d_spec = resolve(
            axes=[page_axis, ct.EMPTY, ct.KV_HEAD, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=cache_view.key_indices_pages.shape,
        )
        pages_3d_spec = resolve(
            axes=[page_axis, ct.EMPTY, ct.KV_HEAD],
            mode=ct.MODE_PREFILL,
            shape=cache_view.value_norms_pages.shape,
        )

        # Use XLA backend — no Pallas dispatch overhead (3.4ms/call savings).
        platform = "xla"
        cfg = self.metadata.get_operation_config("ragged_page_attention_v3_turboquant")

        result = ragged_page_attention_v3_turboquant(
            query,
            key,
            value,
            cache_view.key_indices_pages,
            cache_view.key_signs_pages,
            cache_view.key_norms_pages,
            cache_view.value_indices_pages,
            cache_view.value_norms_pages,
            cache_metadata.context_lens,
            cache_metadata.pages_tables.reshape(-1),
            cache_metadata.query_start_loc,
            request_distribution,
            constants.rotation_matrix,
            constants.qjl_projection,
            constants.key_codebook,
            constants.value_codebook,
            softmax_aux,
            mesh=self.metadata.mesh,
            in_specs=(
                qaxes,
                kvaxes,
                kvaxes,
                pages_4d_spec,
                pages_4d_spec,
                pages_4d_spec,  # ki, ks, kn
                pages_4d_spec,  # vi
                pages_3d_spec,  # vn
                Ps(),
                Ps(),
                Ps(),
                Ps(),  # kv_lens, block_tables, qsl, distribution
                Ps(),
                Ps(),
                Ps(),
                Ps(),  # rotation, projection, key_cb, val_cb
                sinks_axis,
            ),
            out_specs=(
                qaxes,
                pages_4d_spec,
                pages_4d_spec,
                pages_4d_spec,  # ki, ks, kn
                pages_4d_spec,  # vi
                pages_3d_spec,  # vn
            ),
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            sliding_window=sliding_window,
            bits=tq_config.bits,
            qjl_dim=qjl_dim,
            cfg=cfg,
            platform=platform,
        )

        output = result[0]
        cache_view.key_indices_pages = result[1]
        cache_view.key_signs_pages = result[2]
        cache_view.key_norms_pages = result[3]
        cache_view.value_indices_pages = result[4]
        cache_view.value_norms_pages = result[5]

        return AttentionOutput(attention_weights=None, attention_outputs=output, cache_view=cache_view)

    def _forward_v3_standard(
        self,
        query: Float[Array, "total_tokens num_q_heads head_dim"],
        key: Float[Array, "total_tokens num_kv_heads head_dim"],
        value: Float[Array, "total_tokens num_kv_heads head_dim"],
        cache_view: RaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_sinks"] | Float[Array, "num_heads num_sinks"] | None = None,  # noqa
        vmem_limit_bytes: int | None = None,
        chunk_size: int | None = None,
        alibi_slopes: Float[Array, "num_q_heads"] | None = None,  # noqa
        **ignore,
    ) -> AttentionOutput:
        """V3 ragged paged attention over a standard (non-TurboQuant) cache.

        Selects either a DP-local ``shard_map`` path (when DP page sharding
        is enabled and the request count divides the DP size) or a
        replicated path that runs the kernel directly.

        Args:
            query: Ragged queries ``(total_tokens, num_q_heads, head_dim)``.
            key: Ragged keys ``(total_tokens, num_kv_heads, head_dim)``.
            value: Ragged values, same shape convention as ``key``.
            cache_view: Standard ragged-pages cache view holding ``kv_pages``.
            cache_metadata: Per-batch metadata including
                ``request_distribution``.
            softmax_scale: Logits scaling factor.
            logits_soft_cap: Optional soft-cap for logits.
            sliding_window: Optional window size for local attention.
            softmax_aux: Optional sink-token logits: ``(num_q_heads,)``, a shared
                ``(num_sinks,)``, or 2-D ``(num_q_heads | num_kv_heads | 1, num_sinks)``.
                Multiple sinks of one head are reduced with ``logsumexp`` to a single
                per-query-head sink before the kernel, so head sharding is exact.
            vmem_limit_bytes: VMEM budget hint for the kernel.
            chunk_size: Optional chunked-local-attention size ``C`` (Llama4): a query at
                ``q`` attends key ``k`` only if ``k <= q`` and ``k // C == q // C``.
            alibi_slopes: Optional per-query-head ALiBi slopes ``(num_q_heads,)``; the
                kernel adds ``slope_h * (k - q)`` to the scaled logits.
            **ignore: Forward-compatibility kwargs (ignored).

        Returns:
            AttentionOutput: Attention output plus the updated cache view
            with the mutated ``kv_pages``.
        """
        kv_pages = cache_view.kv_pages
        num_model_kv_heads = key.shape[1]
        key, value = _repeat_kv_heads_to_cache_width(key, value, cache_view)
        kv_cache_dtype = getattr(kv_pages, "dtype", None)
        if kv_cache_dtype is not None and (key.dtype != kv_cache_dtype or value.dtype != kv_cache_dtype):
            if jnp.dtype(kv_cache_dtype).itemsize < max(jnp.dtype(key.dtype).itemsize, jnp.dtype(value.dtype).itemsize):
                logger.warning(
                    "Casting key/value from %s to lower-precision KV cache dtype %s; "
                    "this may reduce numerical fidelity.",
                    key.dtype,
                    kv_cache_dtype,
                )
            key = key.astype(kv_cache_dtype)
            value = value.astype(kv_cache_dtype)
        resolver = _runtime_sharding_resolver(self.metadata, cache_view)
        resolve = resolver.resolve
        request_distribution = cache_metadata.request_distribution
        use_rank_major_dp = getattr(request_distribution, "ndim", 0) == 2
        page_axis = dp_page_axis(cache_view)
        kv_token_axis = _kv_axis(cache_view, ct.KV_HEAD)
        kv_page_axis = _kv_axis(cache_view, ct.HEAD)
        q_token_axis = ATTN_DP if use_rank_major_dp else ct.EMPTY

        sinks_axis = None
        alibi_axis = None

        if softmax_aux is not None:
            softmax_aux = reduce_sinks_to_heads(softmax_aux, query.shape[1], num_model_kv_heads)
            sinks_axis = resolve(axes=[ct.HEAD], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)
        if alibi_slopes is not None:
            alibi_slopes = alibi_slopes.astype("f4")
            alibi_axis = resolve(axes=[ct.HEAD], mode=ct.MODE_PREFILL, shape=alibi_slopes.shape)
        alibi_args = () if alibi_slopes is None else (alibi_slopes,)
        alibi_specs = () if alibi_slopes is None else (alibi_axis,)

        qaxes = resolve(axes=[q_token_axis, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=query.shape)
        kvaxes = resolve(axes=[q_token_axis, kv_token_axis, ct.EMPTY], mode=ct.MODE_PREFILL, shape=key.shape)

        kv_pages_spec = resolve(
            axes=[page_axis, ct.EMPTY, kv_page_axis, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )
        page_axis_names = normalize_axis_names(kv_pages_spec[0] if len(kv_pages_spec) > 0 else None)
        page_axis_size = mesh_axis_size(self.metadata.mesh, page_axis_names)
        kv_pages_spec_replicated = resolve(
            axes=[ct.EMPTY, ct.EMPTY, kv_page_axis, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )

        platform = "pallas" if jax.default_backend() == "tpu" else "auto"
        cfg = self.metadata.get_operation_config("ragged_page_attention_v3")
        if not self.autotune:
            cfg = _default_tpu_rpa_v3_cfg(
                cfg,
                query=query,
                key=key,
                kv_pages=kv_pages,
                context_lens=cache_metadata.context_lens,
                pages_tables=cache_metadata.pages_tables,
            )
        common_call_kwargs = dict(
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            vmem_limit_bytes=vmem_limit_bytes,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            cfg=cfg,
            platform=platform,
        )
        # For DP-sharded page buffers, consume per-DP-local block tables and sequence metadata
        # inside the shard_map body. This avoids materializing global page all-gathers.
        can_use_dp_local = (
            ENABLE_DP_LOCAL_PAGE_PATH
            and page_axis == ATTN_DP
            and page_axis_size > 1
            and len(page_axis_names) > 0
            and int(cache_metadata.context_lens.shape[0]) % page_axis_size == 0
        )
        if use_rank_major_dp and not can_use_dp_local:
            raise ValueError("Rank-major DP ragged attention requires the DP-local page path.")
        if can_use_dp_local:
            rows_per_shard = int(cache_metadata.context_lens.shape[0]) // page_axis_size
            max_pages_per_req = int(cache_metadata.pages_tables.shape[1])

            @partial(
                jax.shard_map,
                mesh=self.metadata.mesh,
                in_specs=(qaxes, kvaxes, kvaxes, kv_pages_spec, Ps(), Ps(), Ps(), Ps(), sinks_axis, *alibi_specs),
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
                *local_alibi,
            ):
                shard_idx = axis_index(page_axis_names)
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
                if use_rank_major_dp:
                    local_query_start_loc = full_query_start_loc[shard_idx]
                    local_distribution = full_distribution[shard_idx]
                else:
                    local_query_start_loc = jax.lax.dynamic_slice_in_dim(
                        full_query_start_loc,
                        start_index=row_start,
                        slice_size=rows_per_shard + 1,
                        axis=0,
                    )
                    local_scheduled = local_query_start_loc[1:] - local_query_start_loc[:-1]
                    local_distribution = _request_distribution_bounds(local_scheduled, local_context_lens)
                local_kernel_distribution = _tpu_mixed_request_distribution(local_distribution, cfg)
                local_total = local_distribution[2]

                local_num_pages = jnp.int32(local_kv_pages.shape[0])
                page_offset = shard_idx * local_num_pages
                local_block_tables = local_block_tables - page_offset
                local_block_tables = jnp.clip(local_block_tables, 0, local_num_pages - 1)

                local_output, local_kv_pages = ragged_page_attention_v3(
                    local_query,
                    local_key,
                    local_value,
                    local_kv_pages,
                    local_context_lens,
                    local_block_tables,
                    local_query_start_loc,
                    local_kernel_distribution,
                    local_softmax_aux,
                    alibi_slopes=local_alibi[0] if local_alibi else None,
                    **common_call_kwargs,
                )

                if use_rank_major_dp:
                    return local_output, local_kv_pages

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
                *alibi_args,
            )
        else:
            kernel_distribution = _tpu_mixed_request_distribution(request_distribution, cfg)
            output, kv_pages = ragged_page_attention_v3(
                query,
                key,
                value,
                kv_pages,
                cache_metadata.context_lens,
                cache_metadata.pages_tables.reshape(-1),
                cache_metadata.query_start_loc,
                kernel_distribution,
                softmax_aux,
                alibi_slopes=alibi_slopes,
                in_specs=(
                    qaxes,
                    kvaxes,
                    kvaxes,
                    kv_pages_spec_replicated,
                    Ps(),
                    Ps(),
                    Ps(),
                    Ps(),
                    sinks_axis,
                    *alibi_specs,
                ),
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
        if not ignore.get("normalize_output", True) or (
            ignore.get("dropout_prob", 0.0) > 0 and not ignore.get("deterministic", True)
        ):
            return OperationRegistry.create("paged_attention_reference", self.metadata)(
                query=query,
                key=key,
                value=value,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
                softmax_scale=softmax_scale,
                logits_soft_cap=logits_soft_cap,
                sliding_window=sliding_window,
                softmax_aux=softmax_aux,
                compute_dtype=compute_dtype,
                **ignore,
            )
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
        """GPU dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm dispatch path; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
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

        batch: int = 0
        sequence: int = 0
        head: int = 0
        dim: int = 0
        if is_4d:
            batch, sequence, head, dim = query.shape

        # The reference adapter needs the original BTHD layout to map batched
        # masks/bias onto packed query rows. Flattening only Q/K/V here loses
        # that batch-to-row relationship before the dropout redirect runs.
        use_reference = not ignore.get("normalize_output", True) or (
            ignore.get("dropout_prob", 0.0) > 0 and not ignore.get("deterministic", True)
        )
        query_reshaped = query if use_reference else query.reshape(-1, *query.shape[-2:])
        key_reshaped = key if use_reference else key.reshape(-1, *key.shape[-2:])
        value_reshaped = value if use_reference else value.reshape(-1, *value.shape[-2:])

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
    """Ragged paged attention using serving-style slot mappings (read-only over cache).

    Variant of the ragged paged-attention operation that consumes
    pre-populated paged KV pools (the cache update step is performed
    elsewhere via the cache view). The kernel is selected by
    ``ragged_page_attention_v2`` from ejkernel and accepts ``pages_tables``
    plus ``slot_mapping`` from :class:`RaggedPagesMetadata`.

    Used by the legacy v2 serving path (still active for some Gemma /
    LLaMA-style models). Newer deployments prefer :class:`RaggedPageAttnV3`,
    which uses a coarser ``request_distribution`` instead of per-token slot
    maps. Both share the same :class:`RaggedPagesCacheView` cache layout.

    Registered under ``"ragged_page_attention_v2"``.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """Return the registered name for this attention implementation.

        Returns:
            str: ``"ragged_page_attention_v2"``.
        """
        return "ragged_page_attention_v2"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Return requirements for RaggedPageAttnV2 (slot-mapping based).

        Args:
            mode: Execution mode (ignored; requirements are the same for
                all modes).

        Returns:
            OperationRequirements: V2 requires sequence/context/position
            metadata plus per-token slot mapping into the paged KV pool;
            uses :class:`RaggedPagesCacheView`.
        """
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
    """Ragged paged attention with cache-update support and request-distribution dispatch.

    The current default ragged paged-attention operator. Unlike the v2
    variant, v3 takes the new ``key`` / ``value`` projections as direct
    inputs and is responsible for both scattering them into the paged KV
    pool and emitting outputs for the just-written tokens. Branching
    between decode-only, prefill-only and mixed batches is driven by the
    ``request_distribution`` triple ``(decode_end, prefill_end, total)``
    instead of per-token slot mapping.

    Supports both the standard :class:`RaggedPagesCacheView` and the
    compressed :class:`TurboQuantRaggedPagesCacheView` (handled by
    ``_forward_v3_turboquant``). On TPU the TPU-specific Pallas kernel is
    selected and ``num_kv_pages_per_block`` is clamped to keep VMEM under
    the 16MB scoped limit for wide-head models.

    Registered under ``"ragged_page_attention_v3"``.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """Return the registered name for this attention implementation.

        Returns:
            str: ``"ragged_page_attention_v3"``.
        """
        return "ragged_page_attention_v3"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Return requirements for RaggedPageAttnV3 (request-distribution based).

        Args:
            mode: Execution mode (ignored; requirements are the same for
                all modes).

        Returns:
            OperationRequirements: V3 requires sequence/context/position
            metadata plus the request-distribution triple driving the
            decode/prefill/mixed dispatch; uses :class:`RaggedPagesCacheView`.
        """
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
