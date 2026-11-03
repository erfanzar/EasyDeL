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

"""Multi-Latent Attention (MLA) operations over ragged paged KV caches.

This module provides :class:`MultiLatentRaggedPageAttn` (v1) and
:class:`MultiLatentRaggedPageAttnV2`, both registered :class:`OperationImpl`
classes that wrap ejkernel's MLA paged-attention kernels for the
DeepSeek-style decoder. MLA factorises the standard QK projection into:

* ``queries_nope`` / ``queries_pe`` - non-positional and positional-embedding
  query halves (the latter carries the rotary frequencies).
* ``keys_values`` - the low-rank latent representation that replaces the
  full-rank K/V tensors. May be rank-2 (shared-head) or rank-3 (head-aware).
* ``keys_pe`` - positional-embedding key projections, always rank-2.

The op handles ragged (variable-length) batches via per-request page tables
and ``query_start_loc``, performs request distribution bookkeeping, and
dispatches to one of two ``shard_map`` paths: the DP-local path that slices
context lengths and page tables per shard (active when
``EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH=True`` and the request count divides the
DP axis size), or the globally replicated path that runs the kernel against
the full batch on every shard.

Both versions share the same MLA cache view (:class:`MLARaggedPagesCacheView`)
and request-distribution layout; v2 differs only in the underlying ejkernel
entry point and config dataclass.
"""

from functools import partial

import jax
from eformer.loggings import get_logger
from ejkernel.modules import (
    multi_latent_ragged_page_attention,  # pyright: ignore[reportMissingTypeStubs]
    multi_latent_ragged_page_attention_v2,  # pyright: ignore[reportMissingTypeStubs]
    sparse_multi_latent_ragged_page_attention,  # pyright: ignore[reportMissingTypeStubs]
)
from ejkernel.modules.operations.configs import (
    MultiLatentRaggedPageAttentionConfig,
    MultiLatentRaggedPageAttentionV2Config,
)
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Float
from spectrax import common_types as ct

from easydel.axis import ATTN_DP
from easydel.caching import MLARaggedPagesCacheView, RaggedPagesMetadata
from easydel.infra.sharding import axis_index, mesh_axis_size, normalize_axis_names

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)
from ._paged_common import ENABLE_DP_LOCAL_PAGE_PATH, dp_page_axis, request_distribution_bounds

# Backward-compatible alias for callers importing the former adapter helper.
_dp_page_axis = dp_page_axis

logger = get_logger(__name__)

# dsa_serving_ep_split: shortest per-shard token window run as compact per-window kernel calls; shorter windows use
# one call over all rows with the other shards' rows skipped (every compact call pays the op's fixed metadata and
# cache-write cost). Measured on v5p, one DSA layer at 16 query heads per chip, kv_lora_rank 512 (no rope part), page
# size 128, 513 selected 4-token chunks per query, ep 4; ms per layer, single call vs compact calls: 8-token windows
# 0.42 vs 0.52, 32-token windows 0.60 vs 0.64, 64-token windows 0.87 vs 0.80, 512-token windows 10.8 vs 7.0,
# 8,192-token windows 224.6 vs 161.3.
_EP_SPLIT_COMPACT_MIN_WINDOW = 64


def _mla_kernel_call(dense_fn, ignore: dict, total_tokens: int, common_call_kwargs: dict):
    """Return ``(selected, kernel)`` for the shard_map bodies.

    Without ``sparse_selected`` in ``ignore`` the kernel is ``dense_fn`` with the dense
    call kwargs. With it (DSA top-k serving), the kernel is the sparse MLA op attending
    only to the selected chunks (``sparse_chunk_size`` tokens each) of every query.

    Args:
        dense_fn: Dense MLA ragged kernel (v1 or v2 functional API).
        ignore: Extra forward kwargs; may carry ``sparse_selected`` ``[..., W]`` int chunk
            ids (``-1`` padded, logical per sequence) and ``sparse_chunk_size``.
        total_tokens: Packed query token count.
        common_call_kwargs: Dense call kwargs (scale, soft cap, window, scales, cfg).

    Returns:
        ``(selected [total_tokens, W] int32 or None, kernel(*args, selected=...))``.
    """
    selected = ignore.get("sparse_selected")
    if selected is None:

        def dense_kernel(*args, selected=None):
            del selected
            return dense_fn(*args, **common_call_kwargs)

        return None, dense_kernel

    from ejkernel.modules.operations.configs import SparseMultiLatentRaggedPageAttentionConfig

    selected = jnp.asarray(selected, jnp.int32)
    selected = selected.reshape(total_tokens, selected.shape[-1])
    chunk_size = int(ignore.get("sparse_chunk_size") or 1)
    sparse_kwargs = {k: v for k, v in common_call_kwargs.items() if k != "cfg"}
    dense_cfg = common_call_kwargs.get("cfg")
    sparse_kwargs["cfg"] = SparseMultiLatentRaggedPageAttentionConfig(
        # TPU: 16 consecutive query tokens per program of ejkernel's Pallas query-block prefill kernel. Prefill
        # buckets (>= 256 tokens) whose 16-token blocks each hold one sequence fetch the block's shared selected
        # chunks once; other steps (decode, batches with a block spanning two sequences) keep the per-token kernel.
        # Measured on v5p with 16 heads per chip (16 x 16 = 256 MXU rows per program), latent 512 + 128 rope pad,
        # page 128, 4-token chunks, 513 selected chunks per token. The kernel switches itself off for head counts
        # that are not power-of-two multiples of 16. Other backends keep the XLA default (8 rows per step).
        num_queries_per_block=16 if jax.default_backend() == "tpu" else None,
        vmem_limit_bytes=getattr(dense_cfg, "vmem_limit_bytes", None),
        platform="auto",
    )

    def sparse_kernel(*args, selected):
        return sparse_multi_latent_ragged_page_attention(*args, selected, chunk_size=chunk_size, **sparse_kwargs)

    return selected, sparse_kernel


def _ep_split_sparse_attention(
    kernel,
    *,
    mesh,
    axis_names: tuple[str, ...],
    specs: tuple[Ps, Ps, Ps, Ps, Ps],
    queries_nope: jax.Array,
    queries_pe: jax.Array,
    keys_values: jax.Array,
    keys_pe: jax.Array,
    kv_pages: jax.Array,
    context_lens: jax.Array,
    pages_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    selected: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sparse MLA attention with its query tokens split over the ``axis_names`` mesh axes.

    ``T`` is padded to a multiple of the shard count ``n`` with tokens past ``query_start_loc[num_seqs]``, which the
    kernel neither writes nor attends, and cut into ``n`` contiguous windows of ``T / n`` tokens; shard ``e`` attends
    for window ``e``. Every shard runs the unchanged sparse op once per window on that window alone: the ragged
    metadata clipped to the window (each sequence's query tokens inside it, its context length ending at its last
    token inside it), the same page table. Each token thus gets exactly its sequence, position and selection
    metadata of the unsplit call. The other shards' windows come first and only write their latent tokens into this
    shard's cache replica (their attention output is unused, so XLA drops those kernel calls); the own window comes
    last, after every token of the step is in the cache, and attends over a compact grid of ``T / n`` programs.
    A grid that only skipped the other windows' rows would leave one of the two TensorCores of a v5p megacore chip,
    which split the grid into contiguous halves, with all of the shard's work. Every replica receives each token
    once (a packed cache word shared by two windows keeps the lanes the other window wrote, whichever call runs
    first), so outputs and caches equal the unsplit call's bit for bit.

    Windows shorter than ``_EP_SPLIT_COMPACT_MIN_WINDOW`` tokens (decode steps) instead make one call over all
    ``T`` rows in which the other windows' rows have zero queries and an all ``-1`` selection (the kernel skips
    them), and return the own window's rows: the per-window calls each pay the op's fixed metadata and cache-write
    cost, which outweighs the smaller kernel grid there.

    Args:
        kernel: Sparse kernel from :func:`_mla_kernel_call`.
        mesh: Device mesh.
        axis_names: Mesh axes the query tokens are split over (the expert-parallel axis).
        specs: Unsplit partition specs of ``(queries_nope, queries_pe, keys_values, keys_pe, kv_pages)``; the
            queries' token entry is replaced by ``axis_names``, the others are used as given (token-replicated).
        queries_nope: ``[T, H, L]`` absorbed queries.
        queries_pe: ``[T, H, R]`` rope queries.
        keys_values: ``[T, L]`` new latent tokens.
        keys_pe: ``[T, R]`` new rope keys.
        kv_pages: Paged MLA cache (replicated over ``axis_names``).
        context_lens: ``[S]`` context lengths.
        pages_tables: ``[S, pages_per_seq]`` page table.
        query_start_loc: ``[S + 1]`` cumulative query counts.
        distribution: ``[3]`` request distribution.
        selected: ``[T, W]`` selected chunk ids.

    Returns:
        ``(outputs [T, H, L] sharded over axis_names on the token axis, updated kv_pages)``.
    """
    # Hold the queries in the unsplit call's layout up to the split; each shard then slices its rows locally.
    # Unpinned, XLA propagates their token split back into the projections that produce them, which then partition,
    # and round, differently from the unsplit program (v5p, one GLM-5.3 DSA layer call at ep 2 x tp 2: about 2/3 of
    # the absorbed queries off by bf16 ulps and some top-k near-ties flipped; pinned, outputs and caches bitwise).
    jax_mesh = getattr(mesh, "jax_mesh", mesh)
    queries_nope = jax.lax.with_sharding_constraint(queries_nope, NamedSharding(jax_mesh, specs[0]))
    queries_pe = jax.lax.with_sharding_constraint(queries_pe, NamedSharding(jax_mesh, specs[1]))
    num_shards = mesh_axis_size(mesh, axis_names)
    total = int(queries_nope.shape[0])
    window = -(-total // num_shards)
    padded = window * num_shards
    if padded != total:

        def _pad(x, value=0):
            return jnp.pad(x, [(0, padded - total)] + [(0, 0)] * (x.ndim - 1), constant_values=value)

        queries_nope, queries_pe, keys_values, keys_pe = (
            _pad(x) for x in (queries_nope, queries_pe, keys_values, keys_pe)
        )
        selected = _pad(selected, -1)

    token_entry = axis_names[0] if len(axis_names) == 1 else tuple(axis_names)

    def _split(spec, ndim: int) -> Ps:
        entries = tuple(spec) + (None,) * (ndim - len(tuple(spec)))
        return Ps(token_entry, *entries[1:])

    q_nope_spec, q_pe_spec, keys_values_spec, keys_pe_spec, pages_spec = specs
    q_nope_split = _split(q_nope_spec, queries_nope.ndim)
    q_pe_split = _split(q_pe_spec, queries_pe.ndim)

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            q_nope_split,
            q_pe_split,
            keys_values_spec,
            keys_pe_spec,
            pages_spec,
            Ps(),
            Ps(),
            Ps(),
            Ps(),
            Ps(token_entry, None),
        ),
        out_specs=(q_nope_split, pages_spec),
        check_vma=False,
    )
    def _mapped_split(
        local_queries_nope,
        local_queries_pe,
        all_keys_values,
        all_keys_pe,
        local_kv_pages,
        full_context_lens,
        full_pages_tables,
        full_query_start_loc,
        full_distribution,
        local_selected,
    ):
        own = axis_index(axis_names)
        block_tables = full_pages_tables.reshape(-1)
        if window < _EP_SPLIT_COMPACT_MIN_WINDOW:
            # Small (decode) windows: one call over all rows, the other shards' rows skipped (no attendable chunk).
            start = own * jnp.int32(window)

            def _widen(x, fill):
                full = jnp.full((padded, *x.shape[1:]), fill, x.dtype)
                return jax.lax.dynamic_update_slice_in_dim(full, x, start, axis=0)

            output, local_kv_pages = kernel(
                _widen(local_queries_nope, 0),
                _widen(local_queries_pe, 0),
                all_keys_values,
                all_keys_pe,
                local_kv_pages,
                full_context_lens,
                block_tables,
                full_query_start_loc,
                full_distribution,
                selected=_widen(local_selected, -1),
            )
            return jax.lax.dynamic_slice_in_dim(output, start, window, axis=0), local_kv_pages

        def window_call(index, window_queries_nope, window_queries_pe, window_selected, pages):
            start = index * jnp.int32(window)
            end = start + jnp.int32(window)
            # Window-local ragged metadata: a token's sequence and position are unchanged (see the docstring).
            window_query_start_loc = jnp.clip(full_query_start_loc, start, end) - start
            # Sequences without tokens in the window are never read; clamping keeps their length non-negative.
            window_context_lens = jnp.maximum(full_context_lens - jnp.maximum(full_query_start_loc[1:] - end, 0), 0)
            return kernel(
                window_queries_nope,
                window_queries_pe,
                jax.lax.dynamic_slice_in_dim(all_keys_values, start, window, axis=0),
                jax.lax.dynamic_slice_in_dim(all_keys_pe, start, window, axis=0),
                pages,
                window_context_lens,
                block_tables,
                window_query_start_loc,
                full_distribution,
                selected=window_selected,
            )

        no_queries_nope = jnp.zeros_like(local_queries_nope)
        no_queries_pe = jnp.zeros_like(local_queries_pe)
        no_selection = jnp.full_like(local_selected, -1)
        for step in range(1, num_shards):
            # Cache write only: the attention output of another shard's window is discarded.
            _, local_kv_pages = window_call(
                (own + step) % num_shards, no_queries_nope, no_queries_pe, no_selection, local_kv_pages
            )
        return window_call(own, local_queries_nope, local_queries_pe, local_selected, local_kv_pages)

    output, kv_pages = _mapped_split(
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_pages,
        context_lens,
        pages_tables,
        query_start_loc,
        distribution,
        selected,
    )
    if padded != total:
        output = output[:total]
    return output, kv_pages


def _reshape_query_tensor(x, name: str):
    """Reshape a query tensor to rank-3 ``[tokens, heads, dim]``.

    Rank-4 inputs (e.g. ``[batch, seq, heads, dim]``) are flattened along the
    first two dimensions. Rank-3 inputs are returned unchanged.

    Args:
        x: Query array of rank 3 or 4, or None.
        name: Tensor name used in error messages.

    Returns:
        Rank-3 array, or None if *x* is None.

    Raises:
        ValueError: If *x* is not rank-3 or rank-4.
    """
    if x is None:
        return None
    if x.ndim == 4:
        # Avoid ``-1`` here: a zero-width trailing dim (NoPE MLA, rope_dim=0)
        # makes ``math.prod`` return 0 and JAX's -1 inference divide by zero.
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
    if x.ndim == 3:
        return x
    raise ValueError(f"`{name}` must be rank-3 or rank-4, got rank-{x.ndim}.")


def _reshape_feature_tensor(x, name: str):
    """Reshape a key/value feature tensor to rank-2 or rank-3 for the kernel.

    Handles MLA latent representations which may or may not carry a head
    dimension:

    - Rank-4 ``[batch, seq, heads, dim]`` -> rank-3 ``[tokens, heads, dim]``
      (head-aware path).
    - Rank-3 ``[batch, seq, dim]`` -> rank-2 ``[tokens, dim]``
      (shared-head path).
    - Rank-2 inputs are returned unchanged.

    Args:
        x: Feature array of rank 2, 3, or 4, or None.
        name: Tensor name used in error messages.

    Returns:
        Rank-2 or rank-3 array, or None if *x* is None.

    Raises:
        ValueError: If *x* is not rank-2, rank-3, or rank-4.
    """
    if x is None:
        return None
    if x.ndim == 4:
        # Keep per-head features for kernels that support head-aware KV tensors.
        # No ``-1``: zero-width dims (NoPE MLA) break JAX's -1 inference.
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
    if x.ndim == 3:
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    if x.ndim == 2:
        return x
    raise ValueError(f"`{name}` must be rank-2, rank-3, or rank-4, got rank-{x.ndim}.")


def _resolve_num_queries_per_block(
    *,
    num_q_heads: int,
    requested: int | tuple[int, int, int] | None,
    cfg_requested: int | tuple[int, int, int] | None,
    q_dtype: jnp.dtype | None = None,
    head_axis_size: int = 1,
    query_token_blocks: bool = False,
) -> int | tuple[int, int, int] | None:
    """Pick TPU-safe query blocks for either MLA kernel version.

    Both kernels trace single-query decode tiles, whose packed local head
    count must be divisible by the query block size. Preserve the original
    head-divisor fallback even for v2, including when no size is requested.

    Args:
        num_q_heads: Total number of query heads (before TP sharding).
        requested: Explicitly requested block size (takes priority).
        cfg_requested: Block size from the operation config (used when
            *requested* is None).
        q_dtype: Query dtype; float16/bfloat16 enables 2x packing.
        head_axis_size: Tensor-parallel axis size (number of TP shards).
        query_token_blocks: Accepted for compatibility; v2 also needs safe
            single-query decode blocks.

    Returns:
        A head-compatible block size, or three such sizes for a tuple/list.

    Raises:
        ValueError: If a tuple/list has the wrong length.
    """
    del query_token_blocks
    preferred = requested if requested is not None else cfg_requested
    q_packing = 2 if q_dtype is not None and jnp.finfo(q_dtype).bits == 16 else 1
    local_heads = max(1, int(num_q_heads) // max(1, head_axis_size))
    # align_to mimics the kernel's padding to q_packing multiples.
    aligned_local = local_heads + (-local_heads % q_packing)
    packed_heads = aligned_local // q_packing
    if isinstance(preferred, (tuple, list)):
        if len(preferred) != 3:
            raise ValueError("num_queries_per_block must have exactly three entries.")
        return tuple(
            _resolve_num_queries_per_block(
                num_q_heads=num_q_heads,
                requested=item,
                cfg_requested=None,
                q_dtype=q_dtype,
                head_axis_size=head_axis_size,
            )
            for item in preferred
        )
    preferred = None if preferred is None else max(1, int(preferred))

    if preferred is not None and packed_heads % preferred == 0:
        return preferred

    for candidate in (32, 16, 10, 8, 5, 4, 3, 2, 1):
        if preferred is not None and candidate > preferred:
            continue
        if packed_heads % candidate == 0:
            return candidate

    return 1


# Backward-compatible alias for callers importing the former adapter helper.
_request_distribution_bounds = request_distribution_bounds


def _resolve_distribution(cache_metadata: RaggedPagesMetadata) -> jax.Array:
    """Compute a ``[decode_end, prefill_end, total]`` request distribution vector.

    If ``cache_metadata.request_distribution`` is already set, it is returned
    directly (cast to int32). Otherwise the distribution is inferred from
    ``query_start_loc`` and ``context_lens``.

    Args:
        cache_metadata: Ragged-pages cache metadata describing the current
            batch. Either provides ``request_distribution`` directly or the
            ``query_start_loc`` / ``context_lens`` arrays needed to compute
            it.

    Returns:
        jax.Array: ``int32[3]`` array ``[decode_end, prefill_end, total]``.
    """
    if cache_metadata.request_distribution is not None:
        return jnp.asarray(cache_metadata.request_distribution, dtype=jnp.int32)

    context_lens = jnp.asarray(cache_metadata.context_lens, dtype=jnp.int32)
    query_start_loc = jnp.asarray(cache_metadata.query_start_loc, dtype=jnp.int32)
    scheduled = query_start_loc[1:] - query_start_loc[:-1]
    return _request_distribution_bounds(scheduled, context_lens)


@OperationRegistry.register
class MultiLatentRaggedPageAttn(OperationImpl):
    """Multi-Latent Attention (MLA) operation using ragged paged KV caches.

    This operation implements the MLA decode/prefill attention pattern used by
    DeepSeek-style architectures.  It wraps the ``ejkernel`` paged-attention
    kernel and handles:

    * Splitting queries into non-positional (``queries_nope``) and positional-
      embedding (``queries_pe``) components.
    * Separate ``keys_values`` (the low-rank latent KV representation) and
      ``keys_pe`` (positional-embedding keys) tensors.
    * Ragged (variable-length) batches with page-table-based KV caches.
    * Data-parallel local-page sharding when ``EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH``
      is active, falling back to a globally-replicated page path otherwise.
    * Automatic sharding via ``jax.shard_map`` for TPU/GPU meshes.

    Registered under the name ``"multi_latent_ragged_page_attention_v1"``.
    """

    _kernel_function = staticmethod(multi_latent_ragged_page_attention)
    _config_type = MultiLatentRaggedPageAttentionConfig
    _supports_rank_major_dp = False
    _query_token_blocks = False

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """Return the registered operation name for this implementation."""
        return "multi_latent_ragged_page_attention_v1"

    def _sparse_ep_split_axes(
        self,
        ignore: dict,
        sparse_selected: jax.Array | None,
        *,
        dp_local: bool,
        page_axis,
        query_specs: tuple[Ps, ...],
    ) -> tuple[str, ...]:
        """Mesh axes to split the sparse kernel's query tokens over, or ``()`` to keep the unsplit call.

        The DSA layer passes ``sparse_ep_split=True`` (``config.dsa_serving_ep_split``) with its top-k selection. The
        expert-parallel axis only replicates attention and the MLA pages are replicated over it, so each of its shards
        can attend for a slice of the query tokens (see :func:`_ep_split_sparse_attention`). The split needs an
        expert-parallel axis of more than one device, the globally replicated page path (not the data-parallel local
        one, pages not sharded over data parallelism) and a query sharding that does not already use that axis;
        otherwise the call stays unsplit and a one-time warning names the reason.

        Args:
            ignore: Extra forward kwargs; ``sparse_ep_split`` requests the split.
            sparse_selected: The DSA selection, ``None`` for dense MLA calls (never split).
            dp_local: Whether the data-parallel local page path runs this call.
            page_axis: Logical KV-page axis of the cache view (``EMPTY`` when replicated).
            query_specs: Resolved partition specs of the two query tensors.

        Returns:
            The expert-parallel mesh axis names to split over, or ``()``.
        """
        if not ignore.get("sparse_ep_split") or sparse_selected is None:
            return ()
        mesh = self.metadata.mesh
        names = normalize_axis_names(getattr(self.metadata.partition_axis, "expert_parallel_axis", None))
        names = tuple(name for name in names if mesh_axis_size(mesh, name) > 1)
        if not names:
            return ()  # nothing to split over (ep = 1): the unsplit call is the split call
        reasons = []
        if dp_local:
            reasons.append("the data-parallel local page path runs this call")
        elif page_axis != ct.EMPTY:
            reasons.append("the KV pages are sharded over the data-parallel axis")
        used = {axis for spec in query_specs for entry in spec for axis in normalize_axis_names(entry)}
        overlap = sorted(used.intersection(names))
        if overlap:
            reasons.append(f"the query sharding already uses {overlap}")
        if reasons:
            logger.warning_once(
                "dsa_serving_ep_split is inactive for this sparse MLA call because %s; every expert-parallel shard "
                "attends for all query tokens.",
                "; ".join(reasons),
            )
            return ()
        return names

    def forward_core(
        self,
        queries_nope: Float[Array, "total_tokens num_q_heads qk_nope_dim"],
        queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
        keys_values: Float[Array, "total_tokens kv_lora_rank"],
        keys_pe: Float[Array, "total_tokens qk_pe_dim"],
        cache_view: MLARaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        vmem_limit_bytes: int | None = None,
        mask_value: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        **ignore,
    ) -> AttentionOutput:
        """Run MLA ragged-paged attention with the implementation's kernel and config type.

        This is the primary forward path.  It resolves sharding specs,
        builds the request-distribution vector, and dispatches to either
        the DP-local or globally-replicated ``shard_map`` path depending
        on whether data-parallel page sharding is active.

        Args:
            queries_nope: Non-positional query projections.
            queries_pe: Positional-embedding query projections.
            keys_values: Low-rank latent KV representation (shared across heads)
                or per-head KV when rank-3.
            keys_pe: Positional-embedding key projections (always rank-2).
            cache_view: MLA ragged-pages cache view holding the paged KV buffer.
            cache_metadata: Batch scheduling metadata (context lengths, page
                tables, query start locations, etc.).
            softmax_scale: Multiplicative scale applied before softmax.
            logits_soft_cap: Optional soft-capping of logits before softmax.
            sliding_window: Sliding-window size for local attention.
            vmem_limit_bytes: VMEM budget hint for the kernel.
            mask_value: Value used for masked positions. Defaults to
                ``-0.7 * float32_max``.
            q_scale: Optional quantization scale for queries.
            k_scale: Optional quantization scale for keys.
            v_scale: Optional quantization scale for values.
            **ignore: Extra keyword arguments (silently ignored).

        Returns:
            An ``AttentionOutput`` containing the attended representations and
            the updated ``cache_view`` with mutated KV pages.

        Raises:
            ValueError: If tensor shapes/ranks are inconsistent or if
                ``keys_pe`` is None.
        """
        if keys_pe is None:
            raise ValueError(
                "multi_latent_ragged_page_attention requires `keys_pe`. Pass it as `keys_pe=` (or `k_pe=`)."
            )

        queries_nope = _reshape_query_tensor(queries_nope, "queries_nope")
        queries_pe = _reshape_query_tensor(queries_pe, "queries_pe")
        keys_values = _reshape_feature_tensor(keys_values, "keys_values")
        keys_pe = _reshape_feature_tensor(keys_pe, "keys_pe")
        if queries_nope.shape[:2] != queries_pe.shape[:2]:
            raise ValueError(
                "`queries_nope` and `queries_pe` must share [total_tokens, num_q_heads]. "
                f"Got {queries_nope.shape[:2]} vs {queries_pe.shape[:2]}."
            )

        if keys_values.ndim not in (2, 3):
            raise ValueError(
                "`keys_values` must be rank-2 [total_tokens, kv_lora_rank] or "
                f"rank-3 [total_tokens, num_kv_heads, kv_lora_rank]. Got shape={keys_values.shape}."
            )
        if keys_pe.ndim != 2:
            raise ValueError(f"`keys_pe` must be rank-2 [total_tokens, qk_pe_dim]. Got shape={keys_pe.shape}.")
        if keys_values.shape[0] != queries_nope.shape[0]:
            raise ValueError(
                "`keys_values` token dimension must match `queries_nope` token dimension. "
                f"Got keys_values={keys_values.shape[0]} vs queries_nope={queries_nope.shape[0]}. "
                "Pass explicit MLA tensors (`keys_values`, `keys_pe`) instead of standard value heads."
            )
        if keys_pe.shape[0] != queries_nope.shape[0]:
            raise ValueError(
                "`keys_pe` token dimension must match `queries_nope` token dimension. "
                f"Got keys_pe={keys_pe.shape[0]} vs queries_nope={queries_nope.shape[0]}."
            )

        kv_pages = cache_view.kv_pages
        resolver = self.metadata.runtime_sharding_resolver.with_mesh(self.metadata.mesh)
        resolve = resolver.resolve
        request_distribution = _resolve_distribution(cache_metadata)

        use_rank_major_dp = self._supports_rank_major_dp and getattr(request_distribution, "ndim", 0) == 2
        if use_rank_major_dp:
            request_distribution = jnp.stack(
                (
                    jnp.zeros_like(request_distribution[:, 2]),
                    jnp.zeros_like(request_distribution[:, 2]),
                    request_distribution[:, 2],
                ),
                axis=1,
            ).astype(jnp.int32)
        else:
            request_distribution = jnp.array([0, 0, request_distribution[2]], dtype=jnp.int32)
        page_axis = dp_page_axis(cache_view)
        head_aware_kv = keys_values.ndim == 3
        q_token_axis = ATTN_DP if use_rank_major_dp else ct.EMPTY

        qaxes_nope = resolve(
            axes=[q_token_axis, ct.EMPTY, ct.EMPTY] if head_aware_kv else [q_token_axis, ct.HEAD, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=queries_nope.shape,
        )
        qaxes_pe = resolve(
            axes=[q_token_axis, ct.EMPTY, ct.EMPTY] if head_aware_kv else [q_token_axis, ct.HEAD, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=queries_pe.shape,
        )
        kv_values_axes = resolve(
            axes=[q_token_axis, ct.EMPTY] if keys_values.ndim == 2 else [q_token_axis, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=keys_values.shape,
        )
        keys_pe_axes = resolve(axes=[q_token_axis, ct.EMPTY], mode=ct.MODE_PREFILL, shape=keys_pe.shape)

        kv_pages_spec = resolve(
            axes=[page_axis, ct.EMPTY, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )
        page_axis_names = normalize_axis_names(kv_pages_spec[0] if len(kv_pages_spec) > 0 else None)
        page_axis_size = mesh_axis_size(self.metadata.mesh, page_axis_names)
        kv_pages_spec_replicated = resolve(
            axes=[ct.EMPTY, ct.EMPTY, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )

        platform = "pallas" if jax.default_backend() == "tpu" else "auto"
        cfg = self.metadata.get_operation_config(self.get_impl_name())
        if cfg is None:
            cfg = self.metadata.get_operation_config("multi_latent_ragged_page_attention")

        # Operation configs support both legacy mappings and ejkernel dataclasses.
        cfg_value = cfg.get if isinstance(cfg, dict) else lambda name: getattr(cfg, name, None)
        chunk_prefill_size = cfg_value("chunk_prefill_size")
        num_kv_pages_per_block = cfg_value("num_kv_pages_per_block")
        cfg_num_queries_per_block = cfg_value("num_queries_per_block")

        # Compute TP axis size for head sharding so we resolve the correct
        # num_queries_per_block for the local (sharded) head count.
        head_spec = resolve(
            axes=[ct.HEAD],
            mode=ct.MODE_PREFILL,
            shape=(queries_nope.shape[1],),
        )
        head_axis_names = normalize_axis_names(head_spec[0] if len(head_spec) > 0 else None)
        head_axis_size = mesh_axis_size(self.metadata.mesh, head_axis_names) if not head_aware_kv else 1

        resolved_num_queries_per_block = _resolve_num_queries_per_block(
            num_q_heads=queries_nope.shape[1],
            requested=ignore.get("num_queries_per_block"),
            cfg_requested=cfg_num_queries_per_block,
            q_dtype=queries_nope.dtype,
            head_axis_size=head_axis_size,
            query_token_blocks=self._query_token_blocks,
        )

        # Both versions trace decode tiles that require packed-head-compatible blocks.
        ejkernel_cfg = self._config_type(
            chunk_prefill_size=chunk_prefill_size,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=resolved_num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes if vmem_limit_bytes is not None else cfg_value("vmem_limit_bytes"),
            platform=cfg_value("platform") or platform,
            backend=cfg_value("backend") or "any",
            num_warps=cfg_value("num_warps") if cfg_value("num_warps") is not None else 4,
            num_stages=cfg_value("num_stages") if cfg_value("num_stages") is not None else 1,
        )
        common_call_kwargs = dict(
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            sliding_window=sliding_window,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            cfg=ejkernel_cfg,
        )
        if mask_value is not None:
            common_call_kwargs["mask_value"] = mask_value
        else:
            common_call_kwargs["mask_value"] = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

        can_use_dp_local = (
            ENABLE_DP_LOCAL_PAGE_PATH
            and page_axis == ATTN_DP
            and page_axis_size > 1
            and len(page_axis_names) > 0
            and int(cache_metadata.context_lens.shape[0]) % page_axis_size == 0
        )

        if use_rank_major_dp and not can_use_dp_local:
            raise ValueError("Rank-major DP MLA ragged attention requires the DP-local page path.")

        sparse_selected, _kernel = _mla_kernel_call(
            self._kernel_function, ignore, queries_nope.shape[0], common_call_kwargs
        )
        extra_specs, extra_args = (), ()
        if sparse_selected is not None:
            extra_specs = (resolve(axes=[q_token_axis, ct.EMPTY], mode=ct.MODE_PREFILL, shape=sparse_selected.shape),)
            extra_args = (sparse_selected,)
        ep_split_axes = self._sparse_ep_split_axes(
            ignore,
            sparse_selected,
            dp_local=can_use_dp_local,
            page_axis=page_axis,
            query_specs=(qaxes_nope, qaxes_pe),
        )

        if can_use_dp_local:
            rows_per_shard = int(cache_metadata.context_lens.shape[0]) // page_axis_size
            max_pages_per_req = int(cache_metadata.pages_tables.shape[1])

            @partial(
                jax.shard_map,
                mesh=self.metadata.mesh,
                in_specs=(
                    qaxes_nope,
                    qaxes_pe,
                    kv_values_axes,
                    keys_pe_axes,
                    kv_pages_spec,
                    Ps(),
                    Ps(),
                    Ps(),
                    Ps(),
                    *extra_specs,
                ),
                out_specs=(qaxes_nope, kv_pages_spec),
                check_vma=False,
            )
            def _mapped(
                local_queries_nope,
                local_queries_pe,
                local_keys_values,
                local_keys_pe,
                local_kv_pages,
                full_context_lens,
                full_pages_tables,
                full_query_start_loc,
                full_distribution,
                *extra,
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
                local_total = local_distribution[2]

                local_num_pages = jnp.int32(local_kv_pages.shape[0])
                page_offset = shard_idx * local_num_pages
                local_block_tables = local_block_tables - page_offset
                local_block_tables = jnp.clip(local_block_tables, 0, local_num_pages - 1)

                local_output, local_kv_pages = _kernel(
                    local_queries_nope,
                    local_queries_pe,
                    local_keys_values,
                    local_keys_pe,
                    local_kv_pages,
                    local_context_lens,
                    local_block_tables,
                    local_query_start_loc,
                    local_distribution,
                    selected=extra[0] if extra else None,
                )

                if use_rank_major_dp:
                    return local_output, local_kv_pages

                row_ids = jnp.arange(rows_per_shard, dtype=jnp.int32)[:, None]
                token_ids = jnp.arange(local_queries_nope.shape[0], dtype=jnp.int32)[None, :]
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
                queries_nope,
                queries_pe,
                keys_values,
                keys_pe,
                kv_pages,
                cache_metadata.context_lens,
                cache_metadata.pages_tables,
                cache_metadata.query_start_loc,
                request_distribution,
                *extra_args,
            )
        elif ep_split_axes:
            output, kv_pages = _ep_split_sparse_attention(
                _kernel,
                mesh=self.metadata.mesh,
                axis_names=ep_split_axes,
                specs=(qaxes_nope, qaxes_pe, kv_values_axes, keys_pe_axes, kv_pages_spec_replicated),
                queries_nope=queries_nope,
                queries_pe=queries_pe,
                keys_values=keys_values,
                keys_pe=keys_pe,
                kv_pages=kv_pages,
                context_lens=cache_metadata.context_lens,
                pages_tables=cache_metadata.pages_tables,
                query_start_loc=cache_metadata.query_start_loc,
                distribution=request_distribution,
                selected=sparse_selected,
            )
        else:

            @partial(
                jax.shard_map,
                mesh=self.metadata.mesh,
                in_specs=(
                    qaxes_nope,
                    qaxes_pe,
                    kv_values_axes,
                    keys_pe_axes,
                    kv_pages_spec_replicated,
                    Ps(),
                    Ps(),
                    Ps(),
                    Ps(),
                    *extra_specs,
                ),
                out_specs=(qaxes_nope, kv_pages_spec_replicated),
                check_vma=False,
            )
            def _mapped_global(
                local_queries_nope,
                local_queries_pe,
                local_keys_values,
                local_keys_pe,
                local_kv_pages,
                full_context_lens,
                full_pages_tables,
                full_query_start_loc,
                full_distribution,
                *extra,
            ):
                return _kernel(
                    local_queries_nope,
                    local_queries_pe,
                    local_keys_values,
                    local_keys_pe,
                    local_kv_pages,
                    full_context_lens,
                    full_pages_tables.reshape(-1),
                    full_query_start_loc,
                    full_distribution,
                    selected=extra[0] if extra else None,
                )

            output, kv_pages = _mapped_global(
                queries_nope,
                queries_pe,
                keys_values,
                keys_pe,
                kv_pages,
                cache_metadata.context_lens,
                cache_metadata.pages_tables,
                cache_metadata.query_start_loc,
                request_distribution,
                *extra_args,
            )

        cache_view.kv_pages = kv_pages
        return AttentionOutput(attention_weights=None, attention_outputs=output, cache_view=cache_view)

    def forward_native(
        self,
        queries_nope: Float[Array, "total_tokens num_q_heads qk_nope_dim"],
        queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
        keys_values: Float[Array, "total_tokens kv_lora_rank"],
        keys_pe: Float[Array, "total_tokens qk_pe_dim"],
        cache_view: MLARaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        vmem_limit_bytes: int | None = None,
        mask_value: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        **ignore,
    ):
        """Platform-agnostic forward; delegates directly to :meth:`forward_core`."""
        return self.forward_core(
            queries_nope=queries_nope,
            queries_pe=queries_pe,
            keys_values=keys_values,
            keys_pe=keys_pe,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            sliding_window=sliding_window,
            vmem_limit_bytes=vmem_limit_bytes,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            **ignore,
        )

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward path; delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU forward path; delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward path; delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA forward path; delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm forward path; delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        queries_nope: Float[Array, "total_tokens num_q_heads qk_nope_dim"],
        queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
        keys_values: Float[Array, "total_tokens kv_lora_rank"],
        keys_pe: Float[Array, "total_tokens qk_pe_dim"],
        cache_view: MLARaggedPagesCacheView,
        cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        vmem_limit_bytes: int | None = None,
        mask_value: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        **ignore,
    ) -> AttentionOutput:
        """Dispatch MLA attention through the base-class backend selector.

        Delegates to the parent ``OperationImpl.__call__`` which routes to
        the appropriate ``forward_*`` method based on the current backend
        (TPU, GPU, CPU, etc.).

        Args:
            queries_nope: Non-positional query projections.
            queries_pe: Positional-embedding query projections.
            keys_values: Low-rank latent KV representation.
            keys_pe: Positional-embedding key projections.
            cache_view: MLA ragged-pages cache view.
            cache_metadata: Batch scheduling metadata.
            softmax_scale: Multiplicative scale applied before softmax.
            logits_soft_cap: Optional soft-capping of logits.
            sliding_window: Sliding-window size for local attention.
            vmem_limit_bytes: VMEM budget hint.
            mask_value: Value for masked positions.
            q_scale: Optional quantization scale for queries.
            k_scale: Optional quantization scale for keys.
            v_scale: Optional quantization scale for values.
            **ignore: Extra keyword arguments (silently ignored).

        Returns:
            An ``AttentionOutput`` with attended representations and updated cache.
        """
        output: AttentionOutput = super().__call__(
            queries_nope=queries_nope,
            queries_pe=queries_pe,
            keys_values=keys_values,
            keys_pe=keys_pe,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            sliding_window=sliding_window,
            vmem_limit_bytes=vmem_limit_bytes,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            **ignore,
        )

        return output

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Declare metadata and cache requirements for this operation.

        The operation requires sequence lengths, context lengths, positions,
        query-start locations, page tables, and request distribution
        metadata.  It uses ``MLARaggedPagesCacheView`` as the cache view.

        Args:
            mode: Execution mode (ignored; requirements are the same for
                all modes).

        Returns:
            An ``OperationRequirements`` describing the inputs needed by
            this operation.
        """
        del mode
        return (
            RequirementsBuilder("multi_latent_ragged_page_attention")
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
            .use_cache_view(MLARaggedPagesCacheView)
            .build()
        )


@OperationRegistry.register
class MultiLatentRaggedPageAttnV2(MultiLatentRaggedPageAttn):
    """Multi-Latent Attention v2 with per-case (decode/prefill/mixed) block-size tuning.

    Extends the v1 MLA ragged-paged attention with per-case block-size
    triples for decode, prefill, and mixed workloads.  Uses the same
    ``MLARaggedPagesCacheView`` cache format as v1.

    Registered under ``"multi_latent_ragged_page_attention_v2"``.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """Return the registered operation name.

        Returns:
            str: ``"multi_latent_ragged_page_attention_v2"``.
        """
        return "multi_latent_ragged_page_attention_v2"

    _kernel_function = staticmethod(multi_latent_ragged_page_attention_v2)
    _config_type = MultiLatentRaggedPageAttentionV2Config
    _supports_rank_major_dp = True
    _query_token_blocks = True

    def forward_native(self, *args, **kwargs) -> AttentionOutput:
        """Platform-agnostic forward; delegates to :meth:`forward_core`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: Result of :meth:`forward_core`.
        """
        return self.forward_core(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU dispatch; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU dispatch; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU dispatch; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA dispatch; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm dispatch; delegates to :meth:`forward_native`.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return self.forward_native(*args, **kwargs)

    def __call__(self, **kwargs) -> AttentionOutput:
        """Dispatch through the base-class backend selector.

        Args:
            **kwargs: All keyword arguments accepted by :meth:`forward_core`.

        Returns:
            AttentionOutput: Attended representations and updated cache view.
        """
        output: AttentionOutput = super().__call__(**kwargs)
        return output

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Declare metadata and cache requirements for v2.

        Args:
            mode: Execution mode (ignored; v2 has the same requirements
                across modes).

        Returns:
            OperationRequirements: Identical requirements to v1, scoped to
            the v2 implementation name.
        """
        del mode
        return (
            RequirementsBuilder("multi_latent_ragged_page_attention_v2")
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
            .use_cache_view(MLARaggedPagesCacheView)
            .build()
        )
