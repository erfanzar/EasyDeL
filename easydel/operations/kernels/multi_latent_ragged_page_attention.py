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

"""Multi-latent ragged paged attention operation wrapper."""

from functools import partial

import jax
from eformer import common_types as ct
from ejkernel.modules import (
    multi_latent_ragged_page_attention,  # pyright: ignore[reportMissingTypeStubs]
    multi_latent_ragged_page_attention_v2,  # pyright: ignore[reportMissingTypeStubs]
)
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Float

from easydel.axis import ATTN_DP
from easydel.caching import MLARaggedPagesCacheView, RaggedPagesMetadata
from easydel.utils.helpers import check_bool_flag

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)

ENABLE_DP_LOCAL_PAGE_PATH = check_bool_flag("EASURGE_ENABLE_DP_LOCAL_PAGE_PATH", default=True)


def _normalize_axis_names(axis: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Convert an axis specification into a canonical tuple of axis name strings.

    Args:
        axis: A single axis name, a sequence of axis names, or None.

    Returns:
        A tuple of non-empty axis name strings. Returns ``()`` when *axis* is None.
    """
    if axis is None:
        return ()
    if isinstance(axis, (tuple, list)):
        return tuple(str(a) for a in axis if a)
    return (str(axis),)


def _mesh_axis_size(mesh, axis_names: tuple[str, ...]) -> int:
    """Compute the total size of one or more mesh axes.

    Multiplies together the sizes of the requested axes from a JAX mesh.
    Missing axis names are treated as size 1.

    Args:
        mesh: A ``jax.sharding.Mesh`` (or None).
        axis_names: Tuple of mesh axis names whose sizes are multiplied.

    Returns:
        Product of the sizes of *axis_names* in *mesh*. Returns 1 when
        *mesh* is None or *axis_names* is empty.
    """
    size = 1
    if mesh is None:
        return size
    for axis_name in axis_names:
        size *= int(mesh.shape.get(axis_name, 1))
    return int(size)


def _axis_index(axis_names: tuple[str, ...]) -> jax.Array:
    """Compute a linearized device index across multiple mesh axes.

    For a single axis this returns ``jax.lax.axis_index(axis)``.  For
    multiple axes the indices are combined in row-major order so that
    each device gets a unique int32 index in ``[0, product_of_sizes)``.

    Args:
        axis_names: Tuple of mesh axis names. May be empty.

    Returns:
        A scalar int32 ``jax.Array`` representing the linearized index
        of the current device. Returns ``0`` when *axis_names* is empty.
    """
    if not axis_names:
        return jnp.int32(0)
    idx = jax.lax.axis_index(axis_names[0]).astype(jnp.int32)
    for axis_name in axis_names[1:]:
        axis_size = jax.lax.psum(jnp.int32(1), axis_name)
        idx = idx * axis_size + jax.lax.axis_index(axis_name).astype(jnp.int32)
    return idx


def _dp_page_axis(cache_view: MLARaggedPagesCacheView):
    """Return the sharding axis name for KV pages based on data-parallel size.

    When data parallelism is active (``data_parallel_size > 1``), pages are
    sharded along the ``ATTN_DP`` axis. Otherwise they are replicated.

    Args:
        cache_view: MLA ragged-pages cache view carrying DP metadata.

    Returns:
        ``ATTN_DP`` when the data-parallel size exceeds 1, else ``ct.EMPTY``.
    """
    dp_size = max(1, int(getattr(cache_view.metadata, "data_parallel_size", 1)))
    return ATTN_DP if dp_size > 1 else ct.EMPTY


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
        return x.reshape(-1, x.shape[-2], x.shape[-1])
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
        return x.reshape(-1, x.shape[-2], x.shape[-1])
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    if x.ndim == 2:
        return x
    raise ValueError(f"`{name}` must be rank-2, rank-3, or rank-4, got rank-{x.ndim}.")


def _resolve_num_queries_per_block(
    *,
    num_q_heads: int,
    requested: int | None,
    cfg_requested: int | None,
    q_dtype: jnp.dtype | None = None,
    head_axis_size: int = 1,
) -> int:
    """Pick a TPU-safe ``num_queries_per_block`` for the ejkernel MLA decode path.

    The ejkernel TPU MLA kernel requires (decode):
    ``local_heads_per_q_packing % num_queries_per_block == 0``
    where ``local_heads = num_q_heads / head_axis_size`` (tensor-parallel sharding).

    This function tries the *requested* (or *cfg_requested*) value first, then
    falls back to the largest divisor from a fixed candidate list.

    Args:
        num_q_heads: Total number of query heads (before TP sharding).
        requested: Explicitly requested block size (takes priority).
        cfg_requested: Block size from the operation config (used when
            *requested* is None).
        q_dtype: Query dtype; float16/bfloat16 enables 2x packing.
        head_axis_size: Tensor-parallel axis size (number of TP shards).

    Returns:
        A positive integer block size that evenly divides the packed local
        head count.
    """
    q_packing = 2 if q_dtype is not None and jnp.finfo(q_dtype).bits == 16 else 1
    local_heads = max(1, int(num_q_heads) // max(1, head_axis_size))
    # align_to mimics the kernel's padding to q_packing multiples.
    aligned_local = local_heads + (-local_heads % q_packing)
    packed_heads = aligned_local // q_packing
    preferred = requested if requested is not None else cfg_requested
    preferred = None if preferred is None else max(1, int(preferred))

    if preferred is not None and packed_heads % preferred == 0:
        return preferred

    for candidate in (32, 16, 10, 8, 5, 4, 3, 2, 1):
        if preferred is not None and candidate > preferred:
            continue
        if packed_heads % candidate == 0:
            return candidate

    return 1


def _request_distribution_bounds(scheduled: jax.Array, context_lens: jax.Array) -> jax.Array:
    """Compute a ``[decode_end, prefill_end, total]`` request distribution vector."""
    scheduled = jnp.asarray(scheduled, dtype=jnp.int32)
    context_lens = jnp.asarray(context_lens, dtype=jnp.int32)

    has_tokens = scheduled > 0
    total = jnp.sum(has_tokens).astype(jnp.int32)
    is_decode = (scheduled == 1) & (context_lens > 1) & has_tokens
    decode = jnp.sum(is_decode & has_tokens).astype(jnp.int32)
    prefill_count = jnp.sum((~is_decode) & has_tokens).astype(jnp.int32)
    prefill_end = decode + prefill_count
    return jnp.stack((decode, prefill_end, total)).astype(jnp.int32)


def _resolve_distribution(cache_metadata: RaggedPagesMetadata) -> jax.Array:
    """Compute a ``[decode_end, prefill_end, total]`` request distribution vector.

    If ``cache_metadata.request_distribution`` is already set, it is returned
    directly (cast to int32). Otherwise the distribution is inferred from
    ``query_start_loc`` and ``context_lens``.
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
    * Data-parallel local-page sharding when ``EASURGE_ENABLE_DP_LOCAL_PAGE_PATH``
      is active, falling back to a globally-replicated page path otherwise.
    * Automatic sharding via ``jax.shard_map`` for TPU/GPU meshes.

    Registered under the name ``"multi_latent_ragged_page_attention_v1"``.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """Return the registered operation name for this implementation."""
        return "multi_latent_ragged_page_attention_v1"

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
        """Run the MLA ragged-paged attention kernel (v1 implementation).

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
        manager = self.metadata.partition_manager
        resolve = manager.resolve
        request_distribution = _resolve_distribution(cache_metadata)

        request_distribution = jnp.array([0, 0, request_distribution[2]], dtype=jnp.int32)
        page_axis = _dp_page_axis(cache_view)
        head_aware_kv = keys_values.ndim == 3

        qaxes_nope = resolve(
            axes=[ct.EMPTY, ct.EMPTY, ct.EMPTY] if head_aware_kv else [ct.EMPTY, ct.HEAD, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=queries_nope.shape,
        )
        qaxes_pe = resolve(
            axes=[ct.EMPTY, ct.EMPTY, ct.EMPTY] if head_aware_kv else [ct.EMPTY, ct.HEAD, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=queries_pe.shape,
        )
        kv_values_axes = resolve(
            axes=[ct.EMPTY, ct.EMPTY] if keys_values.ndim == 2 else [ct.EMPTY, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=keys_values.shape,
        )
        keys_pe_axes = resolve(axes=[ct.EMPTY, ct.EMPTY], mode=ct.MODE_PREFILL, shape=keys_pe.shape)

        kv_pages_spec = resolve(
            axes=[page_axis, ct.EMPTY, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )
        page_axis_names = _normalize_axis_names(kv_pages_spec[0] if len(kv_pages_spec) > 0 else None)
        page_axis_size = _mesh_axis_size(self.metadata.mesh, page_axis_names)
        kv_pages_spec_replicated = resolve(
            axes=[ct.EMPTY, ct.EMPTY, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )

        platform = "pallas" if jax.default_backend() == "tpu" else "auto"
        cfg = self.metadata.get_operation_config("multi_latent_ragged_page_attention_v1")
        if cfg is None:
            cfg = self.metadata.get_operation_config("multi_latent_ragged_page_attention")

        # Extract tuning knobs from the EasyDeL operation config (dict).
        chunk_prefill_size = cfg.get("chunk_prefill_size") if cfg else None
        num_kv_pages_per_block = cfg.get("num_kv_pages_per_block") if cfg else None
        cfg_num_queries_per_block = cfg.get("num_queries_per_block") if cfg else None

        # Compute TP axis size for head sharding so we resolve the correct
        # num_queries_per_block for the local (sharded) head count.
        head_spec = resolve(
            axes=[ct.HEAD],
            mode=ct.MODE_PREFILL,
            shape=(queries_nope.shape[1],),
        )
        head_axis_names = _normalize_axis_names(head_spec[0] if len(head_spec) > 0 else None)
        head_axis_size = _mesh_axis_size(self.metadata.mesh, head_axis_names) if not head_aware_kv else 1

        resolved_num_queries_per_block = _resolve_num_queries_per_block(
            num_q_heads=queries_nope.shape[1],
            requested=ignore.get("num_queries_per_block"),
            cfg_requested=cfg_num_queries_per_block,
            q_dtype=queries_nope.dtype,
            head_axis_size=head_axis_size,
        )

        # Build an ejkernel config so the autotuner uses a compatible
        # num_queries_per_block (must divide local packed heads on each shard).
        from ejkernel.modules.operations.configs import MultiLatentRaggedPageAttentionConfig as _MlrpaCfg

        ejkernel_cfg = _MlrpaCfg(
            chunk_prefill_size=chunk_prefill_size,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=resolved_num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
            platform=platform,
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
            ):
                del full_distribution
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

                local_scheduled = local_query_start_loc[1:] - local_query_start_loc[:-1]
                local_distribution = _request_distribution_bounds(local_scheduled, local_context_lens)
                local_total = local_distribution[2]

                local_num_pages = jnp.int32(local_kv_pages.shape[0])
                page_offset = shard_idx * local_num_pages
                local_block_tables = local_block_tables - page_offset
                local_block_tables = jnp.clip(local_block_tables, 0, local_num_pages - 1)

                local_output, local_kv_pages = multi_latent_ragged_page_attention(
                    local_queries_nope,
                    local_queries_pe,
                    local_keys_values,
                    local_keys_pe,
                    local_kv_pages,
                    local_context_lens,
                    local_block_tables,
                    local_query_start_loc,
                    local_distribution,
                    **common_call_kwargs,
                )

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
            ):
                return multi_latent_ragged_page_attention(
                    local_queries_nope,
                    local_queries_pe,
                    local_keys_values,
                    local_keys_pe,
                    local_kv_pages,
                    full_context_lens,
                    full_pages_tables.reshape(-1),
                    full_query_start_loc,
                    full_distribution,
                    **common_call_kwargs,
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
class MultiLatentRaggedPageAttnV2(OperationImpl):
    """Multi-Latent Attention v2 with per-case (decode/prefill/mixed) block-size tuning.

    Extends the v1 MLA ragged-paged attention with per-case block-size
    triples for decode, prefill, and mixed workloads.  Uses the same
    ``MLARaggedPagesCacheView`` cache format as v1.

    Registered under ``"multi_latent_ragged_page_attention_v2"``.
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        return "multi_latent_ragged_page_attention_v2"

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
        """Run the MLA ragged-paged attention kernel (v2 implementation)."""
        if keys_pe is None:
            raise ValueError("multi_latent_ragged_page_attention_v2 requires `keys_pe`.")

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
            raise ValueError(f"`keys_values` must be rank-2 or rank-3. Got shape={keys_values.shape}.")
        if keys_pe.ndim != 2:
            raise ValueError(f"`keys_pe` must be rank-2. Got shape={keys_pe.shape}.")
        if keys_values.shape[0] != queries_nope.shape[0]:
            raise ValueError(
                "`keys_values` token dim must match `queries_nope` token dim. "
                f"Got {keys_values.shape[0]} vs {queries_nope.shape[0]}."
            )
        if keys_pe.shape[0] != queries_nope.shape[0]:
            raise ValueError(
                "`keys_pe` token dim must match `queries_nope` token dim. "
                f"Got {keys_pe.shape[0]} vs {queries_nope.shape[0]}."
            )

        kv_pages = cache_view.kv_pages
        manager = self.metadata.partition_manager
        resolve = manager.resolve
        request_distribution = _resolve_distribution(cache_metadata)
        request_distribution = jnp.array([0, 0, request_distribution[2]], dtype=jnp.int32)
        page_axis = _dp_page_axis(cache_view)
        head_aware_kv = keys_values.ndim == 3

        qaxes_nope = resolve(
            axes=[ct.EMPTY, ct.EMPTY, ct.EMPTY] if head_aware_kv else [ct.EMPTY, ct.HEAD, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=queries_nope.shape,
        )
        qaxes_pe = resolve(
            axes=[ct.EMPTY, ct.EMPTY, ct.EMPTY] if head_aware_kv else [ct.EMPTY, ct.HEAD, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=queries_pe.shape,
        )
        kv_values_axes = resolve(
            axes=[ct.EMPTY, ct.EMPTY] if keys_values.ndim == 2 else [ct.EMPTY, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=keys_values.shape,
        )
        keys_pe_axes = resolve(axes=[ct.EMPTY, ct.EMPTY], mode=ct.MODE_PREFILL, shape=keys_pe.shape)

        kv_pages_spec = resolve(
            axes=[page_axis, ct.EMPTY, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )
        page_axis_names = _normalize_axis_names(kv_pages_spec[0] if len(kv_pages_spec) > 0 else None)
        page_axis_size = _mesh_axis_size(self.metadata.mesh, page_axis_names)
        kv_pages_spec_replicated = resolve(
            axes=[ct.EMPTY, ct.EMPTY, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=kv_pages.shape,
        )

        platform = "pallas" if jax.default_backend() == "tpu" else "auto"
        cfg = self.metadata.get_operation_config("multi_latent_ragged_page_attention_v2")
        if cfg is None:
            cfg = self.metadata.get_operation_config("multi_latent_ragged_page_attention")

        chunk_prefill_size = cfg.get("chunk_prefill_size") if cfg else None
        num_kv_pages_per_block = cfg.get("num_kv_pages_per_block") if cfg else None
        cfg_num_queries_per_block = cfg.get("num_queries_per_block") if cfg else None

        head_spec = resolve(
            axes=[ct.HEAD],
            mode=ct.MODE_PREFILL,
            shape=(queries_nope.shape[1],),
        )
        head_axis_names = _normalize_axis_names(head_spec[0] if len(head_spec) > 0 else None)
        head_axis_size = _mesh_axis_size(self.metadata.mesh, head_axis_names) if not head_aware_kv else 1

        resolved_num_queries_per_block = _resolve_num_queries_per_block(
            num_q_heads=queries_nope.shape[1],
            requested=ignore.get("num_queries_per_block"),
            cfg_requested=cfg_num_queries_per_block,
            q_dtype=queries_nope.dtype,
            head_axis_size=head_axis_size,
        )

        from ejkernel.modules.operations.configs import MultiLatentRaggedPageAttentionV2Config as _MlrpaV2Cfg

        ejkernel_cfg = _MlrpaV2Cfg(
            chunk_prefill_size=chunk_prefill_size,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=resolved_num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
            platform=platform,
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
            ):
                del full_distribution
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

                local_scheduled = local_query_start_loc[1:] - local_query_start_loc[:-1]
                local_distribution = _request_distribution_bounds(local_scheduled, local_context_lens)
                local_total = local_distribution[2]

                local_num_pages = jnp.int32(local_kv_pages.shape[0])
                page_offset = shard_idx * local_num_pages
                local_block_tables = local_block_tables - page_offset
                local_block_tables = jnp.clip(local_block_tables, 0, local_num_pages - 1)

                local_output, local_kv_pages = multi_latent_ragged_page_attention_v2(
                    local_queries_nope,
                    local_queries_pe,
                    local_keys_values,
                    local_keys_pe,
                    local_kv_pages,
                    local_context_lens,
                    local_block_tables,
                    local_query_start_loc,
                    local_distribution,
                    **common_call_kwargs,
                )

                row_ids = jnp.arange(rows_per_shard, dtype=jnp.int32)[:, None]
                token_ids = jnp.arange(local_queries_nope.shape[0], dtype=jnp.int32)[None, :]
                starts = local_query_start_loc[:-1, None]
                ends = local_query_start_loc[1:, None]
                active_rows = row_ids < local_total
                local_token_mask = jnp.any(active_rows & (token_ids >= starts) & (token_ids < ends), axis=0)
                local_output = jnp.where(local_token_mask[:, None, None], local_output, jnp.zeros_like(local_output))
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
            ):
                return multi_latent_ragged_page_attention_v2(
                    local_queries_nope,
                    local_queries_pe,
                    local_keys_values,
                    local_keys_pe,
                    local_kv_pages,
                    full_context_lens,
                    full_pages_tables.reshape(-1),
                    full_query_start_loc,
                    full_distribution,
                    **common_call_kwargs,
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
            )

        cache_view.kv_pages = kv_pages
        return AttentionOutput(attention_weights=None, attention_outputs=output, cache_view=cache_view)

    def forward_native(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_core(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        return self.forward_native(*args, **kwargs)

    def __call__(self, **kwargs) -> AttentionOutput:
        output: AttentionOutput = super().__call__(**kwargs)
        return output

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
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
