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

"""EasyDeL operation adapter for Dual-Stream Ragged Paged Attention.

Thin :class:`~easydel.operations.OperationImpl` over eJKernel's public
:func:`ejkernel.modules.dual_stream_ragged_page_attention` operation for
continuous-batching serving (eSurge) and online RL rollouts of Loop Transformers
using shared global Pass-1 paged KV alongside local sliding-window Pass-2 paged
KV under a single exact softmax normalizer.
"""

from __future__ import annotations

from functools import partial

import jax
from ejkernel.loggings import get_logger
from ejkernel.modules import (
    DualStreamRaggedPageAttentionConfig,
    dual_stream_ragged_page_attention,
    reduce_sinks_to_heads,
)
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Float
from spectrax import common_types as ct

from easydel.axis import ATTN_DP
from easydel.caching import ParallelHybridCacheView, RaggedPagesCacheView, RaggedPagesMetadata, unwrap_metadata
from easydel.infra.sharding import axis_index, mesh_axis_size, normalize_axis_names
from easydel.utils.helpers import check_bool_flag

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)
from ._compressed_window_common import configured_platform, resolve_platform
from ._paged_common import ENABLE_DP_LOCAL_PAGE_PATH, dp_page_axis, request_distribution_bounds
from .ragged_page_attention import (
    _default_tpu_rpa_v3_cfg,
    _kv_axis,
    _repeat_kv_heads_to_cache_width,
    _runtime_sharding_resolver,
    _tpu_mixed_request_distribution,
)

logger = get_logger(__name__)


def _unwrap_ragged_view(view: RaggedPagesCacheView | ParallelHybridCacheView) -> RaggedPagesCacheView:
    """Return the underlying :class:`RaggedPagesCacheView` from a view or parallel-hybrid wrapper."""
    if isinstance(view, ParallelHybridCacheView):
        view = view.transformer
    if not isinstance(view, RaggedPagesCacheView):
        raise TypeError(f"DualStreamRaggedPageAttn requires RaggedPagesCacheView, got {type(view).__name__}.")
    return view


def _resolve_layer_ragged_metadata(
    cache_metadata,
    cache_view: RaggedPagesCacheView,
) -> RaggedPagesMetadata:
    """Unwrap and resolve per-layer group metadata for a ragged page cache view."""
    meta = unwrap_metadata(cache_metadata, "ragged")
    if meta is None:
        meta = cache_metadata
    if hasattr(meta, "for_layer"):
        meta = meta.for_layer(cache_view.layer_index)
    if not isinstance(meta, RaggedPagesMetadata):
        raise TypeError(f"DualStreamRaggedPageAttn requires RaggedPagesMetadata, got {type(meta).__name__}.")
    return meta


@OperationRegistry.register
class DualStreamRaggedPageAttn(OperationImpl):
    """Dual-Stream Ragged Paged Attention (shared global KV + local sliding-window KV).

    Registered under ``"dual_stream_ragged_page_attention"``. Reads Pass-1's
    already-populated ``global_cache_view`` read-only while appending new
    local ``key``/``value`` tokens into ``local_cache_view`` and computing
    exact joint-softmax attention in one fused eJKernel call.
    """

    def __init__(self, metadata: OperationMetadata, *, autotune: bool = True) -> None:
        """Configure dual-stream ragged paged attention.

        Args:
            metadata: Runtime metadata, including explicit ejkernel config overrides.
            autotune: Leave unspecified block sizes to ejkernel's autotuner / heuristics.
                Set to False to use the static TPU serving heuristic instead.
        """
        super().__init__(metadata)
        self.autotune = autotune

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the operation registry key."""
        return "dual_stream_ragged_page_attention"

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Declare required ragged-v3 metadata and paged cache views."""
        del mode
        return (
            RequirementsBuilder("dual_stream_ragged_page_attention")
            .require_metadata(
                MetadataField.SEQ_LENS
                | MetadataField.CONTEXT_LENS
                | MetadataField.POSITIONS
                | MetadataField.QUERY_START_LOC
                | MetadataField.PAGES_TABLES
                | MetadataField.REQUEST_DISTRIBUTION
            )
            .support_cache(CacheType.RAGGED_PAGES | CacheType.HYBRID)
            .use_cache_view(RaggedPagesCacheView)
            .build()
        )

    @jax.named_scope("easydel-dual-stream-ragged-page-attention")
    def forward_native(
        self,
        query: Float[Array, "total_tokens num_q_heads head_dim"],
        key: Float[Array, "total_tokens num_kv_heads head_dim"],
        value: Float[Array, "total_tokens num_kv_heads head_dim"],
        global_cache_view: RaggedPagesCacheView | ParallelHybridCacheView,
        local_cache_view: RaggedPagesCacheView | ParallelHybridCacheView,
        global_cache_metadata: RaggedPagesMetadata,
        local_cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        sliding_window: int = 128,
        overlap: bool = True,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "..."] | None = None,
        vmem_limit_bytes: int | None = None,
        platform: str | None = None,
        **ignore,
    ) -> AttentionOutput:
        """Execute dual-stream ragged paged attention over ragged 3D tensors."""
        del ignore
        orig_local_wrapper = local_cache_view if isinstance(local_cache_view, ParallelHybridCacheView) else None
        g_view = _unwrap_ragged_view(global_cache_view)
        l_view = _unwrap_ragged_view(local_cache_view)
        g_meta = _resolve_layer_ragged_metadata(global_cache_metadata, g_view)
        l_meta = _resolve_layer_ragged_metadata(local_cache_metadata, l_view)

        global_kv_pages = g_view.kv_pages
        local_kv_pages = l_view.kv_pages
        num_model_kv_heads = int(key.shape[1])
        key, value = _repeat_kv_heads_to_cache_width(key, value, l_view)

        kv_cache_dtype = getattr(local_kv_pages, "dtype", None)
        if kv_cache_dtype is not None and (key.dtype != kv_cache_dtype or value.dtype != kv_cache_dtype):
            key = key.astype(kv_cache_dtype)
            value = value.astype(kv_cache_dtype)

        resolver = _runtime_sharding_resolver(self.metadata, l_view)
        resolve = resolver.resolve
        request_distribution = l_meta.request_distribution
        if request_distribution is None:
            scheduled = l_meta.query_start_loc[1:] - l_meta.query_start_loc[:-1]
            request_distribution = request_distribution_bounds(scheduled, l_meta.context_lens)

        use_rank_major_dp = getattr(request_distribution, "ndim", 0) == 2
        page_axis = dp_page_axis(l_view)
        g_page_axis = dp_page_axis(g_view)
        kv_token_axis = _kv_axis(l_view, ct.KV_HEAD)
        kv_page_axis = _kv_axis(l_view, ct.HEAD)
        q_token_axis = ATTN_DP if use_rank_major_dp else ct.EMPTY

        sinks_axis = None
        if softmax_aux is not None:
            softmax_aux = reduce_sinks_to_heads(softmax_aux, int(query.shape[1]), num_model_kv_heads)
            sinks_axis = resolve(axes=[ct.HEAD], mode=ct.MODE_PREFILL, shape=softmax_aux.shape)

        qaxes = resolve(axes=[q_token_axis, ct.HEAD, ct.EMPTY], mode=ct.MODE_PREFILL, shape=query.shape)
        kvaxes = resolve(axes=[q_token_axis, kv_token_axis, ct.EMPTY], mode=ct.MODE_PREFILL, shape=key.shape)
        g_pages_spec = resolve(
            axes=[g_page_axis, ct.EMPTY, kv_page_axis, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=global_kv_pages.shape,
        )
        l_pages_spec = resolve(
            axes=[page_axis, ct.EMPTY, kv_page_axis, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=local_kv_pages.shape,
        )
        g_pages_spec_rep = resolve(
            axes=[ct.EMPTY, ct.EMPTY, kv_page_axis, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=global_kv_pages.shape,
        )
        l_pages_spec_rep = resolve(
            axes=[ct.EMPTY, ct.EMPTY, kv_page_axis, ct.EMPTY, ct.EMPTY],
            mode=ct.MODE_PREFILL,
            shape=local_kv_pages.shape,
        )

        page_axis_names = normalize_axis_names(l_pages_spec[0] if len(l_pages_spec) > 0 else None)
        page_axis_size = mesh_axis_size(self.metadata.mesh, page_axis_names)

        resolved_platform = resolve_platform(platform, configured_platform(self.metadata, self.get_impl_name()))
        if check_bool_flag("FORCE_NATIVE_RUNTIME", False):
            resolved_platform = "xla"
        if resolved_platform is None:
            resolved_platform = "pallas" if jax.default_backend() == "tpu" else "auto"

        raw_cfg = self.metadata.get_operation_config("dual_stream_ragged_page_attention")
        op_cfg = raw_cfg if isinstance(raw_cfg, DualStreamRaggedPageAttentionConfig) else None
        if not self.autotune and op_cfg is None:
            rpa_default = _default_tpu_rpa_v3_cfg(
                None,
                query=query,
                key=key,
                kv_pages=global_kv_pages,
                context_lens=g_meta.context_lens,
                pages_tables=g_meta.pages_tables,
            )
            if rpa_default is not None:
                op_cfg = DualStreamRaggedPageAttentionConfig(
                    sliding_window=int(sliding_window),
                    overlap=bool(overlap),
                    chunk_prefill_size=rpa_default.chunk_prefill_size,
                    num_kv_pages_per_block=rpa_default.num_kv_pages_per_block,
                    num_queries_per_block=rpa_default.num_queries_per_block,
                    vmem_limit_bytes=vmem_limit_bytes,
                    platform=resolved_platform,
                    backend="any",
                )

        common_kwargs = dict(
            softmax_scale=softmax_scale,
            sliding_window=int(sliding_window),
            overlap=bool(overlap),
            logits_soft_cap=logits_soft_cap,
            vmem_limit_bytes=vmem_limit_bytes,
            cfg=op_cfg,
            platform=resolved_platform,
        )

        can_use_dp_local = (
            ENABLE_DP_LOCAL_PAGE_PATH
            and page_axis == ATTN_DP
            and g_page_axis == ATTN_DP
            and page_axis_size > 1
            and len(page_axis_names) > 0
            and int(l_meta.context_lens.shape[0]) % page_axis_size == 0
        )
        if use_rank_major_dp and not can_use_dp_local:
            raise ValueError("Rank-major DP dual-stream ragged attention requires the DP-local page path.")

        if can_use_dp_local:
            rows_per_shard = int(l_meta.context_lens.shape[0]) // page_axis_size
            g_max_pages_per_req = int(g_meta.pages_tables.shape[1])
            l_max_pages_per_req = int(l_meta.pages_tables.shape[1])

            @partial(
                jax.shard_map,
                mesh=self.metadata.mesh,
                in_specs=(
                    qaxes,
                    kvaxes,
                    kvaxes,
                    g_pages_spec,
                    l_pages_spec,
                    Ps(),
                    Ps(),
                    Ps(),
                    Ps(),
                    Ps(),
                    Ps(),
                    sinks_axis,
                ),
                out_specs=(qaxes, l_pages_spec),
                check_vma=False,
            )
            def _mapped(
                local_q,
                local_k,
                local_v,
                shard_g_pages,
                shard_l_pages,
                full_g_lens,
                full_g_tables,
                full_l_lens,
                full_l_tables,
                full_qsl,
                full_dist,
                local_aux,
            ):
                shard_idx = axis_index(page_axis_names)
                row_start = shard_idx * jnp.int32(rows_per_shard)

                shard_g_lens = jax.lax.dynamic_slice_in_dim(full_g_lens, row_start, rows_per_shard, axis=0)
                shard_l_lens = jax.lax.dynamic_slice_in_dim(full_l_lens, row_start, rows_per_shard, axis=0)
                shard_g_bt = jax.lax.dynamic_slice_in_dim(full_g_tables, row_start, rows_per_shard, axis=0).reshape(
                    (rows_per_shard * g_max_pages_per_req,)
                )
                shard_l_bt = jax.lax.dynamic_slice_in_dim(full_l_tables, row_start, rows_per_shard, axis=0).reshape(
                    (rows_per_shard * l_max_pages_per_req,)
                )

                if use_rank_major_dp:
                    shard_qsl = full_qsl[shard_idx]
                    shard_dist = full_dist[shard_idx]
                else:
                    shard_qsl = jax.lax.dynamic_slice_in_dim(full_qsl, row_start, rows_per_shard + 1, axis=0)
                    shard_sched = shard_qsl[1:] - shard_qsl[:-1]
                    shard_dist = request_distribution_bounds(shard_sched, shard_l_lens)

                kernel_dist = _tpu_mixed_request_distribution(shard_dist, op_cfg)
                local_total = shard_dist[2]

                g_num_pages = jnp.int32(shard_g_pages.shape[0])
                l_num_pages = jnp.int32(shard_l_pages.shape[0])
                shard_g_bt = jnp.clip(shard_g_bt - shard_idx * g_num_pages, 0, g_num_pages - 1)
                shard_l_bt = jnp.clip(shard_l_bt - shard_idx * l_num_pages, 0, l_num_pages - 1)

                local_out, updated_l_pages = dual_stream_ragged_page_attention(
                    local_q,
                    local_k,
                    local_v,
                    shard_g_pages,
                    shard_l_pages,
                    shard_g_lens,
                    shard_g_bt,
                    shard_l_lens,
                    shard_l_bt,
                    shard_qsl,
                    kernel_dist,
                    local_aux,
                    **common_kwargs,
                )
                if use_rank_major_dp:
                    return local_out, updated_l_pages

                row_ids = jnp.arange(rows_per_shard, dtype=jnp.int32)[:, None]
                token_ids = jnp.arange(local_q.shape[0], dtype=jnp.int32)[None, :]
                active_rows = row_ids < local_total
                local_token_mask = jnp.any(
                    active_rows & (token_ids >= shard_qsl[:-1, None]) & (token_ids < shard_qsl[1:, None]),
                    axis=0,
                )
                local_out = jnp.where(local_token_mask[:, None, None], local_out, jnp.zeros_like(local_out))
                axis_spec = page_axis_names[0] if len(page_axis_names) == 1 else tuple(page_axis_names)
                out_full = jax.lax.psum(local_out.astype(jnp.float32), axis_spec).astype(local_out.dtype)
                return out_full, updated_l_pages

            output, updated_local_pages = _mapped(
                query,
                key,
                value,
                global_kv_pages,
                local_kv_pages,
                g_meta.context_lens,
                g_meta.pages_tables,
                l_meta.context_lens,
                l_meta.pages_tables,
                l_meta.query_start_loc,
                request_distribution,
                softmax_aux,
            )
        else:
            kernel_dist = _tpu_mixed_request_distribution(request_distribution, op_cfg)
            output, updated_local_pages = dual_stream_ragged_page_attention(
                query,
                key,
                value,
                global_kv_pages,
                local_kv_pages,
                g_meta.context_lens,
                g_meta.pages_tables.reshape(-1),
                l_meta.context_lens,
                l_meta.pages_tables.reshape(-1),
                l_meta.query_start_loc,
                kernel_dist,
                softmax_aux,
                mesh=self.metadata.mesh,
                in_specs=(
                    qaxes,
                    kvaxes,
                    kvaxes,
                    g_pages_spec_rep,
                    l_pages_spec_rep,
                    Ps(),
                    Ps(),
                    Ps(),
                    Ps(),
                    Ps(),
                    Ps(),
                    sinks_axis,
                ),
                out_specs=(qaxes, l_pages_spec_rep),
                **common_kwargs,
            )

        l_view.kv_pages = updated_local_pages
        returned_view = (
            orig_local_wrapper.replace(transformer=l_view) if orig_local_wrapper is not None else l_view
        )
        return AttentionOutput(attention_weights=None, attention_outputs=output, cache_view=returned_view)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU dispatch path (Pallas fused dual-stream paged kernel)."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU dispatch path."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU dispatch path (XLA fallback)."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA dispatch path."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm dispatch path."""
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "..."],
        key: Float[Array, "..."],
        value: Float[Array, "..."],
        global_cache_view: RaggedPagesCacheView | ParallelHybridCacheView,
        local_cache_view: RaggedPagesCacheView | ParallelHybridCacheView,
        global_cache_metadata: RaggedPagesMetadata,
        local_cache_metadata: RaggedPagesMetadata,
        softmax_scale: float | None = None,
        sliding_window: int = 128,
        overlap: bool = True,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "..."] | None = None,
        vmem_limit_bytes: int | None = None,
        platform: str | None = None,
        **ignore,
    ) -> AttentionOutput:
        """Accept either 3D ragged ``[T, H, D]`` or 4D ``[B, T, H, D]`` tensors."""
        is_4d = query.ndim == 4
        orig_shape = query.shape
        q_3d = query.reshape(-1, *query.shape[-2:]) if is_4d else query
        k_3d = key.reshape(-1, *key.shape[-2:]) if key.ndim == 4 else key
        v_3d = value.reshape(-1, *value.shape[-2:]) if value.ndim == 4 else value

        out = super().__call__(
            query=q_3d,
            key=k_3d,
            value=v_3d,
            global_cache_view=global_cache_view,
            local_cache_view=local_cache_view,
            global_cache_metadata=global_cache_metadata,
            local_cache_metadata=local_cache_metadata,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            overlap=overlap,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
            vmem_limit_bytes=vmem_limit_bytes,
            platform=platform,
            **ignore,
        )
        if is_4d:
            out.attention_outputs = out.attention_outputs.reshape(orig_shape)
        return out
