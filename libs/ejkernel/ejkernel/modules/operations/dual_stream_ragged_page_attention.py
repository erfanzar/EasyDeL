# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Dual-Stream Ragged Paged Attention module with automatic platform dispatch.

Implements continuous-batching paged attention for GLM-5.4-style Loop
Transformers (and any architecture sharing a global paged KV pool alongside a
local sliding-window paged KV pool under a single exact float32 softmax):

- **Global stream (read-only):** reads Pass-1's already-populated paged KV cache.
- **Local stream (fused write + read):** writes Pass-2's new local K/V tokens
  into its sliding-window paged KV cache and attends over the local window
  ``sliding_window`` in the same kernel call.
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int32

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import DualStreamRaggedPageAttentionConfig

__all__ = ("DualStreamRaggedPageAttention", "dual_stream_ragged_page_attention")


class DualStreamRaggedPageAttention(
    Kernel[DualStreamRaggedPageAttentionConfig, tuple[Array, Array]]
):
    """Dual-Stream Ragged Paged Attention kernel (shared global KV + local sliding-window KV).

    Dispatches between the fused single-kernel TPU Pallas backend and the
    cross-platform XLA blocked online-softmax fallback registered under
    ``"dual_stream_ragged_page_attention"``.
    """

    def __init__(self):
        """Initialize the dual-stream ragged paged attention kernel."""
        super().__init__(op_id="dual_stream_ragged_page_attention")

    def get_impl(self, cfg: DualStreamRaggedPageAttentionConfig):
        """Resolve the registered backend implementation for ``cfg``."""
        platform = detect_platform("dual_stream_ragged_page_attention", cfg.platform, prefer_pallas=True)
        return kernel_registry.get("dual_stream_ragged_page_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        local_keys: Float[Array, "total_tokens num_kv_heads head_dim"],
        local_values: Float[Array, "total_tokens num_kv_heads head_dim"],
        global_kv_cache: Float[
            Array,
            "num_global_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded",
        ],
        local_kv_cache: Float[
            Array,
            "num_local_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded",
        ],
        global_kv_lens: Int32[Array, "max_num_seqs"],
        global_block_tables: Int32[Array, "max_num_seqs_times_global_pages"],
        local_kv_lens: Int32[Array, "max_num_seqs"],
        local_block_tables: Int32[Array, "max_num_seqs_times_local_pages"],
        query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
        distribution: Int32[Array, "3"],
        softmax_aux: Float[Array, "..."] | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | None = None,
        overlap: bool | None = None,
        logits_soft_cap: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        vmem_limit_bytes: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: DualStreamRaggedPageAttentionConfig,
    ) -> tuple[
        Float[Array, "total_tokens num_q_heads head_dim"],
        Float[Array, "num_local_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
    ]:
        """Execute dual-stream ragged paged attention with the resolved backend."""
        cfg_backend = getattr(cfg, "backend", Backend.ANY)
        if platform is not None:
            cfg = DualStreamRaggedPageAttentionConfig(
                sliding_window=int(sliding_window) if sliding_window is not None else int(cfg.sliding_window),
                overlap=bool(overlap) if overlap is not None else bool(cfg.overlap),
                chunk_prefill_size=cfg.chunk_prefill_size,
                num_kv_pages_per_block=cfg.num_kv_pages_per_block,
                num_queries_per_block=cfg.num_queries_per_block,
                vmem_limit_bytes=vmem_limit_bytes if vmem_limit_bytes is not None else cfg.vmem_limit_bytes,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            queries=queries,
            local_keys=local_keys,
            local_values=local_values,
            global_kv_cache=global_kv_cache,
            local_kv_cache=local_kv_cache,
            global_kv_lens=global_kv_lens,
            global_block_tables=global_block_tables,
            local_kv_lens=local_kv_lens,
            local_block_tables=local_block_tables,
            query_start_loc=query_start_loc,
            distribution=distribution,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            sliding_window=int(sliding_window) if sliding_window is not None else int(cfg.sliding_window),
            overlap=bool(overlap) if overlap is not None else bool(cfg.overlap),
            logits_soft_cap=logits_soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            chunk_prefill_size=cfg.chunk_prefill_size,
            num_kv_pages_per_block=cfg.num_kv_pages_per_block,
            num_queries_per_block=cfg.num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes if vmem_limit_bytes is not None else cfg.vmem_limit_bytes,
        )

    def heuristic_cfg(
        self,
        inv: Invocation[DualStreamRaggedPageAttentionConfig, tuple[Array, Array]] | None,
    ) -> DualStreamRaggedPageAttentionConfig:
        """Return default heuristic configuration."""
        sw = (
            int(inv.kwargs["sliding_window"])
            if (inv is not None and inv.kwargs.get("sliding_window") is not None)
            else 128
        )
        ov = bool(inv.kwargs["overlap"]) if (inv is not None and inv.kwargs.get("overlap") is not None) else True
        return DualStreamRaggedPageAttentionConfig(
            sliding_window=sw,
            overlap=ov,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(
        self,
        inv: Invocation[DualStreamRaggedPageAttentionConfig, tuple[Array, Array]],
    ):
        """Return candidate configurations for optional autotuning."""
        return [
            self.heuristic_cfg(inv),
            DualStreamRaggedPageAttentionConfig(
                num_kv_pages_per_block=4,
                num_queries_per_block=32,
                platform="auto",
                backend="any",
            ),
            DualStreamRaggedPageAttentionConfig(
                num_kv_pages_per_block=8,
                num_queries_per_block=16,
                platform="auto",
                backend="any",
            ),
        ]

    def create_shard_map_wrapper(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        local_keys: Float[Array, "total_tokens num_kv_heads head_dim"],
        local_values: Float[Array, "total_tokens num_kv_heads head_dim"],
        global_kv_cache: Float[
            Array,
            "num_global_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded",
        ],
        local_kv_cache: Float[
            Array,
            "num_local_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded",
        ],
        global_kv_lens: Int32[Array, "max_num_seqs"],
        global_block_tables: Int32[Array, "max_num_seqs_times_global_pages"],
        local_kv_lens: Int32[Array, "max_num_seqs"],
        local_block_tables: Int32[Array, "max_num_seqs_times_local_pages"],
        query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
        distribution: Int32[Array, "3"],
        softmax_aux: Float[Array, "..."] | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | None = None,
        overlap: bool | None = None,
        logits_soft_cap: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        vmem_limit_bytes: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: DualStreamRaggedPageAttentionConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: tuple[PartitionSpec, PartitionSpec] | None = None,
        check_vma: bool = False,
    ):
        """Build a ``shard_map`` wrapper over head / page mesh axes.

        ``in_specs`` covers the 11 required tensors plus ``softmax_aux`` (12
        entries total; entry 11 may be ``None`` when ``softmax_aux is None``).
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        if len(in_specs) not in (11, 12):
            raise ValueError(f"Expected 11 or 12 in_specs entries, got {len(in_specs)}.")

        active_cfg = cfg or self.heuristic_cfg(None)
        has_aux = softmax_aux is not None
        used_in_specs = in_specs[:12] if has_aux else in_specs[:11]

        def _wrapper(q, lk, lv, g_cache, l_cache, g_lens, g_bt, l_lens, l_bt, qsl, dist, *rest):
            aux = rest[0] if rest else None
            return self.run(
                queries=q,
                local_keys=lk,
                local_values=lv,
                global_kv_cache=g_cache,
                local_kv_cache=l_cache,
                global_kv_lens=g_lens,
                global_block_tables=g_bt,
                local_kv_lens=l_lens,
                local_block_tables=l_bt,
                query_start_loc=qsl,
                distribution=dist,
                softmax_aux=aux,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                overlap=overlap,
                logits_soft_cap=logits_soft_cap,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                vmem_limit_bytes=vmem_limit_bytes,
                platform=platform,
                cfg=active_cfg,
            )

        shard_map_fn = shard_map(
            _wrapper,
            mesh=mesh,
            in_specs=used_in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        base_args = (
            queries,
            local_keys,
            local_values,
            global_kv_cache,
            local_kv_cache,
            global_kv_lens,
            global_block_tables,
            local_kv_lens,
            local_block_tables,
            query_start_loc,
            distribution,
        )
        call_args = (*base_args, softmax_aux) if has_aux else base_args
        return shard_map_fn, call_args


_dual_stream_rpa_executor: Executor[DualStreamRaggedPageAttentionConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=2, iters=10),
        persistent=PersistentCache("dual-stream-ragged-page-attention", cfg_type=DualStreamRaggedPageAttentionConfig),
    )
)


def dual_stream_ragged_page_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    local_keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    local_values: Float[Array, "total_tokens num_kv_heads head_dim"],
    global_kv_cache: Float[
        Array,
        "num_global_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded",
    ],
    local_kv_cache: Float[
        Array,
        "num_local_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded",
    ],
    global_kv_lens: Int32[Array, "max_num_seqs"],
    global_block_tables: Int32[Array, "max_num_seqs_times_global_pages"],
    local_kv_lens: Int32[Array, "max_num_seqs"],
    local_block_tables: Int32[Array, "max_num_seqs_times_local_pages"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    softmax_aux: Float[Array, "..."] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    overlap: bool | None = None,
    logits_soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: DualStreamRaggedPageAttentionConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: tuple[PartitionSpec, PartitionSpec] | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_local_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
]:
    """Compute dual-stream ragged paged attention with fused local-cache write.

    Args:
        queries: Ragged queries ``[total_tokens, num_q_heads, head_dim]``.
        local_keys: Local-stream keys ``[total_tokens, num_kv_heads, head_dim]``
            written into ``local_kv_cache``.
        local_values: Local-stream values ``[total_tokens, num_kv_heads, head_dim]``
            written into ``local_kv_cache``.
        global_kv_cache: Read-only global paged KV cache (e.g., Pass-1 shared KV).
        local_kv_cache: Local sliding-window paged KV cache updated in-place.
        global_kv_lens: Per-sequence lengths in ``global_kv_cache``.
        global_block_tables: Flattened page table for ``global_kv_cache``.
        local_kv_lens: Per-sequence (possibly eviction-rebased) lengths in ``local_kv_cache``.
        local_block_tables: Flattened page table for ``local_kv_cache``.
        query_start_loc: Cumulative token offsets ``[max_num_seqs + 1]``.
        distribution: Batch split ``[decode_end, prefill_end, total_seqs]``.
        softmax_aux: Optional attention sink logits.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        sliding_window: Local sliding-window width ``W`` (default 128).
        overlap: If ``True``, global attends causally ``k <= q``; if ``False``,
            global attends disjointly ``k <= q - W``.
        logits_soft_cap: Optional tanh soft cap.
        q_scale: Optional query scale.
        k_scale: Optional key scale.
        v_scale: Optional value scale.
        chunk_prefill_size: Optional prefill chunk size override.
        num_kv_pages_per_block: Optional KV pages per kernel block override.
        num_queries_per_block: Optional queries per kernel block override.
        vmem_limit_bytes: Optional TPU VMEM budget hint.
        platform: Optional backend override (``"pallas"``, ``"xla"``, ...).
        cfg: Optional :class:`DualStreamRaggedPageAttentionConfig`.
        mesh: Optional JAX device mesh for ``shard_map`` execution.
        in_specs: Partition specs for ``shard_map`` inputs.
        out_specs: Partition specs ``(output_spec, local_kv_cache_spec)``.

    Returns:
        Tuple ``(attention_output, updated_local_kv_cache)``.
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    has_explicit_cfg = any(
        x is not None
        for x in (cfg, chunk_prefill_size, num_kv_pages_per_block, num_queries_per_block, vmem_limit_bytes, platform)
    )
    if has_explicit_cfg:
        base = (
            cfg
            if cfg is not None
            else DualStreamRaggedPageAttentionConfig(platform=platform or "auto", backend="any")
        )
        resolved_cfg: DualStreamRaggedPageAttentionConfig | None = DualStreamRaggedPageAttentionConfig(
            sliding_window=int(sliding_window) if sliding_window is not None else int(base.sliding_window),
            overlap=bool(overlap) if overlap is not None else bool(base.overlap),
            chunk_prefill_size=chunk_prefill_size if chunk_prefill_size is not None else base.chunk_prefill_size,
            num_kv_pages_per_block=(
                num_kv_pages_per_block if num_kv_pages_per_block is not None else base.num_kv_pages_per_block
            ),
            num_queries_per_block=(
                num_queries_per_block if num_queries_per_block is not None else base.num_queries_per_block
            ),
            vmem_limit_bytes=vmem_limit_bytes if vmem_limit_bytes is not None else base.vmem_limit_bytes,
            platform=platform or base.platform,
            backend=Backend.ANY if platform == "xla" else base.backend,
        )
    else:
        resolved_cfg = None
    eff_sliding_window = (
        int(sliding_window)
        if sliding_window is not None
        else (int(resolved_cfg.sliding_window) if resolved_cfg is not None else 128)
    )
    eff_overlap = (
        bool(overlap)
        if overlap is not None
        else (bool(resolved_cfg.overlap) if resolved_cfg is not None else True)
    )
    return _dual_stream_rpa_executor(
        DualStreamRaggedPageAttention(),
        queries=queries,
        local_keys=local_keys,
        local_values=local_values,
        global_kv_cache=global_kv_cache,
        local_kv_cache=local_kv_cache,
        global_kv_lens=global_kv_lens,
        global_block_tables=global_block_tables,
        local_kv_lens=local_kv_lens,
        local_block_tables=local_block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=eff_sliding_window,
        overlap=eff_overlap,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=resolved_cfg,
    )
