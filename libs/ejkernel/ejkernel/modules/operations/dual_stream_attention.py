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

"""Dual-Stream Hybrid Attention module with automatic platform dispatch.

Implements GLM-5.4-style dual-stream attention where queries attend
simultaneously over a global KV stream (e.g. Pass-1 cached keys/values) and a
local sliding-window KV stream (e.g. Pass-2 local keys/values of width ``W``)
under a single joint softmax normalizer.

On TPU, sequences meeting SplashAttention alignment run through two Pallas
SplashAttention kernels merged in fp32 with an exact ``custom_vjp`` backward
pass. Everywhere else (CPU/GPU, decode, explicit masks, or unaligned shapes)
the operation runs on the exact XLA blocked online-softmax fallback.
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    BwdParams,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    FwdParams,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import DualStreamAttentionConfig

_OPTIONAL_ARRAY_FIELDS = (
    "g_q_seg",
    "g_kv_seg",
    "l_q_seg",
    "l_kv_seg",
    "global_mask",
    "local_mask",
)


class DualStreamAttention(Kernel[DualStreamAttentionConfig, Array]):
    """Dual-Stream Hybrid Attention kernel (global + local sliding-window KV).

    Dispatches between the TPU Pallas SplashAttention dual-stream kernel and
    the cross-platform XLA blocked online-softmax fallback registered under
    ``"dual_stream_attention"``.
    """

    def __init__(self):
        """Initialize the dual-stream attention kernel."""
        super().__init__(op_id="dual_stream_attention")

    def get_impl(self, cfg: DualStreamAttentionConfig):
        """Resolve the registered backend implementation for ``cfg``."""
        platform = detect_platform("dual_stream_attention", cfg.platform, prefer_pallas=True)
        return kernel_registry.get("dual_stream_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch num_q_heads q_len head_dim"],
        k_global: Float[Array, "batch num_kv_heads kv_g_len head_dim"],
        v_global: Float[Array, "batch num_kv_heads kv_g_len head_dim"],
        k_local: Float[Array, "batch num_kv_heads kv_l_len head_dim"],
        v_local: Float[Array, "batch num_kv_heads kv_l_len head_dim"],
        g_q_seg: Int[Array, "batch q_len"] | None = None,
        g_kv_seg: Int[Array, "batch kv_g_len"] | None = None,
        l_q_seg: Int[Array, "batch q_len"] | None = None,
        l_kv_seg: Int[Array, "batch kv_l_len"] | None = None,
        global_mask: Bool[Array, "batch mask_heads q_len kv_g_len"] | None = None,
        local_mask: Bool[Array, "batch mask_heads q_len kv_l_len"] | None = None,
        softmax_scale: float | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: DualStreamAttentionConfig,
    ) -> Float[Array, "batch num_q_heads q_len head_dim"]:
        """Execute dual-stream attention with the resolved backend."""
        cfg_backend = getattr(cfg, "backend", Backend.ANY)
        if platform is not None:
            cfg = DualStreamAttentionConfig(
                fwd_params=getattr(cfg, "fwd_params", None),
                bwd_params=getattr(cfg, "bwd_params", None),
                sliding_window=int(getattr(cfg, "sliding_window", 128)),
                overlap=bool(getattr(cfg, "overlap", True)),
                q_chunk_size=int(getattr(cfg, "q_chunk_size", 512)),
                kv_chunk_size=int(getattr(cfg, "kv_chunk_size", 1024)),
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            k_global=k_global,
            v_global=v_global,
            k_local=k_local,
            v_local=v_local,
            g_q_seg=g_q_seg,
            g_kv_seg=g_kv_seg,
            l_q_seg=l_q_seg,
            l_kv_seg=l_kv_seg,
            global_mask=global_mask,
            local_mask=local_mask,
            softmax_scale=softmax_scale,
            sliding_window=int(getattr(cfg, "sliding_window", 128)),
            overlap=bool(getattr(cfg, "overlap", True)),
            fwd_params=getattr(cfg, "fwd_params", None),
            bwd_params=getattr(cfg, "bwd_params", None),
            q_chunk_size=int(getattr(cfg, "q_chunk_size", 512)),
            kv_chunk_size=int(getattr(cfg, "kv_chunk_size", 1024)),
        )

    def heuristic_cfg(self, inv: Invocation[DualStreamAttentionConfig, Array]) -> DualStreamAttentionConfig:
        """Return default heuristic configuration."""
        del inv
        return DualStreamAttentionConfig(
            fwd_params=FwdParams(q_blocksize=512, kv_blocksize=512),
            bwd_params=BwdParams(q_blocksize=512, kv_blocksize=512),
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[DualStreamAttentionConfig, Array]):
        """Return candidate configurations for optional autotuning."""
        return [
            self.heuristic_cfg(inv),
            DualStreamAttentionConfig(
                fwd_params=FwdParams(q_blocksize=256, kv_blocksize=256),
                bwd_params=BwdParams(q_blocksize=256, kv_blocksize=256),
                platform="auto",
                backend="any",
            ),
        ]

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch num_q_heads q_len head_dim"],
        k_global: Float[Array, "batch num_kv_heads kv_g_len head_dim"],
        v_global: Float[Array, "batch num_kv_heads kv_g_len head_dim"],
        k_local: Float[Array, "batch num_kv_heads kv_l_len head_dim"],
        v_local: Float[Array, "batch num_kv_heads kv_l_len head_dim"],
        g_q_seg: Int[Array, "batch q_len"] | None = None,
        g_kv_seg: Int[Array, "batch kv_g_len"] | None = None,
        l_q_seg: Int[Array, "batch q_len"] | None = None,
        l_kv_seg: Int[Array, "batch kv_l_len"] | None = None,
        global_mask: Bool[Array, "batch mask_heads q_len kv_g_len"] | None = None,
        local_mask: Bool[Array, "batch mask_heads q_len kv_l_len"] | None = None,
        softmax_scale: float | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: DualStreamAttentionConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Build a differentiable ``shard_map`` wrapper over ``(batch, head)`` axes.

        Optional non-``None`` array operands (``g_q_seg``, ``g_kv_seg``,
        ``l_q_seg``, ``l_kv_seg``, ``global_mask``, ``local_mask``) ride after
        the five required tensors in that exact order, and ``in_specs`` must
        carry one :class:`PartitionSpec` per supplied operand.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        optional_values = {
            "g_q_seg": g_q_seg,
            "g_kv_seg": g_kv_seg,
            "l_q_seg": l_q_seg,
            "l_kv_seg": l_kv_seg,
            "global_mask": global_mask,
            "local_mask": local_mask,
        }
        extras = [(name, optional_values[name]) for name in _OPTIONAL_ARRAY_FIELDS if optional_values[name] is not None]
        n_operands = 5 + len(extras)
        if len(in_specs) != n_operands:
            raise ValueError(
                f"in_specs must have {n_operands} entries for 5 required operands + {[n for n, _ in extras]}, "
                f"got {len(in_specs)}."
            )
        extra_names = tuple(name for name, _ in extras)
        active_cfg = cfg or self.heuristic_cfg(None)

        def _wrapper(q, kg, vg, kl, vl, *rest):
            return self.run(
                query=q,
                k_global=kg,
                v_global=vg,
                k_local=kl,
                v_local=vl,
                softmax_scale=softmax_scale,
                platform=platform,
                cfg=active_cfg,
                **dict(zip(extra_names, rest, strict=True)),
            )

        shard_map_fn = shard_map(
            _wrapper,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        call_args = (query, k_global, v_global, k_local, v_local, *(val for _, val in extras))
        return shard_map_fn, call_args


_dual_stream_attention_executor: Executor[DualStreamAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=2, iters=10),
        persistent=PersistentCache("dual-stream-attention", cfg_type=DualStreamAttentionConfig),
    )
)


def dual_stream_attention(
    query: Float[Array, "batch num_q_heads q_len head_dim"],
    k_global: Float[Array, "batch num_kv_heads kv_g_len head_dim"],
    v_global: Float[Array, "batch num_kv_heads kv_g_len head_dim"],
    k_local: Float[Array, "batch num_kv_heads kv_l_len head_dim"],
    v_local: Float[Array, "batch num_kv_heads kv_l_len head_dim"],
    g_q_seg: Int[Array, "batch q_len"] | None = None,
    g_kv_seg: Int[Array, "batch kv_g_len"] | None = None,
    l_q_seg: Int[Array, "batch q_len"] | None = None,
    l_kv_seg: Int[Array, "batch kv_l_len"] | None = None,
    global_mask: Bool[Array, "batch mask_heads q_len kv_g_len"] | None = None,
    local_mask: Bool[Array, "batch mask_heads q_len kv_l_len"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    overlap: bool | None = None,
    q_chunk_size: int | None = None,
    kv_chunk_size: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: DualStreamAttentionConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "batch num_q_heads q_len head_dim"]:
    """Compute exact joint-softmax attention over global + local sliding-window KV streams.

    Args:
        query: Queries ``[batch, num_q_heads, q_len, head_dim]``.
        k_global: Global-stream keys ``[batch, num_kv_heads, kv_g_len, head_dim]``.
        v_global: Global-stream values ``[batch, num_kv_heads, kv_g_len, head_dim]``.
        k_local: Local-stream keys ``[batch, num_kv_heads, kv_l_len, head_dim]``.
        v_local: Local-stream values ``[batch, num_kv_heads, kv_l_len, head_dim]``.
        g_q_seg: Optional global-stream query segment IDs ``[batch, q_len]``.
        g_kv_seg: Optional global-stream KV segment IDs ``[batch, kv_g_len]``.
        l_q_seg: Optional local-stream query segment IDs ``[batch, q_len]``.
        l_kv_seg: Optional local-stream KV segment IDs ``[batch, kv_l_len]``.
        global_mask: Optional explicit boolean mask ``[batch, 1|Hq, q_len, kv_g_len]``.
        local_mask: Optional explicit boolean mask ``[batch, 1|Hq, q_len, kv_l_len]``.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        sliding_window: Local sliding-window width ``W`` (default 128).
        overlap: If ``True`` (default), Stream 1 uses causal ``k <= q``; if ``False``,
            Stream 1 uses disjoint shifted causal ``k <= q - W``.
        q_chunk_size: Query chunk size for blocked XLA fallback (default 512).
        kv_chunk_size: Global-KV chunk size for blocked XLA fallback (default 1024).
        platform: Optional backend override (``"pallas"``, ``"xla"``, ...).
        cfg: Optional :class:`DualStreamAttentionConfig`.
        mesh: Optional JAX device mesh for ``shard_map`` execution.
        in_specs: Partition specs matching ``(query, k_global, v_global, k_local, v_local, *optional_non_none)``.
        out_specs: Output partition spec.

    Returns:
        Attention output ``[batch, num_q_heads, q_len, head_dim]``.
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    base = cfg if cfg is not None else DualStreamAttentionConfig(platform=platform or "auto", backend="any")
    resolved_cfg = DualStreamAttentionConfig(
        fwd_params=base.fwd_params,
        bwd_params=base.bwd_params,
        sliding_window=int(sliding_window) if sliding_window is not None else int(getattr(base, "sliding_window", 128)),
        overlap=bool(overlap) if overlap is not None else bool(getattr(base, "overlap", True)),
        q_chunk_size=int(q_chunk_size) if q_chunk_size is not None else int(getattr(base, "q_chunk_size", 512)),
        kv_chunk_size=int(kv_chunk_size) if kv_chunk_size is not None else int(getattr(base, "kv_chunk_size", 1024)),
        platform=platform or base.platform,
        backend=Backend.ANY if platform == "xla" else base.backend,
    )
    return _dual_stream_attention_executor(
        DualStreamAttention(),
        query=query,
        k_global=k_global,
        v_global=v_global,
        k_local=k_local,
        v_local=v_local,
        g_q_seg=g_q_seg,
        g_kv_seg=g_kv_seg,
        l_q_seg=l_q_seg,
        l_kv_seg=l_kv_seg,
        global_mask=global_mask,
        local_mask=local_mask,
        softmax_scale=softmax_scale,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=resolved_cfg,
    )
