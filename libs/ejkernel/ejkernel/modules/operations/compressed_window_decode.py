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

"""Compressed-window decode attention module (DeepSeek-V4).

Shared-KV (``K == V``) attention with an explicit per-``(slot, query, kv)``
additive bias and per-head learnable attention sinks. The KV axis (a
sliding-window token ring concatenated with a small number of compressed
entries) is bounded and resident in VMEM, so the TPU kernel is a single fused
pass rather than a flash-style online scan.

This is the fused kernel for DeepSeek-V4's serving decode step: it replaces the
per-token dense ``einsum -> sink softmax -> einsum`` math that previously ran
once per scanned grid column. The caller (the model / eSurge runner) builds the
KV axis and the additive bias — including the Lightning-Indexer top-k gating,
which stays a pre-kernel gather producing ``bias`` — so the kernel itself is a
pure attention primitive.
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float

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
from .configs import CompressedWindowDecodeConfig


class CompressedWindowDecode(Kernel[CompressedWindowDecodeConfig, Array]):
    """Shared-KV sink-augmented decode attention with an explicit additive bias.

    Dispatches to a Pallas TPU kernel (when available) or the XLA reference,
    both registered under ``"compressed_window_decode"``. The XLA reference is
    the correctness baseline (DeepSeek-V4's dense ``_attend`` math).
    """

    def __init__(self):
        """Initialize the compressed-window decode attention kernel."""
        super().__init__(op_id="compressed_window_decode")

    def get_impl(self, cfg: CompressedWindowDecodeConfig):
        """Resolve the registry implementation for the requested platform.

        Args:
            cfg: Configuration selecting platform/backend.

        Returns:
            The registered kernel callable.
        """
        platform = detect_platform("compressed_window_decode", cfg.platform, prefer_pallas=True)
        return kernel_registry.get("compressed_window_decode", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch num_heads q_len head_dim"],
        kv: Float[Array, "batch kv_len head_dim"],
        bias: Float[Array, "batch q_len kv_len"],
        softmax_aux: Float[Array, "num_heads"] | None = None,
        softmax_scale: float | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: CompressedWindowDecodeConfig,
    ) -> Float[Array, "batch num_heads q_len head_dim"]:
        """Execute compressed-window decode attention.

        Args:
            query: Rotated queries ``[batch, num_heads, q_len, head_dim]``.
            kv: Shared key/value axis ``[batch, kv_len, head_dim]`` (``K == V``).
            bias: Additive fp32 mask ``[batch, q_len, kv_len]`` (head-broadcast).
            softmax_aux: Optional per-head sink logits ``[num_heads]``.
            softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
            platform: Optional platform override.
            cfg: Kernel configuration.

        Returns:
            Attention output ``[batch, num_heads, q_len, head_dim]``.
        """
        cfg_backend = getattr(cfg, "backend", Backend.ANY)
        if platform is not None:
            cfg = CompressedWindowDecodeConfig(
                fwd_params=getattr(cfg, "fwd_params", None),
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            kv=kv,
            bias=bias,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            fwd_params=getattr(cfg, "fwd_params", None),
        )

    def heuristic_cfg(self, inv: Invocation[CompressedWindowDecodeConfig, Array]) -> CompressedWindowDecodeConfig:
        """Return the default configuration (no block tuning; KV is unblocked)."""
        return CompressedWindowDecodeConfig(fwd_params=None, platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[CompressedWindowDecodeConfig, Array]):
        """Return the single deterministic configuration (nothing to autotune)."""
        return [self.heuristic_cfg(inv)]

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch num_heads q_len head_dim"],
        kv: Float[Array, "batch kv_len head_dim"],
        bias: Float[Array, "batch q_len kv_len"],
        softmax_aux: Float[Array, "num_heads"] | None = None,
        softmax_scale: float | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: CompressedWindowDecodeConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Build a ``shard_map`` wrapper for head-parallel (TP) execution.

        The single shared KV head is replicated across the tensor-parallel axis
        and the query/output heads are sharded, so each device runs the kernel
        over its local heads with no in-kernel collectives.

        Args:
            query: Queries ``[batch, num_heads, q_len, head_dim]``.
            kv: Shared KV axis ``[batch, kv_len, head_dim]``.
            bias: Additive fp32 bias ``[batch, q_len, kv_len]``.
            softmax_aux: Optional per-head sink logits ``[num_heads]``.
            softmax_scale: Logit scale.
            platform: Optional platform override.
            cfg: Kernel configuration.
            mesh: Device mesh.
            in_specs: Partition specs for ``(query, kv, bias, softmax_aux)``.
            out_specs: Partition spec for the output.
            check_vma: Whether to check for valid memory access patterns.

        Returns:
            Tuple ``(shard_map_fn, call_args)``.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapper(query, kv, bias, softmax_aux):
            """Shard-map body delegating to ``run`` with captured static params."""
            return self.run(
                query=query,
                kv=kv,
                bias=bias,
                softmax_aux=softmax_aux,
                softmax_scale=softmax_scale,
                platform=platform,
                cfg=cfg or self.heuristic_cfg(None),
            )

        shard_map_fn = shard_map(
            _wrapper,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        call_args = (query, kv, bias, softmax_aux)
        return shard_map_fn, call_args


_compressed_window_decode_executor: Executor[CompressedWindowDecodeConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=2, iters=10),
        persistent=PersistentCache("compressed-window-decode", cfg_type=CompressedWindowDecodeConfig),
    )
)


def compressed_window_decode(
    query: Float[Array, "batch num_heads q_len head_dim"],
    kv: Float[Array, "batch kv_len head_dim"],
    bias: Float[Array, "batch q_len kv_len"],
    softmax_aux: Float[Array, "num_heads"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: CompressedWindowDecodeConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "batch num_heads q_len head_dim"]:
    """Fused shared-KV sink-augmented decode attention (DeepSeek-V4).

    Computes, per batch row ``b``, head ``h``, query position ``s``::

        logits[b, h, s, l] = scale * (q[b, h, s] . kv[b, l]) + bias[b, s, l]
        out[b, h, s]       = (softmax over l of [logits, sink[h]]) @ kv[b]

    where the sink column's probability mass is dropped before weighting the
    values. The single shared KV head is broadcast over all query heads.

    Args:
        query: Rotated queries ``[batch, num_heads, q_len, head_dim]`` (``q_len``
            is 1 for a serving decode step).
        kv: Shared key/value axis ``[batch, kv_len, head_dim]`` (``K == V``): the
            sliding-window token ring concatenated with the compressed entries.
        bias: Additive fp32 mask ``[batch, q_len, kv_len]`` broadcast over heads
            (ring-slot validity, compressed-entry causal thresholds, and
            Lightning-Indexer top-k gating already folded in by the caller).
        softmax_aux: Optional per-head sink logits ``[num_heads]``.
        softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
        platform: Optional platform override (``"pallas"``, ``"xla"``, ...).
        cfg: Optional :class:`~.configs.CompressedWindowDecodeConfig`.
        mesh: Optional device mesh for ``shard_map`` (tensor-parallel) execution.
        in_specs: Partition specs for ``(query, kv, bias, softmax_aux)``.
        out_specs: Output partition spec.

    Returns:
        Attention output ``[batch, num_heads, q_len, head_dim]``.
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"
    return _compressed_window_decode_executor(
        CompressedWindowDecode(),
        query=query,
        kv=kv,
        bias=bias,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )
