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

"""Reduce-scatter matmul operation module.

This module implements a fused reduce-scatter matmul collective that computes::

    output = reduce_scatter(x @ y.T, scatter_dim=0)

where the K dimension of both ``x`` and ``y`` is already sharded across TP
devices (each device holds a ``k_shard``-sized slice), and the M dimension of
the result is scattered so that each device owns ``m // tp_size`` rows.

The operation is equivalent to::

    # full computation on device d
    partial = x_shard @ y_shard.T          # [m, n]
    # all-reduce, then scatter M dim 0:
    output_local = reduce_scatter(partial)  # [m // tp_size, n]

but fused into a single communication-computation primitive.

Supported Platforms:
    - GPU (Triton/CUDA): Overlapped compute and NCCL collective via the
      Triton ``reduce_scatter_matmul`` kernel.
    - XLA fallback: ``lax.all_reduce`` followed by a local slice.

Entry Points:
    :class:`ReduceScatterMatmul` -- ``Kernel`` subclass for the
    executor/autotuner framework.

    :func:`reduce_scatter_matmul` -- functional entry point wrapping the
    kernel in an :class:`~ejkernel.ops.Executor` with config caching.

Distributed Execution:
    The operation can be wrapped in ``shard_map`` by supplying ``mesh``.
    In that case ``in_specs`` and ``out_specs`` default to sharding K along
    ``axis_name`` for inputs and M along ``axis_name`` for the output.

Note:
    The block sizes in :class:`ReduceScatterMatmulConfig` are upper bounds;
    the actual tiling is snapped to the largest divisor of the tensor
    dimension that does not exceed the configured value.
"""

from __future__ import annotations

import os
from typing import Literal

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import ReduceScatterMatmulConfig


def _largest_divisor_leq(
    x: int,
    upper: int,
    candidates: tuple[int, ...] = (512, 256, 128, 64, 32, 16, 8, 4, 2, 1),
) -> int:
    """Return the largest divisor of ``x`` that is also <= ``upper``.

    First tries each value in ``candidates`` in decreasing order; if none
    qualify, falls back to a linear scan from ``min(x, upper)`` downward.
    Always returns at least 1.

    Args:
        x: Value whose divisors are enumerated (clamped to >= 1).
        upper: Maximum allowed return value (clamped to >= 1).
        candidates: Ordered sequence of preferred values to try first.

    Returns:
        Largest integer d such that ``d <= upper`` and ``x % d == 0``.
    """
    x = int(max(1, x))
    upper = int(max(1, upper))
    for candidate in candidates:
        if candidate <= upper and candidate <= x and x % candidate == 0:
            return candidate
    for candidate in range(min(x, upper), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def _infer_axis_size(axis_name: str) -> int | None:
    """Infer collective axis size from the active ``shard_map`` context.

    Queries the mapped axis size by issuing a ``lax.psum`` of 1.  Must be
    called inside a ``shard_map`` body where ``axis_name`` is in scope.

    Args:
        axis_name: The sharded axis name (must be active in the current
            ``shard_map`` context).

    Returns:
        Integer axis size, or ``None`` if the query fails (e.g. the axis is
        not currently mapped, or the value is not statically concrete).
    """
    try:
        return jax.core.concrete_or_error(
            int,
            jax.lax.psum(jnp.array(1, dtype=jnp.int32), axis_name=axis_name),
            f"collective axis '{axis_name}' size must be static.",
        )
    except Exception:
        return None


def _resolve_tp_size(tp_size: int | None, axis_name: str) -> int:
    """Resolve tensor-parallel world size.

    Resolution order:
    1. Use ``tp_size`` if explicitly provided and >= 1.
    2. Attempt to infer from the active ``shard_map`` axis (``_infer_axis_size``).
    3. Fall back to the total JAX device count (``jax.device_count()``).

    Args:
        tp_size: Explicit override, or ``None`` to auto-resolve.
        axis_name: Name of the sharded axis (used for inference fallback).

    Returns:
        Resolved integer world size >= 1.

    Raises:
        ValueError: If the resolved value is < 1.
    """
    resolved = int(tp_size) if tp_size is not None else (_infer_axis_size(axis_name) or int(jax.device_count()))
    if resolved < 1:
        raise ValueError(f"tp_size must be >= 1, got {resolved}.")
    return resolved


class ReduceScatterMatmul(Kernel[ReduceScatterMatmulConfig, Array]):
    """Fused distributed reduce-scatter matmul kernel.

    Computes ``reduce_scatter(x @ y.T, scatter_dim=0)`` where the K dimension
    of ``x`` and ``y`` is already sharded across TP devices.  Each device
    contributes a partial result over its K shard; the reduce-scatter
    accumulates across all ranks and returns the M-scattered slice
    ``[m // tp_size, n]`` local to each device.

    Block sizes are resolved from :class:`ReduceScatterMatmulConfig` at runtime
    by snapping to the largest divisor of each dimension that does not exceed
    the configured value (see :func:`_largest_divisor_leq`).

    Platforms:
        - GPU: Triton kernel with overlapped NCCL collective.
        - Fallback: XLA all-reduce + local slice (enabled via ``platform="xla"``).
    """

    def __init__(self):
        super().__init__(op_id="reduce_scatter_matmul")

    def get_impl(self, cfg: ReduceScatterMatmulConfig):
        """Get the kernel implementation for the given configuration.

        Args:
            cfg: Kernel configuration specifying platform and backend.

        Returns:
            Callable kernel implementation from the registry.

        Raises:
            ValueError: If no matching implementation is found.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def create_shard_map_wrapper(
        self,
        x: Float[Array, "m k_shard"],
        y: Float[Array, "n k_shard"],
        axis_name: str,
        collective_id: int | None = 0,
        precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
        tp_size: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: ReduceScatterMatmulConfig,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Build a shard_map-wrapped callable and its input arguments.

        Returns:
            Tuple of (shard_mapped_fn, call_args).
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        if in_specs is None:
            in_specs = (PartitionSpec(None, axis_name), PartitionSpec(None, axis_name))
        if out_specs is None:
            out_specs = PartitionSpec(axis_name, None)

        mesh_tp_size = int(mesh.shape[axis_name])
        if tp_size is None:
            tp_size = mesh_tp_size
        else:
            tp_size = int(tp_size)
            if tp_size < 1:
                raise ValueError(f"tp_size must be >= 1, got {tp_size}.")
            if tp_size != mesh_tp_size:
                raise ValueError(
                    f"tp_size ({tp_size}) must match mesh axis '{axis_name}' size ({mesh_tp_size}) in shard_map mode."
                )

        def _wrapped(x: Float[Array, "m k_shard"], y: Float[Array, "n k_shard"]):
            return self.run(
                x=x,
                y=y,
                axis_name=axis_name,
                collective_id=collective_id,
                precision=precision,
                tp_size=tp_size,
                platform=platform,
                cfg=cfg,
            )

        call_args = (x, y)
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"

        return (
            shard_map(
                _wrapped,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=check_vma,
            ),
            call_args,
        )

    def run(
        self,
        x: Float[Array, "m k_shard"],
        y: Float[Array, "n k_shard"],
        axis_name: str,
        collective_id: int | None = 0,
        precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
        tp_size: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: ReduceScatterMatmulConfig,
    ) -> Float[Array, "m_local n"]:
        """Execute reduce-scatter matmul with the selected backend.

        Computes ``reduce_scatter(x @ y.T, scatter_dim=0)`` with automatic
        block-size selection derived from *cfg* and the input shapes.

        Args:
            x: Input tensor of shape ``(m, k_shard)`` where K is sharded.
            y: Weight tensor of shape ``(n, k_shard)`` where K is sharded.
            axis_name: Name of the sharded axis for collective ops.
            collective_id: Barrier semaphore allocation id.
            precision: JAX matmul precision.
            tp_size: Tensor-parallel world size.
            platform: Optional platform override.
            cfg: Kernel configuration with block sizes and settings.

        Returns:
            Result of shape ``(m_local, n)`` where M is scattered.
        """
        if platform is not None:
            cfg = ReduceScatterMatmulConfig(
                block_m=cfg.block_m,
                block_n=cfg.block_n,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        impl = self.get_impl(cfg)
        resolved_tp_size = _resolve_tp_size(tp_size, axis_name)
        world = resolved_tp_size
        m_total = int(x.shape[0])
        n_total = int(y.shape[0])
        k_shard = int(x.shape[1])

        m_block = max(1, m_total // world)
        m_half = max(1, m_block // 2)
        bm = _largest_divisor_leq(m_half, int(cfg.block_m))
        bn = _largest_divisor_leq(n_total, int(cfg.block_n))
        bk = _largest_divisor_leq(k_shard, int(cfg.block_k))

        return impl(
            x=x,
            y=y,
            axis_name=axis_name,
            bm=bm,
            bn=bn,
            bk=bk,
            tp_size=resolved_tp_size,
            collective_id=collective_id,
            precision=precision,
        )

    def heuristic_cfg(self, inv: Invocation[ReduceScatterMatmulConfig, Array]) -> ReduceScatterMatmulConfig:
        """Return default heuristic configuration for any platform."""
        return ReduceScatterMatmulConfig(
            block_m=128,
            block_n=128,
            block_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[ReduceScatterMatmulConfig, Array]):
        """Return candidate configurations for autotuning."""
        candidates = []
        for block_m, block_n, block_k in ((128, 128, 128), (256, 128, 128), (256, 256, 128), (256, 256, 256)):
            candidates.append(
                ReduceScatterMatmulConfig(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=4,
                    num_stages=2,
                    platform="auto",
                    backend="any",
                )
            )
        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[ReduceScatterMatmulConfig, Array]):
        """Return GPU candidates for TileLang and XLA reduce-scatter matmul.

        Three tile knobs: ``(block_m, block_n, block_k)``. On H100,
        wgmma instructions prefer power-of-two tiles >= 64; we also try
        deeper pipelines (num_stages=3) for large-K shapes.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        tilelang_configs = (
            (64, 128, 64, 4, 2),
            (128, 128, 64, 4, 2),
            (128, 128, 128, 4, 3),
            (128, 256, 64, 8, 3),
            (256, 128, 64, 8, 3),
            (256, 256, 64, 8, 3),
            (256, 256, 128, 8, 3),
        )
        xla_configs = (
            (128, 128, 128),
            (256, 128, 128),
            (256, 256, 128),
            (256, 256, 256),
            (512, 256, 128),
            (512, 256, 256),
        )
        candidates: list[ReduceScatterMatmulConfig] = []
        if "tilelang" in platforms:
            for block_m, block_n, block_k, num_warps, num_stages in tilelang_configs:
                candidates.append(
                    ReduceScatterMatmulConfig(
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            for block_m, block_n, block_k in xla_configs:
                candidates.append(
                    ReduceScatterMatmulConfig(
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        num_warps=4,
                        num_stages=2,
                        platform="xla",
                        backend="any",
                    )
                )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[ReduceScatterMatmulConfig, Array]):
        """Return TPU candidates for Pallas and XLA reduce-scatter matmul paths."""
        base_configs = ((128, 128, 128), (256, 128, 128), (256, 256, 128), (256, 256, 256))
        return [
            ReduceScatterMatmulConfig(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=4,
                num_stages=2,
                platform=platform,
                backend=backend,
            )
            for platform, backend in (("pallas", "tpu"), ("xla", "any"))
            for block_m, block_n, block_k in base_configs
        ]


_reduce_scatter_matmul_executor: Executor[ReduceScatterMatmulConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("reduce-scatter-matmul"),
    )
)


def reduce_scatter_matmul(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    axis_name: str,
    /,
    *,
    collective_id: int | None = 0,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    tp_size: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: ReduceScatterMatmulConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "m_local n"]:
    """Reduce-scatter matmul with automatic backend selection and autotuning.

    Computes ``reduce_scatter(x @ y.T, scatter_dim=0)`` using the best
    available kernel for the current hardware.  When *mesh* is provided
    the operation is executed via ``shard_map``; otherwise it runs inside
    an existing sharded context.

    Note:
        This operation intentionally does not perform runtime fallback
        between distributed backends. If the selected platform cannot
        execute a given shape/constraint, the call fails. Choose
        ``platform``/``cfg.platform`` explicitly for your deployment.

    Args:
        x: Input tensor of shape ``(m, k_shard)`` where K is sharded.
        y: Weight tensor of shape ``(n, k_shard)`` where K is sharded.
        axis_name: Name of the sharded axis for collective ops.
        collective_id: Barrier semaphore allocation id.
        precision: JAX matmul precision.
        tp_size: Tensor-parallel world size.
        platform: Optional platform override.
        cfg: Optional kernel configuration override.
        mesh: If provided, wraps the call in ``shard_map``.
        in_specs: Optional input partition specs for ``shard_map``.
        out_specs: Optional output partition spec for ``shard_map``.

    Returns:
        Result of shape ``(m_local, n)`` where M is scattered.
    """
    method = "shard_map" if mesh is not None else None
    if method == "shard_map":
        if in_specs is None:
            in_specs = (PartitionSpec(None, axis_name), PartitionSpec(None, axis_name))
        if out_specs is None:
            out_specs = PartitionSpec(axis_name, None)
    return _reduce_scatter_matmul_executor(
        ReduceScatterMatmul(),
        x=x,
        y=y,
        axis_name=axis_name,
        collective_id=collective_id,
        precision=precision,
        tp_size=tp_size,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )
