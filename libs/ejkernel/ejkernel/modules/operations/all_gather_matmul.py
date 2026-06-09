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

"""All-gather + matmul fused operation for tensor-parallel linear layers.

In a tensor-parallel setup the LHS activation matrix ``x`` is split along its
row (M) dimension across devices.  The RHS weight matrix ``y`` is split along
the column (N) dimension.  A standard all-gather followed by a local matmul
gives the same result as a global matmul, but fusing the two operations allows
overlapping the communication with computation.

Operation:
    ``result = all_gather(x, axis=0) @ y``

    where ``x`` has shape ``(m_local, k)`` on each device and the gathered
    ``x`` has shape ``(m, k)`` with ``m = m_local * tp_size``.

Memory layout:
    - ``x``:  ``(m_local, k)``   — LHS shard on the current device.
    - ``y``:  ``(k, n_local)``   — RHS shard; or ``(n_local, k)`` when
      ``rhs_transpose=True``.
    - result: ``(m, n_local)``   — output shard, no further collective needed.

Block-size clamping:
    ``block_n`` and ``block_k`` from the config are silently clamped to the
    largest divisor of ``n_local`` and ``k`` respectively that is ``<=`` the
    requested value.  This prevents shape-mismatch errors when matrix
    dimensions are not multiples of the requested block sizes.

Platform support:
    The XLA implementation is always available as a fallback.  An optimised
    Pallas TPU kernel is available when ``platform="pallas"`` or ``"auto"``
    on TPU hardware.
"""

from __future__ import annotations

import os
from typing import Literal

import jax
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import AllGatherMatmulConfig


def _largest_divisor_leq(
    x: int, upper: int, candidates: tuple[int, ...] = (512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
) -> int:
    """Return the largest divisor of ``x`` that is also ``<= upper``.

    Checks ``candidates`` (in descending order) first for speed, then falls
    back to a linear search.  Always returns at least 1.

    Args:
        x: The integer whose divisors are searched.
        upper: Maximum allowed return value.
        candidates: Fast-path candidates to try before the exhaustive search.

    Returns:
        Largest integer ``d`` such that ``d <= upper``, ``d <= x``, and
        ``x % d == 0``.
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


def _inv_xy_rhs_transpose(inv: Invocation[AllGatherMatmulConfig, Array]) -> tuple[Array, Array, bool]:
    """Extract ``x``, ``y``, and ``rhs_transpose`` from an invocation.

    Supports both keyword-argument (``inv.kwargs["x"]``) and positional-
    argument (``inv.args[0]``, ``inv.args[1]``) calling conventions.

    Args:
        inv: The ``Invocation`` object passed to heuristic/candidate methods.

    Returns:
        Tuple of ``(x, y, rhs_transpose)``.

    Raises:
        ValueError: If neither ``x``/``y`` keyword args nor two positional
            args are present in the invocation.
    """
    x = inv.kwargs.get("x")
    y = inv.kwargs.get("y")
    if x is None or y is None:
        if len(inv.args) < 2:
            raise ValueError("AllGatherMatmul invocation missing x/y.")
        x = inv.args[0]
        y = inv.args[1]
    rhs_transpose = bool(inv.kwargs.get("rhs_transpose", False))
    return x, y, rhs_transpose


class AllGatherMatmul(Kernel[AllGatherMatmulConfig, Array]):
    """Fused all-gather + matmul for tensor-parallel linear layers.

    Computes ``all_gather(x, axis=0) @ y`` where ``x`` is the locally held
    row-shard of the activation matrix and ``y`` is the locally held
    column-shard of the weight matrix.

    The kernel automatically clamps ``block_n``/``block_k`` from the config to
    the largest divisors of the actual column/contraction dimensions so that
    shapes are always valid regardless of the requested block sizes.

    Platform support:
        - XLA: always available; used on CPU and as a fallback on GPU.
        - Pallas TPU: selected automatically on TPU hardware.

    Typical use:
        Wrap with ``shard_map`` (or pass ``mesh``/``in_specs``/``out_specs``
        to the functional ``all_gather_matmul``) to run inside an existing
        sharded execution context.
    """

    def __init__(self):
        super().__init__(op_id="all_gather_matmul")

    def get_impl(self, cfg: AllGatherMatmulConfig):
        """Get the kernel implementation for the given configuration.

        Args:
            cfg: Kernel configuration specifying platform and backend.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def create_shard_map_wrapper(
        self,
        x: Float[Array, "m_local k"],
        y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
        axis_name: str,
        rhs_transpose: bool = False,
        collective_id: int | None = 0,
        precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
        tp_size: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: AllGatherMatmulConfig,
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
            if rhs_transpose:
                in_specs = (PartitionSpec(axis_name, None), PartitionSpec(axis_name, None))
            else:
                in_specs = (PartitionSpec(axis_name, None), PartitionSpec(None, axis_name))
        if out_specs is None:
            out_specs = PartitionSpec(None, axis_name)

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

        def _wrapped(
            x: Float[Array, "m_local k"],
            y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
        ):
            return self.run(
                x=x,
                y=y,
                axis_name=axis_name,
                rhs_transpose=rhs_transpose,
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
        x: Float[Array, "m_local k"],
        y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
        axis_name: str,
        rhs_transpose: bool = False,
        collective_id: int | None = 0,
        precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
        tp_size: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: AllGatherMatmulConfig,
    ) -> Float[Array, "m n_local"]:
        """Execute all-gather matmul with the selected backend.

        Computes ``all_gather(x, axis=0) @ y`` with automatic block-size
        clamping derived from *cfg*.

        Args:
            x: Local LHS shard of shape ``(m_local, k)``.
            y: Local RHS shard of shape ``(k, n_local)`` or ``(n_local, k)``
                when *rhs_transpose* is True.
            axis_name: Name of the sharded axis for collective ops.
            rhs_transpose: Whether *y* is stored transposed.
            collective_id: Barrier semaphore allocation id.
            precision: JAX matmul precision.
            tp_size: Tensor-parallel world size.
            platform: Optional platform override.
            cfg: Kernel configuration with block sizes and settings.

        Returns:
            Result of shape ``(m, n_local)``.
        """
        if platform is not None:
            cfg = AllGatherMatmulConfig(
                block_n=cfg.block_n,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        impl = self.get_impl(cfg)
        k = int(x.shape[1])
        n_local = int(y.shape[0] if rhs_transpose else y.shape[1])
        bn = _largest_divisor_leq(n_local, int(cfg.block_n))
        bk = _largest_divisor_leq(k, int(cfg.block_k))
        return impl(
            x=x,
            y=y,
            axis_name=axis_name,
            rhs_transpose=rhs_transpose,
            bn=bn,
            bk=bk,
            tp_size=tp_size,
            collective_id=collective_id,
            precision=precision,
        )

    def heuristic_cfg(self, inv: Invocation[AllGatherMatmulConfig, Array]) -> AllGatherMatmulConfig:
        """Return default heuristic configuration for any platform."""
        return AllGatherMatmulConfig(
            block_n=128,
            block_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def heuristic_cfg_tpu(self, inv: Invocation[AllGatherMatmulConfig, Array]) -> AllGatherMatmulConfig:
        """Return TPU/Pallas-oriented default configuration."""
        x, y, rhs_transpose = _inv_xy_rhs_transpose(inv)
        k = int(x.shape[1])
        n_local = int(y.shape[0] if rhs_transpose else y.shape[1])
        block_n = _largest_divisor_leq(n_local, 256)
        block_k = _largest_divisor_leq(k, 256)
        return AllGatherMatmulConfig(
            block_n=block_n,
            block_k=block_k,
            num_warps=4,
            num_stages=2,
            platform="pallas",
            backend="tpu",
        )

    def candidate_cfgs(self, inv: Invocation[AllGatherMatmulConfig, Array]):
        """Return candidate configurations for autotuning."""
        candidates = []
        for block_n, block_k in ((128, 128), (256, 128), (256, 256), (512, 256)):
            candidates.append(
                AllGatherMatmulConfig(
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=4,
                    num_stages=2,
                    platform="auto",
                    backend="any",
                )
            )
        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[AllGatherMatmulConfig, Array]):
        """Return GPU candidates for TileLang and XLA all-gather matmul paths.

        Two main knobs:

        * ``block_n`` ∈ {64, 128, 256, 512} — output column tile.
        * ``block_k`` ∈ {64, 128, 256} — contraction tile (deeper = more
          pipelining, more SMEM).

        Block sizes are silently clamped to the largest divisor of the
        actual matrix dimension that is still ``<=`` the requested value,
        so unaligned shapes still work.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        configs = (
            (64, 128, 4, 2),
            (128, 64, 4, 2),
            (128, 128, 4, 2),
            (128, 128, 8, 3),
            (256, 128, 8, 3),
            (256, 256, 8, 3),
            (512, 256, 8, 3),
        )
        candidates: list[AllGatherMatmulConfig] = []
        if "tilelang" in platforms:
            for block_n, block_k, num_warps, num_stages in configs:
                candidates.append(
                    AllGatherMatmulConfig(
                        block_n=block_n,
                        block_k=block_k,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        platform="tilelang",
                        backend="gpu",
                    )
                )
        if "xla" in platforms:
            for block_n, block_k in ((128, 128), (256, 128), (256, 256), (512, 256)):
                candidates.append(
                    AllGatherMatmulConfig(
                        block_n=block_n,
                        block_k=block_k,
                        num_warps=4,
                        num_stages=2,
                        platform="xla",
                        backend="any",
                    )
                )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[AllGatherMatmulConfig, Array]):
        """Return TPU/Pallas candidate configurations for autotuning."""
        x, y, rhs_transpose = _inv_xy_rhs_transpose(inv)
        k = int(x.shape[1])
        n_local = int(y.shape[0] if rhs_transpose else y.shape[1])

        targets = ((128, 128), (256, 128), (256, 256), (512, 256), (512, 512))
        seen: set[tuple[int, int]] = set()
        candidates: list[AllGatherMatmulConfig] = []
        for target_n, target_k in targets:
            block_n = _largest_divisor_leq(n_local, target_n)
            block_k = _largest_divisor_leq(k, target_k)
            key = (block_n, block_k)
            if key in seen:
                continue
            seen.add(key)
            for platform, backend in (("pallas", "tpu"), ("xla", "any")):
                candidates.append(
                    AllGatherMatmulConfig(
                        block_n=block_n,
                        block_k=block_k,
                        num_warps=4,
                        num_stages=2,
                        platform=platform,
                        backend=backend,
                    )
                )

        if not candidates:
            candidates.append(self.heuristic_cfg_tpu(inv))
        return candidates

    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_all_gather_matmul_executor: Executor[AllGatherMatmulConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("all-gather-matmul"),
    )
)


def all_gather_matmul(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    /,
    *,
    rhs_transpose: bool = False,
    collective_id: int | None = 0,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    tp_size: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: AllGatherMatmulConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "m n_local"]:
    """All-gather matmul with automatic backend selection and autotuning.

    Computes ``all_gather(x, axis=0) @ y`` using the best available kernel
    for the current hardware.  When *mesh* is provided the operation is
    executed via ``shard_map``; otherwise it runs inside an existing
    sharded context.

    Note:
        This operation intentionally does not perform runtime fallback
        between distributed backends. If the selected platform cannot
        execute a given shape/constraint, the call fails. Choose
        ``platform``/``cfg.platform`` explicitly for your deployment.

    Args:
        x: Local LHS shard of shape ``(m_local, k)``.
        y: Local RHS shard of shape ``(k, n_local)`` or ``(n_local, k)``
            when *rhs_transpose* is True.
        axis_name: Name of the sharded axis for collective ops.
        rhs_transpose: Whether *y* is stored transposed.
        collective_id: Barrier semaphore allocation id.
        precision: JAX matmul precision.
        tp_size: Tensor-parallel world size.
        platform: Optional platform override.
        cfg: Optional kernel configuration override.
        mesh: If provided, wraps the call in ``shard_map``.
        in_specs: Optional input partition specs for ``shard_map``.
        out_specs: Optional output partition spec for ``shard_map``.

    Returns:
        Result of shape ``(m, n_local)``.
    """
    method = "shard_map" if mesh is not None else None
    if method == "shard_map":
        if in_specs is None:
            if rhs_transpose:
                in_specs = (PartitionSpec(axis_name, None), PartitionSpec(axis_name, None))
            else:
                in_specs = (PartitionSpec(axis_name, None), PartitionSpec(None, axis_name))
        if out_specs is None:
            out_specs = PartitionSpec(None, axis_name)
    return _all_gather_matmul_executor(
        AllGatherMatmul(),
        x=x,
        y=y,
        axis_name=axis_name,
        rhs_transpose=rhs_transpose,
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
