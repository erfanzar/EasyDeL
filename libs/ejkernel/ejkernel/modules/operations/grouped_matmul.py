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


"""Grouped matrix multiplication kernel module with automatic optimization.

This module implements grouped matrix multiplication, an efficient operation for
batched matrix multiplication with variable group sizes. This is particularly
useful for mixture-of-experts models, grouped convolutions, and other scenarios
where different groups of inputs need to be multiplied with different weight matrices.

Key Features:
    - Variable group size support for dynamic routing
    - Configurable tiling (block_m, block_n, block_k) for cache optimization
    - Support for RHS transposition (for weight matrix layouts)
    - Optional output accumulation for multi-pass operations
    - Distributed execution via shard_map wrapper
    - Multiple platform support (Triton/Pallas/CUDA/XLA)

Unlike standard batched matrix multiplication which assumes uniform batch sizes,
grouped matmul handles variable-sized groups efficiently by:
    1. Processing groups of different sizes in a single operation
    2. Optimized memory access patterns for grouped computation
    3. Support for both transposed and non-transposed RHS matrices
    4. Optional accumulation into existing output tensors

Mathematical Formulation:
    Given:
        - lhs: [m, k] where m = sum(group_sizes)
        - rhs: [num_groups, k, n] (or [num_groups, n, k] if transposed)
        - group_sizes: [num_groups] defining partitions of lhs

    Output[i:j, :] = lhs[i:j, :] @ rhs[g, :, :]
    where i:j spans the g-th group

Use Cases:
    - Mixture-of-Experts (MoE) layers with dynamic token routing
    - Grouped linear layers with different weights per group
    - Dynamic routing architectures in transformers
    - Efficient expert parallelism in large models

Example:
    >>> from ejkernel.modules.operations import grouped_matmul
    >>> import jax.numpy as jnp
    >>>
    >>> # MoE-style grouped computation
    >>> lhs = jnp.ones((100, 64))  # 100 tokens, 64 features
    >>> rhs = jnp.ones((4, 64, 128))  # 4 experts, input->output
    >>> group_sizes = jnp.array([30, 25, 25, 20])  # tokens per expert
    >>> output = grouped_matmul(lhs, rhs, group_sizes)  # [100, 128]
    >>>
    >>> # With transposed weights
    >>> rhs_t = jnp.ones((4, 128, 64))  # transposed layout
    >>> output = grouped_matmul(lhs, rhs_t, group_sizes, transpose_rhs=True)

References:
    - Mixture of Experts (Switch Transformer): https://arxiv.org/abs/2101.03961
    - Megablocks: https://arxiv.org/abs/2211.15841
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal

import jax
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, DTypeLike, Float, Int

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
from .configs import GroupedMatmulConfig


class GroupedMatmul(Kernel[GroupedMatmulConfig, Array]):
    """Grouped Matrix Multiplication with custom optimization logic.

    Performs efficient matrix multiplication for grouped inputs, where each group
    can have a different size. This is essential for mixture-of-experts (MoE) models
    where tokens are dynamically routed to different experts.

    Features:
        - Variable group size support via group_sizes array
        - Configurable tiling for memory and compute efficiency
        - Support for RHS transposition
        - Optional output accumulation (for multi-pass operations)
        - Group offset for partial computation
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    Typical use cases:
        - MoE layer computation (different tokens to different experts)
        - Grouped linear layers
        - Dynamic routing architectures
    """

    def __init__(self, version: int = 1):
        """Initialize Grouped Matmul module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.

        Args:
            version: Grouped matmul implementation version to dispatch.
                Supported values are 1, 2, and 3.
        """
        match version:
            case 1:
                op_id = "grouped_matmul"
            case 2:
                op_id = "grouped_matmulv2"
            case 3:
                op_id = "grouped_matmulv3"
            case _:
                raise ValueError(f"Unsupported grouped matmul version: {version}")
        super().__init__(op_id=op_id)

    def get_impl(self, cfg: GroupedMatmulConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for grouped matmul

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def create_shard_map_wrapper(
        self,
        lhs: Float[Array, "m k"],
        rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
        group_sizes: Int[Array, "num_groups_or_shards"],
        preferred_element_type: DTypeLike = jnp.float32,
        group_offset: Int[Array, "..."] | None = None,
        existing_out: Float[Array, "m n"] | None = None,
        rhs_scale: Float[Array, "num_groups num_blocks 1 n"] | None = None,
        rhs_bias: Float[Array, "num_groups 1 n"] | None = None,
        transpose_rhs: bool = False,
        interpret: bool = False,
        do_padding: bool = True,
        precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
        out_shard_callback: Callable[[Float[Array, "m n"]], Float[Array, "m n"]] | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: GroupedMatmulConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed grouped matmul execution.

        Wraps the grouped matrix multiplication operation for execution across
        a distributed device mesh using JAX's shard_map primitive.

        Args:
            lhs: Left-hand side matrix [m, k] (typically activations)
            rhs: Right-hand side grouped matrices [num_groups, k, n] or [num_groups, n, k]
            group_sizes: Size of each group [num_groups], sum(group_sizes) == m
            preferred_element_type: Preferred dtype for computation (default: float32)
            group_offset: Optional offset into groups for partial computation
            existing_out: Optional existing output to accumulate into [m, n]
            transpose_rhs: Whether RHS matrices are transposed
            interpret: Use interpreted mode (for debugging)
            do_padding: Whether to pad lhs to block_m alignment
            precision: Computation precision setting
            out_shard_callback: Optional callback to apply to output after computation
            platform: Platform to use for kernel execution
            cfg: Configuration for the kernel (block sizes, etc.)
            mesh: JAX device mesh for distributed execution
            in_specs: Input partition specs (must match length of tensor args)
            out_specs: Output partition spec
            check_vma: Whether to check for valid memory access patterns

        Returns:
            Tuple of (shard_map_fn, call_args, callback) where callback handles
            unpadding of the result if padding was applied
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        mSize, padded_size = lhs.shape[0], 0
        if mSize % cfg.block_m:
            padded_size = cfg.block_m - mSize % cfg.block_m
            lhs = jax.lax.pad(lhs, jnp.array(0.0, dtype=lhs.dtype), [(0, padded_size, 0), (0, 0, 0)])

        def _wrapped_grouped_matmul(
            lhs: Float[Array, "m k"],
            rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
            group_sizes: Int[Array, "num_groups_or_shards"],
        ) -> Float[Array, "m n"]:
            """Shard-local grouped matmul forwarding to self.run."""
            out = self.run(
                lhs=lhs,
                rhs=rhs,
                group_sizes=group_sizes,
                preferred_element_type=preferred_element_type,
                group_offset=group_offset,
                existing_out=existing_out,
                rhs_scale=rhs_scale,
                rhs_bias=rhs_bias,
                transpose_rhs=transpose_rhs,
                interpret=interpret,
                precision=precision,
                platform=platform,
                do_padding=False,
                cfg=cfg or self.heuristic_cfg(None),
            )
            if out_shard_callback is not None:
                out = out_shard_callback(out)
            return out

        call_args = (lhs, rhs, group_sizes)

        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"

        shard_map_fn = shard_map(
            _wrapped_grouped_matmul,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        def callback(out, cfg):
            """Strip padding rows from the output if padding was applied."""
            if padded_size > 0:
                out = out[:mSize]
            return out

        return shard_map_fn, call_args, callback

    def run(
        self,
        lhs: Float[Array, "m k"],
        rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
        group_sizes: Int[Array, "num_groups_or_shards"],
        preferred_element_type: DTypeLike = jnp.float32,
        group_offset: Int[Array, "..."] | None = None,
        existing_out: Float[Array, "m n"] | None = None,
        rhs_scale: Float[Array, "num_groups num_blocks 1 n"] | None = None,
        rhs_bias: Float[Array, "num_groups 1 n"] | None = None,
        transpose_rhs: bool = False,
        interpret: bool = False,
        do_padding: bool = True,
        precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
        out_shard_callback: Callable[[Float[Array, "m n"]], Float[Array, "m n"]] | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: GroupedMatmulConfig,
    ) -> Float[Array, "m n"]:
        """Execute grouped matrix multiplication.

        Tiling is computed internally from ``cfg.block_m/block_n/block_k``
        (clamped to the actual matrix dimensions). The ``lhs`` rows may be
        padded before the call and the padding is stripped from the output if
        ``do_padding=True``.

        Args:
            lhs: Left-hand side matrix [m, k] (activations).
            rhs: Right-hand side matrices [num_groups, k, n] or
                [num_groups, n, k] when ``transpose_rhs=True``.
            group_sizes: Number of rows per group [num_groups].
                Must satisfy ``sum(group_sizes) == m``.
            preferred_element_type: Accumulator/output dtype (default:
                ``jnp.float32``).
            group_offset: Optional scalar index of the first active group.
            existing_out: Optional pre-allocated output [m, n] to accumulate
                into.
            rhs_scale: Dequantisation scale, ``grouped_matmulv3`` only
                [num_groups, num_blocks, 1, n].
            rhs_bias: Per-group bias, ``grouped_matmulv3`` only
                [num_groups, 1, n].
            transpose_rhs: RHS is stored transposed (default: False).
            interpret: Pallas interpreted mode for debugging (default: False).
            do_padding: Pad lhs and trim output to block_m alignment
                (default: True).
            precision: JAX matmul precision (default: ``Precision.DEFAULT``).
            out_shard_callback: Callback applied to output shard inside
                ``shard_map``.
            platform: Optional platform override.
            cfg: Kernel configuration providing block sizes and platform.

        Returns:
            Matrix multiplication result [m, n].

        Raises:
            ValueError: If ``rhs_scale``/``rhs_bias`` are passed to a v1/v2
                kernel.
        """
        if preferred_element_type is None:
            preferred_element_type = jnp.float32

        if self.op_id != "grouped_matmulv3" and (rhs_scale is not None or rhs_bias is not None):
            raise ValueError("rhs_scale and rhs_bias are only supported by grouped_matmulv3.")

        if platform is not None:
            cfg = GroupedMatmulConfig(
                block_m=cfg.block_m,
                block_n=cfg.block_n,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
                bypass_xla_tiling=cfg.bypass_xla_tiling,
            )

        resolved_platform = detect_platform(self.op_id, cfg.platform)
        impl = self.get_impl(cfg)
        tiling = None
        mSize, kSize, nSize = lhs.shape[0], lhs.shape[1], rhs.shape[2]
        padded_size = 0

        if cfg.bypass_xla_tiling and resolved_platform == "xla":
            ...
        else:
            if do_padding:
                if mSize % cfg.block_m:
                    padded_size = cfg.block_m - mSize % cfg.block_m
                    lhs = jax.lax.pad(lhs, jnp.array(0.0, dtype=lhs.dtype), [(0, padded_size, 0), (0, 0, 0)])
            tiling = (min(cfg.block_m, mSize), min(cfg.block_k, kSize), min(cfg.block_n, nSize))

        impl_kwargs = dict(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            preferred_element_type=preferred_element_type,
            tiling=tiling,
            group_offset=group_offset,
            existing_out=existing_out,
            transpose_rhs=transpose_rhs,
            interpret=interpret,
            precision=precision,
        )
        if self.op_id == "grouped_matmulv3":
            impl_kwargs["rhs_scale"] = rhs_scale
            impl_kwargs["rhs_bias"] = rhs_bias
        out = impl(**impl_kwargs)
        if do_padding and padded_size > 0:
            out = out[:mSize]
        return out

    def heuristic_cfg(self, inv: Invocation[GroupedMatmulConfig, Array]) -> GroupedMatmulConfig:
        """Provide default configuration with block sizes.

        Selects balanced block sizes suitable for typical grouped matmul workloads.
        The default 128x128x128 tiling provides good cache utilization for most
        problem sizes.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with 128x128x128 blocks, 4 warps, 2 stages
        """
        return GroupedMatmulConfig(
            block_m=128,
            block_n=128,
            block_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[GroupedMatmulConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates configurations with different block sizes to explore the
        performance space. Grouped matmul benefits from various tile sizes
        depending on group size distribution and matrix dimensions.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations

        """
        block_configs = [
            (128, 128, 128),
            (256, 256, 128),
            (512, 512, 128),
            (512, 512, 256),
            (512, 512, 512),
            (1024, 1024, 128),
            (1024, 1024, 256),
            (1024, 1024, 512),
            (1024, 1024, 1024),
        ]

        candidates = []
        for block_m, block_n, block_k in block_configs:
            candidates.append(
                GroupedMatmulConfig(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=None,
                    num_stages=None,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[GroupedMatmulConfig, Array]):
        """Generate GPU candidates for grouped matmul across TileLang and XLA.

        Grouped matmul (MoE) tile selection on H100:

        * ``block_m`` ∈ {64, 128, 256} — driven by avg group size.
        * ``block_n`` ∈ {64, 128, 256} — N is usually fixed (intermediate
          dim or model dim); the sweep finds the best within the budget.
        * ``block_k`` ∈ {32, 64, 128} — K must divide cleanly; 128 prefers
          a deep pipeline.
        * ``num_warps`` ∈ {4, 8}; ``num_stages`` ∈ {2, 3, 4}.

        XLA path also gets a wider sweep since XLA's tiling hint is
        passed through to its grouped-matmul implementation.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        tilelang_configs = [
            (64, 128, 64, 4, 2),
            (128, 64, 64, 4, 2),
            (128, 128, 32, 4, 2),
            (128, 128, 64, 4, 2),
            (128, 128, 64, 8, 3),
            (128, 256, 64, 8, 3),
            (256, 128, 64, 8, 3),
            (256, 128, 32, 8, 3),
            (256, 256, 64, 8, 3),
            (128, 128, 128, 4, 3),
        ]
        xla_configs = [
            (128, 128, 128),
            (128, 256, 128),
            (256, 256, 128),
            (256, 256, 256),
            (512, 512, 128),
            (512, 512, 256),
            (1024, 1024, 256),
        ]
        candidates: list[GroupedMatmulConfig] = []
        if "tilelang" in platforms:
            for block_m, block_n, block_k, num_warps, num_stages in tilelang_configs:
                candidates.append(
                    GroupedMatmulConfig(
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
                    GroupedMatmulConfig(
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        num_warps=None,
                        num_stages=None,
                        platform="xla",
                        backend="any",
                    )
                )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[GroupedMatmulConfig, Array]):
        """Generate TPU candidates for Pallas and XLA grouped matmul."""
        block_configs = [
            (128, 128, 128),
            (256, 256, 128),
            (512, 512, 128),
            (512, 512, 256),
            (1024, 1024, 256),
        ]
        return [
            GroupedMatmulConfig(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=None,
                num_stages=None,
                platform=platform,
                backend=backend,
            )
            for platform, backend in (("pallas", "tpu"), ("xla", "any"))
            for block_m, block_n, block_k in block_configs
        ]


_grouped_matmul_executor: Executor[GroupedMatmulConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("grouped-matmul"),
    )
)


def grouped_matmul(
    lhs: Float[Array, "m k"],
    rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
    group_sizes: Int[Array, "num_groups_or_shards"],
    group_offset: Int[Array, "..."] | None = None,
    existing_out: Float[Array, "m n"] | None = None,
    /,
    *,
    preferred_element_type: DTypeLike = jnp.float32,
    rhs_scale: Float[Array, "num_groups num_blocks 1 n"] | None = None,
    rhs_bias: Float[Array, "num_groups 1 n"] | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    do_padding: bool = True,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    use_v2: bool = False,
    use_v3: bool = False,
    out_shard_callback: Callable[[Float[Array, "m n"]], Float[Array, "m n"]] | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: GroupedMatmulConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "m n"]:
    """Execute grouped matrix multiplication with automatic optimization.

    Performs efficient batched matrix multiplication with variable group sizes,
    optimised for Mixture-of-Experts routing where different token groups are
    multiplied by different weight matrices.

    Tiling is derived automatically from ``cfg.block_m``, ``cfg.block_n``,
    and ``cfg.block_k`` (or the autotuned equivalent). LHS rows are optionally
    padded to ``block_m`` alignment before calling the kernel and trimmed
    from the output afterwards.

    Args:
        lhs: Left-hand side activation matrix [m, k], where
            ``m = sum(group_sizes)``.
        rhs: Right-hand side grouped weight matrices.
            Shape ``[num_groups, k, n]`` normally, or ``[num_groups, n, k]``
            when ``transpose_rhs=True``.
        group_sizes: Number of rows in ``lhs`` belonging to each group.
            Shape ``[num_groups]``.  Must sum to ``m``.
        group_offset: Optional scalar index of the first active group.
            Used for partial computation over a subset of groups.
        existing_out: Optional pre-allocated output buffer [m, n] to
            accumulate results into instead of allocating fresh storage.
        preferred_element_type: Dtype for the accumulator/output
            (default: ``jnp.float32``).
        rhs_scale: Per-block dequantisation scale used by ``grouped_matmulv3``
            only.  Shape ``[num_groups, num_blocks, 1, n]``.
            Raises ``ValueError`` if passed to v1/v2.
        rhs_bias: Per-group bias added after the matmul, used by
            ``grouped_matmulv3`` only.  Shape ``[num_groups, 1, n]``.
            Raises ``ValueError`` if passed to v1/v2.
        transpose_rhs: If True, ``rhs`` is stored as ``[num_groups, n, k]``
            and will be transposed before the matmul (default: False).
        interpret: Run in Pallas interpreted mode for debugging (default: False).
        do_padding: Pad ``lhs`` rows to ``block_m`` alignment before the
            kernel call and strip padding from the output (default: True).
        precision: JAX matmul precision (default: ``Precision.DEFAULT``).
        use_v2: Use ``grouped_matmulv2`` kernel (default: False).
        use_v3: Use ``grouped_matmulv3`` kernel with ``rhs_scale``/``rhs_bias``
            support (default: False).  Mutually exclusive with ``use_v2``.
        out_shard_callback: Optional function applied to the output shard
            inside ``shard_map`` before returning.
        platform: Force a specific platform
            (``"triton"``, ``"pallas"``, ``"cuda"``, ``"xla"``, ``"auto"``).
        cfg: Optional ``GroupedMatmulConfig`` override.
        mesh: JAX device mesh for ``shard_map`` distributed execution.
        in_specs: Input partition specs for ``shard_map``.
        out_specs: Output partition spec for ``shard_map``.

    Returns:
        Matrix multiplication result [m, n].

    Raises:
        ValueError: If both ``use_v2`` and ``use_v3`` are True, or if
            ``rhs_scale``/``rhs_bias`` are passed to a v1/v2 kernel.

    Example:
        >>> lhs = jnp.ones((100, 64))          # 100 tokens, 64 features
        >>> rhs = jnp.ones((4, 64, 128))        # 4 experts, 64→128
        >>> group_sizes = jnp.array([30, 25, 25, 20])
        >>>
        >>> out = grouped_matmul(lhs, rhs, group_sizes)            # [100, 128]
        >>>
        >>> rhs_t = jnp.ones((4, 128, 64))
        >>> out = grouped_matmul(lhs, rhs_t, group_sizes, transpose_rhs=True)
        >>>
        >>> out = grouped_matmul(lhs, rhs, group_sizes, existing_out=prev_out)
        >>>
        >>> out = grouped_matmul(lhs, rhs, group_sizes, platform="pallas")
    """

    if use_v2 and use_v3:
        raise ValueError("use_v2 and use_v3 are mutually exclusive.")

    version = 3 if use_v3 else 2 if use_v2 else 1

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    return _grouped_matmul_executor(
        GroupedMatmul(version=version),
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
        existing_out=existing_out,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
        do_padding=do_padding,
        precision=precision,
        out_shard_callback=out_shard_callback,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )
