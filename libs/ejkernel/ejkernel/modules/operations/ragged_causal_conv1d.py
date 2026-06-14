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

"""Public ragged causal depthwise conv1d operation.

This module owns the eJKernel operation wrapper for the short causal
depthwise convolution used before GDN/GDR recurrence in packed inference. It
targets continuous batching: decode and prefill requests of different lengths
share one flat token buffer, while ``query_start_loc`` and ``state_indices``
map tokens back to per-request rolling convolution state.

The XLA implementation carries the algorithmic reference. This wrapper owns
executor/config dispatch plus the feature-axis ``shard_map`` path used by
tensor-parallel Qwen3 Next style layouts.
"""

from __future__ import annotations

import os
from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.modules.base import mesh_to_jax_mesh
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
from .configs import RaggedCausalConv1DConfig


def mesh_axis_size(mesh: object | None, axis_name: object | None) -> int:
    """Return the size of ``axis_name`` in a JAX/SpectraX-like mesh.

    Works with any mesh exposing a ``shape`` mapping or sequence. A tuple/list of
    axis names is treated as a combined axis whose size is the product of the
    member sizes. Missing axes and unreadable shapes resolve to ``1``.

    Args:
        mesh: Mesh-like object with a ``shape`` attribute, or ``None``.
        axis_name: Single mesh axis name, a tuple/list of axis names, or
            ``None``.

    Returns:
        The (possibly product) axis size as an ``int``; ``1`` when ``mesh`` or
        ``axis_name`` is ``None``, the mesh has no ``shape``, or the axis is
        absent.
    """
    if mesh is None or axis_name is None:
        return 1
    if isinstance(axis_name, (tuple, list)):
        product = 1
        for name in axis_name:
            product *= mesh_axis_size(mesh, name)
        return int(product)
    shape = getattr(mesh, "shape", None)
    if shape is None:
        return 1
    if hasattr(shape, "get"):
        return int(shape.get(axis_name, 1))
    try:
        return int(shape[axis_name])
    except Exception:
        return 1


def _reorder_concatenated_tensor_for_sharding(
    concatenated_tensor: jax.Array,
    split_sizes: tuple[int, ...],
    n_shards: int,
    dim: int,
) -> jax.Array:
    """Arrange fused feature groups into per-shard interleaved layout.

    Reorders a concatenated feature axis such as ``[A | B | C]`` into
    ``[A0 | B0 | C0 | A1 | B1 | C1 | ...]`` so each tensor-parallel shard gets
    a slice of every original group rather than one contiguous group. Each group
    is split into ``n_shards`` equal chunks along ``dim``, the chunks are
    interleaved, and the result is reshaped back to the original shape.

    Args:
        concatenated_tensor: Tensor whose ``dim`` axis is the concatenation of
            the fused feature groups.
        split_sizes: Sizes of the fused groups along ``dim``; must sum to the
            ``dim`` axis length and each must be divisible by ``n_shards``.
        n_shards: Number of tensor-parallel shards to interleave for.
        dim: Axis along which the groups are concatenated (negative values are
            normalized against ``ndim``).

    Returns:
        A tensor with the same shape and dtype as ``concatenated_tensor`` but
        with the feature groups interleaved per shard.
    """
    if dim < 0:
        dim += concatenated_tensor.ndim
    old_shape = concatenated_tensor.shape
    new_shape = (*old_shape[:dim], int(n_shards), -1, *old_shape[dim + 1 :])
    split_tensors = []
    start_offset = 0
    for split_size in split_sizes:
        split_tensor = jax.lax.slice_in_dim(
            concatenated_tensor,
            start_offset,
            start_offset + int(split_size),
            axis=dim,
        )
        split_tensors.append(split_tensor.reshape(new_shape))
        start_offset += int(split_size)
    reordered_tensor = jnp.concatenate(split_tensors, axis=dim + 1)
    return reordered_tensor.reshape(old_shape)


class RaggedCausalConv1D(Kernel[RaggedCausalConv1DConfig, tuple[Array, Array]]):
    """Executor-driven wrapper for ragged causal depthwise conv1d.

    Implements the :class:`~ejkernel.ops.Kernel` protocol for the short causal
    depthwise convolution applied to a packed (continuously batched) token
    buffer. ``run`` dispatches to the registered backend implementation, while
    ``create_shard_map_wrapper`` builds the executor-compatible sharded path
    for feature/head-axis partitioned packed buffers.

    Attributes:
        version: Cache-busting version tag for the persistent config cache.
    """

    version = "1"

    def __init__(self):
        """Initialize the ragged causal conv1d kernel.

        Registers the operation under ``op_id="ragged_causal_conv1d"`` for
        registry and persistent-cache lookups.
        """
        super().__init__(op_id="ragged_causal_conv1d")

    def get_impl(self, cfg: RaggedCausalConv1DConfig):
        """Resolve the backend implementation for a config.

        Args:
            cfg: Operation config carrying the requested ``platform`` and
                ``backend``.

        Returns:
            The registered callable implementing ``ragged_causal_conv1d`` for the
            detected platform and the config's backend.
        """
        platform = detect_platform("ragged_causal_conv1d", cfg.platform)
        return kernel_registry.get("ragged_causal_conv1d", platform=platform, backend=cfg.backend)

    def create_shard_map_wrapper(
        self,
        x: jnp.ndarray,
        conv_state: jnp.ndarray,
        kernel: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        state_indices: jnp.ndarray,
        distribution: jnp.ndarray,
        *,
        d_conv: int | None = None,
        apply_silu: bool | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedCausalConv1DConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: tuple[PartitionSpec, ...] | None = None,
        check_vma: bool = False,
    ):
        """Create a ``shard_map`` wrapper for executor-driven ragged conv1d.

        Args:
            x: Packed input stream, shape ``[num_tokens, conv_dim]``.
            conv_state: Rolling state pool,
                ``[num_slots, conv_dim, d_conv]``.
            kernel: Depthwise convolution kernel, ``[conv_dim, d_conv]``.
            query_start_loc: CSR-style cumulative request offsets.
            state_indices: Request-to-state-slot mapping.
            distribution: Runtime distribution vector.
            d_conv: Convolution window size.
            apply_silu: Whether to apply SiLU to the output.
            platform: Optional implementation-family override.
            cfg: Optional operation config.
            mesh: Mesh used by ``jax.shard_map``.
            in_specs: Input partition specs matching call arguments.
            out_specs: Output partition specs for ``(output, state)``.
            check_vma: Forwarded to ``jax.shard_map``.

        Returns:
            ``(shard_map_fn, call_args)`` for the executor's
            ``method="shard_map"`` path.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        run_cfg = cfg or RaggedCausalConv1DConfig(
            d_conv=int(d_conv if d_conv is not None else 4),
            apply_silu=bool(apply_silu if apply_silu is not None else True),
            platform=platform or "auto",
        )
        run_d_conv = int(d_conv if d_conv is not None else run_cfg.d_conv)
        run_apply_silu = bool(apply_silu if apply_silu is not None else run_cfg.apply_silu)

        call_args = (x, conv_state, kernel, query_start_loc, state_indices, distribution)
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"

        def _wrapped(local_x, local_state, local_kernel, local_qsl, local_si, local_dist):
            """Run ragged conv1d on a per-shard slice of the packed buffers.

            Args:
                local_x: Per-shard input-stream slice.
                local_state: Per-shard conv-state slice.
                local_kernel: Per-shard kernel slice.
                local_qsl: Replicated query-start-loc offsets.
                local_si: Replicated state-indices mapping.
                local_dist: Replicated distribution vector.

            Returns:
                Tuple ``(output, updated_conv_state)`` for this shard.
            """
            return self.run(
                local_x,
                local_state,
                local_kernel,
                local_qsl,
                local_si,
                local_dist,
                d_conv=run_d_conv,
                apply_silu=run_apply_silu,
                platform=platform,
                cfg=run_cfg,
            )

        shard_map_fn = jax.shard_map(
            _wrapped,
            mesh=mesh_to_jax_mesh(mesh),
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        return shard_map_fn, call_args

    def run(
        self,
        x: jnp.ndarray,
        conv_state: jnp.ndarray,
        kernel: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        state_indices: jnp.ndarray,
        distribution: jnp.ndarray,
        *,
        d_conv: int | None = None,
        apply_silu: bool | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedCausalConv1DConfig,
        **_,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run the selected ragged conv1d backend implementation.

        Args:
            x: Packed input stream, shape ``[num_tokens, conv_dim]``.
            conv_state: Rolling convolution state pool.
            kernel: Depthwise convolution kernel.
            query_start_loc: CSR-style cumulative request offsets.
            state_indices: Request-to-state-slot mapping.
            distribution: Runtime distribution vector whose third entry is
                the number of valid request rows.
            d_conv: Convolution window size.
            apply_silu: Whether to apply SiLU to the output.
            platform: Optional implementation-family override.
            cfg: Required operation config selected by the executor.

        Returns:
            ``(output, updated_conv_state)``.
        """
        if platform is not None:
            cfg = RaggedCausalConv1DConfig(
                d_conv=cfg.d_conv,
                apply_silu=cfg.apply_silu,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            x,
            conv_state,
            kernel,
            query_start_loc,
            state_indices,
            distribution,
            d_conv=int(d_conv if d_conv is not None else cfg.d_conv),
            apply_silu=bool(apply_silu if apply_silu is not None else cfg.apply_silu),
        )

    def heuristic_cfg(self, inv: Invocation[RaggedCausalConv1DConfig, tuple[Array, Array]]):
        """Return the default XLA config for an invocation.

        Reads ``d_conv`` (default ``4``) and ``apply_silu`` (default ``True``)
        from the invocation kwargs and pins the platform to XLA.

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A :class:`RaggedCausalConv1DConfig` on ``platform="xla"`` with
            ``backend="any"``.
        """
        d_conv = int(inv.kwargs.get("d_conv", 4))
        apply_silu = bool(inv.kwargs.get("apply_silu", True))
        return RaggedCausalConv1DConfig(d_conv=d_conv, apply_silu=apply_silu, platform="xla", backend="any")

    def candidate_cfgs(self, inv: Invocation[RaggedCausalConv1DConfig, tuple[Array, Array]]):
        """Return the default candidate config list (heuristic only).

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A single-element list with the heuristic XLA config.
        """
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_gpu(self, inv: Invocation[RaggedCausalConv1DConfig, tuple[Array, Array]]):
        """Return GPU autotuning candidates (heuristic XLA only).

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A single-element list with the heuristic XLA config.
        """
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_tpu(self, inv: Invocation[RaggedCausalConv1DConfig, tuple[Array, Array]]):
        """Return TPU autotuning candidates (heuristic XLA only).

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A single-element list with the heuristic XLA config.
        """
        return [self.heuristic_cfg(inv)]

    candidate_cfgs_xla = candidate_cfgs
    candidate_cfgs_any = candidate_cfgs
    candidate_cfgs_shard_map = candidate_cfgs
    candidate_cfgs_shard_map_xla = candidate_cfgs
    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_executor: Executor[RaggedCausalConv1DConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("ragged_causal_conv1d"),
    )
)


def ragged_causal_conv1d(
    x: jnp.ndarray,
    conv_state: jnp.ndarray,
    kernel: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    *,
    d_conv: int = 4,
    apply_silu: bool = True,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RaggedCausalConv1DConfig | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Execute ragged causal conv1d through the eJKernel operation stack.

    Args:
        x: Packed input stream, shape ``[num_tokens, conv_dim]``.
        conv_state: Per-slot rolling state pool,
            ``[num_slots, conv_dim, d_conv]``.
        kernel: Depthwise convolution kernel, ``[conv_dim, d_conv]``.
        query_start_loc: CSR-style cumulative request offsets.
        state_indices: Request-to-state-slot mapping.
        distribution: Runtime distribution vector; ``distribution[2]`` is the
            number of valid request rows.
        d_conv: Convolution window size.
        apply_silu: Whether to apply SiLU after accumulation.
        platform: Optional implementation-family override.
        cfg: Optional operation config.

    Returns:
        ``(output, updated_conv_state)``.
    """
    return _executor(
        RaggedCausalConv1D(),
        x,
        conv_state,
        kernel,
        query_start_loc,
        state_indices,
        distribution,
        d_conv=d_conv,
        apply_silu=apply_silu,
        platform=platform,
        _cfg=cfg or RaggedCausalConv1DConfig(d_conv=d_conv, apply_silu=apply_silu, platform="xla"),
    )


def ragged_causal_conv1d_head_sharded(
    x: jnp.ndarray,
    conv_state: jnp.ndarray,
    kernel: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    *,
    split_sizes: tuple[int, ...],
    mesh: object | None,
    head_axis: object | None,
    d_conv: int,
    apply_silu: bool = True,
    pre_sharded: bool = False,
    cfg: RaggedCausalConv1DConfig | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run ragged conv1d with optional feature-axis tensor parallelism.

    When a mesh and non-trivial ``head_axis`` are provided and every fused
    feature group in ``split_sizes`` is divisible by the tensor-parallel size,
    the helper routes through the executor's ``method="shard_map"`` path.
    Otherwise it falls back to the unsharded public operation.

    Args:
        x: Packed input stream, shape ``[num_tokens, conv_dim]``.
        conv_state: Rolling convolution state pool.
        kernel: Depthwise convolution kernel.
        query_start_loc: CSR-style cumulative request offsets.
        state_indices: Request-to-state-slot mapping.
        distribution: Runtime distribution vector.
        split_sizes: Sizes of the fused feature groups that form
            ``conv_dim``.
        mesh: Optional mesh carrying ``head_axis``.
        head_axis: Logical mesh axis used to shard the feature dimension.
        d_conv: Convolution window size.
        apply_silu: Whether to apply SiLU after accumulation.
        pre_sharded: Whether ``x`` and ``kernel`` are already in the
            per-shard interleaved layout.
        cfg: Optional operation config.

    Returns:
        ``(output, updated_conv_state)``.
    """
    tp_size = mesh_axis_size(mesh, head_axis)
    can_head_shard = (
        mesh is not None
        and head_axis is not None
        and int(tp_size) > 1
        and all(int(size) % int(tp_size) == 0 for size in split_sizes)
    )
    if not can_head_shard:
        return ragged_causal_conv1d(
            x,
            conv_state,
            kernel,
            query_start_loc,
            state_indices,
            distribution,
            d_conv=d_conv,
            apply_silu=apply_silu,
            cfg=cfg,
        )

    if not pre_sharded:
        x = _reorder_concatenated_tensor_for_sharding(x, split_sizes, int(tp_size), -1)
        kernel = _reorder_concatenated_tensor_for_sharding(kernel, split_sizes, int(tp_size), 0)

    feature_spec = PartitionSpec(None, head_axis)
    state_spec = PartitionSpec(None, head_axis, None)
    kernel_spec = PartitionSpec(head_axis, None)
    replicated = PartitionSpec()
    run_cfg = cfg or RaggedCausalConv1DConfig(d_conv=d_conv, apply_silu=apply_silu, platform="xla")

    return _executor(
        RaggedCausalConv1D(),
        x,
        conv_state,
        kernel,
        query_start_loc,
        state_indices,
        distribution,
        d_conv=d_conv,
        apply_silu=apply_silu,
        method="shard_map",
        mesh=mesh,
        in_specs=(feature_spec, state_spec, kernel_spec, replicated, replicated, replicated),
        out_specs=(feature_spec, state_spec),
        check_vma=False,
        _cfg=run_cfg,
    )


ragged_causal_conv1d_op = ragged_causal_conv1d
