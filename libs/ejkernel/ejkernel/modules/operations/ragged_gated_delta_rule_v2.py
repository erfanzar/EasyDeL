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

"""Public ragged GDN v2 operation for packed inference.

This module exposes the eJKernel operation surface used by EasyDeL's
continuous-batching GDN path. The backend implementations live under
``ejkernel.kernels``; this file owns the public ``Kernel`` wrapper, executor,
configuration dispatch, and optional tensor-parallel ``shard_map`` routing.

The operation consumes a packed mixed-QKV token buffer plus per-token GDN gates
and a recurrent state pool. It returns the updated recurrent state pool and the
packed token output. On TPU, the selected Pallas implementation may use the
fused recurrent scan and decode kernels. On XLA, the wrapper routes to the
portable reference implementation.
"""

from __future__ import annotations

import os
from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array

from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2 import (
    KERNEL_TILE_POLICIES,
    KernelTilePolicy,
    normalize_kernel_tile_policy,
    ragged_gated_delta_rule_decode_only,
    ragged_gated_delta_rule_mixed_prefill,
    set_gdn_kernel_tile_policy,
)
from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.kernels._xla.ragged_gated_delta_rule_v2._xla_impl_fwd import ragged_gated_delta_rule
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
from .configs import RaggedGatedDeltaRuleV2Config


def _normalize_runtime_dtype(runtime_dtype: object | None) -> jnp.dtype | None:
    """Coerce a runtime dtype specifier into a canonical ``jnp.dtype``.

    Args:
        runtime_dtype: Optional dtype specifier (e.g. a ``jnp.dtype``, NumPy
            dtype, type, or dtype-like string). ``None`` selects the backend
            default and is passed through unchanged.

    Returns:
        The normalized ``jnp.dtype`` for a non-``None`` input, or ``None`` when
        ``runtime_dtype`` is ``None``.
    """
    if runtime_dtype is None:
        return None
    return jnp.dtype(runtime_dtype)


def mesh_axis_size(mesh: object | None, axis_name: object | None) -> int:
    """Return the size of ``axis_name`` in a JAX/SpectraX-like mesh.

    The mesh size is read from its ``shape`` mapping, supporting both
    ``dict``-like ``shape`` objects (via ``shape.get``) and indexable shapes.
    Tuple/list axis names are treated as a composite axis whose size is the
    product of the named axes.

    Args:
        mesh: Mesh-like object exposing a ``shape`` attribute, or ``None``.
        axis_name: Axis name, a tuple/list of axis names for a composite axis,
            or ``None``.

    Returns:
        The size of the requested axis. Returns ``1`` when ``mesh`` or
        ``axis_name`` is ``None``, when the mesh has no ``shape``, or when the
        axis cannot be resolved (missing key or indexing failure).
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
    """Reorder a fused ``[A|B|C]`` feature axis into per-shard interleaved layout.

    Given a tensor whose ``dim`` axis concatenates several feature blocks (e.g.
    ``[A|B|C]``), each block is split into ``n_shards`` contiguous chunks and the
    chunks are interleaved so the axis becomes ``[A0|B0|C0|A1|B1|C1|...]``. This
    matches the layout expected when the fused axis is later sharded into
    ``n_shards`` tensor-parallel partitions, so each shard holds a contiguous
    ``A``/``B``/``C`` slice.

    Args:
        concatenated_tensor: Tensor whose ``dim`` axis holds the fused feature
            blocks.
        split_sizes: Sizes of the consecutive feature blocks along ``dim`` (e.g.
            ``(key_dim, key_dim, value_dim)``). Their sum must equal the length
            of ``dim``, and each must be divisible by ``n_shards``.
        n_shards: Number of tensor-parallel shards to interleave for.
        dim: Axis carrying the fused features; negative values are normalized
            against ``concatenated_tensor.ndim``.

    Returns:
        A tensor with the same shape as ``concatenated_tensor`` whose ``dim``
        axis is reordered into per-shard interleaved layout.
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


class RaggedGatedDeltaRuleV2(Kernel[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]]):
    """Executor-driven wrapper for packed mixed-QKV ragged GDN.

    The class separates public operation concerns from backend algorithm code:
    ``get_impl`` resolves the registered backend implementation, ``run`` calls
    it with normalized static arguments, and ``create_shard_map_wrapper`` builds
    executor-compatible sharding wrappers for tensor-parallel head layouts.
    """

    version = "1"

    def __init__(self):
        """Initialize the operation with its registry id ``ragged_gated_delta_rule_v2``."""
        super().__init__(op_id="ragged_gated_delta_rule_v2")

    def get_impl(self, cfg: RaggedGatedDeltaRuleV2Config):
        """Resolve the registered backend implementation for a configuration.

        The effective platform is resolved by ``detect_platform`` from
        ``cfg.platform`` (which may be ``"auto"``), and the resulting
        ``(platform, backend)`` pair is looked up in the kernel registry.

        Args:
            cfg: Operation configuration carrying ``platform`` and ``backend``
                used for registry dispatch.

        Returns:
            The callable backend implementation registered for
            ``ragged_gated_delta_rule_v2`` under the resolved platform/backend.
        """
        platform = detect_platform("ragged_gated_delta_rule_v2", cfg.platform)
        return kernel_registry.get("ragged_gated_delta_rule_v2", platform=platform, backend=cfg.backend)

    def create_shard_map_wrapper(
        self,
        mixed_qkv: jnp.ndarray,
        b: jnp.ndarray,
        a: jnp.ndarray,
        recurrent_state: jnp.ndarray,
        A_log: jnp.ndarray,
        dt_bias: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        state_indices: jnp.ndarray,
        distribution: jnp.ndarray,
        has_initial_state: jnp.ndarray | None = None,
        *,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        chunk_size: int | None = None,
        use_qk_norm_in_gdn: bool = True,
        apply_silu_in_gdr: bool = False,
        use_recurrent_scan_prefill: bool = False,
        runtime_dtype: object | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        split_qkv_for_head_shard: bool = False,
        cfg: RaggedGatedDeltaRuleV2Config | None = None,
        mesh: object | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: tuple[PartitionSpec, ...] | None = None,
        check_vma: bool = False,
    ):
        """Create a ``shard_map`` wrapper for tensor-parallel ragged GDN v2.

        Args:
            mixed_qkv: Packed token buffer with concatenated query, key, and
                value projections.
            b: Per-token beta gate input, shape ``[tokens, n_v]``.
            a: Per-token decay gate input, shape ``[tokens, n_v]``.
            recurrent_state: State pool, shape
                ``[num_slots, n_v, d_k, d_v]``.
            A_log: Per-value-head log-``A`` (log-decay) parameter.
            dt_bias: Per-value-head ``dt`` bias.
            query_start_loc: CSR-style request offsets.
            state_indices: Request-to-state-slot mapping.
            distribution: Packed runtime distribution: int32 ``[decode_end,
                prefill_end, total]`` triple, where ``total`` is the number of
                valid sequences.
            has_initial_state: Optional per-request state validity mask. When
                omitted, every request is treated as having an initial state.
            n_kq: Number of query/key heads.
            n_v: Number of value heads.
            d_k: Query/key head dimension.
            d_v: Value head dimension.
            chunk_size: Prefill chunk size.
            use_qk_norm_in_gdn: Whether to normalize Q/K in the GDN recurrence.
            apply_silu_in_gdr: Whether to apply SiLU to the mixed QKV input.
            use_recurrent_scan_prefill: Whether to route prefill through the
                TPU recurrent-scan implementation when selected by backend.
            runtime_dtype: Optional runtime dtype override.
            platform: Optional implementation family override.
            split_qkv_for_head_shard: If true, split the mixed projection into
                Q/K/V before sharding the head axis.
            cfg: Optional operation config.
            mesh: Mesh used by ``jax.shard_map``.
            in_specs: Input partition specs matching the call arguments.
            out_specs: Output partition specs for ``(state, output)``.
            check_vma: Forwarded to ``jax.shard_map``.

        Returns:
            A tuple accepted by the executor's ``method="shard_map"`` path:
            either ``(shard_map_fn, call_args)`` or
            ``(shard_map_fn, call_args, output_callback)`` for split-QKV output
            flattening.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        if has_initial_state is None:
            has_initial_state = jnp.ones(state_indices.shape[0], dtype=jnp.bool_)
            if len(in_specs) in (9, 11):
                in_specs = (*in_specs, PartitionSpec())

        run_cfg = cfg or RaggedGatedDeltaRuleV2Config(
            chunk_size=int(chunk_size if chunk_size is not None else 64),
            platform=platform or "auto",
        )
        run_chunk_size = int(chunk_size if chunk_size is not None else run_cfg.chunk_size)

        if split_qkv_for_head_shard:
            num_tokens = mixed_qkv.shape[0]
            key_dim = int(n_kq * d_k)
            query = mixed_qkv[..., :key_dim].reshape(num_tokens, n_kq, d_k)
            key = mixed_qkv[..., key_dim : key_dim * 2].reshape(num_tokens, n_kq, d_k)
            value = mixed_qkv[..., key_dim * 2 :].reshape(num_tokens, n_v, d_v)

            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                query = jnp.repeat(query, repeat_factor, axis=1)
                key = jnp.repeat(key, repeat_factor, axis=1)

            call_args = (
                query,
                key,
                value,
                b,
                a,
                recurrent_state,
                A_log,
                dt_bias,
                query_start_loc,
                state_indices,
                distribution,
                has_initial_state,
            )
            assert len(in_specs) == len(
                call_args
            ), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"

            def _wrapped_split(
                local_query,
                local_key,
                local_value,
                local_b,
                local_a,
                local_state,
                local_A_log,
                local_dt_bias,
                local_qsl,
                local_si,
                local_dist,
                local_has_initial_state,
            ):
                """Run the GDN v2 backend on one head-parallel shard of split Q/K/V projections.

                Used as the ``jax.shard_map`` body for the ``split_qkv_for_head_shard`` path. Each shard holds a
                slice of the head axis for the separately-passed query/key/value tensors (with the key/query group
                already repeated to match the value heads). The closure re-concatenates them into the packed
                ``mixed_qkv`` layout that :meth:`run` expects, infers the per-shard head counts and head dimensions
                from the local tensor shapes, and dispatches to :meth:`run` with the captured runtime options
                (``run_chunk_size``, ``use_qk_norm_in_gdn``, ``apply_silu_in_gdr``, ``use_recurrent_scan_prefill``,
                ``runtime_dtype``, ``platform``, ``run_cfg``).

                Args:
                    local_query: This shard's query heads, shape ``[tokens, local_n_kq, d_k]``.
                    local_key: This shard's key heads, shape ``[tokens, local_n_kq, d_k]``.
                    local_value: This shard's value heads, shape ``[tokens, local_n_v, d_v]``.
                    local_b: This shard's per-token beta gate input.
                    local_a: This shard's per-token decay gate input.
                    local_state: This shard's slice of the recurrent state pool.
                    local_A_log: This shard's per-value-head log-decay parameter.
                    local_dt_bias: This shard's per-value-head ``dt`` bias.
                    local_qsl: CSR-style request start offsets (replicated).
                    local_si: Request-to-state-slot mapping (replicated).
                    local_dist: Packed runtime distribution triple (replicated).
                    local_has_initial_state: Per-request state validity mask (replicated).

                Returns:
                    A tuple ``(new_state, output)`` where ``new_state`` is the updated state slice and ``output`` is
                    reshaped to ``[tokens, local_n_v, d_v]`` so the value head axis can be sharded by ``out_specs``.
                """
                local_n_kq = local_query.shape[1]
                local_n_v = local_value.shape[1]
                local_d_k = local_query.shape[2]
                local_d_v = local_value.shape[2]
                local_mixed_qkv = jnp.concatenate(
                    [
                        local_query.reshape(local_query.shape[0], -1),
                        local_key.reshape(local_key.shape[0], -1),
                        local_value.reshape(local_value.shape[0], -1),
                    ],
                    axis=-1,
                )
                new_state, output = self.run(
                    mixed_qkv=local_mixed_qkv,
                    b=local_b,
                    a=local_a,
                    recurrent_state=local_state,
                    A_log=local_A_log,
                    dt_bias=local_dt_bias,
                    query_start_loc=local_qsl,
                    state_indices=local_si,
                    distribution=local_dist,
                    has_initial_state=local_has_initial_state,
                    n_kq=local_n_kq,
                    n_v=local_n_v,
                    d_k=local_d_k,
                    d_v=local_d_v,
                    chunk_size=run_chunk_size,
                    use_qk_norm_in_gdn=use_qk_norm_in_gdn,
                    apply_silu_in_gdr=apply_silu_in_gdr,
                    use_recurrent_scan_prefill=use_recurrent_scan_prefill,
                    runtime_dtype=runtime_dtype,
                    platform=platform,
                    cfg=run_cfg,
                )
                return new_state, output.reshape(output.shape[0], local_n_v, local_d_v)

            shard_map_fn = jax.shard_map(
                _wrapped_split,
                mesh=mesh_to_jax_mesh(mesh),
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=check_vma,
            )

            def _flatten_output(outs, cfg):
                """Flatten the per-head-sharded shard_map output back to the packed value layout.

                Output callback returned alongside the split-QKV ``shard_map_fn``. After ``shard_map`` gathers the
                value head axis, this collapses the ``[tokens, n_v, d_v]`` output produced by :func:`_wrapped_split`
                into the flat ``[tokens, n_v * d_v]`` layout expected by callers, leaving the recurrent state
                unchanged.

                Args:
                    outs: The ``(new_state, output)`` tuple produced by the shard_map call.
                    cfg: Operation config supplied by the executor; unused and ignored.

                Returns:
                    A tuple ``(new_state, flattened_output)`` with ``output`` reshaped to ``[tokens, -1]``.
                """
                _ = cfg
                new_state, output = outs
                return new_state, output.reshape(output.shape[0], -1)

            return shard_map_fn, call_args, _flatten_output

        call_args = (
            mixed_qkv,
            b,
            a,
            recurrent_state,
            A_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"

        def _wrapped(
            local_mixed_qkv,
            local_b,
            local_a,
            local_state,
            local_A_log,
            local_dt_bias,
            local_qsl,
            local_si,
            local_dist,
            local_has_initial_state,
        ):
            """Run the GDN v2 backend on one tensor-parallel shard of the packed ``mixed_qkv`` buffer.

            Used as the ``jax.shard_map`` body for the default (non-split) path. Each shard receives its slice of
            the packed query/key/value buffer and recurrent state and dispatches directly to :meth:`run`, forwarding
            the captured head geometry (``n_kq``, ``n_v``, ``d_k``, ``d_v``) and runtime options (``run_chunk_size``,
            ``use_qk_norm_in_gdn``, ``apply_silu_in_gdr``, ``use_recurrent_scan_prefill``, ``runtime_dtype``,
            ``platform``, ``run_cfg``).

            Args:
                local_mixed_qkv: This shard's packed query/key/value token buffer.
                local_b: This shard's per-token beta gate input.
                local_a: This shard's per-token decay gate input.
                local_state: This shard's slice of the recurrent state pool.
                local_A_log: This shard's per-value-head log-decay parameter.
                local_dt_bias: This shard's per-value-head ``dt`` bias.
                local_qsl: CSR-style request start offsets (replicated).
                local_si: Request-to-state-slot mapping (replicated).
                local_dist: Packed runtime distribution triple (replicated).
                local_has_initial_state: Per-request state validity mask (replicated).

            Returns:
                The ``(new_state, output)`` tuple returned by :meth:`run` for this shard.
            """
            return self.run(
                mixed_qkv=local_mixed_qkv,
                b=local_b,
                a=local_a,
                recurrent_state=local_state,
                A_log=local_A_log,
                dt_bias=local_dt_bias,
                query_start_loc=local_qsl,
                state_indices=local_si,
                distribution=local_dist,
                has_initial_state=local_has_initial_state,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                chunk_size=run_chunk_size,
                use_qk_norm_in_gdn=use_qk_norm_in_gdn,
                apply_silu_in_gdr=apply_silu_in_gdr,
                use_recurrent_scan_prefill=use_recurrent_scan_prefill,
                runtime_dtype=runtime_dtype,
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
        mixed_qkv: jnp.ndarray,
        b: jnp.ndarray,
        a: jnp.ndarray,
        recurrent_state: jnp.ndarray,
        A_log: jnp.ndarray,
        dt_bias: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        state_indices: jnp.ndarray,
        distribution: jnp.ndarray,
        has_initial_state: jnp.ndarray | None = None,
        *,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        chunk_size: int | None = None,
        use_qk_norm_in_gdn: bool = True,
        apply_silu_in_gdr: bool = False,
        use_recurrent_scan_prefill: bool = False,
        runtime_dtype: object | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: RaggedGatedDeltaRuleV2Config,
        **_,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run the selected ragged GDN v2 backend implementation.

        Args:
            mixed_qkv: Packed token buffer with shape
                ``[tokens, 2 * n_kq * d_k + n_v * d_v]``.
            b: Per-token beta gate input.
            a: Per-token decay gate input.
            recurrent_state: Recurrent state pool.
            A_log: Per-value-head log-``A`` parameter.
            dt_bias: Per-value-head ``dt`` bias.
            query_start_loc: CSR-style request offsets.
            state_indices: Request-to-state-slot mapping.
            distribution: Runtime distribution vector for decode and prefill.
            has_initial_state: Optional per-request state validity mask.
            n_kq: Number of query/key heads.
            n_v: Number of value heads.
            d_k: Query/key head dimension.
            d_v: Value head dimension.
            chunk_size: Prefill chunk size.
            use_qk_norm_in_gdn: Whether to normalize Q/K in the recurrence.
            apply_silu_in_gdr: Whether to apply SiLU to mixed QKV.
            use_recurrent_scan_prefill: Whether to enable recurrent-scan
                prefill in backends that support it.
            runtime_dtype: Optional dtype override applied by the backend.
            platform: Optional implementation family override.
            cfg: Required operation config selected by the executor.

        Returns:
            ``(updated_recurrent_state, output)``.
        """
        runtime_dtype = _normalize_runtime_dtype(runtime_dtype)
        if platform is not None:
            cfg = RaggedGatedDeltaRuleV2Config(
                chunk_size=cfg.chunk_size,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            recurrent_state=recurrent_state,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            has_initial_state=has_initial_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            chunk_size=int(chunk_size if chunk_size is not None else cfg.chunk_size),
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            apply_silu_in_gdr=apply_silu_in_gdr,
            use_recurrent_scan_prefill=use_recurrent_scan_prefill,
            runtime_dtype=runtime_dtype,
        )

    def _run_unsharded(
        self,
        mixed_qkv: jnp.ndarray,
        b: jnp.ndarray,
        a: jnp.ndarray,
        recurrent_state: jnp.ndarray,
        A_log: jnp.ndarray,
        dt_bias: jnp.ndarray,
        query_start_loc: jnp.ndarray,
        state_indices: jnp.ndarray,
        distribution: jnp.ndarray,
        has_initial_state: jnp.ndarray | None,
        *,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        chunk_size: int,
        use_qk_norm_in_gdn: bool,
        apply_silu_in_gdr: bool,
        use_recurrent_scan_prefill: bool,
        runtime_dtype: object | None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None,
        cfg: RaggedGatedDeltaRuleV2Config | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run ragged GDN v2 through the normal executor path.

        This method is used when no tensor-parallel head sharding is requested
        or possible. It exists on the operation class so there is no separate
        module-level bypass around the executor/config machinery. All arguments
        are forwarded verbatim to the executor; when ``cfg`` is ``None`` a default
        config is built from ``chunk_size`` and ``platform``.

        Args:
            mixed_qkv: Packed token buffer with shape
                ``[tokens, 2 * n_kq * d_k + n_v * d_v]``.
            b: Per-token beta gate input.
            a: Per-token decay gate input.
            recurrent_state: Recurrent state pool, shape ``[num_slots, n_v, d_k, d_v]``.
            A_log: Per-value-head log-``A`` parameter.
            dt_bias: Per-value-head ``dt`` bias.
            query_start_loc: CSR-style request offsets.
            state_indices: Request-to-state-slot mapping.
            distribution: Runtime distribution vector for decode and prefill.
            has_initial_state: Optional per-request state validity mask.
            n_kq: Number of query/key heads.
            n_v: Number of value heads.
            d_k: Query/key head dimension.
            d_v: Value head dimension.
            chunk_size: Prefill chunk size.
            use_qk_norm_in_gdn: Whether to normalize Q/K in the recurrence.
            apply_silu_in_gdr: Whether to apply SiLU to the mixed QKV input.
            use_recurrent_scan_prefill: Whether to enable recurrent-scan prefill
                in backends that support it.
            runtime_dtype: Optional dtype override applied by the backend.
            platform: Optional implementation family override.
            cfg: Optional operation config; a default is built when ``None``.

        Returns:
            ``(updated_recurrent_state, output)``.
        """
        return _executor(
            self,
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            recurrent_state=recurrent_state,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            has_initial_state=has_initial_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            chunk_size=chunk_size,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            apply_silu_in_gdr=apply_silu_in_gdr,
            use_recurrent_scan_prefill=use_recurrent_scan_prefill,
            runtime_dtype=runtime_dtype,
            platform=platform,
            _cfg=cfg or RaggedGatedDeltaRuleV2Config(chunk_size=chunk_size, platform=platform or "auto"),
        )

    def heuristic_cfg(self, inv: Invocation[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]]):
        """Return the default configuration for an invocation.

        The ``chunk_size`` is taken from the invocation keyword arguments
        (defaulting to ``64``) and the portable XLA backend (``platform="xla"``,
        ``backend="any"``) is selected so the heuristic is valid on any device.

        Args:
            inv: Invocation carrying the call arguments and keyword arguments;
                only ``inv.kwargs["chunk_size"]`` is consulted.

        Returns:
            A ``RaggedGatedDeltaRuleV2Config`` targeting the XLA backend with the
            requested chunk size.
        """
        chunk_size = int(inv.kwargs.get("chunk_size", 64))
        return RaggedGatedDeltaRuleV2Config(chunk_size=chunk_size, platform="xla", backend="any")

    def candidate_cfgs(self, inv: Invocation[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]]):
        """Enumerate autotuning candidates for the generic/XLA path.

        Args:
            inv: Invocation context (unused; the candidate set is fixed).

        Returns:
            A list of XLA-backend configs spanning chunk sizes ``16``, ``32``,
            and ``64``.
        """
        return [RaggedGatedDeltaRuleV2Config(chunk_size=c, platform="xla", backend="any") for c in (16, 32, 64)]

    def candidate_cfgs_gpu(self, inv: Invocation[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]]):
        """Enumerate autotuning candidates for GPU execution.

        The candidate chunk sizes are ``16``, ``32``, ``64`` plus the invocation's
        requested ``chunk_size`` (default ``64``), de-duplicated while preserving
        order. All candidates use the portable XLA backend, as there is no
        dedicated GPU GDN v2 kernel.

        Args:
            inv: Invocation whose ``inv.kwargs["chunk_size"]`` augments the
                default chunk-size set.

        Returns:
            A list of XLA-backend configs across the de-duplicated chunk sizes.
        """
        chunk_size = int(inv.kwargs.get("chunk_size", 64))
        chunk_sizes = tuple(dict.fromkeys((16, 32, 64, chunk_size)))
        return [RaggedGatedDeltaRuleV2Config(chunk_size=c, platform="xla", backend="any") for c in chunk_sizes]

    def candidate_cfgs_tpu(self, inv: Invocation[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]]):
        """Enumerate autotuning candidates for TPU execution.

        The candidate chunk sizes are ``16``, ``32``, ``64`` plus the invocation's
        requested ``chunk_size`` (default ``64``), de-duplicated while preserving
        order. Both the TPU Pallas backend and the portable XLA fallback are
        offered for each chunk size, Pallas first.

        Args:
            inv: Invocation whose ``inv.kwargs["chunk_size"]`` augments the
                default chunk-size set.

        Returns:
            A list of configs: Pallas/TPU candidates for every chunk size,
            followed by XLA fallback candidates for every chunk size.
        """
        chunk_size = int(inv.kwargs.get("chunk_size", 64))
        chunk_sizes = tuple(dict.fromkeys((16, 32, 64, chunk_size)))
        return [RaggedGatedDeltaRuleV2Config(chunk_size=c, platform="pallas", backend="tpu") for c in chunk_sizes] + [
            RaggedGatedDeltaRuleV2Config(chunk_size=c, platform="xla", backend="any") for c in chunk_sizes
        ]

    candidate_cfgs_xla = candidate_cfgs
    candidate_cfgs_any = candidate_cfgs
    candidate_cfgs_shard_map = candidate_cfgs
    candidate_cfgs_shard_map_xla = candidate_cfgs
    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_executor: Executor[RaggedGatedDeltaRuleV2Config, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("ragged_gated_delta_rule_v2"),
    )
)


def ragged_gated_delta_rule_v2(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    has_initial_state: jnp.ndarray | None = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = True,
    pre_sharded_mixed_qkv: bool = False,
    flat_tp_shard: bool = False,
    apply_silu_in_gdr: bool = False,
    use_recurrent_scan_prefill: bool = False,
    runtime_dtype: object | None = None,
    mesh: object | None = None,
    head_axis: object | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RaggedGatedDeltaRuleV2Config | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Execute packed-inference ragged GDN v2 through eJKernel.

    Args:
        mixed_qkv: Packed Q/K/V token buffer with shape
            ``[tokens, 2 * n_kq * d_k + n_v * d_v]``.
        b: Per-token beta gate input, shape ``[tokens, n_v]``.
        a: Per-token decay gate input, shape ``[tokens, n_v]``.
        recurrent_state: Recurrent state pool, shape
            ``[num_slots, n_v, d_k, d_v]``.
        A_log: Per-value-head log-``A`` parameter.
        dt_bias: Per-value-head ``dt`` bias.
        query_start_loc: CSR-style cumulative token offsets per request.
        state_indices: Mapping from request id to recurrent-state slot.
        distribution: Runtime distribution as an int32 ``[decode_end, prefill_end,
            total]`` triple, where ``total`` (``distribution[2]``) is the number
            of valid sequences.
        has_initial_state: Optional per-request mask indicating whether the
            existing recurrent state should be loaded.
        n_kq: Number of query/key heads.
        n_v: Number of value heads.
        d_k: Query/key head dimension.
        d_v: Value head dimension.
        chunk_size: Prefill chunk size.
        use_qk_norm_in_gdn: Whether to normalize Q/K inside the recurrence.
        pre_sharded_mixed_qkv: Whether ``mixed_qkv`` is already in flat
            tensor-parallel shard layout.
        flat_tp_shard: Prefer flat mixed-QKV head sharding when possible.
        apply_silu_in_gdr: Whether to apply SiLU to the mixed QKV input.
        use_recurrent_scan_prefill: Enable recurrent-scan prefill in supported
            backends.
        runtime_dtype: Runtime dtype override.
        mesh: Optional mesh for tensor-parallel ``shard_map`` execution.
        head_axis: Logical mesh axis used for head sharding.
        platform: Optional implementation family override.
        cfg: Optional operation config.

    Returns:
        ``(updated_recurrent_state, output)`` where ``output`` is packed as
        ``[tokens, n_v * d_v]``.
    """
    if has_initial_state is None:
        has_initial_state = jnp.ones(state_indices.shape[0], dtype=jnp.bool_)

    op = RaggedGatedDeltaRuleV2()
    tp_size = mesh_axis_size(mesh, head_axis)
    key_dim = int(n_kq * d_k)
    value_dim = int(n_v * d_v)
    can_flat_shard = (
        mesh is not None
        and head_axis is not None
        and int(tp_size) > 1
        and n_kq % int(tp_size) == 0
        and n_v % int(tp_size) == 0
        and key_dim % int(tp_size) == 0
        and value_dim % int(tp_size) == 0
    )
    can_split_shard = (
        mesh is not None and head_axis is not None and int(tp_size) > 1 and n_v % int(tp_size) == 0 and n_v % n_kq == 0
    )

    if not (flat_tp_shard and can_flat_shard) and not can_split_shard:
        return op._run_unsharded(
            mixed_qkv,
            b,
            a,
            recurrent_state,
            A_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            chunk_size=chunk_size,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            apply_silu_in_gdr=apply_silu_in_gdr,
            use_recurrent_scan_prefill=use_recurrent_scan_prefill,
            runtime_dtype=runtime_dtype,
            platform=platform,
            cfg=cfg,
        )

    if pre_sharded_mixed_qkv and not (flat_tp_shard and can_flat_shard):
        raise ValueError(
            "pre_sharded_mixed_qkv requires the flat TP shard path; "
            "the split Q/K/V shard path expects unsharded mixed_qkv layout."
        )

    beta_spec = PartitionSpec(None, head_axis)
    state_spec = PartitionSpec(None, head_axis, None, None)
    head_param_spec = PartitionSpec(head_axis)
    replicated = PartitionSpec()
    run_cfg = cfg or RaggedGatedDeltaRuleV2Config(chunk_size=chunk_size, platform=platform or "auto")

    if flat_tp_shard and can_flat_shard:
        mixed_spec = PartitionSpec(None, head_axis)
        local_n_kq_static = n_kq // int(tp_size)
        local_n_v_static = n_v // int(tp_size)

        if not pre_sharded_mixed_qkv:
            mixed_qkv = _reorder_concatenated_tensor_for_sharding(
                mixed_qkv,
                (key_dim, key_dim, value_dim),
                int(tp_size),
                -1,
            )

        return _executor(
            op,
            mixed_qkv,
            b,
            a,
            recurrent_state,
            A_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            n_kq=local_n_kq_static,
            n_v=local_n_v_static,
            d_k=d_k,
            d_v=d_v,
            chunk_size=chunk_size,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            apply_silu_in_gdr=apply_silu_in_gdr,
            use_recurrent_scan_prefill=use_recurrent_scan_prefill,
            runtime_dtype=runtime_dtype,
            platform=platform,
            method="shard_map",
            mesh=mesh,
            in_specs=(
                mixed_spec,
                beta_spec,
                beta_spec,
                state_spec,
                head_param_spec,
                head_param_spec,
                replicated,
                replicated,
                replicated,
                replicated,
            ),
            out_specs=(state_spec, mixed_spec),
            check_vma=False,
            _cfg=run_cfg,
        )

    token_head_spec = PartitionSpec(None, head_axis, None)
    return _executor(
        op,
        mixed_qkv,
        b,
        a,
        recurrent_state,
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        apply_silu_in_gdr=apply_silu_in_gdr,
        use_recurrent_scan_prefill=use_recurrent_scan_prefill,
        runtime_dtype=runtime_dtype,
        platform=platform,
        split_qkv_for_head_shard=True,
        method="shard_map",
        mesh=mesh,
        in_specs=(
            token_head_spec,
            token_head_spec,
            token_head_spec,
            beta_spec,
            beta_spec,
            state_spec,
            head_param_spec,
            head_param_spec,
            replicated,
            replicated,
            replicated,
            replicated,
        ),
        out_specs=(state_spec, token_head_spec),
        check_vma=False,
        _cfg=run_cfg,
    )


ragged_gated_delta_rule_v2_op = ragged_gated_delta_rule_v2

__all__ = (
    "KERNEL_TILE_POLICIES",
    "KernelTilePolicy",
    "RaggedGatedDeltaRuleV2",
    "normalize_kernel_tile_policy",
    "ragged_gated_delta_rule",
    "ragged_gated_delta_rule_decode_only",
    "ragged_gated_delta_rule_mixed_prefill",
    "ragged_gated_delta_rule_v2",
    "ragged_gated_delta_rule_v2_op",
    "set_gdn_kernel_tile_policy",
)
