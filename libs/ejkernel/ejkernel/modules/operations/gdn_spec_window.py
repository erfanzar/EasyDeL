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

"""Public GDN speculative-window state-scan operation.

Speculative decoding on GDN hybrids needs the recurrent state after every
window step (per accepted-prefix candidate). The XLA backend carries the
canonical per-step reference; the TPU Pallas backend fuses the whole window
into one VMEM-resident kernel. A head-sharded helper mirrors the ragged conv
op so tensor-parallel callers can run the scan under ``shard_map``.
"""

from __future__ import annotations

import os
from typing import Literal

import jax
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, Float

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
from .configs import GdnSpecWindowStatesConfig
from .ragged_causal_conv1d import mesh_axis_size


class GdnSpecWindowStates(Kernel[GdnSpecWindowStatesConfig, Array]):
    """Executor-driven wrapper for the GDN speculative-window state scan.

    Implements the :class:`~ejkernel.ops.Kernel` protocol: ``run`` dispatches
    to the registered backend implementation and
    ``create_shard_map_wrapper`` builds the tensor-parallel head-sharded path.

    Attributes:
        version: Cache-busting version tag for the persistent config cache.
    """

    version = "1"

    def __init__(self):
        """Register the operation under ``op_id="gdn_spec_window_states"``."""
        super().__init__(op_id="gdn_spec_window_states")

    def get_impl(self, cfg: GdnSpecWindowStatesConfig):
        """Resolve the backend implementation for a config.

        Args:
            cfg: Operation config carrying the requested ``platform``/
                ``backend``.

        Returns:
            The registered callable implementing ``gdn_spec_window_states``.
        """
        platform = detect_platform("gdn_spec_window_states", cfg.platform)
        return kernel_registry.get("gdn_spec_window_states", platform=platform, backend=cfg.backend)

    def create_shard_map_wrapper(
        self,
        mixed_qkv: Float[Array, "batch num_steps qkv_dim"],
        b: Float[Array, "batch num_steps num_v_heads"],
        a: Float[Array, "batch num_steps num_v_heads"],
        A_log: Float[Array, "num_v_heads"],
        dt_bias: Float[Array, "num_v_heads"],
        step_valid: Bool[Array, "batch num_steps"],
        recurrent_state: Float[Array, "batch num_v_heads key_dim value_dim"],
        *,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        apply_silu: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: GdnSpecWindowStatesConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: PartitionSpec | tuple[PartitionSpec | None, ...] | None = None,
        check_vma: bool = False,
    ):
        """Create a ``shard_map`` wrapper for TP head-sharded execution.

        ``n_kq``/``n_v`` are the GLOBAL head counts; the per-shard counts are
        derived from the mesh axis size carried by ``in_specs`` head axis (the
        caller passes local counts implicitly via the sharded tensors, so the
        wrapped run divides the head counts by the shard count).

        Args:
            mixed_qkv: Packed rows; sharded on the feature axis (interleaved
                per-shard concat layout, as produced by the head-sharded conv).
            b: Raw beta rows; sharded on the head axis.
            a: Raw alpha rows; sharded on the head axis.
            A_log: Per-head decay parameter; sharded on the head axis.
            dt_bias: Per-head bias; sharded on the head axis.
            step_valid: Replicated per-step mask.
            recurrent_state: State pool; sharded on the head axis.
            n_kq: GLOBAL number of key/query heads.
            n_v: GLOBAL number of value heads.
            d_k: Key/query head dim.
            d_v: Value head dim.
            apply_silu: Apply SiLU to ``mixed_qkv`` inside the op.
            platform: Optional implementation-family override.
            cfg: Optional operation config.
            mesh: Mesh used by :func:`jax.shard_map`. Must not be ``None``.
            in_specs: Partition specs for the seven call arguments.
            out_specs: Partition spec for the stacked-state output.
            check_vma: Forwarded to :func:`jax.shard_map`.

        Returns:
            Tuple ``(shard_map_fn, call_args)``.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        run_cfg = cfg or GdnSpecWindowStatesConfig(platform=platform or "auto")
        call_args = (mixed_qkv, b, a, A_log, dt_bias, step_valid, recurrent_state)
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"

        # Head axis size — every sharded head count divides by it (validated by
        # the caller's head-sharding gate).
        head_axis = in_specs[-1][1] if in_specs[-1] is not None else None
        tp_size = int(mesh_axis_size(mesh, head_axis)) if head_axis is not None else 1
        local_n_kq = n_kq // tp_size
        local_n_v = n_v // tp_size

        def _wrapped(x, b_, a_, alog_, dtb_, valid_, state_):
            """Run the window scan on a per-shard slice of the inputs."""
            return self.run(
                x,
                b_,
                a_,
                alog_,
                dtb_,
                valid_,
                state_,
                n_kq=local_n_kq,
                n_v=local_n_v,
                d_k=d_k,
                d_v=d_v,
                apply_silu=apply_silu,
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
        mixed_qkv: Float[Array, "batch num_steps qkv_dim"],
        b: Float[Array, "batch num_steps num_v_heads"],
        a: Float[Array, "batch num_steps num_v_heads"],
        A_log: Float[Array, "num_v_heads"],
        dt_bias: Float[Array, "num_v_heads"],
        step_valid: Bool[Array, "batch num_steps"],
        recurrent_state: Float[Array, "batch num_v_heads key_dim value_dim"],
        *,
        n_kq: int,
        n_v: int,
        d_k: int,
        d_v: int,
        apply_silu: bool = False,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: GdnSpecWindowStatesConfig,
        **_,
    ) -> Float[Array, "num_steps batch num_v_heads key_dim value_dim"]:
        """Run the selected backend implementation of the window state scan.

        Args:
            mixed_qkv: Post-conv packed rows ``[batch, num_steps, qkv_dim]``.
            b: Raw beta rows ``[batch, num_steps, n_v]``.
            a: Raw alpha rows ``[batch, num_steps, n_v]``.
            A_log: Per-head log-decay parameter ``[n_v]``.
            dt_bias: Per-head delta-time bias ``[n_v]``.
            step_valid: ``[batch, num_steps]`` per-step validity mask.
            recurrent_state: Window-entry state ``[batch, n_v, d_k, d_v]``.
            n_kq: Number of key/query heads (local when shard-mapped).
            n_v: Number of value heads (local when shard-mapped).
            d_k: Key/query head dim.
            d_v: Value head dim.
            apply_silu: Apply SiLU to ``mixed_qkv`` before slicing.
            platform: Optional implementation-family override.
            cfg: Operation config selected by the executor.
            **_: Ignored extra kwargs forwarded by the executor.

        Returns:
            ``[num_steps, batch, n_v, d_k, d_v]`` per-prefix states.
        """
        if platform is not None:
            cfg = GdnSpecWindowStatesConfig(
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            mixed_qkv,
            b,
            a,
            A_log,
            dt_bias,
            step_valid,
            recurrent_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            apply_silu=apply_silu,
        )

    def heuristic_cfg(self, inv: Invocation[GdnSpecWindowStatesConfig, Array]):
        """Return the default config for an invocation (XLA reference).

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A :class:`GdnSpecWindowStatesConfig` on ``platform="xla"``.
        """
        return GdnSpecWindowStatesConfig(platform="xla", backend="any")

    def candidate_cfgs(self, inv: Invocation[GdnSpecWindowStatesConfig, Array]):
        """Return the default candidate config list (heuristic only)."""
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_tpu(self, inv: Invocation[GdnSpecWindowStatesConfig, Array]):
        """Return TPU candidates (XLA reference plus fused Pallas)."""
        return [
            GdnSpecWindowStatesConfig(platform="xla", backend="any"),
            GdnSpecWindowStatesConfig(platform="pallas", backend="tpu"),
        ]

    candidate_cfgs_gpu = candidate_cfgs
    candidate_cfgs_xla = candidate_cfgs
    candidate_cfgs_any = candidate_cfgs
    heuristic_cfg_shard_map = heuristic_cfg
    candidate_cfgs_shard_map = candidate_cfgs
    candidate_cfgs_shard_map_xla = candidate_cfgs
    candidate_cfgs_shard_map_gpu = candidate_cfgs
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_executor: Executor[GdnSpecWindowStatesConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("gdn_spec_window_states"),
    )
)


def gdn_spec_window_states(
    mixed_qkv: Float[Array, "batch num_steps qkv_dim"],
    b: Float[Array, "batch num_steps num_v_heads"],
    a: Float[Array, "batch num_steps num_v_heads"],
    A_log: Float[Array, "num_v_heads"],
    dt_bias: Float[Array, "num_v_heads"],
    step_valid: Bool[Array, "batch num_steps"],
    recurrent_state: Float[Array, "batch num_v_heads key_dim value_dim"],
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    apply_silu: bool = False,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: GdnSpecWindowStatesConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | tuple[PartitionSpec | None, ...] | None = None,
    check_vma: bool = False,
) -> Float[Array, "num_steps batch num_v_heads key_dim value_dim"]:
    """Execute the GDN speculative-window state scan through the operation stack.

    Args:
        mixed_qkv: Post-conv packed rows ``[batch, num_steps, qkv_dim]``.
        b: Raw beta rows ``[batch, num_steps, n_v]``.
        a: Raw alpha rows ``[batch, num_steps, n_v]``.
        A_log: Per-head log-decay parameter ``[n_v]``.
        dt_bias: Per-head delta-time bias ``[n_v]``.
        step_valid: ``[batch, num_steps]`` per-step validity mask.
        recurrent_state: Window-entry state ``[batch, n_v, d_k, d_v]``.
        n_kq: GLOBAL number of key/query heads.
        n_v: GLOBAL number of value heads.
        d_k: Key/query head dim.
        d_v: Value head dim.
        apply_silu: Apply SiLU to ``mixed_qkv`` before slicing.
        platform: Optional implementation-family override.
        cfg: Optional operation config.
        mesh: Optional mesh for ``shard_map`` execution.
        in_specs: Optional input partition specs for ``shard_map``.
        out_specs: Optional output partition spec for ``shard_map``.
        check_vma: Forwarded to ``jax.shard_map``.

    Returns:
        ``[num_steps, batch, n_v, d_k, d_v]`` per-prefix recurrent states.
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"
    if cfg is None:
        cfg = GdnSpecWindowStatesConfig(platform=platform or "auto")
    return _executor(
        GdnSpecWindowStates(),
        mixed_qkv,
        b,
        a,
        A_log,
        dt_bias,
        step_valid,
        recurrent_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        apply_silu=apply_silu,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=check_vma,
        _cfg=cfg,
    )


gdn_spec_window_states_op = gdn_spec_window_states
