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

"""Standalone differentiable all-gather collective.

Concatenates per-device shards over a mesh axis.  Two algorithm families:

    - ``one_shot`` (Pallas TPU): every device DMAs its shard directly into
      every peer's output window in a single communication phase.  Optimal
      for the latency-bound small-message regime.
    - ``ring`` (XLA): delegates to ``lax.all_gather``.

Calling conventions:
    - ``mesh=`` provided → wrapped in ``shard_map`` over a globally sharded
      input; the gathered (replicated) result is returned.
    - ``mesh=None`` → must be called inside an existing shard_map /
      manual-axes context where ``axis_name`` is active.

The backward pass is the transpose dual: a reduce-scatter of the output
cotangent.
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import AllGatherConfig


def _spec_on_axis(axis_name: str | tuple[str, ...], axis: int) -> PartitionSpec:
    """Build a PartitionSpec placing ``axis_name`` at position ``axis``."""
    parts: list[str | None] = [None] * (axis + 1)
    parts[axis] = axis_name
    return PartitionSpec(*parts)


class AllGather(Kernel[AllGatherConfig, Array]):
    """Differentiable all-gather over a mesh axis.

    Platform support:
        - XLA: always available (``lax.all_gather``).
        - Pallas TPU: one-shot direct-exchange kernel for small, aligned
          shards gathered along axis 0.
    """

    def __init__(self):
        super().__init__(op_id="all_gather")

    def get_impl(self, cfg: AllGatherConfig):
        """Get the kernel implementation for the given configuration.

        Args:
            cfg: Kernel configuration specifying platform and backend.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def create_shard_map_wrapper(
        self,
        x: Float[Array, "..."],
        axis_name: str | tuple[str, ...],
        gather_axis: int = 0,
        tiled: bool = True,
        mode: str | None = None,
        tp_size: int | None = None,
        collective_id: int | None = 0,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: AllGatherConfig,
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
            in_specs = (_spec_on_axis(axis_name, gather_axis),)
        if out_specs is None:
            out_specs = PartitionSpec()

        _axis_names_tuple = axis_name if isinstance(axis_name, tuple) else (axis_name,)
        mesh_tp_size = 1
        for _axis in _axis_names_tuple:
            mesh_tp_size *= int(mesh.shape[_axis])
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

        def _wrapped(x: Float[Array, "..."]):
            return self.run(
                x=x,
                axis_name=axis_name,
                gather_axis=gather_axis,
                tiled=tiled,
                mode=mode,
                tp_size=tp_size,
                collective_id=collective_id,
                platform=platform,
                cfg=cfg,
            )

        call_args = (x,)
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
        x: Float[Array, "..."],
        axis_name: str | tuple[str, ...],
        gather_axis: int = 0,
        tiled: bool = True,
        mode: str | None = None,
        tp_size: int | None = None,
        collective_id: int | None = 0,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: AllGatherConfig,
    ) -> Float[Array, "..."]:
        """Execute the all-gather with the selected backend.

        Args:
            x: Local shard.
            axis_name: Name of the sharded axis for the collective.
            gather_axis: Concatenation axis.
            tiled: Tiled gather semantics.
            mode: Optional per-call algorithm override; defaults to *cfg.mode*.
            tp_size: Collective world size.
            collective_id: Barrier semaphore allocation id.
            platform: Optional platform override.
            cfg: Kernel configuration.

        Returns:
            The gathered array.
        """
        if platform is not None:
            cfg = AllGatherConfig(
                mode=cfg.mode,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        impl = self.get_impl(cfg)
        return impl(
            x=x,
            axis_name=axis_name,
            gather_axis=gather_axis,
            tiled=tiled,
            mode=mode if mode is not None else cfg.mode,
            tp_size=tp_size,
            collective_id=collective_id,
        )

    def heuristic_cfg(self, inv: Invocation[AllGatherConfig, Array]) -> AllGatherConfig:
        """Return default heuristic configuration for any platform."""
        return AllGatherConfig(mode="auto", platform="auto", backend="any")

    def heuristic_cfg_tpu(self, inv: Invocation[AllGatherConfig, Array]) -> AllGatherConfig:
        """Return TPU/Pallas-oriented default configuration."""
        return AllGatherConfig(mode="auto", platform="pallas", backend="tpu")

    def candidate_cfgs(self, inv: Invocation[AllGatherConfig, Array]):
        """Return candidate configurations for autotuning."""
        return [AllGatherConfig(mode="auto", platform="auto", backend="any")]

    def candidate_cfgs_tpu(self, inv: Invocation[AllGatherConfig, Array]):
        """Return TPU candidate configurations for autotuning."""
        return [
            AllGatherConfig(mode="one_shot", platform="pallas", backend="tpu"),
            AllGatherConfig(mode="ring", platform="pallas", backend="tpu"),
            AllGatherConfig(mode="auto", platform="xla", backend="any"),
        ]

    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_all_gather_executor: Executor[AllGatherConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("all-gather"),
    )
)


def all_gather(
    x: Float[Array, "..."],
    axis_name: str | tuple[str, ...],
    /,
    *,
    gather_axis: int = 0,
    tiled: bool = True,
    mode: str | None = None,
    tp_size: int | None = None,
    collective_id: int | None = 0,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: AllGatherConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "..."]:
    """Differentiable all-gather with automatic backend selection.

    Concatenates per-device shards along ``gather_axis`` over ``axis_name``.
    When *mesh* is provided the operation is wrapped in ``shard_map``;
    otherwise it must run inside an existing sharded context.

    Args:
        x: Local shard (manual-axes mode) or the globally sharded array
            (mesh mode).
        axis_name: Name of the sharded axis for the collective.
        gather_axis: Concatenation axis.
        tiled: Tiled gather semantics (no new leading axis).
        mode: Optional algorithm override (``"auto"``/``"one_shot"``/``"ring"``).
        tp_size: Collective world size; inferred when ``None``.
        collective_id: Barrier semaphore allocation id.
        platform: Optional platform override.
        cfg: Optional kernel configuration override.
        mesh: If provided, wraps the call in ``shard_map``.
        in_specs: Optional input partition specs for ``shard_map``.
        out_specs: Optional output partition spec for ``shard_map``.

    Returns:
        The gathered array.
    """
    method = "shard_map" if mesh is not None else None
    if method == "shard_map":
        if in_specs is None:
            in_specs = (_spec_on_axis(axis_name, gather_axis),)
        if out_specs is None:
            out_specs = PartitionSpec()
    return _all_gather_executor(
        AllGather(),
        x=x,
        axis_name=axis_name,
        gather_axis=gather_axis,
        tiled=tiled,
        mode=mode,
        tp_size=tp_size,
        collective_id=collective_id,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )
