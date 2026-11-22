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

"""Standalone differentiable dense all-to-all collective.

Exchanges shards across a mesh axis: the local array is split along
``split_axis``, one piece goes to every peer, and received pieces are
concatenated along ``concat_axis``.  Typical uses are sequence-parallel head
exchange around attention and dense MoE token exchange.

Calling conventions:
    - ``mesh=`` provided → wrapped in ``shard_map``; the global input is
      sharded on ``concat_axis`` and the result comes back sharded on
      ``split_axis``.
    - ``mesh=None`` → must be called inside an existing shard_map /
      manual-axes context where ``axis_name`` is active.

The backward pass is the inverse all-to-all (split/concat axes swapped),
provided by JAX's native autodiff.
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
from .configs import AllToAllConfig


def _spec_on_axis(axis_name: str | tuple[str, ...], axis: int) -> PartitionSpec:
    """Build a PartitionSpec placing ``axis_name`` at position ``axis``."""
    parts: list = [None] * (axis + 1)
    parts[axis] = axis_name
    return PartitionSpec(*parts)


class AllToAll(Kernel[AllToAllConfig, Array]):
    """Differentiable dense all-to-all over a mesh axis.

    Platform support:
        - XLA: always available (``lax.all_to_all``).
    """

    def __init__(self):
        super().__init__(op_id="all_to_all")

    def get_impl(self, cfg: AllToAllConfig):
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
        split_axis: int,
        concat_axis: int,
        tiled: bool = True,
        tp_size: int | None = None,
        collective_id: int | None = 0,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: AllToAllConfig,
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
            in_specs = (_spec_on_axis(axis_name, concat_axis),)
        if out_specs is None:
            out_specs = _spec_on_axis(axis_name, split_axis)

        def _wrapped(x: Float[Array, "..."]):
            return self.run(
                x=x,
                axis_name=axis_name,
                split_axis=split_axis,
                concat_axis=concat_axis,
                tiled=tiled,
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
        split_axis: int,
        concat_axis: int,
        tiled: bool = True,
        tp_size: int | None = None,
        collective_id: int | None = 0,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: AllToAllConfig,
    ) -> Float[Array, "..."]:
        """Execute the all-to-all with the selected backend.

        Args:
            x: Local shard.
            axis_name: Name(s) of the sharded axis for the collective.
            split_axis: Axis split into per-peer pieces.
            concat_axis: Axis received pieces are concatenated along.
            tiled: Tiled semantics.
            tp_size: Collective world size (validation only).
            collective_id: Barrier semaphore allocation id (TPU engines).
            platform: Optional platform override.
            cfg: Kernel configuration.

        Returns:
            The exchanged array.
        """
        if platform is not None:
            cfg = AllToAllConfig(
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            x=x,
            axis_name=axis_name,
            split_axis=split_axis,
            concat_axis=concat_axis,
            tiled=tiled,
            tp_size=tp_size,
            collective_id=collective_id,
        )

    def heuristic_cfg(self, inv: Invocation[AllToAllConfig, Array]) -> AllToAllConfig:
        """Return default heuristic configuration for any platform."""
        return AllToAllConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[AllToAllConfig, Array]):
        """Return candidate configurations for autotuning."""
        return [AllToAllConfig(platform="auto", backend="any")]


_all_to_all_executor: Executor[AllToAllConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("all-to-all"),
    )
)


def all_to_all(
    x: Float[Array, "..."],
    axis_name: str | tuple[str, ...],
    /,
    *,
    split_axis: int,
    concat_axis: int,
    tiled: bool = True,
    tp_size: int | None = None,
    collective_id: int | None = 0,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: AllToAllConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> Float[Array, "..."]:
    """Differentiable dense all-to-all with automatic backend selection.

    When *mesh* is provided the operation is wrapped in ``shard_map`` (the
    global input is sharded on ``concat_axis`` and the result returns
    sharded on ``split_axis``); otherwise it must run inside an existing
    sharded context.

    Args:
        x: Local shard (manual-axes mode) or the globally sharded array
            (mesh mode).
        axis_name: Name(s) of the sharded axis for the collective.
        split_axis: Axis split into per-peer pieces.
        concat_axis: Axis received pieces are concatenated along.
        tiled: Tiled semantics (no new leading axis).
        tp_size: Collective world size (validation only).
        collective_id: Barrier semaphore allocation id (TPU engines).
        platform: Optional platform override.
        cfg: Optional kernel configuration override.
        mesh: If provided, wraps the call in ``shard_map``.
        in_specs: Optional input partition specs for ``shard_map``.
        out_specs: Optional output partition spec for ``shard_map``.

    Returns:
        The exchanged array.
    """
    method = "shard_map" if mesh is not None else None
    if method == "shard_map":
        if in_specs is None:
            in_specs = (_spec_on_axis(axis_name, concat_axis),)
        if out_specs is None:
            out_specs = _spec_on_axis(axis_name, split_axis)
    return _all_to_all_executor(
        AllToAll(),
        x=x,
        axis_name=axis_name,
        split_axis=split_axis,
        concat_axis=concat_axis,
        tiled=tiled,
        tp_size=tp_size,
        collective_id=collective_id,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )
