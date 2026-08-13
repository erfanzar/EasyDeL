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

"""Ragged row gather (MoE permute/unpermute primitive).

``out[i] = x[indices[i]]`` with negative indices producing zero rows.

Engine status: XLA only.  A from-scratch SparseCore engine was built and
measured on v5p (2026-08-10): correct, but 7-23x slower than the TC gather —
the SC scalar sequencer's per-DMA-descriptor cost (~7.6us/row) cannot
compete with XLA's vectorized gather on gen-5 SparseCores.  This op id is
the registry landing pad for SC engines on newer TPU generations.
"""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import RaggedGatherConfig


class RaggedGather(Kernel[RaggedGatherConfig, Array]):
    """Row gather by index with a negative-index invalid convention.

    Platform support:
        - XLA: always available (safe take + mask).
    """

    def __init__(self):
        super().__init__(op_id="ragged_gather")

    def get_impl(self, cfg: RaggedGatherConfig):
        """Get the kernel implementation for the given configuration.

        Args:
            cfg: Kernel configuration specifying platform and backend.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Float[Array, "n d"],
        indices: Int[Array, "m"],
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: RaggedGatherConfig,
    ) -> Float[Array, "m d"]:
        """Execute the gather with the selected backend.

        Args:
            x: Source table ``[n, d]``.
            indices: Row indices ``[m]``; negative entries yield zero rows.
            platform: Optional platform override.
            cfg: Kernel configuration.

        Returns:
            Gathered rows ``[m, d]``.
        """
        if platform is not None:
            cfg = RaggedGatherConfig(
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(x=x, indices=indices)

    def heuristic_cfg(self, inv: Invocation[RaggedGatherConfig, Array]) -> RaggedGatherConfig:
        """Return default heuristic configuration for any platform."""
        return RaggedGatherConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RaggedGatherConfig, Array]):
        """Return candidate configurations for autotuning."""
        return [RaggedGatherConfig(platform="auto", backend="any")]


_ragged_gather_executor: Executor[RaggedGatherConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("ragged-gather"),
    )
)


def ragged_gather(
    x: Float[Array, "n d"],
    indices: Int[Array, "m"],
    /,
    *,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RaggedGatherConfig | None = None,
) -> Float[Array, "m d"]:
    """Gather rows of ``x`` by ``indices``; negative indices yield zero rows.

    Differentiable (gather transposes to scatter-add via native JAX AD).

    Args:
        x: Source table ``[n, d]``.
        indices: Row indices ``[m]``; entries ``< 0`` mark invalid slots.
        platform: Optional platform override.
        cfg: Optional kernel configuration override.

    Returns:
        Gathered rows ``[m, d]`` in ``x``'s dtype.
    """
    return _ragged_gather_executor(
        RaggedGather(),
        x=x,
        indices=indices,
        platform=platform,
        _cfg=cfg,
    )
