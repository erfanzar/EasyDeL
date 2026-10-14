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

"""Weighted grouped gather-sum (fused MoE top-k combine).

``out[t] = sum_j weights[t*k + j] * x[indices[t*k + j]]`` with negative
indices contributing zero; weighting and accumulation in float32.

Engine status: XLA only — see ``ragged_gather`` for the measured v5p
SparseCore verdict; this op id is the landing pad for SC engines on newer
TPU generations.
"""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import RaggedGatherReduceConfig


class RaggedGatherReduce(Kernel[RaggedGatherReduceConfig, Array]):
    """Weighted grouped row gather-sum (MoE top-k combine).

    Platform support:
        - XLA: always available (safe take + f32 weighted group-sum).
    """

    def __init__(self):
        super().__init__(op_id="ragged_gather_reduce")

    def get_impl(self, cfg: RaggedGatherReduceConfig):
        """Get the kernel implementation for the given configuration.

        Args:
            cfg: Kernel configuration specifying platform and backend.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Float[Array, "n d"],
        indices: Int[Array, "mk"],
        weights: Float[Array, "mk"],
        reduce_group_size: int,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: RaggedGatherReduceConfig,
    ) -> Float[Array, "m d"]:
        """Execute the combine with the selected backend.

        Args:
            x: Expert-output table ``[n, d]``.
            indices: Flat slot indices ``[m * reduce_group_size]``.
            weights: Per-slot combine weights, same length as ``indices``.
            reduce_group_size: Slots summed per output row (MoE ``top_k``).
            platform: Optional platform override.
            cfg: Kernel configuration.

        Returns:
            Combined rows ``[m, d]``.
        """
        if platform is not None:
            cfg = RaggedGatherReduceConfig(
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(x=x, indices=indices, weights=weights, reduce_group_size=reduce_group_size)

    def heuristic_cfg(self, inv: Invocation[RaggedGatherReduceConfig, Array]) -> RaggedGatherReduceConfig:
        """Return default heuristic configuration for any platform."""
        return RaggedGatherReduceConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RaggedGatherReduceConfig, Array]):
        """Return candidate configurations for autotuning."""
        return [RaggedGatherReduceConfig(platform="auto", backend="any")]


_ragged_gather_reduce_executor: Executor[RaggedGatherReduceConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("ragged-gather-reduce"),
    )
)


def ragged_gather_reduce(
    x: Float[Array, "n d"],
    indices: Int[Array, "mk"],
    weights: Float[Array, "mk"],
    /,
    *,
    reduce_group_size: int,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: RaggedGatherReduceConfig | None = None,
) -> Float[Array, "m d"]:
    """Weighted grouped gather-sum: the fused MoE top-k combine.

    Differentiable via native JAX AD (gather transpose = scatter-add;
    per-slot weight grads are row dots).

    Args:
        x: Expert-output table ``[n, d]``.
        indices: Flat slot indices ``[m * reduce_group_size]``; entries
            ``< 0`` mark invalid slots contributing zero.
        weights: Per-slot combine weights, same length as ``indices``.
        reduce_group_size: Slots summed per output row (the MoE ``top_k``).
        platform: Optional platform override.
        cfg: Optional kernel configuration override.

    Returns:
        Combined rows ``[m, d]`` in ``x``'s dtype.
    """
    return _ragged_gather_reduce_executor(
        RaggedGatherReduce(),
        x=x,
        indices=indices,
        weights=weights,
        reduce_group_size=reduce_group_size,
        platform=platform,
        _cfg=cfg,
    )
