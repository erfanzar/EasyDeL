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

"""Sinkhorn-Knopp doubly-stochastic projection."""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Float

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import SinkhornKnoppConfig


class SinkhornKnopp(Kernel[SinkhornKnoppConfig, Array]):
    """Alternating row/column normalisation onto the doubly-stochastic manifold.

    The matrices are typically tiny (DeepSeek-V4's hyper-connections normalise a
    4x4), so the cost is dispatch rather than arithmetic: unrolled, each
    iteration contributes two reductions that XLA cannot fuse across, turning 20
    iterations into ~117 device ops for a few hundred elements.

    Platform support:
        - Pallas/TPU: every iteration inside one program.
        - XLA: always available; the reference loop and the numerical contract.
    """

    def __init__(self):
        super().__init__(op_id="sinkhorn_knopp")

    def get_impl(self, cfg: SinkhornKnoppConfig):
        """Get the kernel implementation for the given configuration.

        Args:
            cfg: Kernel configuration specifying platform and backend.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        matrix: Float[Array, "batch seq rows cols"],
        n_iters: int = 20,
        eps: float = 1e-6,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: SinkhornKnoppConfig,
    ) -> Float[Array, "batch seq rows cols"]:
        """Execute the projection with the selected backend.

        Args:
            matrix: Strictly positive matrices ``[batch, seq, rows, cols]``.
            n_iters: Sinkhorn-Knopp iterations (static).
            eps: Denominator floor.
            platform: Optional platform override.
            cfg: Kernel configuration.

        Returns:
            Normalised matrices, same shape and dtype.
        """
        if platform is not None:
            cfg = SinkhornKnoppConfig(
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(matrix, n_iters=n_iters, eps=eps)

    def heuristic_cfg(self, inv: Invocation[SinkhornKnoppConfig, Array]) -> SinkhornKnoppConfig:
        """Return default heuristic configuration for any platform."""
        return SinkhornKnoppConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[SinkhornKnoppConfig, Array]):
        """Return candidate configurations (nothing to autotune: no tiling)."""
        return [SinkhornKnoppConfig(platform="auto", backend="any")]


_sinkhorn_knopp_executor: Executor[SinkhornKnoppConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("sinkhorn-knopp"),
    )
)


def sinkhorn_knopp(
    matrix: Float[Array, "batch seq rows cols"],
    /,
    n_iters: int = 20,
    eps: float = 1e-6,
    *,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: SinkhornKnoppConfig | None = None,
) -> Float[Array, "batch seq rows cols"]:
    """Project ``matrix`` onto the doubly-stochastic manifold.

    Alternates column and row normalisation for a fixed number of iterations.
    The trailing normalisation is over columns, so those sum to one to
    floating-point exactness while rows are converged rather than exact.

    Args:
        matrix: Strictly positive matrices ``[batch, seq, rows, cols]``.
        n_iters: Sinkhorn-Knopp iterations (static).
        eps: Denominator floor, added before every division.
        platform: Optional platform override.
        cfg: Optional kernel configuration override.

    Returns:
        Normalised matrices, same shape and dtype.
    """
    return _sinkhorn_knopp_executor(
        SinkhornKnopp(),
        matrix,
        n_iters=n_iters,
        eps=eps,
        platform=platform,
        _cfg=cfg,
    )
