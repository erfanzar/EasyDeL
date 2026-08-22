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

"""Fused exact top-k.

One operation covering the three shapes that actually occur in this stack,
because they want different things from the same reduction:

===================== ===================== ======== ==========================
call site             shape                 k        wants
===================== ===================== ======== ==========================
MoE router            ``[tokens, experts]`` small     values + indices
DSA indexer           ``[b, s, entries]``   large     indices
sampling top-k filter ``[reqs, vocab]``     per-row   keep-mask
===================== ===================== ======== ==========================

The blockwise-superset Pallas path is exact but costs ``k`` reduction passes,
so :meth:`TopK.heuristic_cfg` only selects it where that trade wins: a wide
reduction axis with a small static ``k``. Everything else -- a narrow axis
(where a Pallas launch costs more than the reduction), a large ``k``, or a
per-row dynamic ``k`` -- routes to the XLA reference, which is not a fallback
for lack of a kernel but the measured-better path for those regimes.
"""

from __future__ import annotations

import os
import typing as tp

import jax
from jaxtyping import Array

from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from .configs import TopKConfig

#: Below this reduction width a Pallas launch costs more than the reduction.
_MIN_WIDTH_FOR_PALLAS = 4096

#: The superset costs ``k`` passes, so it stops paying once ``k`` grows.
_MAX_K_FOR_PALLAS = 32


class TopK(Kernel[TopKConfig, tp.Any]):
    """Exact top-k with mode-aware backend selection."""

    def __init__(self) -> None:
        """Register the operation under the ``topk`` id."""
        super().__init__(op_id="topk")

    def get_impl(self, cfg: TopKConfig) -> tp.Callable[..., tp.Any]:
        """Resolve the registered implementation for ``cfg``.

        Args:
            cfg: Selected configuration.

        Returns:
            The registered callable.
        """
        platform = Platform(cfg.platform) if cfg.platform not in (None, "auto") else None
        backend = Backend(cfg.backend) if cfg.backend not in (None, "any") else Backend.ANY
        return kernel_registry.get("topk", platform=platform, backend=backend)

    def run(
        self,
        operand: Array,
        k: int | Array,
        axis: int = -1,
        mode: str = "values",
        mask_fill: float | None = None,
        *,
        cfg: TopKConfig,
    ) -> tuple[Array, Array] | Array:
        """Execute the selected implementation.

        Args:
            operand: Input array.
            k: Number kept; static int, or a per-row array for ``mode="mask"``.
            axis: Reduction axis.
            mode: ``"values"``, ``"mask"`` or ``"filter"``.
            mask_fill: Fill value for ``mode="filter"``.
            cfg: Selected configuration.

        Returns:
            Mode-dependent result; see the XLA reference.
        """
        return self.get_impl(cfg)(operand, k, axis=axis, mode=mode, mask_fill=mask_fill)

    def heuristic_cfg(self, inv: Invocation[TopKConfig, tp.Any]) -> TopKConfig:
        """Pick a platform from the reduction width, ``k``, and mode.

        Args:
            inv: The invocation being configured.

        Returns:
            The configuration to run with.
        """
        operand = inv.args[0] if inv.args else None
        k = inv.args[1] if len(inv.args) > 1 else inv.kwargs.get("k")
        mode = inv.kwargs.get("mode", "values")
        axis = inv.kwargs.get("axis", -1)

        width = 0
        if operand is not None and getattr(operand, "ndim", 0):
            axis_n = axis if axis >= 0 else operand.ndim + axis
            width = int(operand.shape[axis_n])

        use_pallas = (
            mode == "values"
            and isinstance(k, int)
            and 0 < k <= _MAX_K_FOR_PALLAS
            and width >= _MIN_WIDTH_FOR_PALLAS
            # Mosaic only lowers for real on TPU; on CPU/GPU the Pallas path
            # would fall into interpret mode, which is slower and not the point.
            and jax.default_backend() == "tpu"
        )
        return TopKConfig(platform="pallas" if use_pallas else "xla", backend="tpu" if use_pallas else "any")

    def candidate_cfgs(self, inv: Invocation[TopKConfig, tp.Any]) -> list[TopKConfig]:
        """Candidates for autotuning: the heuristic pick plus the XLA reference.

        Args:
            inv: The invocation being configured.

        Returns:
            Configurations to time against each other.
        """
        chosen = self.heuristic_cfg(inv)
        xla = TopKConfig(platform="xla", backend="any")
        return [chosen] if chosen.platform == "xla" else [chosen, xla]


_topk_executor = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("topk"),
    )
)


def topk(
    operand: Array,
    k: int | Array,
    axis: int = -1,
    mode: str = "values",
    mask_fill: float | None = None,
    *,
    cfg: TopKConfig | None = None,
) -> tuple[Array, Array] | Array:
    """Exact top-k along ``axis``.

    Args:
        operand: Input array.
        k: Number kept. Static ``int`` for ``mode="values"``; may be a per-row
            traced array for ``mode="mask"``/``"filter"``.
        axis: Reduction axis; defaults to the last.
        mode: ``"values"`` returns ``(values, indices)`` with ``jax.lax.top_k``
            semantics. ``"mask"`` returns a boolean keep-mask. ``"filter"``
            returns ``operand`` with dropped entries set to ``mask_fill``.
        mask_fill: Replacement value for ``mode="filter"``.

    Returns:
        Mode-dependent; see ``mode``.

    Example:
        >>> values, indices = topk(logits, k=6)
        >>> keep = topk(logits, per_row_k, mode="mask")
    """
    return _topk_executor(
        TopK(),
        operand,
        k,
        axis=axis,
        mode=mode,
        mask_fill=mask_fill,
        _cfg=cfg,
    )


__all__ = ("TopK", "topk")
