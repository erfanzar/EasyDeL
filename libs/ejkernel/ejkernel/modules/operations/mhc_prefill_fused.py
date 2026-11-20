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


"""Prefill-size mHC site: stream projection, read (gates + collapse) and residual write. APPROXIMATE class.

Serving prefill only (forward, more than 128 tokens per device on the Pallas path); callers opt in explicitly.
:func:`mhc_prefill_project` reduces flattened streams ``[T, hc * D]`` to channel-major statistics ``[rows, T]``
(projection logits and the sum of squares; partial over a sharded hidden axis, so the caller ``psum``s them).
:func:`mhc_prefill_read` turns the reduced statistics into the decode gate table ``[T, lanes]`` and the collapse
``sum_h pre_h x_h``. :func:`mhc_prefill_write` computes ``post_i * out + sum_j comb_ji x_j`` from the table. The XLA
implementation (priority 0) is the reference and fallback; the Pallas TPU path accumulates the projection, the sum of
squares and the write in another order, so it is not bitwise equal to the unfused model path.
"""

from __future__ import annotations

from typing import Literal

from jax import Array

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.kernels._xla.mhc_prefill_fused._xla_impl import stats_rows, supports
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel

from ..base import detect_platform
from .configs import MHCPrefillFusedConfig


def _resolve(op_id: str, cfg: MHCPrefillFusedConfig, platform: str | None):
    if platform is not None:
        cfg = MHCPrefillFusedConfig(platform=platform, backend=Backend.ANY if platform == "xla" else cfg.backend)
    return kernel_registry.get(op_id, platform=detect_platform(op_id, cfg.platform), backend=cfg.backend)


class _Fixed:
    def heuristic_cfg(self, inv: Invocation) -> MHCPrefillFusedConfig:
        """Tiles come from a VMEM rule inside the kernels: nothing to tune."""
        return MHCPrefillFusedConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation):
        """Return the single heuristic configuration."""
        return [self.heuristic_cfg(inv)]


class MHCPrefillProject(_Fixed, Kernel[MHCPrefillFusedConfig, Array]):
    """Prefill mHC projection statistics (Pallas TPU; XLA reference)."""

    def __init__(self):
        """Initialize under ``op_id="mhc_prefill_project"``."""
        super().__init__(op_id="mhc_prefill_project")

    def run(self, streams, fn, platform=None, *, cfg):
        """Dispatch to the selected backend; see :func:`mhc_prefill_project`."""
        return _resolve(self.op_id, cfg, platform)(streams, fn)


class MHCPrefillRead(_Fixed, Kernel[MHCPrefillFusedConfig, tuple[Array, Array]]):
    """Prefill mHC gates + collapse (Pallas TPU; XLA reference)."""

    def __init__(self):
        """Initialize under ``op_id="mhc_prefill_read"``."""
        super().__init__(op_id="mhc_prefill_read")

    def run(self, stats, base, scale, streams, global_width, norm_eps=1e-6, n_iters=20, eps=1e-6, platform=None, *, cfg):
        """Dispatch to the selected backend; see :func:`mhc_prefill_read`."""
        return _resolve(self.op_id, cfg, platform)(
            stats, base, scale, streams, global_width=global_width, norm_eps=norm_eps, n_iters=n_iters, eps=eps
        )


class MHCPrefillWrite(_Fixed, Kernel[MHCPrefillFusedConfig, Array]):
    """Prefill mHC residual write from a gate table (Pallas TPU; XLA reference)."""

    def __init__(self):
        """Initialize under ``op_id="mhc_prefill_write"``."""
        super().__init__(op_id="mhc_prefill_write")

    def run(self, streams, out, table, platform=None, *, cfg):
        """Dispatch to the selected backend; see :func:`mhc_prefill_write`."""
        return _resolve(self.op_id, cfg, platform)(streams, out, table)


_executor = Executor(ConfigSelectorChain(cache=ConfigCache(), policy=AutotunePolicy(allow_autotune=False)))


def mhc_prefill_project(
    streams: Array,
    fn: Array,
    /,
    *,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: MHCPrefillFusedConfig | None = None,
) -> Array:
    """Projection logits and sum of squares of a prefill token block.

    Args:
        streams: Streams flattened per token ``[T, hc * D]``.
        fn: Projection ``[(2 + hc) * hc, hc * D]`` (this block's feature columns).
        platform: Implementation override (``"xla"`` / ``"pallas"``).
        cfg: Optional dispatch configuration.

    Returns:
        ``stats [mhc_prefill_stats_rows(mix), T]`` float32: logits in rows ``[0, mix)``, sum of squares in row ``mix``.

    Raises:
        ValueError: Inconsistent shapes.
    """
    return _executor(MHCPrefillProject(), streams, fn, platform=platform, _cfg=cfg)


def mhc_prefill_read(
    stats: Array,
    base: Array,
    scale: Array,
    streams: Array,
    /,
    global_width: int,
    norm_eps: float = 1e-6,
    n_iters: int = 20,
    eps: float = 1e-6,
    *,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: MHCPrefillFusedConfig | None = None,
) -> tuple[Array, Array]:
    """Gates and collapse of a prefill token block from reduced projection statistics.

    Args:
        stats: Float32 ``[stats_rows, T]`` from :func:`mhc_prefill_project`, summed over feature shards.
        base: Gate offsets ``[(2 + hc) * hc]``.
        scale: Gate/mixer multipliers ``[3]``.
        streams: Streams flattened per token ``[T, hc * D]``.
        global_width: ``hc`` times the unsharded stream width (RMS denominator, static).
        norm_eps: RMS epsilon (static).
        n_iters: Sinkhorn iterations (positive, static).
        eps: Gate floor and normalization offset (static).
        platform: Implementation override (``"xla"`` / ``"pallas"``).
        cfg: Optional dispatch configuration.

    Returns:
        ``(table [T, lanes] float32, collapsed [T, D] in streams.dtype)``; the table has the decode gate-table layout.

    Raises:
        ValueError: Inconsistent shapes.
    """
    return _executor(
        MHCPrefillRead(),
        stats,
        base,
        scale,
        streams,
        global_width=global_width,
        norm_eps=norm_eps,
        n_iters=n_iters,
        eps=eps,
        platform=platform,
        _cfg=cfg,
    )


def mhc_prefill_write(
    streams: Array,
    out: Array,
    table: Array,
    /,
    *,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: MHCPrefillFusedConfig | None = None,
) -> Array:
    """Residual write ``post_i * out + sum_j comb_ji x_j`` from a :func:`mhc_prefill_read` gate table.

    Args:
        streams: Streams flattened per token ``[T, hc * D]``.
        out: Sub-layer output ``[T, D]``.
        table: Gate table ``[T, lanes]`` from :func:`mhc_prefill_read` on the same streams.
        platform: Implementation override (``"xla"`` / ``"pallas"``).
        cfg: Optional dispatch configuration.

    Returns:
        New streams ``[T, hc * D]`` in ``streams.dtype``.

    Raises:
        ValueError: Inconsistent shapes.
    """
    return _executor(MHCPrefillWrite(), streams, out, table, platform=platform, _cfg=cfg)


def mhc_prefill_supports(tokens: int, hc: int, width: int) -> bool:
    """Whether the Pallas TPU prefill kernels take ``tokens`` rows of ``hc`` streams of ``width`` features."""
    return supports(tokens, hc, width)


mhc_prefill_stats_rows = stats_rows

__all__ = (
    "MHCPrefillFusedConfig",
    "MHCPrefillProject",
    "MHCPrefillRead",
    "MHCPrefillWrite",
    "mhc_prefill_project",
    "mhc_prefill_read",
    "mhc_prefill_stats_rows",
    "mhc_prefill_supports",
    "mhc_prefill_write",
)
