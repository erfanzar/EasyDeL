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


"""Decode-size mHC read (gates + collapse) and residual write over a packed gate table. APPROXIMATE class.

Serving decode only (forward, at most 128 tokens per device on the Pallas path); callers opt in explicitly.
:func:`mhc_decode_read` turns projection logits ``[T, (2 + hc) * hc]`` and flattened streams ``[T, hc * D]`` into a gate
table ``[splits, T, lanes]`` (read gates, write gates and the Sinkhorn mixer per token; layout in
:func:`mhc_decode_gate_layout`) and the collapse ``sum_h pre_h x_h``. :func:`mhc_decode_write` computes
``post_i * out + sum_j comb_ji x_j`` from that table. The XLA implementation (priority 0) is the reference and fallback;
the Pallas TPU read matches it bit for bit and the write to one bf16 ulp, so a chain of layers is not bitwise equal.
"""

from __future__ import annotations

from typing import Literal

from jax import Array

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.kernels._xla.mhc_decode_fused._xla_impl import gate_table_layout, supports, table_gates
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel

from ..base import detect_platform
from .configs import MHCDecodeFusedConfig


def _resolve(op_id: str, cfg: MHCDecodeFusedConfig, platform: str | None):
    if platform is not None:
        cfg = MHCDecodeFusedConfig(platform=platform, backend=Backend.ANY if platform == "xla" else cfg.backend)
    return kernel_registry.get(op_id, platform=detect_platform(op_id, cfg.platform), backend=cfg.backend)


class MHCDecodeRead(Kernel[MHCDecodeFusedConfig, tuple[Array, Array]]):
    """Decode mHC gates + collapse (Pallas TPU; XLA reference)."""

    def __init__(self):
        """Initialize under ``op_id="mhc_decode_read"``."""
        super().__init__(op_id="mhc_decode_read")

    def run(self, logits, base, scale, streams, n_iters=20, eps=1e-6, platform=None, *, cfg):
        """Dispatch to the selected backend; see :func:`mhc_decode_read`."""
        return _resolve(self.op_id, cfg, platform)(logits, base, scale, streams, n_iters=n_iters, eps=eps)

    def heuristic_cfg(self, inv: Invocation) -> MHCDecodeFusedConfig:
        """Fixed tiles from the TPU info: nothing to tune."""
        return MHCDecodeFusedConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation):
        """Return the single heuristic configuration."""
        return [self.heuristic_cfg(inv)]


class MHCDecodeWrite(Kernel[MHCDecodeFusedConfig, Array]):
    """Decode mHC residual write from a gate table (Pallas TPU; XLA reference)."""

    def __init__(self):
        """Initialize under ``op_id="mhc_decode_write"``."""
        super().__init__(op_id="mhc_decode_write")

    def run(self, streams, out, table, platform=None, *, cfg):
        """Dispatch to the selected backend; see :func:`mhc_decode_write`."""
        return _resolve(self.op_id, cfg, platform)(streams, out, table)

    def heuristic_cfg(self, inv: Invocation) -> MHCDecodeFusedConfig:
        """Fixed tiles from the TPU info: nothing to tune."""
        return MHCDecodeFusedConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation):
        """Return the single heuristic configuration."""
        return [self.heuristic_cfg(inv)]


_executor = Executor(ConfigSelectorChain(cache=ConfigCache(), policy=AutotunePolicy(allow_autotune=False)))


def mhc_decode_read(
    logits: Array,
    base: Array,
    scale: Array,
    streams: Array,
    /,
    n_iters: int = 20,
    eps: float = 1e-6,
    *,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: MHCDecodeFusedConfig | None = None,
) -> tuple[Array, Array]:
    """Gates and collapse of a decode token block.

    Args:
        logits: Float32 projection logits ``[T, (2 + hc) * hc]`` (RMS scale applied).
        base: Gate offsets ``[(2 + hc) * hc]``.
        scale: Gate/mixer multipliers ``[3]``.
        streams: Streams flattened per token ``[T, hc * D]`` (stream ``h`` in columns ``[h D, (h + 1) D)``).
        n_iters: Sinkhorn iterations (positive, static).
        eps: Gate floor and normalization offset (static).
        platform: Implementation override (``"xla"`` / ``"pallas"``).
        cfg: Optional dispatch configuration.

    Returns:
        ``(table [splits, T, lanes] float32, collapsed [T, D] in streams.dtype)``; read the gates with
        :func:`mhc_decode_table_gates`.

    Raises:
        ValueError: Inconsistent shapes.
    """
    return _executor(
        MHCDecodeRead(), logits, base, scale, streams, n_iters=n_iters, eps=eps, platform=platform, _cfg=cfg
    )


def mhc_decode_write(
    streams: Array,
    out: Array,
    table: Array,
    /,
    *,
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: MHCDecodeFusedConfig | None = None,
) -> Array:
    """Residual write ``post_i * out + sum_j comb_ji x_j`` from a :func:`mhc_decode_read` gate table.

    Args:
        streams: Streams flattened per token ``[T, hc * D]``.
        out: Sub-layer output ``[T, D]``.
        table: Gate table ``[splits, T, lanes]`` from :func:`mhc_decode_read` on the same streams.
        platform: Implementation override (``"xla"`` / ``"pallas"``).
        cfg: Optional dispatch configuration.

    Returns:
        New streams ``[T, hc * D]`` in ``streams.dtype``.

    Raises:
        ValueError: Inconsistent shapes.
    """
    return _executor(MHCDecodeWrite(), streams, out, table, platform=platform, _cfg=cfg)


def mhc_decode_supports(tokens: int, hc: int, width: int) -> bool:
    """Whether the Pallas TPU kernels take ``tokens`` rows of ``hc`` streams of ``width`` features."""
    return supports(tokens, hc, width)


mhc_decode_gate_layout = gate_table_layout
mhc_decode_table_gates = table_gates

__all__ = (
    "MHCDecodeFusedConfig",
    "MHCDecodeRead",
    "MHCDecodeWrite",
    "mhc_decode_gate_layout",
    "mhc_decode_read",
    "mhc_decode_supports",
    "mhc_decode_table_gates",
    "mhc_decode_write",
)
