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


"""XLA registry entries (priority 0 fallback and reference) for the prefill-size mHC project / read / write."""

from __future__ import annotations

from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import mhc_prefill_project_xla, mhc_prefill_read_xla, mhc_prefill_write_xla


@kernel_registry.register("mhc_prefill_project", Platform.XLA, Backend.ANY, priority=0)
def mhc_prefill_project(
    streams: Float[Array, "num_tokens hc_width"],
    fn: Float[Array, "mix hc_width"],
) -> Float[Array, "stats_rows num_tokens"]:
    """Projection logits and sum of squares of a prefill token block (XLA reference).

    Args:
        streams: Streams flattened per token ``[T, hc * D]``.
        fn: Projection ``[(2 + hc) * hc, hc * D]``.

    Returns:
        ``stats [stats_rows, T]`` float32: logits in rows ``[0, mix)``, sum of squares in row ``mix``.
    """
    return mhc_prefill_project_xla(streams, fn)


@kernel_registry.register("mhc_prefill_read", Platform.XLA, Backend.ANY, priority=0)
def mhc_prefill_read(
    stats: Float[Array, "stats_rows num_tokens"],
    base: Float[Array, "mix"],
    scale: Float[Array, "3"],
    streams: Float[Array, "num_tokens hc_width"],
    global_width: int,
    norm_eps: float = 1e-6,
    n_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[Float[Array, "num_tokens lanes"], Float[Array, "num_tokens width"]]:
    """Gates and collapse of a prefill token block (XLA reference).

    Args:
        stats: Reduced projection statistics ``[stats_rows, T]`` float32.
        base: Gate offsets ``[(2 + hc) * hc]``.
        scale: Gate/mixer multipliers ``[3]``.
        streams: Streams flattened per token ``[T, hc * D]``.
        global_width: ``hc`` times the unsharded stream width (RMS denominator).
        norm_eps: RMS epsilon.
        n_iters: Sinkhorn iterations (positive).
        eps: Gate floor and normalization offset.

    Returns:
        ``(table [T, lanes] float32, collapsed [T, D] in streams.dtype)``.
    """
    return mhc_prefill_read_xla(
        stats, base, scale, streams, global_width=global_width, norm_eps=norm_eps, n_iters=n_iters, eps=eps
    )


@kernel_registry.register("mhc_prefill_write", Platform.XLA, Backend.ANY, priority=0)
def mhc_prefill_write(
    streams: Float[Array, "num_tokens hc_width"],
    out: Float[Array, "num_tokens width"],
    table: Float[Array, "num_tokens lanes"],
) -> Float[Array, "num_tokens hc_width"]:
    """Residual write ``post_i * out + sum_j comb_ji x_j`` from a gate table (XLA reference).

    Args:
        streams: Streams flattened per token ``[T, hc * D]``.
        out: Sub-layer output ``[T, D]``.
        table: Gate table ``[T, lanes]`` from ``mhc_prefill_read`` on the same streams.

    Returns:
        New streams ``[T, hc * D]`` in ``streams.dtype``.
    """
    return mhc_prefill_write_xla(streams, out, table)


__all__ = ("mhc_prefill_project", "mhc_prefill_read", "mhc_prefill_write")
