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


"""XLA registry entries (priority 0 fallback and reference) for the decode-size mHC read and write."""

from __future__ import annotations

from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import mhc_decode_read_xla, mhc_decode_write_xla


@kernel_registry.register("mhc_decode_read", Platform.XLA, Backend.ANY, priority=0)
def mhc_decode_read(
    logits: Float[Array, "num_tokens mix"],
    base: Float[Array, "mix"],
    scale: Float[Array, "3"],
    streams: Float[Array, "num_tokens hc_width"],
    n_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[Float[Array, "splits num_tokens lanes"], Float[Array, "num_tokens width"]]:
    """Gates and collapse of a decode token block (XLA reference).

    Args:
        logits: Float32 projection logits ``[T, (2 + hc) * hc]`` (RMS scale applied).
        base: Gate offsets ``[(2 + hc) * hc]``.
        scale: Gate/mixer multipliers ``[3]``.
        streams: Streams flattened per token ``[T, hc * D]``.
        n_iters: Sinkhorn iterations (positive).
        eps: Gate floor and normalization offset.

    Returns:
        ``(table [1, T, lanes] float32, collapsed [T, D] in streams.dtype)``.
    """
    return mhc_decode_read_xla(logits, base, scale, streams, n_iters=n_iters, eps=eps)


@kernel_registry.register("mhc_decode_write", Platform.XLA, Backend.ANY, priority=0)
def mhc_decode_write(
    streams: Float[Array, "num_tokens hc_width"],
    out: Float[Array, "num_tokens width"],
    table: Float[Array, "splits num_tokens lanes"],
) -> Float[Array, "num_tokens hc_width"]:
    """Residual write ``post_i * out + sum_j comb_ji x_j`` from a gate table (XLA reference).

    Args:
        streams: Streams flattened per token ``[T, hc * D]``.
        out: Sub-layer output ``[T, D]``.
        table: Gate table ``[splits, T, lanes]`` from ``mhc_decode_read`` on the same streams.

    Returns:
        New streams ``[T, hc * D]`` in ``streams.dtype``.
    """
    return mhc_decode_write_xla(streams, out, table)


__all__ = ("mhc_decode_read", "mhc_decode_write")
