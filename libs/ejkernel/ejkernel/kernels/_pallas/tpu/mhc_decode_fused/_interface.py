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


"""Pallas TPU registry entries for the decode-size mHC read (gates + collapse) and residual write. APPROXIMATE."""

from __future__ import annotations

from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.mhc_decode_fused._xla_impl import (
    mhc_decode_read_xla,
    mhc_decode_write_xla,
    supports,
    validate_read,
    validate_write,
)
from ._pallas_impl import mhc_decode_read_pallas, mhc_decode_write_pallas


@kernel_registry.register("mhc_decode_read", Platform.PALLAS, Backend.TPU, priority=10)
def mhc_decode_read(
    logits: Float[Array, "num_tokens mix"],
    base: Float[Array, "mix"],
    scale: Float[Array, "3"],
    streams: Float[Array, "num_tokens hc_width"],
    n_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[Float[Array, "splits num_tokens lanes"], Float[Array, "num_tokens width"]]:
    """Gates and collapse of a decode token block (Pallas TPU); same contract as the XLA reference.

    Gates and collapse match the XLA reference bit for bit. Shapes the kernel does not take (more than 128 tokens, a
    stream count that is not a power of two, a width that is not a multiple of 128) run the XLA reference.

    Args:
        logits: Float32 projection logits ``[T, (2 + hc) * hc]`` (RMS scale applied).
        base: Gate offsets ``[(2 + hc) * hc]``.
        scale: Gate/mixer multipliers ``[3]``.
        streams: Streams flattened per token ``[T, hc * D]``.
        n_iters: Sinkhorn iterations (positive).
        eps: Gate floor and normalization offset.

    Returns:
        ``(table [splits, T, lanes] float32, collapsed [T, D] in streams.dtype)``.
    """
    tokens, hc, width = validate_read(logits, base, scale, streams, n_iters)
    if not supports(tokens, hc, width):
        return mhc_decode_read_xla(logits, base, scale, streams, n_iters=n_iters, eps=eps)
    return mhc_decode_read_pallas(logits, base, scale, streams, n_iters=int(n_iters), eps=float(eps))


@kernel_registry.register("mhc_decode_write", Platform.PALLAS, Backend.TPU, priority=10)
def mhc_decode_write(
    streams: Float[Array, "num_tokens hc_width"],
    out: Float[Array, "num_tokens width"],
    table: Float[Array, "splits num_tokens lanes"],
) -> Float[Array, "num_tokens hc_width"]:
    """Residual write ``post_i * out + sum_j comb_ji x_j`` from a gate table (Pallas TPU). APPROXIMATE.

    Within one bf16 ulp of the XLA reference (a sequential float32 four-term sum and one final rounding). Unsupported
    shapes run the XLA reference.

    Args:
        streams: Streams flattened per token ``[T, hc * D]``.
        out: Sub-layer output ``[T, D]``.
        table: Gate table ``[splits, T, lanes]`` from ``mhc_decode_read`` on the same streams.

    Returns:
        New streams ``[T, hc * D]`` in ``streams.dtype``.
    """
    tokens, hc, width = validate_write(streams, out, table)
    if not supports(tokens, hc, width):
        return mhc_decode_write_xla(streams, out, table)
    return mhc_decode_write_pallas(streams, out, table)


__all__ = ("mhc_decode_read", "mhc_decode_write")
