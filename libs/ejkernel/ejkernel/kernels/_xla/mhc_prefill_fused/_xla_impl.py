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


"""XLA reference of the prefill-size mHC site: stream projection, gates + collapse, and residual write.

A site runs in three steps over streams flattened per token ``[T, hc * D]`` (stream ``h`` in columns
``[h D, (h + 1) D)``):

1. ``mhc_prefill_project`` returns ``stats [stats_rows, T]`` float32, channel-major: the projection ``fn @ streams.T``
   in rows ``[0, mix)`` and the sum of squares of the streams in row ``mix`` (the remaining rows are zero). Over a
   sharded hidden axis both are partial sums; the caller ``psum``s them.
2. ``mhc_prefill_read`` applies the RMS scale ``rsqrt(sumsq / global_width + norm_eps)`` to the logits, computes the
   gates (``mhc_coefficients`` arithmetic) and the collapse ``sum_h pre_h x_h`` in float32, and returns the gates as the
   decode gate table ``[T, lanes]`` (layout of :func:`ejkernel.kernels._xla.mhc_decode_fused.gate_table_layout`).
3. ``mhc_prefill_write`` computes ``post_i * out + sum_j comb_ji x_j`` with the gates rounded to the stream dtype.

The arithmetic is the unfused one (same formulas as the easydel manifold hyper-connection); only the operand layout
(the flattened streams) differs, which changes the projection's accumulation order.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp

from ..mhc_coefficients._interface import mhc_coefficients as coefficients_reference
from ..mhc_decode_fused._xla_impl import LANES, build_table, gate_table_layout, table_gates


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def hc_from_mix(mix: int) -> int:
    """Stream count ``hc`` with ``(2 + hc) * hc == mix`` (``0`` when there is none)."""
    hc = round((-2 + (4 + 4 * mix) ** 0.5) / 2)
    return hc if hc > 0 and (2 + hc) * hc == mix else 0


def stats_rows(mix: int) -> int:
    """Rows of the projection statistics: ``mix`` logits plus the sum of squares, rounded up to 8."""
    return _round_up(mix + 1, 8)


def supports(tokens: int, hc: int, width: int) -> bool:
    """Whether the Pallas TPU kernels take ``tokens`` rows of ``hc`` streams of ``width`` features.

    Prefill-size blocks only: more than :data:`LANES` tokens (decode-size blocks belong to the ``mhc_decode_*``
    kernels), a multiple of :data:`LANES`, a power-of-two stream count and a lane-aligned width.
    """
    return tokens > LANES and tokens % LANES == 0 and hc > 0 and hc & (hc - 1) == 0 and width % LANES == 0


def validate_project(streams: jax.Array, fn: jax.Array) -> tuple[int, int, int]:
    """Check projection shapes; return ``(tokens, hc, width)``.

    Raises:
        ValueError: Inconsistent shapes.
    """
    if streams.ndim != 2 or fn.ndim != 2 or fn.shape[1] != streams.shape[1]:
        raise ValueError(f"mhc_prefill_project: streams {streams.shape}, fn {fn.shape}")
    hc = hc_from_mix(fn.shape[0])
    if hc == 0 or streams.shape[1] % hc:
        raise ValueError(f"mhc_prefill_project: fn {fn.shape} is not [(2 + hc) * hc, hc * D] for streams")
    return streams.shape[0], hc, streams.shape[1] // hc


def validate_read(stats, base, scale, streams, global_width: int, n_iters: int) -> tuple[int, int, int]:
    """Check read shapes; return ``(tokens, hc, width)``.

    Raises:
        ValueError: Inconsistent shapes or non-positive static arguments.
    """
    mix = base.shape[0] if base.ndim == 1 else -1
    hc = hc_from_mix(mix)
    if hc == 0 or stats.ndim != 2 or streams.ndim != 2 or stats.shape != (stats_rows(mix), streams.shape[0]):
        raise ValueError(f"mhc_prefill_read: stats {stats.shape}, base {base.shape}, streams {streams.shape}")
    if streams.shape[1] % hc or scale.shape != (3,) or int(n_iters) < 1 or int(global_width) < 1:
        raise ValueError(f"mhc_prefill_read: streams {streams.shape}, scale {scale.shape}, n_iters {n_iters}")
    return streams.shape[0], hc, streams.shape[1] // hc


def validate_write(streams: jax.Array, out: jax.Array, table: jax.Array) -> tuple[int, int, int]:
    """Check write shapes; return ``(tokens, hc, width)``.

    Raises:
        ValueError: Inconsistent shapes.
    """
    if streams.ndim != 2 or out.ndim != 2 or table.ndim != 2:
        raise ValueError(f"mhc_prefill_write: streams {streams.shape}, out {out.shape}, table {table.shape}")
    tokens, total = streams.shape
    width = out.shape[1]
    hc = total // width if width else 0
    if hc < 1 or hc * width != total or out.shape[0] != tokens or table.shape[0] != tokens:
        raise ValueError(f"mhc_prefill_write: streams {streams.shape}, out {out.shape}, table {table.shape}")
    if table.shape[1] != gate_table_layout(hc)[1]:
        raise ValueError(f"mhc_prefill_write: table {table.shape} does not match {hc} streams")
    return tokens, hc, width


def mhc_prefill_project_xla(streams: jax.Array, fn: jax.Array) -> jax.Array:
    """Projection logits and sum of squares (XLA reference); see ``ejkernel.modules.mhc_prefill_project``."""
    validate_project(streams, fn)
    f32 = jnp.float32
    compute = jnp.result_type(streams.dtype, fn.dtype)
    logits = jnp.einsum("mk,tk->mt", fn.astype(compute), streams.astype(compute), preferred_element_type=f32)
    sumsq = jnp.sum(jnp.square(streams.astype(f32)), axis=-1)[None, :]
    pad = stats_rows(fn.shape[0]) - fn.shape[0] - 1
    return jnp.concatenate([logits, sumsq, jnp.zeros((pad, streams.shape[0]), f32)], axis=0)


def scaled_logits(stats: jax.Array, mix: int, global_width: int, norm_eps: float) -> jax.Array:
    """Token-major ``logits * rsqrt(sumsq / global_width + norm_eps)`` ``[T, mix]`` from (reduced) statistics."""
    scale = jax.lax.rsqrt(stats[mix : mix + 1, :] / global_width + norm_eps)
    return (stats[:mix, :] * scale).T


def mhc_prefill_read_xla(
    stats, base, scale, streams, global_width: int, norm_eps: float = 1e-6, n_iters: int = 20, eps: float = 1e-6
):
    """Gates and collapse (XLA reference); see ``ejkernel.modules.mhc_prefill_read``."""
    tokens, hc, width = validate_read(stats, base, scale, streams, global_width, n_iters)
    f32 = jnp.float32
    logits = scaled_logits(stats.astype(f32), base.shape[0], int(global_width), float(norm_eps))
    pre, post, comb = coefficients_reference(logits, base.astype(f32), scale.astype(f32), hc, int(n_iters), float(eps))
    x = streams.reshape(tokens, hc, width)
    collapsed = jnp.sum(pre[..., None] * x.astype(f32), axis=1).astype(streams.dtype)
    return build_table(pre, post, comb)[0], collapsed


def mhc_prefill_write_xla(streams, out, table):
    """Residual write from a gate table (XLA reference); see ``ejkernel.modules.mhc_prefill_write``."""
    tokens, hc, width = validate_write(streams, out, table)
    _, post, comb = table_gates(table[None], hc)
    dtype = streams.dtype
    x = streams.reshape(tokens, hc, width)
    mixed = jnp.einsum("tji,tjd->tid", comb.astype(dtype), x)
    new = post.astype(dtype)[..., None] * out.astype(dtype)[:, None, :] + mixed
    return new.reshape(tokens, hc * width)


__all__ = (
    "LANES",
    "gate_table_layout",
    "hc_from_mix",
    "mhc_prefill_project_xla",
    "mhc_prefill_read_xla",
    "mhc_prefill_write_xla",
    "scaled_logits",
    "stats_rows",
    "supports",
    "validate_project",
    "validate_read",
    "validate_write",
)
