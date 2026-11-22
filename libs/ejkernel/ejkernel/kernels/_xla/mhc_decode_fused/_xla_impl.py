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


"""XLA reference of the decode-size mHC read (gates + collapse) and residual write over a packed gate table.

The gate table ``[splits, T, lanes]`` float32 carries, per token, the read gates ``pre`` at lanes ``[0, hc)``, the write
gates ``post`` at ``[slab, slab + hc)`` and the mixer ``comb[j, i]`` at ``(2 + j) * slab + i`` (``slab`` is ``hc``
rounded up to 8, ``lanes`` a multiple of 128). Every split holds the same gates; the XLA reference emits one split.
The arithmetic is the unfused one: the XLA ``mhc_coefficients``, ``sum_h pre_h x_h`` in float32, and
``post_i * out + sum_j comb_ji x_j`` with the gates rounded to the stream dtype first.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp

from ..mhc_coefficients._interface import mhc_coefficients as coefficients_reference

LANES = 128
MAX_TOKENS = LANES


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def gate_table_layout(hc: int) -> tuple[int, int]:
    """Return ``(slab, lanes)`` of the gate table for ``hc`` streams (see the module docstring)."""
    slab = _round_up(hc, 8)
    return slab, _round_up((2 + hc) * slab, LANES)


def table_gates(table: jax.Array, hc: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(pre [T, hc], post [T, hc], comb [T, hc, hc])`` float32 from a gate table ``[splits, T, lanes]``."""
    slab, _ = gate_table_layout(hc)
    first = table[0]
    comb = first[:, 2 * slab : (2 + hc) * slab].reshape(first.shape[0], hc, slab)[:, :, :hc]
    return first[:, :hc], first[:, slab : slab + hc], comb


def supports(tokens: int, hc: int, width: int) -> bool:
    """Whether the Pallas TPU kernels take ``tokens`` rows of ``hc`` streams of ``width`` features."""
    return 0 < tokens <= MAX_TOKENS and hc > 0 and hc & (hc - 1) == 0 and width % LANES == 0


def validate_read(logits: jax.Array, base: jax.Array, scale: jax.Array, streams: jax.Array, n_iters: int):
    """Check read shapes; return ``(tokens, hc, width)``.

    Raises:
        ValueError: Inconsistent shapes or a non-positive iteration count.
    """
    if logits.ndim != 2 or streams.ndim != 2:
        raise ValueError(f"mhc_decode_read: expected 2-D logits and streams, got {logits.shape}, {streams.shape}")
    tokens, mix = logits.shape
    hc = round((-2 + (4 + 4 * mix) ** 0.5) / 2)
    if hc < 1 or (2 + hc) * hc != mix or streams.shape[0] != tokens or streams.shape[1] % hc:
        raise ValueError(f"mhc_decode_read: logits {logits.shape}, streams {streams.shape}")
    if base.shape != (mix,) or scale.shape != (3,) or int(n_iters) < 1:
        raise ValueError(f"mhc_decode_read: base {base.shape}, scale {scale.shape}, n_iters {n_iters}")
    return tokens, hc, streams.shape[1] // hc


def validate_write(streams: jax.Array, out: jax.Array, table: jax.Array):
    """Check write shapes; return ``(tokens, hc, width)``.

    Raises:
        ValueError: Inconsistent shapes.
    """
    if streams.ndim != 2 or out.ndim != 2 or table.ndim != 3:
        raise ValueError(f"mhc_decode_write: streams {streams.shape}, out {out.shape}, table {table.shape}")
    tokens, total = streams.shape
    width = out.shape[-1]
    hc = total // width if width else 0
    if hc < 1 or hc * width != total or out.shape[0] != tokens or table.shape[1] != tokens:
        raise ValueError(f"mhc_decode_write: streams {streams.shape}, out {out.shape}, table {table.shape}")
    if table.shape[2] != gate_table_layout(hc)[1]:
        raise ValueError(f"mhc_decode_write: table {table.shape} does not match {hc} streams")
    return tokens, hc, width


def build_table(pre: jax.Array, post: jax.Array, comb: jax.Array) -> jax.Array:
    """Pack ``pre [T, hc]``, ``post [T, hc]`` and ``comb [T, hc, hc]`` into a one-split gate table (data movement)."""
    tokens, hc = pre.shape
    slab, lanes = gate_table_layout(hc)
    pad = [(0, 0), (0, slab - hc)]
    parts = [
        jnp.pad(pre.astype(jnp.float32), pad),
        jnp.pad(post.astype(jnp.float32), pad),
        jnp.pad(comb.astype(jnp.float32), [(0, 0), (0, 0), (0, slab - hc)]).reshape(tokens, hc * slab),
    ]
    table = jnp.concatenate(parts, axis=-1)
    return jnp.pad(table, [(0, 0), (0, lanes - table.shape[-1])])[None]


def mhc_decode_read_xla(logits, base, scale, streams, n_iters: int = 20, eps: float = 1e-6):
    """Gates and collapse (XLA reference); see :func:`ejkernel.modules.operations.mhc_decode_read`."""
    tokens, hc, width = validate_read(logits, base, scale, streams, n_iters)
    f32 = jnp.float32
    pre, post, comb = coefficients_reference(
        logits.astype(f32), base.astype(f32), scale.astype(f32), hc, int(n_iters), float(eps)
    )
    x = streams.reshape(tokens, hc, width)
    collapsed = jnp.sum(pre[..., None] * x.astype(f32), axis=1).astype(streams.dtype)
    return build_table(pre, post, comb), collapsed


def mhc_decode_write_xla(streams, out, table):
    """Residual write from a gate table (XLA reference); see :func:`ejkernel.modules.operations.mhc_decode_write`."""
    tokens, hc, width = validate_write(streams, out, table)
    _, post, comb = table_gates(table, hc)
    dtype = streams.dtype
    x = streams.reshape(tokens, hc, width)
    mixed = jnp.einsum("tji,tjd->tid", comb.astype(dtype), x)
    new = post.astype(dtype)[..., None] * out.astype(dtype)[:, None, :] + mixed
    return new.reshape(tokens, hc * width)


__all__ = (
    "LANES",
    "MAX_TOKENS",
    "build_table",
    "gate_table_layout",
    "mhc_decode_read_xla",
    "mhc_decode_write_xla",
    "supports",
    "table_gates",
    "validate_read",
    "validate_write",
)
