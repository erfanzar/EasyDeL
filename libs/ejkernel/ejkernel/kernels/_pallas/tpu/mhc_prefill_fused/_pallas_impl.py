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

"""Prefill-size mHC site as three Pallas TPU kernels over token tiles. APPROXIMATE class, opt-in.

Each kernel is one sweep over row tiles of the streams flattened per token ``[T, hc * D]`` (``grid = T / tile``,
parallel, so the TensorCores split the tiles):

* :func:`mhc_prefill_project_pallas`: ``stats [rows, T]`` float32, channel-major: ``fn @ x.T`` on the MXU (bf16
  operands, float32 accumulation) in rows ``[0, mix)`` and the float32 sum of squares in row ``mix``.
* :func:`mhc_prefill_read_pallas`: RMS scale, the packed coefficient arithmetic with the tokens on lanes (gates,
  softmax and Sinkhorn are elementwise over vregs), an in-kernel transpose to the token-major gate table
  ``[T, lanes]`` and the collapse ``sum_h pre_h x_h`` in float32 with the four-term tree ``(x0 + x2) + (x1 + x3)``.
* :func:`mhc_prefill_write_pallas`: ``post_i * out + sum_j comb_ji x_j`` with the gates rounded to the stream dtype, a
  float32 sum and one final rounding (what TPU XLA's fused write does; on TPU v5p at 32768 x 4 x 1024 it matched the
  in-model XLA write on all but 0.01% of the elements, each within one bf16 ulp).

Approximate against the unfused XLA path: the projection and the sum of squares accumulate in another order, and the
mix's float32 sum order may differ from XLA's einsum. Tiles come from a VMEM rule (:func:`token_tile`).
"""

from __future__ import annotations

import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...._xla.mhc_decode_fused._xla_impl import gate_table_layout
from ...._xla.mhc_prefill_fused._xla_impl import LANES, stats_rows, supports, validate_project, validate_read
from ...._xla.mhc_prefill_fused._xla_impl import validate_write as _validate_write

_F32 = jnp.float32
_NT = (((1,), (1,)), ((), ()))  # [m, k] x [n, k]
_TILES = (1024, 512, 256, 128)


def _tpu_field(name: str, default: int) -> int:
    try:
        return int(getattr(pltpu.get_tpu_info(), name))
    except Exception:  # not a TPU (interpret mode) or an older runtime
        return default


def _vmem_capacity() -> int:
    return _tpu_field("vmem_capacity_bytes", 64 << 20)


def _params(estimate: int) -> pltpu.CompilerParams:
    capacity = _vmem_capacity()
    limit = int(min(capacity - (2 << 20), max(32 << 20, estimate + estimate // 2)))
    return pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=limit)


def _chunk(width: int) -> int:
    for size in (512, 256, 128):
        if width % size == 0:
            return size
    return width


def token_tile(tokens: int, per_row: int, fixed: int = 0) -> int:
    """Largest token tile (1024, 512, 256 or 128 rows dividing ``tokens``) whose VMEM estimate fits half the VMEM.

    Args:
        tokens: Rows of the call.
        per_row: Estimated VMEM bytes per tile row.
        fixed: Estimated VMEM bytes independent of the tile.

    Returns:
        The tile, or 128 when no larger one fits.
    """
    budget = _vmem_capacity() // 2
    for tile in _TILES:
        if tokens % tile == 0 and fixed + tile * per_row <= budget:
            return tile
    return LANES


def _pick(table, lanes, k):
    """Column ``k`` of ``table`` as ``[T, 1]``: one value summed with zeros, so exact."""
    return jnp.sum(jnp.where(lanes == k, table, 0.0), axis=1, keepdims=True)


def _project_kernel(fn_ref, x_ref, out_ref, *, mix):
    tile, total = x_ref.shape
    logits = lax.dot_general(fn_ref[...], x_ref[...], _NT, preferred_element_type=_F32)
    acc = jnp.zeros((tile, LANES), _F32)
    for k in range(total // LANES):
        part = x_ref[:, k * LANES : (k + 1) * LANES].astype(_F32)
        acc = acc + part * part
    sumsq = jnp.sum(acc.T, axis=0, keepdims=True)
    rows = lax.broadcasted_iota(jnp.int32, logits.shape, 0)
    out_ref[...] = jnp.where(rows == mix, sumsq, logits)


def mhc_prefill_project_pallas(streams: jax.Array, fn: jax.Array, *, interpret: bool | None = None) -> jax.Array:
    """Projection logits and sum of squares of a prefill token block.

    Args:
        streams: Streams flattened per token ``[T, hc * D]``.
        fn: Projection ``[(2 + hc) * hc, hc * D]`` (cast to the stream dtype for the MXU).
        interpret: Pallas interpret mode (default: off TPU only).

    Returns:
        ``stats [stats_rows(mix), T]`` float32 (partial sums over this block's features).

    Raises:
        ValueError: Unsupported shapes.
    """
    tokens, hc, width = validate_project(streams, fn)
    if not supports(tokens, hc, width):
        raise ValueError(f"mhc_prefill_project: unsupported tokens={tokens} hc={hc} width={width}")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    mix, total = fn.shape
    rows = stats_rows(mix)
    item = jnp.dtype(streams.dtype).itemsize
    per_row = 3 * total * item + 6 * LANES * 4 + 2 * rows * 4
    fixed = 2 * rows * total * item
    tile = token_tile(tokens, per_row, fixed)
    weight = jnp.pad(fn.astype(streams.dtype), [(0, rows - mix), (0, 0)])
    return pl.pallas_call(
        lambda *refs: _project_kernel(*refs, mix=mix),
        grid=(tokens // tile,),
        in_specs=[pl.BlockSpec((rows, total), lambda t: (0, 0)), pl.BlockSpec((tile, total), lambda t: (t, 0))],
        out_specs=pl.BlockSpec((rows, tile), lambda t: (0, t)),
        out_shape=jax.ShapeDtypeStruct((rows, tokens), _F32),
        compiler_params=_params(fixed + tile * per_row),
        interpret=interpret,
        name="mhc_prefill_project",
    )(weight, streams)


def _read_kernel(s_ref, b_ref, c_ref, x_ref, table_ref, out_ref, scratch, **kw):
    hc, mix, slab, chunk = kw["hc"], kw["mix"], kw["slab"], kw["chunk"]
    eps, n_iters = kw["eps"], kw["n_iters"]
    stats, base, scale = s_ref[...], b_ref[...], c_ref[...]
    tile = stats.shape[1]
    rms = lax.rsqrt(stats[mix : mix + 1, :] / kw["global_width"] + kw["norm_eps"])
    logits = stats[:mix, :] * rms
    pre = jax.nn.sigmoid(logits[:hc, :] * scale[0, 0] + base[:hc, :]) + eps
    post = 2 * jax.nn.sigmoid(logits[hc : 2 * hc, :] * scale[1, 0] + base[hc : 2 * hc, :])
    mixer = (logits[2 * hc :, :] * scale[2, 0] + base[2 * hc :, :]).reshape(hc, hc, tile)
    matrix = jnp.exp(mixer - jnp.max(mixer, axis=1, keepdims=True))
    matrix = matrix / jnp.sum(matrix, axis=1, keepdims=True) + eps
    matrix = matrix / (jnp.sum(matrix, axis=0, keepdims=True) + eps)

    def body(_, value):
        value = value / (jnp.sum(value, axis=1, keepdims=True) + eps)
        return value / (jnp.sum(value, axis=0, keepdims=True) + eps)

    matrix = lax.fori_loop(0, n_iters - 1, body, matrix)
    scratch[...] = jnp.zeros(scratch.shape, _F32)
    scratch[0:hc, :] = pre
    scratch[slab : slab + hc, :] = post
    for row in range(hc):
        scratch[(2 + row) * slab : (2 + row) * slab + hc, :] = matrix[row]
    table = scratch[...].T
    table_ref[...] = table
    lanes = lax.broadcasted_iota(jnp.int32, table.shape, 1)
    gates = [_pick(table, lanes, h) for h in range(hc)]
    width = out_ref.shape[1]
    for k in range(width // chunk):
        terms = [gates[h] * x_ref[:, pl.ds(h * width + k * chunk, chunk)].astype(_F32) for h in range(hc)]
        while len(terms) > 1:
            half = len(terms) // 2
            terms = [terms[i] + terms[i + half] for i in range(half)]
        out_ref[:, pl.ds(k * chunk, chunk)] = terms[0].astype(out_ref.dtype)


def mhc_prefill_read_pallas(
    stats: jax.Array,
    base: jax.Array,
    scale: jax.Array,
    streams: jax.Array,
    *,
    global_width: int,
    norm_eps: float = 1e-6,
    n_iters: int = 20,
    eps: float = 1e-6,
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Gates and collapse of a prefill token block from reduced projection statistics.

    Args:
        stats: Float32 ``[stats_rows(mix), T]`` from :func:`mhc_prefill_project_pallas`, summed over feature shards.
        base: Gate offsets ``[(2 + hc) * hc]``.
        scale: Gate/mixer multipliers ``[3]``.
        streams: Streams flattened per token ``[T, hc * D]``.
        global_width: ``hc`` times the unsharded stream width (the RMS denominator).
        norm_eps: RMS epsilon.
        n_iters: Sinkhorn iterations (positive).
        eps: Gate floor and normalization offset.
        interpret: Pallas interpret mode (default: off TPU only).

    Returns:
        ``table [T, lanes]`` float32 (decode gate-table layout) and ``collapsed [T, D]`` in ``streams.dtype``.

    Raises:
        ValueError: Unsupported shapes.
    """
    tokens, hc, width = validate_read(stats, base, scale, streams, global_width, n_iters)
    if not supports(tokens, hc, width):
        raise ValueError(f"mhc_prefill_read: unsupported tokens={tokens} hc={hc} width={width}")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    mix = base.shape[0]
    rows = stats.shape[0]
    slab, lanes = gate_table_layout(hc)
    chunk = _chunk(width)
    item = jnp.dtype(streams.dtype).itemsize
    per_row = 2 * (rows * 4 + hc * width * item + lanes * 4 + width * item) + 3 * lanes * 4 + (hc + 3) * chunk * 4
    tile = token_tile(tokens, per_row)
    kw = dict(
        hc=hc,
        mix=mix,
        slab=slab,
        chunk=chunk,
        eps=float(eps),
        n_iters=int(n_iters),
        norm_eps=float(norm_eps),
        global_width=float(global_width),
    )
    return pl.pallas_call(
        lambda *refs: _read_kernel(*refs, **kw),
        grid=(tokens // tile,),
        in_specs=[
            pl.BlockSpec((rows, tile), lambda t: (0, t)),
            pl.BlockSpec((mix, 1), lambda t: (0, 0)),
            pl.BlockSpec((3, 1), lambda t: (0, 0)),
            pl.BlockSpec((tile, hc * width), lambda t: (t, 0)),
        ],
        out_specs=(pl.BlockSpec((tile, lanes), lambda t: (t, 0)), pl.BlockSpec((tile, width), lambda t: (t, 0))),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, lanes), _F32),
            jax.ShapeDtypeStruct((tokens, width), streams.dtype),
        ),
        scratch_shapes=[pltpu.VMEM((lanes, tile), _F32)],
        compiler_params=_params(tile * per_row),
        interpret=interpret,
        name="mhc_prefill_read",
    )(stats.astype(_F32), base.astype(_F32)[:, None], scale.astype(_F32)[:, None], streams)


def _write_kernel(x_ref, o_ref, t_ref, y_ref, *, hc, slab, width, chunk):
    table = t_ref[...]
    lanes = lax.broadcasted_iota(jnp.int32, table.shape, 1)
    dtype = y_ref.dtype

    def gate(k):
        return _pick(table, lanes, k).astype(dtype).astype(_F32)

    post = [gate(slab + i) for i in range(hc)]
    comb = [[gate((2 + j) * slab + i) for i in range(hc)] for j in range(hc)]
    for k in range(width // chunk):
        out = o_ref[:, pl.ds(k * chunk, chunk)].astype(_F32)
        xs = [x_ref[:, pl.ds(j * width + k * chunk, chunk)].astype(_F32) for j in range(hc)]
        for i in range(hc):
            acc = comb[0][i] * xs[0]
            for j in range(1, hc):
                acc = acc + comb[j][i] * xs[j]
            # One final rounding: what XLA's fused residual write does on TPU (measured, pf-mhc W1).
            y_ref[:, pl.ds(i * width + k * chunk, chunk)] = (post[i] * out + acc).astype(dtype)


def mhc_prefill_write_pallas(
    streams: jax.Array, out: jax.Array, table: jax.Array, *, interpret: bool | None = None
) -> jax.Array:
    """Residual write ``post_i * out + sum_j comb_ji x_j`` from a :func:`mhc_prefill_read_pallas` gate table.

    Args:
        streams: Streams flattened per token ``[T, hc * D]``.
        out: Sub-layer output ``[T, D]``.
        table: Gate table ``[T, lanes]`` from :func:`mhc_prefill_read_pallas` on the same streams.
        interpret: Pallas interpret mode (default: off TPU only).

    Returns:
        New streams ``[T, hc * D]`` in ``streams.dtype``.

    Raises:
        ValueError: Unsupported shapes.
    """
    tokens, hc, width = _validate_write(streams, out, table)
    if not supports(tokens, hc, width):
        raise ValueError(f"mhc_prefill_write: unsupported streams {streams.shape} / out {out.shape}")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    slab, lanes = gate_table_layout(hc)
    chunk = _chunk(width)
    item = jnp.dtype(streams.dtype).itemsize
    total = hc * width
    per_row = 2 * (2 * total * item + width * item + lanes * 4) + (5 * hc + 2) * LANES * 4 + (hc + 3) * chunk * 4
    tile = token_tile(tokens, per_row)
    return pl.pallas_call(
        lambda *refs: _write_kernel(*refs, hc=hc, slab=slab, width=width, chunk=chunk),
        grid=(tokens // tile,),
        in_specs=[
            pl.BlockSpec((tile, total), lambda t: (t, 0)),
            pl.BlockSpec((tile, width), lambda t: (t, 0)),
            pl.BlockSpec((tile, lanes), lambda t: (t, 0)),
        ],
        out_specs=pl.BlockSpec((tile, total), lambda t: (t, 0)),
        out_shape=jax.ShapeDtypeStruct((tokens, total), streams.dtype),
        compiler_params=_params(tile * per_row),
        interpret=interpret,
        name="mhc_prefill_write",
    )(streams, out.astype(streams.dtype), table.astype(_F32))


__all__ = ("mhc_prefill_project_pallas", "mhc_prefill_read_pallas", "mhc_prefill_write_pallas", "token_tile")
