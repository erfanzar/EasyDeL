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

"""Fused mHC (manifold hyper-connection) collapse / expand, Pallas TPU, with an analytic backward.

The contract is :mod:`ejkernel.kernels._xla.mhc_fused._xla_impl_fwd` (read its docstring first). The streams
``x [B, S, hc, W]`` are the site's big operand (``[32768, 4, 1024]`` bf16 = 256 MiB per device); everything else is
per-token scalars or ``[rows, W]``. XLA runs a site as several loop fusions that each re-read the streams; here
every kernel is ONE sweep over the rows with a ``[tile_rows, hc * W]`` block resident, and the per-token gate /
stats values travel as one lane-dense ``[rows, 128]`` fp32 row (:data:`LANES`; layout in the XLA module).

Why three calls per site and not one: the predictor logits contract the (tensor-parallel sharded) hidden and need
a ``psum`` before the gates exist, and the expand needs the sub-layer output computed in between. So the minimum
is stats -> (psum, gates in XLA) -> collapse, then expand after the sub-layer.

Kernels (each a single grid over row tiles):

* ``stats``: ``sumsq[s]`` and the ``fn`` projection ``mixp[s, m]`` from one read of the streams. The projection is
  an MXU dot ``[tile, hc W] x [hc W, 128]`` with ``fn`` padded to 128 rows (row ``m + 1`` = ``fn[m]``). The
  reference runs it in fp32 at HIGHEST precision; the kernel reproduces that with bf16 split pieces (a bf16 stream
  times a bf16 ``fn`` is exact in one dot; an fp32 ``fn`` is ``fn_pieces`` dots; an fp32 stream splits too, with
  the bf16x6 term rule), so the result is the fp32 product summed in a different order (~1e-6 relative). Sums over
  a hidden shard are partial; the wrapper ``psum``s them over ``axis_name``.
* ``collapse``: ``collapsed = sum_h pre_h x_h`` -- one read.
* ``expand``: ``x'_i = post_i out + sum_j comb_ji x_j`` -- one read of the streams + the sub-layer output, one
  write. The gates are quantised to the streams' dtype first (the residual write casts ``post`` / ``comb``).
* ``expand_bwd``: ``dx_j = sum_i comb_ji g_i``, ``dout = sum_i post_i g_i``, ``dpost_i = <g_i, out>``,
  ``dcomb_ji = <x_j, g_i>`` -- reads ``g`` and ``x``, writes ``dx``.
* ``collapse_bwd_dpre``: ``dpre_h = <g_c, x_h>`` -- one read (the gate VJP needs it before ``dx``).
* ``collapse_bwd_dx``: ``dx_h = pre_h g_c + (dlogits @ fn)_h + 2 dsumsq x_h`` and the ``fn`` cotangent
  ``dfn = dlogits^T x`` accumulated across the grid -- one read, one write.

Layout: the kernels view the streams as ``[rows, hc * W]`` (``x.reshape(B * S, hc * W)`` at the op boundary), so
stream ``h`` is the lane-aligned slice ``[:, h W:(h + 1) W]`` and no block has a sublane dimension of ``hc``. In
the compiled HLO of both paths no relayout copy of the stream shape was observed. Blocks need ``W % 128 == 0`` and
``tile_rows % 128 == 0`` (the ``[tile, 128] -> [128, tile]`` fp32 transpose in ``collapse_bwd_dx``). VMEM is
requested from a working-set estimate (:func:`vmem_estimate_bytes`), capped at the local capacity minus 1 MiB.

Stream passes per site (256 MiB each at the shape above): forward ~4.5 (stats 1, collapse 1, expand 1 read +
1 write, plus the ``[rows, W]`` operands), backward ~7 in the kernels plus the ``dx`` sum XLA adds between the two
ops' cotangents. A fused "boundary" primitive (expand of one site + collapse of the next sharing the stream read)
would remove that add; it is not implemented.

Measured on one TPU v5p chip at ``[1, 32768, 4, 1024]`` bf16 with an fp32 predictor (one mHC site = collapse +
expand): XLA einsum site 1.186 ms forward / 3.503 ms forward+backward (about 4x the HBM roofline), Pallas site
9.277 / 12.212 ms. The Pallas collapse alone (1.16 ms) is about 2x faster than XLA's collapse (2.4 ms); the loss is
in ``expand`` / ``expand_bwd``, which are VPU-bound (the per-token ``hc x hc`` mix is ``hc^2`` broadcast FMAs per
element with the gates extracted by masked lane reductions). XLA therefore stays the default for both ops.

Differentiation under ``shard_map``: both ops are written for ``jax.shard_map(..., check_vma=False)``, whose
transpose divides a replicated output's cotangent by the size of its unmentioned mesh axes and sums a replicated
input's cotangent over them. The backward rules therefore return per-device (partial) gate / parameter
cotangents and let the transpose do the reduction; the only collective inside a backward is the ``psum`` of the
logit cotangent in ``mhc_collapse`` (the transpose of the forward stats ``psum``). This matches what JAX's own AD
does for the XLA reference body, so the two implementations agree under the same ``shard_map``.

Off TPU the kernels run in Pallas interpret mode.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float

from ...._xla.mhc_fused._xla_impl_fwd import (
    LANES,
    check_hc,
    gates_from_stats,
    mix_width,
    pack_gates,
    resolve_hidden_size,
    unpack_gates,
)

_F32 = jnp.float32
_BF16 = jnp.bfloat16
_DEFAULT_VMEM = 16 * 2**20  # Mosaic's default scoped limit (and a v4's whole VMEM)
_NN = (((1,), (0,)), ((), ()))  # [m, k] x [k, n]
_NT = (((1,), (1,)), ((), ()))  # [m, k] x [n, k]


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def _vmem_capacity() -> int:
    """Physical VMEM of the local TPU generation (16 MiB off TPU or when the query is unavailable)."""
    try:
        info = pltpu.get_tpu_info()
    except Exception:  # off TPU the query raises; the default scoped limit is then kept
        return _DEFAULT_VMEM
    cap = int(getattr(info, "vmem_capacity_bytes", 0) or 0)
    return cap if cap > 0 else _DEFAULT_VMEM


def n_pieces(dtype: tp.Any, requested: int) -> int:
    """Number of bf16 pieces an operand of ``dtype`` is split into for the MXU dots.

    Args:
        dtype: Operand dtype.
        requested: Pieces for a non-bf16 operand.

    Returns:
        ``1`` for bf16 operands (exact in one piece), else ``requested``.
    """
    return 1 if jnp.dtype(dtype) == jnp.dtype(_BF16) else int(requested)


def _truncate_to_bf16(v: Array) -> Array:
    """fp32 ``v`` with the low 16 mantissa bits cleared: exactly representable in bf16, kept as fp32.

    A bit mask instead of an ``f32 -> bf16 -> f32`` round trip: inside ``jit`` XLA's excess-precision
    simplification may fold that round trip away (the residual pieces then become zero and an fp32 operand silently
    degrades to one bf16 piece; measured on TPU as a ~1e-3 relative gate error). Bit operations cannot be folded.
    """
    bits = lax.bitcast_convert_type(v.astype(_F32), jnp.uint32)
    return lax.bitcast_convert_type(bits & jnp.uint32(0xFFFF0000), _F32)


def _split_pieces(v: Array, n: int) -> list[Array]:
    """``v`` as ``n`` bf16 pieces whose fp32 sum is ``v`` to ~8n bits (``n = 1``: a plain cast).

    Each leading piece is a truncation (exact in bf16), so ``r - p`` is exact in fp32; the last piece rounds.
    """
    if n == 1:
        return [v.astype(_BF16)]
    r = v.astype(_F32)
    pieces = []
    for _ in range(n - 1):
        p = _truncate_to_bf16(r)
        pieces.append(p.astype(_BF16))
        r = r - p
    pieces.append(r.astype(_BF16))
    return pieces


def _dot_pieces(a_pieces: list[Array], b_pieces: list[Array], dims) -> Array:
    """fp32 ``a @ b`` from bf16 pieces: the terms ``a_i b_j`` with ``i + j < max(len a, len b)``."""
    n = max(len(a_pieces), len(b_pieces))
    acc = None
    for i, a in enumerate(a_pieces):
        for j, b in enumerate(b_pieces):
            if i + j >= n:
                continue
            t = lax.dot_general(a, b, dims, preferred_element_type=_F32)
            acc = t if acc is None else acc + t
    return acc


def _lane(v: Array, lane: int) -> Array:
    """``v[:, lane]`` of a ``[rows, LANES]`` fp32 value as ``[rows, 1]`` (a masked lane reduction; no lane slice)."""
    mask = lax.broadcasted_iota(jnp.int32, v.shape, 1) == lane
    return jnp.sum(jnp.where(mask, v, 0.0), axis=1, keepdims=True)


def _onehot(shape: tuple[int, int], lane: int) -> Array:
    return (lax.broadcasted_iota(jnp.int32, shape, 1) == lane).astype(_F32)


def _stats_kernel(x_ref, fn_ref, o_ref, *, hc: int, width: int, x_pieces: int):
    """``[tile, LANES]``: lane 0 = sum of squares, lanes ``1..mix`` = ``x @ fn^T`` (fn rows shifted by one)."""
    xp = _split_pieces(x_ref[...], x_pieces)
    fnp = [fn_ref[p] for p in range(fn_ref.shape[0])]
    mixp = _dot_pieces(xp, fnp, _NT)  # [tile, LANES]; lane 0 hits fn's zero row
    sumsq = None
    for h in range(hc):
        xh = x_ref[:, h * width : (h + 1) * width].astype(_F32)
        s = jnp.sum(xh * xh, axis=1, keepdims=True)
        sumsq = s if sumsq is None else sumsq + s
    o_ref[...] = mixp + sumsq * _onehot(mixp.shape, 0)


def _collapse_kernel(x_ref, g_ref, o_ref, *, hc: int, width: int):
    """``collapsed = sum_h pre_h x_h`` in fp32, stored in the streams' dtype."""
    gates = g_ref[...]
    acc = None
    for h in range(hc):
        t = _lane(gates, h) * x_ref[:, h * width : (h + 1) * width].astype(_F32)
        acc = t if acc is None else acc + t
    o_ref[...] = acc.astype(o_ref.dtype)


def _expand_kernel(x_ref, out_ref, g_ref, o_ref, *, hc: int, width: int, quant_dtype):
    """``x'_i = post_i out + sum_j comb_ji x_j`` with the gates quantised to ``quant_dtype``."""
    gates = g_ref[...]

    def q(v):
        return v.astype(quant_dtype).astype(_F32)

    out = out_ref[...].astype(_F32)
    xs = [x_ref[:, j * width : (j + 1) * width].astype(_F32) for j in range(hc)]
    for i in range(hc):
        acc = q(_lane(gates, hc + i)) * out
        for j in range(hc):
            acc = acc + q(_lane(gates, 2 * hc + j * hc + i)) * xs[j]
        o_ref[:, i * width : (i + 1) * width] = acc.astype(o_ref.dtype)


def _expand_bwd_kernel(g_ref, x_ref, out_ref, gt_ref, dx_ref, dout_ref, dg_ref, *, hc: int, width: int, quant_dtype):
    """VJP of :func:`_expand_kernel`: ``dx``, ``dout`` and the gate cotangents (``dpost`` / ``dcomb`` lanes)."""
    gates = gt_ref[...]

    def q(v):
        return v.astype(quant_dtype).astype(_F32)

    out = out_ref[...].astype(_F32)
    gs = [g_ref[:, i * width : (i + 1) * width].astype(_F32) for i in range(hc)]
    xs = [x_ref[:, j * width : (j + 1) * width].astype(_F32) for j in range(hc)]
    shape = gates.shape
    dg = jnp.zeros(shape, _F32)
    dout = None
    for i in range(hc):
        t = q(_lane(gates, hc + i)) * gs[i]
        dout = t if dout is None else dout + t
        dg = dg + jnp.sum(gs[i] * out, axis=1, keepdims=True) * _onehot(shape, hc + i)
    dout_ref[...] = dout.astype(dout_ref.dtype)
    for j in range(hc):
        acc = None
        for i in range(hc):
            t = q(_lane(gates, 2 * hc + j * hc + i)) * gs[i]
            acc = t if acc is None else acc + t
            dg = dg + jnp.sum(xs[j] * gs[i], axis=1, keepdims=True) * _onehot(shape, 2 * hc + j * hc + i)
        dx_ref[:, j * width : (j + 1) * width] = acc.astype(dx_ref.dtype)
    dg_ref[...] = dg


def _dpre_kernel(x_ref, gc_ref, o_ref, *, hc: int, width: int):
    """``dpre_h = <g_collapsed, x_h>`` per row, lanes ``0..hc`` of a ``[tile, LANES]`` row (rest zero)."""
    gc = gc_ref[...].astype(_F32)
    row = None
    for h in range(hc):
        t = jnp.sum(gc * x_ref[:, h * width : (h + 1) * width].astype(_F32), axis=1, keepdims=True) * _onehot(
            (gc.shape[0], LANES), h
        )
        row = t if row is None else row + t
    o_ref[...] = row


def _dx_kernel(
    x_ref, gc_ref, gt_ref, ds_ref, fn_ref, dx_ref, dfn_ref, *, hc: int, width: int, x_pieces: int, grad_pieces: int
):
    """``dx_h = pre_h g_c + (dstats @ fn)_h + 2 dsumsq x_h``; ``dfn += dstats^T x`` (accumulated over the grid).

    ``dstats`` lane 0 is ``dsumsq`` (it meets ``fn``'s zero row in the dot, so it contributes nothing there) and
    lanes ``1..mix`` are the logit cotangents. The same lane shift makes ``dfn`` row ``m + 1`` the gradient of
    ``fn[m]``; row 0 collects ``dsumsq^T x`` and is discarded by the wrapper.
    """

    @pl.when(pl.program_id(0) == 0)
    def _init():
        dfn_ref[...] = jnp.zeros_like(dfn_ref)

    gates = gt_ref[...]
    ds = ds_ref[...]
    gc = gc_ref[...].astype(_F32)
    fnp = [fn_ref[p] for p in range(fn_ref.shape[0])]
    proj = _dot_pieces(_split_pieces(ds, grad_pieces), fnp, _NN)  # [tile, hc W]
    two_dsumsq = 2.0 * _lane(ds, 0)
    for h in range(hc):
        xh = x_ref[:, h * width : (h + 1) * width].astype(_F32)
        dxh = _lane(gates, h) * gc + proj[:, h * width : (h + 1) * width] + two_dsumsq * xh
        dx_ref[:, h * width : (h + 1) * width] = dxh.astype(dx_ref.dtype)
    ds_t = ds.T  # [LANES, tile] fp32 (32-bit 2D transpose; tile % 128 == 0)
    dfn_ref[...] += _dot_pieces(_split_pieces(ds_t, grad_pieces), _split_pieces(x_ref[...], x_pieces), _NN)


class _Cfg(tp.NamedTuple):
    hc: int
    width: int
    hidden_size: int
    sinkhorn_iters: int
    eps: float
    rms_eps: float
    axis_name: tp.Any
    tile_rows: int
    fn_pieces: int
    grad_pieces: int
    interpret: bool


def vmem_estimate_bytes(tile_rows: int, hc: int, width: int, itemsize: int, *, fn_pieces: int = 3) -> int:
    """Working set of the heaviest program (``expand_bwd`` / ``collapse_bwd_dx``) at ``tile_rows``.

    Counts double-buffered stream / cotangent / ``[rows, W]`` / gate blocks, the ``fn`` pieces and the ``dfn``
    accumulator, plus the fp32 stream slabs the kernels hold (``3 hc`` of ``[tile, W]`` and one ``[tile, hc W]``
    projection).

    Args:
        tile_rows: Rows per grid step.
        hc: Number of streams.
        width: Hidden width per device.
        itemsize: Bytes per stream element.
        fn_pieces: bf16 pieces of the predictor.

    Returns:
        Estimated bytes of VMEM.
    """
    hw = hc * width
    stream_blk = tile_rows * hw * itemsize
    row_blk = tile_rows * width * itemsize
    gate_blk = tile_rows * LANES * 4
    fn_blk = fn_pieces * LANES * hw * 2
    dfn_blk = LANES * hw * 4
    slabs = (2 * hc + hc) * tile_rows * width * 4 + tile_rows * hw * 4
    inputs = 2 * (2 * stream_blk + 2 * row_blk + 3 * gate_blk)
    outputs = 2 * (stream_blk + row_blk + gate_blk)
    return inputs + outputs + fn_blk + 2 * dfn_blk + slabs


def vmem_limit_bytes(tile_rows: int, hc: int, width: int, itemsize: int, *, fn_pieces: int = 3) -> int | None:
    """Scoped-VMEM request for the kernels.

    Args:
        tile_rows: Rows per grid step.
        hc: Number of streams.
        width: Hidden width per device.
        itemsize: Bytes per stream element.
        fn_pieces: bf16 pieces of the predictor.

    Returns:
        ``None`` (Mosaic's default) on a part with at most 16 MiB of VMEM, else ``2 x`` the estimate clamped to
        ``[16 MiB, capacity - 1 MiB]`` (63 MiB on a 64 MiB part; a full-capacity request is refused by Mosaic).

    Raises:
        ValueError: The estimated working set alone exceeds the cap (a compile-time VMEM OOM, named here).
    """
    capacity = _vmem_capacity()
    if capacity <= _DEFAULT_VMEM:
        return None
    need = vmem_estimate_bytes(tile_rows, hc, width, itemsize, fn_pieces=fn_pieces)
    cap = capacity - 2**20
    if need > cap:
        raise ValueError(
            f"mhc: tile_rows={tile_rows} needs ~{need / 2**20:.0f} MiB of VMEM at [{tile_rows}, {hc} x {width}] "
            f"(itemsize {itemsize}), over the {cap / 2**20:.0f} MiB scoped limit of this part; lower tile_rows"
        )
    return int(min(cap, max(_DEFAULT_VMEM, 2 * need)))


def _params(cfg: _Cfg, itemsize: int, semantics: tuple[str, ...]) -> pltpu.CompilerParams:
    limit = vmem_limit_bytes(cfg.tile_rows, cfg.hc, cfg.width, itemsize, fn_pieces=cfg.fn_pieces)
    if limit is None:
        return pltpu.CompilerParams(dimension_semantics=semantics)
    return pltpu.CompilerParams(dimension_semantics=semantics, vmem_limit_bytes=limit)


def _pad_rows(a: Array, rows_pad: int) -> Array:
    return a if a.shape[0] == rows_pad else jnp.pad(a, ((0, rows_pad - a.shape[0]),) + ((0, 0),) * (a.ndim - 1))


def _row_block(tile: int, width: int) -> pl.BlockSpec:
    return pl.BlockSpec((tile, width), lambda i: (i, 0))


def _const_block(shape: tuple[int, ...]) -> pl.BlockSpec:
    return pl.BlockSpec(shape, lambda i: (0,) * len(shape))


def _prep_fn_pieces(fn: Array, cfg: _Cfg) -> Array:
    """``[n_fn, LANES, hc W]`` bf16: row ``m + 1`` = piece of ``fn[m]``; rows 0 and ``mix + 1 ..`` zero."""
    mix = mix_width(cfg.hc)
    fn2 = fn.reshape(mix, cfg.hc * cfg.width)
    stacked = jnp.stack(_split_pieces(fn2, n_pieces(fn.dtype, cfg.fn_pieces)))
    return jnp.pad(stacked, ((0, 0), (1, LANES - 1 - mix), (0, 0)))


def _stats_call(x2: Array, fnp: Array, cfg: _Cfg) -> Array:
    rows, hw = x2.shape
    tile = cfg.tile_rows
    rows_pad = _round_up(rows, tile)
    out = pl.pallas_call(
        functools.partial(_stats_kernel, hc=cfg.hc, width=cfg.width, x_pieces=n_pieces(x2.dtype, cfg.fn_pieces)),
        grid=(rows_pad // tile,),
        in_specs=[_row_block(tile, hw), _const_block(fnp.shape)],
        out_specs=_row_block(tile, LANES),
        out_shape=jax.ShapeDtypeStruct((rows_pad, LANES), _F32),
        compiler_params=_params(cfg, x2.dtype.itemsize, ("parallel",)),
        interpret=cfg.interpret,
    )(_pad_rows(x2, rows_pad), fnp)
    return out[:rows]


def _collapse_call(x2: Array, gates: Array, cfg: _Cfg) -> Array:
    rows, hw = x2.shape
    tile = cfg.tile_rows
    rows_pad = _round_up(rows, tile)
    out = pl.pallas_call(
        functools.partial(_collapse_kernel, hc=cfg.hc, width=cfg.width),
        grid=(rows_pad // tile,),
        in_specs=[_row_block(tile, hw), _row_block(tile, LANES)],
        out_specs=_row_block(tile, cfg.width),
        out_shape=jax.ShapeDtypeStruct((rows_pad, cfg.width), x2.dtype),
        compiler_params=_params(cfg, x2.dtype.itemsize, ("parallel",)),
        interpret=cfg.interpret,
    )(_pad_rows(x2, rows_pad), _pad_rows(gates, rows_pad))
    return out[:rows]


def _expand_call(x2: Array, out2: Array, gates: Array, cfg: _Cfg) -> Array:
    rows, hw = x2.shape
    tile = cfg.tile_rows
    rows_pad = _round_up(rows, tile)
    new = pl.pallas_call(
        functools.partial(_expand_kernel, hc=cfg.hc, width=cfg.width, quant_dtype=x2.dtype),
        grid=(rows_pad // tile,),
        in_specs=[_row_block(tile, hw), _row_block(tile, cfg.width), _row_block(tile, LANES)],
        out_specs=_row_block(tile, hw),
        out_shape=jax.ShapeDtypeStruct((rows_pad, hw), x2.dtype),
        compiler_params=_params(cfg, x2.dtype.itemsize, ("parallel",)),
        interpret=cfg.interpret,
    )(_pad_rows(x2, rows_pad), _pad_rows(out2, rows_pad), _pad_rows(gates, rows_pad))
    return new[:rows]


def _expand_bwd_call(g2: Array, x2: Array, out2: Array, gates: Array, cfg: _Cfg) -> tuple[Array, Array, Array]:
    rows, hw = x2.shape
    tile = cfg.tile_rows
    rows_pad = _round_up(rows, tile)
    dx, dout, dg = pl.pallas_call(
        functools.partial(_expand_bwd_kernel, hc=cfg.hc, width=cfg.width, quant_dtype=x2.dtype),
        grid=(rows_pad // tile,),
        in_specs=[_row_block(tile, hw), _row_block(tile, hw), _row_block(tile, cfg.width), _row_block(tile, LANES)],
        out_specs=[_row_block(tile, hw), _row_block(tile, cfg.width), _row_block(tile, LANES)],
        out_shape=[
            jax.ShapeDtypeStruct((rows_pad, hw), x2.dtype),
            jax.ShapeDtypeStruct((rows_pad, cfg.width), out2.dtype),
            jax.ShapeDtypeStruct((rows_pad, LANES), _F32),
        ],
        compiler_params=_params(cfg, x2.dtype.itemsize, ("parallel",)),
        interpret=cfg.interpret,
    )(_pad_rows(g2, rows_pad), _pad_rows(x2, rows_pad), _pad_rows(out2, rows_pad), _pad_rows(gates, rows_pad))
    return dx[:rows], dout[:rows], dg[:rows]


def _dpre_call(x2: Array, gc: Array, cfg: _Cfg) -> Array:
    rows, hw = x2.shape
    tile = cfg.tile_rows
    rows_pad = _round_up(rows, tile)
    out = pl.pallas_call(
        functools.partial(_dpre_kernel, hc=cfg.hc, width=cfg.width),
        grid=(rows_pad // tile,),
        in_specs=[_row_block(tile, hw), _row_block(tile, cfg.width)],
        out_specs=_row_block(tile, LANES),
        out_shape=jax.ShapeDtypeStruct((rows_pad, LANES), _F32),
        compiler_params=_params(cfg, x2.dtype.itemsize, ("parallel",)),
        interpret=cfg.interpret,
    )(_pad_rows(x2, rows_pad), _pad_rows(gc, rows_pad))
    return out[:rows]


def _dx_call(x2: Array, gc: Array, gates: Array, dstats: Array, fnp: Array, cfg: _Cfg) -> tuple[Array, Array]:
    rows, hw = x2.shape
    tile = cfg.tile_rows
    rows_pad = _round_up(rows, tile)
    dx, dfn = pl.pallas_call(
        functools.partial(
            _dx_kernel,
            hc=cfg.hc,
            width=cfg.width,
            x_pieces=n_pieces(x2.dtype, cfg.fn_pieces),
            grad_pieces=cfg.grad_pieces,
        ),
        grid=(rows_pad // tile,),
        in_specs=[
            _row_block(tile, hw),
            _row_block(tile, cfg.width),
            _row_block(tile, LANES),
            _row_block(tile, LANES),
            _const_block(fnp.shape),
        ],
        out_specs=[_row_block(tile, hw), _const_block((LANES, hw))],
        out_shape=[jax.ShapeDtypeStruct((rows_pad, hw), x2.dtype), jax.ShapeDtypeStruct((LANES, hw), _F32)],
        compiler_params=_params(cfg, x2.dtype.itemsize, ("arbitrary",)),
        interpret=cfg.interpret,
    )(_pad_rows(x2, rows_pad), _pad_rows(gc, rows_pad), _pad_rows(gates, rows_pad), _pad_rows(dstats, rows_pad), fnp)
    return dx[:rows], dfn


def _gates(stats: Array, base: Array, scale: Array, cfg: _Cfg):
    return gates_from_stats(
        stats,
        base,
        scale,
        hc=cfg.hc,
        hidden_size=cfg.hidden_size,
        rms_eps=cfg.rms_eps,
        sinkhorn_iters=cfg.sinkhorn_iters,
        eps=cfg.eps,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _collapse(x, fn, base, scale, cfg: _Cfg):
    return _collapse_fwd(x, fn, base, scale, cfg)[0]


def _collapse_fwd(x, fn, base, scale, cfg: _Cfg):
    batch, seq, hc, width = x.shape
    rows = batch * seq
    x2 = x.reshape(rows, hc * width)
    fnp = _prep_fn_pieces(fn, cfg)
    stats = _stats_call(x2, fnp, cfg)
    if cfg.axis_name is not None:
        stats = lax.psum(stats, cfg.axis_name)
    pre, post, comb = _gates(stats, base, scale, cfg)
    gates = pack_gates(pre, post, comb)
    collapsed = _collapse_call(x2, gates, cfg)
    outs = (post.reshape(batch, seq, hc), comb.reshape(batch, seq, hc, hc), collapsed.reshape(batch, seq, width))
    return outs, (x2, fn, fnp, stats, gates, base, scale)


def _collapse_bwd(cfg: _Cfg, res, cts):
    x2, fn, fnp, stats, gates, base, scale = res
    d_post, d_comb, d_collapsed = cts
    batch, seq, width = d_collapsed.shape
    hc, mix = cfg.hc, mix_width(cfg.hc)
    rows = batch * seq
    gc = d_collapsed.reshape(rows, width)
    dpre = _dpre_call(x2, gc, cfg)[:, :hc]  # partial over the hidden shard
    # Per-device gate VJP, exactly as JAX's AD runs the reference body: the logit cotangent is summed over the
    # hidden shards (the transpose of the forward psum); base / scale cotangents stay per-device partials that the
    # enclosing shard_map transpose reduces.
    _, vjp = jax.vjp(lambda s, b, c: _gates(s, b, c, cfg), stats, base, scale)
    d_stats, d_base, d_scale = vjp((dpre, d_post.reshape(rows, hc), d_comb.reshape(rows, hc, hc)))
    if cfg.axis_name is not None:
        d_stats = lax.psum(d_stats, cfg.axis_name)
    dx2, dfn_acc = _dx_call(x2, gc, gates, d_stats, fnp, cfg)
    d_fn = dfn_acc[1 : 1 + mix].reshape(mix, hc, width).astype(fn.dtype)
    return dx2.reshape(batch, seq, hc, width), d_fn, d_base.astype(base.dtype), d_scale.astype(scale.dtype)


_collapse.defvjp(_collapse_fwd, _collapse_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _expand(x, out, post, comb, cfg: _Cfg):
    return _expand_fwd(x, out, post, comb, cfg)[0]


def _expand_fwd(x, out, post, comb, cfg: _Cfg):
    batch, seq, hc, width = x.shape
    rows = batch * seq
    x2 = x.reshape(rows, hc * width)
    out2 = out.reshape(rows, width)
    gates = pack_gates(jnp.zeros((rows, hc), _F32), post.reshape(rows, hc), comb.reshape(rows, hc, hc))
    new = _expand_call(x2, out2, gates, cfg)
    # dtype carriers for the gate cotangents (residuals must be arrays)
    return new.reshape(x.shape), (x2, out2, gates, jnp.zeros((), post.dtype), jnp.zeros((), comb.dtype))


def _expand_bwd(cfg: _Cfg, res, ct):
    x2, out2, gates, post_like, comb_like = res
    batch, seq, hc, width = ct.shape
    rows = batch * seq
    # <g, out> and <x, g> are partial over a hidden shard; the enclosing shard_map transpose sums them.
    dx2, dout2, dg = _expand_bwd_call(ct.reshape(rows, hc * width), x2, out2, gates, cfg)
    _, d_post, d_comb = unpack_gates(dg, hc)
    return (
        dx2.reshape(ct.shape),
        dout2.reshape(batch, seq, width),
        d_post.reshape(batch, seq, hc).astype(post_like.dtype),
        d_comb.reshape(batch, seq, hc, hc).astype(comb_like.dtype),
    )


_expand.defvjp(_expand_fwd, _expand_bwd)


def _cfg(
    x: Array,
    *,
    hidden_size: int | None,
    sinkhorn_iters: int,
    eps: float,
    rms_eps: float,
    axis_name,
    tile_rows: int,
    fn_pieces: int,
    grad_pieces: int,
    interpret: bool | None,
) -> _Cfg:
    if x.ndim != 4:
        raise ValueError(f"mhc: streams must be [batch, seq, hc, width], got {x.shape}")
    _, _, hc, width = x.shape
    check_hc(hc)
    if width % LANES:
        raise ValueError(f"mhc: width={width} must be a multiple of {LANES}")
    if tile_rows < LANES or tile_rows % LANES:
        raise ValueError(f"mhc: tile_rows={tile_rows} must be a positive multiple of {LANES}")
    if fn_pieces < 1 or grad_pieces < 1:
        raise ValueError("mhc: fn_pieces and grad_pieces must be >= 1")
    if sinkhorn_iters < 1:
        raise ValueError(f"mhc: sinkhorn_iters={sinkhorn_iters} must be >= 1")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    return _Cfg(
        hc=int(hc),
        width=int(width),
        hidden_size=resolve_hidden_size(width, hidden_size, axis_name),
        sinkhorn_iters=int(sinkhorn_iters),
        eps=float(eps),
        rms_eps=float(rms_eps),
        axis_name=axis_name,
        tile_rows=int(tile_rows),
        fn_pieces=int(fn_pieces),
        grad_pieces=int(grad_pieces),
        interpret=bool(interpret),
    )


def mhc_collapse_pallas(
    x: Float[Array, "batch seq hc width"],
    fn: Float[Array, "mix hc width"],
    base: Float[Array, "mix"],
    scale: Float[Array, "three"],
    *,
    hidden_size: int | None = None,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    rms_eps: float = 1e-6,
    axis_name: str | tuple[str, ...] | None = None,
    tile_rows: int = 256,
    fn_pieces: int = 3,
    grad_pieces: int = 3,
    interpret: bool | None = None,
) -> tuple[Float[Array, "batch seq hc"], Float[Array, "batch seq hc hc"], Float[Array, "batch seq width"]]:
    """Fused mHC collapse: ``(post, comb, collapsed)`` -- the contract of the XLA ``mhc_collapse``.

    Args:
        x: Residual streams ``[B, S, hc, W]``; ``W`` the hidden shard this device holds (``W % 128 == 0``).
        fn: Gate predictor ``[(2 + hc) hc, hc, W]``, the same hidden shard.
        base: Gate biases ``[(2 + hc) hc]``.
        scale: Logit scales ``[3]``.
        hidden_size: FULL hidden width (RMS denominator ``hc * hidden_size``); ``None`` takes it from the streams
            (``W`` times the size of ``axis_name``).
        sinkhorn_iters: Sinkhorn-Knopp iterations.
        eps: Sinkhorn / ``pre`` floor.
        rms_eps: RMS epsilon.
        axis_name: Mesh axis (or axes) the hidden is sharded over; the stats and the logit cotangent are summed
            over it. Requires a ``shard_map(check_vma=False)`` context.
        tile_rows: Rows per grid step (multiple of 128); the working set scales with it, see
            :func:`vmem_estimate_bytes`.
        fn_pieces: bf16 pieces an fp32 ``fn`` (and an fp32 stream) splits into for the MXU projection: 3 = fp32
            accuracy, 2 = ~16 bits, 1 = plain bf16. bf16 operands always use one piece (exact).
        grad_pieces: Pieces for the fp32 logit cotangent in the backward dots.
        interpret: Force Pallas interpret mode (default: interpret off TPU).

    Returns:
        ``post [B, S, hc]`` fp32, ``comb [B, S, hc, hc]`` fp32, ``collapsed [B, S, W]`` in ``x.dtype``.

    Raises:
        ValueError: Invalid shapes or arguments.
    """
    cfg = _cfg(
        x,
        hidden_size=hidden_size,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        rms_eps=rms_eps,
        axis_name=axis_name,
        tile_rows=tile_rows,
        fn_pieces=fn_pieces,
        grad_pieces=grad_pieces,
        interpret=interpret,
    )
    mix = mix_width(cfg.hc)
    if tuple(fn.shape) != (mix, cfg.hc, cfg.width):
        raise ValueError(f"mhc: fn must be [{mix}, {cfg.hc}, {cfg.width}], got {fn.shape}")
    if base.shape != (mix,) or scale.shape != (3,):
        raise ValueError(f"mhc: base must be [{mix}] and scale [3], got {base.shape} / {scale.shape}")
    return _collapse(x, fn, base, scale, cfg)


def mhc_expand_pallas(
    x: Float[Array, "batch seq hc width"],
    out: Float[Array, "batch seq width"],
    post: Float[Array, "batch seq hc"],
    comb: Float[Array, "batch seq hc hc"],
    *,
    tile_rows: int = 256,
    interpret: bool | None = None,
) -> Float[Array, "batch seq hc width"]:
    """Fused mHC expand -- the contract of the XLA ``mhc_expand``.

    Computes ``post[..., None] * out[..., None, :] + einsum("bsji,bsjd->bsid", comb, x)`` with the gates quantised
    to ``x.dtype`` first and fp32 accumulation. Neither direction has a collective: under a hidden-sharded
    ``shard_map(check_vma=False)`` the ``post`` / ``comb`` cotangents are per-device partials that the transpose
    sums.

    Args:
        x: Residual streams ``[B, S, hc, W]`` (``W % 128 == 0``).
        out: Sub-layer output ``[B, S, W]``.
        post: Write gates ``[B, S, hc]``.
        comb: Mixer ``[B, S, hc, hc]`` indexed ``[source, target]``.
        tile_rows: Rows per grid step (multiple of 128).
        interpret: Force Pallas interpret mode (default: interpret off TPU).

    Returns:
        New streams ``[B, S, hc, W]`` in ``x.dtype``.

    Raises:
        ValueError: Invalid shapes or arguments.
    """
    cfg = _cfg(
        x,
        hidden_size=x.shape[-1] if x.ndim == 4 else None,
        sinkhorn_iters=1,
        eps=0.0,
        rms_eps=0.0,
        axis_name=None,
        tile_rows=tile_rows,
        fn_pieces=1,
        grad_pieces=1,
        interpret=interpret,
    )
    batch, seq, hc, width = x.shape
    if out.shape != (batch, seq, width) or post.shape != (batch, seq, hc) or comb.shape != (batch, seq, hc, hc):
        raise ValueError(f"mhc: out {out.shape}, post {post.shape}, comb {comb.shape} do not match streams {x.shape}")
    return _expand(x, out, post, comb, cfg)


__all__ = ("mhc_collapse_pallas", "mhc_expand_pallas", "n_pieces", "vmem_estimate_bytes", "vmem_limit_bytes")
