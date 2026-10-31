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

"""Row permutation by contiguous runs as a tile-safe streaming copy through VMEM (Pallas TPU).

A run table ``(dst_start, src_start, length)`` describes ``out[dst + j] = x[src + j]``. Typical producers are MoE
expert-parallel receive buffers: the buffer is source-major / expert-minor, the grouped matmul wants it expert-major,
and both directions are permutations made of a few hundred contiguous runs. A per-row ``x[index]`` gather moves the
same bytes at a fraction of HBM bandwidth.

Why not one HBM-to-HBM DMA per run: HBM refs are tiled (``(16, 128)`` for 2-byte dtypes, ``(8, 128)`` for 4-byte)
and Mosaic rejects a DMA whose row offset is not a tile multiple ("Offsets along tiled dimensions must be aligned to
tiles"). Run starts are arbitrary and so is ``dst - src``, so no HBM-to-HBM copy of a run can be aligned on both
ends. The check only runs in the TPU compiler (``pl.lower_as_mlir`` emits the module without it), so alignment must
hold by construction: every HBM row offset below is ``TILE * k`` and carries ``pl.multiple_of``.

The streaming form, runs sorted by destination (padding runs last, skipped):

1. a run is walked in chunks of ``chunk_rows`` source rows; each chunk's window ``[S_lo, S_lo + L)`` (``S_lo`` the
   chunk start rounded down to the tile and clamped inside ``x``, ``L = chunk_rows + TILE``) is one aligned
   HBM->VMEM DMA;
2. the window is rotated along rows by ``(b - a) mod L`` (``pltpu.roll`` in 32 bits; 2-byte floats are widened
   exactly to float32 and narrowed back), so row ``j`` holds destination row ``align_down(pos) + j``;
3. the carried partial destination tile (fewer than ``TILE`` rows) is merged into the window head, the complete
   tiles are written by aligned VMEM->HBM DMAs (the dynamic tile count is decomposed into ``pl.when``-guarded powers
   of two, DMA shapes being static) and the leftover rows become the carry for the next chunk or run;
4. a destination gap (padding, uncovered rows) is filled: with ``init`` the partial tile is read-modify-written and
   the tile the next run starts inside is reloaded (rows keep ``init``); without ``init`` the gap is written with
   zeros from a VMEM zero buffer, so uncovered rows are exactly zero (the XLA reference's contract).

Every row is read once and written once (two HBM passes at DMA rates). Rows are moved, not computed, so the result
is bit-identical to ``x[index]`` on every covered row.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ...._xla.row_permute._xla_impl import (
    invert_runs,
    resolve_out_rows,
    runs_to_index,
    sort_runs_by_destination,
    validate_runs,
)

TILE = 16  # rows per HBM tile for 2-byte dtypes; a multiple of the 4-byte tile (8), so aligned for both
LANES = 128
DEFAULT_CHUNK_ROWS = 128  # source rows per window (window = chunk_rows + TILE); best of 64/128/256/512/1024 on v5
_VMEM_MARGIN = 4 * 1024 * 1024
_VMEM_CAP = 64 * 1024 * 1024


def _rem(a, b: int):
    """``a % b`` for a non-negative traced ``a`` (``lax.rem``; jnp's Python-style ``%`` lowers through ``sign``)."""
    return lax.rem(a, jnp.int32(b))


def _div(a, b: int):
    """``a // b`` for a non-negative traced ``a`` (``lax.div``, see ``_rem``)."""
    return lax.div(a, jnp.int32(b))


def _tile_start(k):
    """Row offset ``TILE * k`` with the alignment hint Mosaic needs on dynamic tiled slices."""
    if isinstance(k, int):
        return k * TILE
    return pl.multiple_of(k * TILE, TILE)


def _column_block(width: int) -> int:
    """Lanes processed per rotate step (bounds the vector temporaries)."""
    for cb in (512, 256, 128):
        if width % cb == 0:
            return cb
    raise ValueError(f"width ({width}) must be a multiple of {LANES}")


def check_supported(x_shape: tuple[int, ...], dtype, out_rows: int) -> str | None:
    """Return why the Pallas kernel cannot run this shape/dtype (``None`` when it can).

    Args:
        x_shape: Source shape.
        dtype: Source dtype.
        out_rows: Destination rows.

    Returns:
        A human-readable reason, or ``None``.
    """
    if len(x_shape) != 2:
        return f"x must be rank 2, got shape {x_shape}"
    rows, width = x_shape
    dt = jnp.dtype(dtype)
    if not (dt.itemsize == 4 or dt == jnp.dtype(jnp.bfloat16)):
        # float16 is refused: Mosaic on TPU v5 does not legalize float16 vector stores (measured).
        return f"permute_rows_runs supports 4-byte dtypes and bfloat16, got {dt}"
    if rows % TILE or out_rows % TILE or width % LANES:
        return f"rows ({rows}, out {out_rows}) must be multiples of {TILE} and width ({width}) of {LANES}"
    if rows < 2 * TILE or out_rows < 2 * TILE:
        return f"rows ({rows}) and out_rows ({out_rows}) must be at least {2 * TILE}"
    return None


def _kernel(
    runs_ref,
    x_hbm,
    *rest,
    n_runs: int,
    chunk_rows: int,
    rows: int,
    out_rows: int,
    width: int,
    has_init: bool,
    wide: bool,
):
    """Walk the destination-sorted run table, streaming each run through VMEM in aligned windows."""
    if has_init:
        _init_hbm, o_hbm, vin, vout, vcarry, vtile, sem = rest
        vzero = None
    else:
        o_hbm, vin, vout, vcarry, vtile, vzero, sem = rest
    dtype = vin.dtype
    L = chunk_rows + TILE
    LT = L // TILE
    max_chunk_tiles = chunk_rows // TILE  # (b + n) // TILE <= (TILE - 1 + chunk_rows) // TILE
    cb = _column_block(width)
    col_blocks = [slice(c * cb, (c + 1) * cb) for c in range(width // cb)]
    row16 = lax.broadcasted_iota(jnp.int32, (TILE, cb), 0)

    def widen(v):
        return v.astype(jnp.float32) if wide else v

    def narrow(v):
        return v.astype(dtype) if wide else v

    def out_tiles(start_tile, n_tiles: int):
        return o_hbm.at[pl.ds(_tile_start(start_tile), n_tiles * TILE)]

    def copy_sync(src, dst):
        cp = pltpu.make_async_copy(src, dst, sem)
        cp.start()
        cp.wait()

    def write_tile_pieces(src_vmem, dst_tile0, n_tiles, max_tiles: int):
        """DMA tiles ``[0, n_tiles)`` of ``src_vmem`` to output tiles ``[dst_tile0, dst_tile0 + n_tiles)``.

        ``n_tiles <= max_tiles`` is dynamic and DMA shapes are static, so the count is decomposed into powers of two;
        each piece is built and started inside its ``pl.when`` (no out-of-range slice is ever emitted unguarded).
        """
        pieces = []
        off = jnp.int32(0)
        for bit in range(max(max_tiles, 1).bit_length() - 1, -1, -1):
            size = 1 << bit
            has = (jnp.right_shift(n_tiles, bit) & 1) == 1

            def piece(off=off, size=size):
                return pltpu.make_async_copy(
                    src_vmem.at[pl.ds(_tile_start(off), size * TILE)], out_tiles(dst_tile0 + off, size), sem
                )

            pl.when(has)(lambda piece=piece: piece().start())
            pieces.append((has, piece))
            off = off + jnp.where(has, jnp.int32(size), jnp.int32(0))
        return pieces

    def wait_pieces(pieces):
        for has, piece in pieces:
            pl.when(has)(lambda piece=piece: piece().wait())

    def chunk(src0, pos, n):
        """Move ``n <= chunk_rows`` source rows from ``src0`` to destination ``pos``; returns ``pos + n``."""
        s_lo = jnp.minimum(_div(src0, TILE), jnp.int32((rows - L) // TILE))  # the window stays inside x
        a = src0 - s_lo * TILE  # the chunk's first row inside the window (a + n <= L)
        copy_sync(x_hbm.at[pl.ds(_tile_start(s_lo), L)], vin)
        b = _rem(pos, TILE)
        shift = _rem(b - a + L, L)
        for cols in col_blocks:
            # row j of the rotated window = vin[j - b + a]: destination row align_down(pos) + j
            blk = pltpu.roll(widen(vin[:, cols]), shift, 0)
            head = jnp.where(row16 < b, widen(vcarry[:, cols]), blk[:TILE])
            vout[:TILE, cols] = narrow(head)
            vout[TILE:, cols] = narrow(blk[TILE:])
        n_tiles = _div(b + n, TILE)
        pieces = write_tile_pieces(vout, _div(pos, TILE), n_tiles, max_chunk_tiles)
        vcarry[...] = vout[pl.ds(_tile_start(n_tiles), TILE), :]  # leftover rows [TILE * n_tiles, b + n)
        wait_pieces(pieces)
        return pos + n

    def mask_carry(r):
        """Zero the carry rows ``>= r``."""
        for cols in col_blocks:
            kept = widen(vcarry[:, cols])
            vcarry[:, cols] = narrow(jnp.where(row16 < r, kept, jnp.zeros_like(kept)))

    def zero_tiles(t0, count):
        """Write zeros to output tiles ``[t0, t0 + count)`` from the VMEM zero buffer."""
        n_full = _div(count, LT)

        def body(k, carry):
            copy_sync(vzero, out_tiles(t0 + k * LT, LT))
            return carry

        lax.fori_loop(0, n_full, body, jnp.int32(0))
        wait_pieces(write_tile_pieces(vzero, t0 + n_full * LT, count - n_full * LT, LT - 1))

    def gap_zero(pos, dst):
        """Fill ``[pos, dst)`` with zeros; afterwards the carry holds rows ``[align_down(dst), dst)``."""
        r = _rem(pos, TILE)
        tp, td = _div(pos, TILE), _div(dst, TILE)
        mask_carry(r)

        @pl.when(tp != td)
        def _():
            @pl.when(r > 0)
            def _():
                copy_sync(vcarry, out_tiles(tp, 1))

            first = tp + (r > 0).astype(jnp.int32)
            zero_tiles(first, td - first)
            vcarry[...] = jnp.zeros(vcarry.shape, vcarry.dtype)

    def flush_carry_init(pos):
        """Read-modify-write the partial destination tile ``[align_down(pos), pos)`` held in the carry."""
        r = _rem(pos, TILE)

        @pl.when(r > 0)
        def _():
            tile = _div(pos, TILE)
            copy_sync(out_tiles(tile, 1), vtile)
            for cols in col_blocks:
                vtile[:, cols] = narrow(jnp.where(row16 < r, widen(vcarry[:, cols]), widen(vtile[:, cols])))
            copy_sync(vtile, out_tiles(tile, 1))

    def load_carry_init(dst):
        """Start writing inside an existing tile: its rows ``[align_down(dst), dst)`` become the carry."""

        @pl.when(_rem(dst, TILE) > 0)
        def _():
            copy_sync(out_tiles(_div(dst, TILE), 1), vcarry)

    def gap(pos, dst):
        if has_init:
            flush_carry_init(pos)
            load_carry_init(dst)
        else:
            gap_zero(pos, dst)
        return dst

    vcarry[...] = jnp.zeros(vcarry.shape, vcarry.dtype)
    if not has_init:
        vzero[...] = jnp.zeros(vzero.shape, vzero.dtype)

    def run_body(i, pos):
        dst, src, length = runs_ref[3 * i], runs_ref[3 * i + 1], runs_ref[3 * i + 2]

        def active(pos):
            pos = lax.cond(dst != pos, lambda p: gap(p, dst), lambda p: p, pos)
            n_chunks = _div(length + (chunk_rows - 1), chunk_rows)

            def chunk_body(k, pos):
                n = jnp.minimum(jnp.int32(chunk_rows), length - k * chunk_rows)
                return chunk(src + k * chunk_rows, pos, n)

            return lax.fori_loop(0, n_chunks, chunk_body, pos)

        return lax.cond(length > 0, active, lambda p: p, pos)

    pos = lax.fori_loop(0, n_runs, run_body, jnp.int32(0))
    if has_init:
        flush_carry_init(pos)
    else:
        gap_zero(pos, jnp.int32(out_rows))


def permute_rows_runs_pallas(
    x: Array,
    runs: Array,
    *,
    init: Array | None = None,
    out_rows: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    interpret: bool = False,
) -> Array:
    """``out[dst + j] = x[src + j]`` for every run ``(dst, src, length)`` of ``runs``.

    Args:
        x: ``[rows, D]``; 4-byte dtypes are moved as is, 2-byte floats are widened exactly to float32 for the row
            rotate. ``rows`` must be a multiple of ``TILE`` (16) and at least 32; ``D`` a multiple of 128.
        runs: ``[R, 3]`` int32 ``(dst_start, src_start, length)`` in any order; ``length == 0`` runs are skipped.
            Destination ranges must not overlap and every run must lie inside ``x`` / the output (not checked).
        init: Optional ``[out_rows, D]`` buffer the output aliases; rows no run covers keep its values. ``None``:
            uncovered rows are zero.
        out_rows: Destination rows (default ``x.shape[0]`` or ``init.shape[0]``); a multiple of ``TILE``.
        chunk_rows: Source rows per streamed window (a positive multiple of ``TILE``, clamped to ``rows - TILE``).
            VMEM holds ``2`` (``3`` without ``init``) windows of ``(chunk_rows + TILE) x D`` in ``x.dtype``.
        interpret: Pallas interpret mode (debugging only).

    Returns:
        ``[out_rows, D]`` in ``x.dtype``.

    Raises:
        ValueError: On an unsupported shape/dtype or a malformed run table.
    """
    runs = jnp.asarray(runs)
    validate_runs(runs)
    out_rows = resolve_out_rows(x, init, out_rows)
    reason = check_supported(tuple(x.shape), x.dtype, out_rows)
    if reason is not None:
        raise ValueError(reason)
    if chunk_rows <= 0 or chunk_rows % TILE:
        raise ValueError(f"chunk_rows must be a positive multiple of {TILE}, got {chunk_rows}")
    rows, width = x.shape
    chunk_rows = min(int(chunk_rows), rows - TILE)
    wide = jnp.dtype(x.dtype).itemsize == 2
    has_init = init is not None
    L = chunk_rows + TILE
    if runs.shape[0] == 0:
        runs = jnp.zeros((1, 3), jnp.int32)  # one padding run: scalar prefetch needs a non-empty table
    flat_runs = sort_runs_by_destination(runs).reshape(-1)

    itemsize = jnp.dtype(x.dtype).itemsize
    buffers = ((3 if not has_init else 2) * L + 2 * TILE) * width * itemsize
    temporaries = 8 * L * _column_block(width) * 4
    vmem_limit = min(buffers + temporaries + _VMEM_MARGIN, _VMEM_CAP)

    in_specs = [pl.BlockSpec(memory_space=pl.ANY)]
    operands = [flat_runs, x]
    aliases = {}
    if has_init:
        in_specs.append(pl.BlockSpec(memory_space=pl.ANY))
        operands.append(init)
        aliases = {2: 0}  # operand 2 (after the scalar-prefetch table and x) is the output buffer
    scratch = [
        pltpu.VMEM((L, width), x.dtype),  # vin: the source window
        pltpu.VMEM((L, width), x.dtype),  # vout: the rotated window, DMA source
        pltpu.VMEM((TILE, width), x.dtype),  # vcarry: the partial destination tile
        pltpu.VMEM((TILE, width), x.dtype),  # vtile: one-tile staging for read-modify-writes
    ]
    if not has_init:
        scratch.append(pltpu.VMEM((L, width), x.dtype))  # vzero: zeros for destination gaps
    scratch.append(pltpu.SemaphoreType.DMA)

    kernel = functools.partial(
        _kernel,
        n_runs=int(runs.shape[0]),
        chunk_rows=chunk_rows,
        rows=rows,
        out_rows=out_rows,
        width=width,
        has_init=has_init,
        wide=wide,
    )
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=in_specs,
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=scratch,
        ),
        out_shape=jax.ShapeDtypeStruct((out_rows, width), x.dtype),
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",), vmem_limit_bytes=int(vmem_limit)),
        interpret=interpret,
    )(*operands)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _permute(x, runs, inverse_runs, init, out_rows, src_rows, chunk_rows, interpret):
    del inverse_runs, src_rows
    return permute_rows_runs_pallas(x, runs, init=init, out_rows=out_rows, chunk_rows=chunk_rows, interpret=interpret)


def _permute_fwd(x, runs, inverse_runs, init, out_rows, src_rows, chunk_rows, interpret):
    out = _permute(x, runs, inverse_runs, init, out_rows, src_rows, chunk_rows, interpret)
    return out, (inverse_runs, None if init is None else runs)


def _permute_bwd(out_rows, src_rows, chunk_rows, interpret, res, g):
    inverse_runs, runs = res
    dx = permute_rows_runs_pallas(g, inverse_runs, out_rows=src_rows, chunk_rows=chunk_rows, interpret=interpret)
    d_init = None
    if runs is not None:
        _, covered = runs_to_index(runs, out_rows)
        d_init = jnp.where(covered[:, None], jnp.zeros_like(g), g)
    return dx, None, None, d_init


_permute.defvjp(_permute_fwd, _permute_bwd)


def permute_rows_runs(
    x: Array,
    runs: Array,
    inverse_runs: Array | None = None,
    init: Array | None = None,
    *,
    out_rows: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    interpret: bool = False,
) -> Array:
    """Differentiable run permutation: the VJP is the same kernel along the inverse run table.

    The gradient equals the gather's when every source row is read by at most one run (an injective table, e.g. a
    permutation or a partial permutation); source rows no run reads get a zero gradient. Tables that read a source
    row twice need the gather (its VJP sums duplicates).

    Args:
        x: ``[rows, D]`` source rows (see :func:`permute_rows_runs_pallas` for shape/dtype limits).
        runs: ``[R, 3]`` int32 ``(dst_start, src_start, length)``.
        inverse_runs: The inverse table (default :func:`invert_runs` of ``runs``); any order.
        init: Optional ``[out_rows, D]`` values for uncovered rows (its gradient is the cotangent on those rows).
        out_rows: Destination rows.
        chunk_rows: Streamed window rows.
        interpret: Pallas interpret mode (debugging only).

    Returns:
        ``[out_rows, D]`` in ``x.dtype``.
    """
    runs = jnp.asarray(runs, jnp.int32)
    inverse_runs = invert_runs(runs) if inverse_runs is None else jnp.asarray(inverse_runs, jnp.int32)
    out_rows = resolve_out_rows(x, init, out_rows)
    return _permute(x, runs, inverse_runs, init, out_rows, int(x.shape[0]), int(chunk_rows), bool(interpret))


__all__ = (
    "DEFAULT_CHUNK_ROWS",
    "TILE",
    "check_supported",
    "invert_runs",
    "permute_rows_runs",
    "permute_rows_runs_pallas",
    "runs_to_index",
)
