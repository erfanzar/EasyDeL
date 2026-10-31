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

"""Pallas TPU row gathers for MoE token dispatch and expert-output combine.

A bfloat16 ``[N, H]`` row covers 1/16 of a ``(16, 128)`` tile, and Mosaic only DMAs at tile-aligned offsets of tiled
dimensions, so single rows cannot be fetched from the native layout. The source is first repacked by one streaming
kernel into int32 ``[N * H / 256, 128]``: word ``c`` of a row holds column ``c`` (low 16 bits) and column ``c + H / 2``
(high 16 bits). A source row then spans ``H / 256`` packed rows, i.e. whole ``(8, 128)`` int32 tiles when ``H`` is a
multiple of 2048, so its DMA offset is tile aligned. The gather kernels put every row copy of a grid step in flight on
one DMA semaphore, wait once, unpack with strided sublane loads and write the native bfloat16 layout.

Sparse gather-sum (``skip_negative``, e.g. one tensor-parallel shard's routes with ~1/tp of the slots live) visits only
the live slots; see :func:`gather_sum_rows`.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32
I32 = jnp.int32
LANES = 128
SUBLANES = 8
INDEX_BLOCK = SUBLANES * LANES
"""Rows per gather step: SMEM index blocks are ``(rows / 128, 128)`` and need ``rows / 128`` to be a multiple of 8."""
COMPUTE_ROWS = 128
DEFAULT_VMEM_LIMIT = 64 * 2**20
VMEM_BUDGET = 48 * 2**20
LIVE_WAIT_ROWS = 64
"""Row copies retired per semaphore wait in the sparse (``skip_negative``) gather-sum."""
_PACK_BLOCKS = (256, 128, 64, 32, 16)


def packed_rows(hidden: int) -> int:
    """int32 ``(·, 128)`` rows holding one packed bfloat16 row of width ``hidden``."""
    return hidden // (2 * LANES)


def _pack_block(n: int) -> int | None:
    return next((tb for tb in _PACK_BLOCKS if n % tb == 0), None)


def _source_reason(shape: tuple[int, ...], dtype) -> str | None:
    if len(shape) != 2:
        return f"rank-{len(shape)} source (needs [rows, hidden])"
    if jnp.dtype(dtype) != jnp.bfloat16:
        return f"dtype {jnp.dtype(dtype).name} (needs bfloat16)"
    n, h = shape
    if h % (2 * LANES * SUBLANES):
        return f"hidden size {h} is not a multiple of {2 * LANES * SUBLANES}"
    if _pack_block(n) is None:
        return f"{n} source rows is not a multiple of {_PACK_BLOCKS[-1]}"
    return None


def gather_unsupported_reason(src_shape, dtype, rows: int, *, has_dot: bool = False) -> str | None:
    """Why :func:`gather_rows` cannot run these shapes (``None`` when it can)."""
    reason = _source_reason(tuple(src_shape), dtype)
    if reason is not None:
        return reason
    n, h = src_shape
    if rows % INDEX_BLOCK:
        return f"{rows} gathered rows is not a multiple of {INDEX_BLOCK}"
    if n < INDEX_BLOCK:
        return f"{n} source rows is fewer than {INDEX_BLOCK}"
    need = INDEX_BLOCK * h * 2 * (3 + (2 if has_dot else 0))
    if need > VMEM_BUDGET:
        return f"a {INDEX_BLOCK}-row step needs {need >> 20} MiB of VMEM (budget {VMEM_BUDGET >> 20} MiB)"
    return None


def gather_sum_block_tokens(k: int) -> int:
    """Tokens per gather-sum step: ``tokens * k`` a multiple of :data:`INDEX_BLOCK` and tokens a multiple of 16."""
    tb = INDEX_BLOCK // math.gcd(k, INDEX_BLOCK)
    return tb * 16 // math.gcd(tb, 16)


def gather_sum_unsupported_reason(src_shape, dtype, tokens: int, k: int, out_dtype=None) -> str | None:
    """Why :func:`gather_sum_rows` cannot run these shapes (``None`` when it can)."""
    reason = _source_reason(tuple(src_shape), dtype)
    if reason is not None:
        return reason
    m, h = src_shape
    tb = gather_sum_block_tokens(k)
    if tokens % tb:
        return f"{tokens} tokens is not a multiple of {tb}"
    if m < tb * k:
        return f"{m} source rows is fewer than {tb * k}"
    need = tb * k * h * 2 + 2 * tb * h * jnp.dtype(out_dtype or dtype).itemsize
    if need > VMEM_BUDGET:
        return f"a {tb}-token step needs {need >> 20} MiB of VMEM (budget {VMEM_BUDGET >> 20} MiB)"
    return None


def _unpack(words):
    lo = lax.bitcast_convert_type(lax.shift_left(words, jnp.int32(16)), F32)
    hi = lax.bitcast_convert_type(words & jnp.int32(-65536), F32)
    return lo, hi


def _clamped_index(idx_ref, j, n_src: int):
    row = idx_ref[lax.div(j, LANES), lax.rem(j, LANES)]
    return jnp.minimum(jnp.maximum(row, 0), n_src - 1)


# ------------------------------------------------------------------------------------------------------------- pack
def _pack_kernel(x_ref, o_ref, *, block, sub, half):
    # bf16 -> f32 is exact (f32 bits = bf16 bits << 16).
    for a in range(sub):
        lo = lax.bitcast_convert_type(x_ref[:, a * LANES : (a + 1) * LANES].astype(F32), I32)
        hi = lax.bitcast_convert_type(x_ref[:, half + a * LANES : half + (a + 1) * LANES].astype(F32), I32)
        o_ref[pl.ds(a, block, stride=sub), :] = lax.shift_right_logical(lo, jnp.int32(16)) | (hi & jnp.int32(-65536))


def pack_rows(x: jax.Array, *, vmem_limit_bytes: int | None = None) -> jax.Array:
    """bfloat16 ``[N, H]`` -> int32 ``[N * H / 256, 128]`` (word ``c`` = column ``c`` | column ``c + H/2`` << 16)."""
    n, h = x.shape
    sub, block = packed_rows(h), _pack_block(n)
    return pl.pallas_call(
        functools.partial(_pack_kernel, block=block, sub=sub, half=h // 2),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block, h), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block * sub, LANES), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((n * sub, LANES), I32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=int(vmem_limit_bytes or DEFAULT_VMEM_LIMIT)
        ),
        name=f"moe_row_pack_b{block}",
    )(x)


# ----------------------------------------------------------------------------------------------------------- gather
def _gather_kernel(idx_ref, *refs, block, sub, half, n_src, has_scale, has_dot):
    it = iter(refs)
    src_hbm = next(it)
    scale_ref = next(it) if has_scale else None
    dot_in = next(it) if has_dot else None
    out_ref = next(it)
    dot_out = next(it) if has_dot else None
    buf, sem = next(it), next(it)

    def issue(j, carry):
        row = _clamped_index(idx_ref, j, n_src)
        pltpu.make_async_copy(
            src_hbm.at[pl.ds(pl.multiple_of(row * sub, sub), sub)],
            buf.at[pl.ds(pl.multiple_of(j * sub, sub), sub)],
            sem,
        ).start()
        return carry

    lax.fori_loop(0, block, issue, 0, unroll=8)
    # One descriptor covering the whole buffer: its byte count equals the `block` row copies signalled.
    pltpu.make_async_copy(src_hbm.at[pl.ds(0, block * sub)], buf, sem).wait()
    for c in range(block // COMPUTE_ROWS):
        r0, r1 = c * COMPUTE_ROWS, (c + 1) * COMPUTE_ROWS
        factor = scale_ref[r0:r1, :] if has_scale else None
        acc = None
        for a in range(sub):
            lo, hi = _unpack(buf[pl.ds(r0 * sub + a, COMPUTE_ROWS, stride=sub), :])
            cols_lo = slice(a * LANES, (a + 1) * LANES)
            cols_hi = slice(half + a * LANES, half + (a + 1) * LANES)
            if has_dot:
                term = lo * dot_in[r0:r1, cols_lo].astype(F32) + hi * dot_in[r0:r1, cols_hi].astype(F32)
                acc = term if acc is None else acc + term
            if has_scale:
                lo, hi = lo * factor, hi * factor
            out_ref[r0:r1, cols_lo] = lo.astype(out_ref.dtype)
            out_ref[r0:r1, cols_hi] = hi.astype(out_ref.dtype)
        if has_dot:
            dot_out[r0:r1, :] = jnp.sum(acc, axis=1, keepdims=True)


def gather_rows(
    x: jax.Array,
    indices: jax.Array,
    scale: jax.Array | None = None,
    dot_with: jax.Array | None = None,
    *,
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """``out[i] = scale[i] * x[indices[i]]`` and ``dot[i] = <x[indices[i]], dot_with[i]>`` by per-row DMA.

    Shapes must satisfy :func:`gather_unsupported_reason`.
    """
    n, h = x.shape
    rows = indices.shape[0]
    block, sub = INDEX_BLOCK, packed_rows(h)
    has_scale, has_dot = scale is not None, dot_with is not None
    in_specs = [
        pl.BlockSpec((block // LANES, LANES), lambda i: (i, 0), memory_space=pltpu.SMEM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ]
    args = [indices.astype(I32).reshape(rows // LANES, LANES), pack_rows(x, vmem_limit_bytes=vmem_limit_bytes)]
    if has_scale:
        in_specs.append(pl.BlockSpec((block, 1), lambda i: (i, 0)))
        args.append(scale.astype(F32).reshape(rows, 1))
    if has_dot:
        in_specs.append(pl.BlockSpec((block, h), lambda i: (i, 0)))
        args.append(dot_with)
    out_specs = [pl.BlockSpec((block, h), lambda i: (i, 0))]
    out_shape = [jax.ShapeDtypeStruct((rows, h), x.dtype)]
    if has_dot:
        out_specs.append(pl.BlockSpec((block, 1), lambda i: (i, 0)))
        out_shape.append(jax.ShapeDtypeStruct((rows, 1), F32))
    res = pl.pallas_call(
        functools.partial(
            _gather_kernel, block=block, sub=sub, half=h // 2, n_src=n, has_scale=has_scale, has_dot=has_dot
        ),
        grid=(rows // block,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=[pltpu.VMEM((block * sub, LANES), I32), pltpu.SemaphoreType.DMA(())],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=int(vmem_limit_bytes or DEFAULT_VMEM_LIMIT)
        ),
        name=f"moe_row_gather_s{int(has_scale)}_d{int(has_dot)}",
    )(*args)
    return (res[0], res[1][:, 0]) if has_dot else (res[0], None)


# ------------------------------------------------------------------------------------------------------- gather-sum
def _gather_sum_kernel(idx_ref, w_ref, src_hbm, out_ref, buf, sem, *, block, k, sub, half, n_src):
    # Copy j of token t lands in buffer row block (j * block + t), so each copy index is a contiguous run of tokens.
    def issue_all(slot, carry):
        t, j = lax.div(slot, k), lax.rem(slot, k)
        row = _clamped_index(idx_ref, slot, n_src)
        pltpu.make_async_copy(
            src_hbm.at[pl.ds(pl.multiple_of(row * sub, sub), sub)],
            buf.at[pl.ds(pl.multiple_of((j * block + t) * sub, sub), sub)],
            sem,
        ).start()
        return carry

    lax.fori_loop(0, block * k, issue_all, 0, unroll=8)
    # One descriptor covering the whole buffer: its byte count equals the ``block * k`` row copies signalled.
    pltpu.make_async_copy(src_hbm.at[pl.ds(0, block * k * sub)], buf, sem).wait()
    _weighted_sum(w_ref, out_ref, buf, block=block, k=k, sub=sub, half=half, mask_dead=False)


def _gather_sum_live_kernel(
    count_ref, key_ref, row_ref, w_ref, src_hbm, out_ref, buf, sem, *, block, k, sub, half, wait_rows
):
    # Grid step i owns entries [0, count[i]) of its (key, row) list block (:func:`live_slot_lists`): buffer row
    # ``key mod (block * k)`` (laid out as in the dense kernel) receives source row ``row``. Only live slots are
    # visited; dead slots leave stale buffer rows, which their zeroed weights mask in the sum.
    n = count_ref[pl.program_id(0)]
    span = block * k

    def issue(q, carry):
        r, c = lax.div(q, LANES), lax.rem(q, LANES)
        dst = lax.rem(key_ref[r, c], span)
        row = row_ref[r, c]
        pltpu.make_async_copy(
            src_hbm.at[pl.ds(pl.multiple_of(row * sub, sub), sub)],
            buf.at[pl.ds(pl.multiple_of(dst * sub, sub), sub)],
            sem,
        ).start()
        return carry

    lax.fori_loop(0, n, issue, 0)

    def waiter(rows):
        # A descriptor of ``rows`` source rows: its byte count equals that many row copies signalled.
        def wait(q, carry):
            pltpu.make_async_copy(src_hbm.at[pl.ds(0, rows * sub)], buf.at[pl.ds(0, rows * sub)], sem).wait()
            return carry

        return wait

    lax.fori_loop(0, lax.div(n, wait_rows), waiter(wait_rows), 0)
    lax.fori_loop(0, lax.rem(n, wait_rows), waiter(1), 0)
    _weighted_sum(w_ref, out_ref, buf, block=block, k=k, sub=sub, half=half, mask_dead=True)


def _weighted_sum(w_ref, out_ref, buf, *, block, k, sub, half, mask_dead):
    """``out[t] = sum_j w[t, j] * unpack(buffer row j * block + t)`` with fp32 accumulation, in slot order."""
    w = w_ref[...].astype(F32)
    for a in range(sub):
        acc_lo = acc_hi = None
        for j in range(k):
            lo, hi = _unpack(buf[pl.ds(j * block * sub + a, block, stride=sub), :])
            wj = w[:, j : j + 1]
            term_lo, term_hi = lo * wj, hi * wj
            if mask_dead:
                # Skipped slots leave stale (possibly non-finite) buffer rows; the wrapper zeroed their weights.
                live = jnp.broadcast_to(wj, term_lo.shape) != 0.0
                term_lo = jnp.where(live, term_lo, 0.0)
                term_hi = jnp.where(live, term_hi, 0.0)
            acc_lo = term_lo if acc_lo is None else acc_lo + term_lo
            acc_hi = term_hi if acc_hi is None else acc_hi + term_hi
        out_ref[:, a * LANES : (a + 1) * LANES] = acc_lo.astype(out_ref.dtype)
        out_ref[:, half + a * LANES : half + (a + 1) * LANES] = acc_hi.astype(out_ref.dtype)


def live_slot_lists(indices: jax.Array, block: int, n_src: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per grid step of ``block`` tokens, its live (non-negative) slots listed ahead of its dead ones.

    Args:
        indices: ``[tokens, k]`` source rows, negative = dead slot; ``tokens`` a multiple of ``block``.
        block: Tokens per grid step.
        n_src: Source rows (live indices are clamped to ``n_src - 1``).

    Returns:
        ``(counts [tokens / block], keys, rows)``, ``keys`` / ``rows`` int32 ``[tokens * k / 128, 128]``. Step ``i``
        owns the ``block * k`` entries from ``i * block * k``; its first ``counts[i]`` are live, and entry ``q``
        copies source row ``rows[q]`` into buffer row ``keys[q] mod (block * k)`` (``j * block + t`` for slot ``j``
        of local token ``t``).

    Raises:
        ValueError: When ``2 * tokens * k`` does not fit int32.
    """
    tokens, k = indices.shape
    nb, span = tokens // block, block * k
    if 2 * tokens * k >= 2**31:
        raise ValueError(f"gather_sum_rows(skip_negative=True): {tokens} x {k} slots overflow the int32 sort keys")
    t = jnp.arange(tokens, dtype=I32)
    dst = jnp.arange(k, dtype=I32)[None, :] * block + (t % block)[:, None]
    live = indices >= 0
    key = (t // block)[:, None] * (2 * span) + jnp.where(live, 0, span) + dst
    rows = jnp.clip(indices, 0, n_src - 1).astype(I32)
    keys, rows = lax.sort((key.astype(I32).reshape(-1), rows.reshape(-1)), num_keys=1)
    counts = jnp.sum(live.reshape(nb, span), axis=1, dtype=I32)
    return counts, keys.reshape(-1, LANES), rows.reshape(-1, LANES)


def gather_sum_rows(
    y: jax.Array,
    indices: jax.Array,
    weights: jax.Array,
    out_dtype=None,
    *,
    skip_negative: bool = False,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
    """``out[t] = sum_j weights[t, j] * y[indices[t, j]]`` (fp32 accumulation) by per-row DMA.

    With ``skip_negative`` a negative index marks an inactive slot: it contributes nothing (whatever its weight) and
    issues no DMA. An XLA sort lists each grid step's live slots first (:func:`live_slot_lists`) and the kernel
    visits only those, so sparse slot patterns (e.g. only this device's routes of a token) pay only for live rows; a
    scalar loop that tested every slot would cost more than the row copies themselves. Without ``skip_negative``
    indices are clamped to range. Shapes must satisfy :func:`gather_sum_unsupported_reason`.
    """
    m, h = y.shape
    tokens, k = indices.shape
    block, sub = gather_sum_block_tokens(k), packed_rows(h)
    span = block * k
    out_shape = jax.ShapeDtypeStruct((tokens, h), jnp.dtype(out_dtype or y.dtype))
    scratch = [pltpu.VMEM((span * sub, LANES), I32), pltpu.SemaphoreType.DMA(())]
    params = pltpu.CompilerParams(
        dimension_semantics=("parallel",), vmem_limit_bytes=int(vmem_limit_bytes or DEFAULT_VMEM_LIMIT)
    )
    packed = pack_rows(y, vmem_limit_bytes=vmem_limit_bytes)
    if not skip_negative:
        return pl.pallas_call(
            functools.partial(_gather_sum_kernel, block=block, k=k, sub=sub, half=h // 2, n_src=m),
            grid=(tokens // block,),
            in_specs=[
                pl.BlockSpec((span // LANES, LANES), lambda i: (i, 0), memory_space=pltpu.SMEM),
                pl.BlockSpec((block, k), lambda i: (i, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((block, h), lambda i: (i, 0)),
            out_shape=out_shape,
            scratch_shapes=scratch,
            compiler_params=params,
            name=f"moe_row_gather_sum_k{k}",
        )(indices.astype(I32).reshape(tokens * k // LANES, LANES), weights.astype(F32), packed)
    weights = jnp.where(indices >= 0, weights.astype(F32), 0.0)
    counts, keys, rows = live_slot_lists(indices, block, m)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(tokens // block,),
        in_specs=[
            pl.BlockSpec((span // LANES, LANES), lambda i, c: (i, 0), memory_space=pltpu.SMEM),
            pl.BlockSpec((span // LANES, LANES), lambda i, c: (i, 0), memory_space=pltpu.SMEM),
            pl.BlockSpec((block, k), lambda i, c: (i, 0)),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec((block, h), lambda i, c: (i, 0)),
        scratch_shapes=scratch,
    )
    return pl.pallas_call(
        functools.partial(_gather_sum_live_kernel, block=block, k=k, sub=sub, half=h // 2, wait_rows=LIVE_WAIT_ROWS),
        grid_spec=grid_spec,
        out_shape=out_shape,
        compiler_params=params,
        name=f"moe_row_gather_sum_k{k}_live",
    )(counts, keys, rows, weights, packed)
