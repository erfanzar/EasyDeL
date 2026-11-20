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

"""Pallas TPU short-row phase of ``ragged_kda``: every row of 1..W raw tokens in one kernel, pools updated in place.

One ``pallas_call`` on a grid of ``(rows, head blocks)``. Per grid step (row ``r``, heads ``hb``):

* the row's conv windows and recurrent state are read from its read slot ``g + a`` (block DMAs driven by the
  scalar-prefetched slot table, pipelined across rows); rows without an initial state start from zeros;
* the row's ``W`` raw q / k / v tokens (gathered per row by XLA into ``[R, W, D]`` fp32) run the depthwise causal
  conv with SiLU, the L2 norm and the per-channel KDA delta rule token by token, with the state held in VMEM. Windows
  and state are rounded to their pool dtypes after every token, as separate decode steps store them;
* after token ``j`` the windows and the state are DMAed into slot ``g + j - (c - 1)`` for ``c - 1 <= j < lens``
  (``c`` the commit length), directly into the pools, which are input/output aliases: no other slot is written and no
  pool-shaped copy is made.

The recurrence per head is the serving decode step's (``ragged_kda_decode``): ``S = S * exp(g)``,
``delta = beta * (v - k^T S)``, ``S += k delta^T``, ``o = q^T S``, with both contractions as MXU matvecs at
``HIGHEST``. Rows must own disjoint slot groups and read inside their own group (as eSurge schedules them).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ...._xla.ragged_kda._xla_impl import RowPlan, conv_compute_dtype, conv_output_dtype, is_lowfloat

L2_EPS = 1e-6  # the op's L2-norm epsilon (``kda_chunk._xla_impl.l2norm``)
_F32 = jnp.dtype(jnp.float32)


def supported(head_dim: int, v_dim: int, dtypes) -> bool:
    """Whether the kernel runs these head sizes and stream / pool dtypes (lane-aligned heads, no 1-byte floats)."""
    if int(head_dim) % 128 or int(v_dim) % 128:
        return False
    return not any(is_lowfloat(d) for d in dtypes)


def pick_block_h(num_heads: int) -> int:
    """Heads per grid step: 8 when it divides the head count, else all heads (the block rule: 8-multiple or full)."""
    return 8 if num_heads % 8 == 0 else int(num_heads)


def _vmem_limit_bytes() -> int:
    try:
        capacity = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:  # pragma: no cover - only reached off TPU
        capacity = 64 * 1024 * 1024
    return int(capacity * 0.75)


def _rnd(x, dtype):
    """``x`` (fp32) rounded through ``dtype``."""
    return x if jnp.dtype(dtype) == _F32 else x.astype(dtype).astype(jnp.float32)


def _short_kernel(
    rslot_ref,
    base_ref,
    lens_ref,
    commit_ref,
    init_ref,
    xq_ref,
    xk_ref,
    xv_ref,
    eg_ref,
    b_ref,
    wq_ref,
    wk_ref,
    wv_ref,
    cq_ref,
    ck_ref,
    cv_ref,
    st_ref,
    o_ref,
    cq_out,
    ck_out,
    cv_out,
    rec_out,
    s_scr,
    q_scr,
    k_scr,
    v_scr,
    hq_scr,
    hk_scr,
    hv_scr,
    capq,
    capk,
    capv,
    caprec,
    sems,
    *,
    width: int,
    d_conv: int,
    block_h: int,
    head_dim: int,
    v_dim: int,
    computes: tuple,
    outs: tuple,
    pool_dtypes: tuple,
    rec_dtype,
    use_qk_l2norm: bool,
):
    """One row and head block: ``width`` tokens of conv + recurrence in VMEM, captures DMAed into the pools."""
    del rslot_ref  # used by the block index maps only
    r = pl.program_id(0)
    hb = pl.program_id(1)
    f32 = jnp.float32
    n = lens_ref[r]
    commit = commit_ref[r]
    base = base_ref[r]
    kcols = block_h * head_dim
    vcols = block_h * v_dim
    scale = 1.0 / (head_dim**0.5)

    def contract(vec, s):
        # sum_k vec[h, k] * s[h, k, v] -> [h, v] (the decode kernel's MXU matvec)
        return lax.dot_general(
            vec.reshape(block_h, 1, head_dim),
            s,
            (((2,), (1,)), ((0,), (0,))),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=f32,
        ).reshape(block_h, v_dim)

    def l2(x):
        return x * lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + L2_EPS)

    def captured(j):
        return jnp.logical_and(j < n, j >= commit - 1)

    def copies(j):
        dest = base + j - (commit - 1)
        kc = pl.ds(pl.multiple_of(hb * kcols, 128), kcols)
        vc = pl.ds(pl.multiple_of(hb * vcols, 128), vcols)
        return (
            pltpu.make_async_copy(capq.at[j], cq_out.at[dest, :, kc], sems.at[0, j]),
            pltpu.make_async_copy(capk.at[j], ck_out.at[dest, :, kc], sems.at[1, j]),
            pltpu.make_async_copy(capv.at[j], cv_out.at[dest, :, vc], sems.at[2, j]),
            pltpu.make_async_copy(caprec.at[j], rec_out.at[dest, pl.ds(hb * block_h, block_h)], sems.at[3, j]),
        )

    @pl.when(n > 0)
    def _row():
        init = init_ref[r] != 0
        streams = (xq_ref, xk_ref, xv_ref)
        heads = (q_scr, k_scr, v_scr)
        hist_scr = (hq_scr, hk_scr, hv_scr)
        caps = (capq, capk, capv)
        dims = (head_dim, head_dim, v_dim)
        hists = []
        taps_w = []
        for c_ref, w_ref in ((cq_ref, wq_ref), (ck_ref, wk_ref), (cv_ref, wv_ref)):
            blk = jnp.where(init, c_ref[...].astype(f32), 0.0)
            hists.append([blk[t : t + 1, :] for t in range(d_conv)])
            wt = w_ref[...]
            taps_w.append([wt[t : t + 1, :] for t in range(d_conv)])
        s_scr[...] = jnp.where(init, st_ref[...].astype(f32), 0.0)
        for j in range(width):
            for i in range(3):
                xc = _rnd(streams[i][j : j + 1, :], computes[i])
                taps = [*hists[i][j + 1 : j + d_conv], xc]
                mixed = taps[0] * taps_w[i][0]
                for t in range(1, d_conv):
                    mixed = mixed + taps[t] * taps_w[i][t]
                y = _rnd(_rnd(jax.nn.silu(mixed), computes[i]), outs[i])
                hists[i].append(_rnd(xc, pool_dtypes[i]))
                for t in range(d_conv):
                    hist_scr[i][t : t + 1, :] = hists[i][j + 1 + t]
                caps[i][j] = hist_scr[i][...].astype(caps[i].dtype)
                d = dims[i]
                for h in range(block_h):
                    heads[i][h : h + 1, :] = y[:, h * d : (h + 1) * d]
            q = q_scr[...]
            k = k_scr[...]
            if use_qk_l2norm:
                q = _rnd(l2(q), outs[0])
                k = _rnd(l2(k), outs[1])
            q = q * scale
            s = s_scr[...] * eg_ref[j][:, :, None]
            delta = (v_scr[...] - contract(k, s)) * b_ref[j]
            s = s + k[:, :, None] * delta[:, None, :]
            o_ref[j] = contract(q, s).astype(o_ref.dtype)
            s = _rnd(s, rec_dtype)
            s_scr[...] = s
            caprec[j] = s.astype(caprec.dtype)

            @pl.when(captured(j))
            def _start(j=j):
                for cp in copies(j):
                    cp.start()

        for j in range(width):

            @pl.when(captured(j))
            def _wait(j=j):
                for cp in copies(j):
                    cp.wait()


def ragged_kda_short_pallas(
    out: Array,
    pools: tuple[Array, Array, Array, Array],
    *,
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    weights: tuple[Array, Array, Array],
    plan: RowPlan,
    short_row_max: int,
    use_qk_l2norm: bool,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    """Every short row of a step in one Pallas kernel (``short_phase``'s contract on raw q / k / v with conv pools).

    Args:
        out: ``[T, H, V]`` fp32 output buffer.
        pools: ``(q_conv_state [S, H*K, dc], k_conv_state, v_conv_state [S, H*V, dc], recurrent_state [S, H, K, V])``.
        query: Raw queries ``[T, H, K]``.
        key: Raw keys ``[T, H, K]``.
        value: Raw values ``[T, H, V]``.
        beta: ``[T, H]``.
        decay: ``[T, H, K]`` per-channel log-decay.
        weights: ``(q, k, v)`` conv taps ``[dim, dc]``.
        plan: The step's row plan.
        short_row_max: ``W``.
        use_qk_l2norm: L2-normalise queries and keys.

    Returns:
        ``(out, pools)`` with the short rows' outputs and captured slots written (pools updated in place).
    """
    q_pool, k_pool, v_pool, rec = pools
    num_tokens, num_heads, head_dim = query.shape
    v_dim = value.shape[-1]
    num_slots = rec.shape[0]
    num_rows = plan.lens.shape[0]
    width = int(short_row_max)
    d_conv = int(weights[0].shape[-1])
    f32 = jnp.float32
    i32 = jnp.int32
    bh = pick_block_h(num_heads)

    lens = jnp.where(plan.short, plan.lens, 0).astype(i32)
    pos = jnp.arange(width, dtype=i32)
    valid = pos[None, :] < lens[:, None]  # [R, W]
    tok = jnp.clip(plan.starts[:, None] + pos[None, :], 0, num_tokens - 1)

    def rows(x: Array) -> Array:
        g = x[tok].astype(f32)
        return jnp.where(valid.reshape(valid.shape + (1,) * (g.ndim - 2)), g, 0.0)

    xq = rows(query.reshape(num_tokens, num_heads * head_dim))
    xk = rows(key.reshape(num_tokens, num_heads * head_dim))
    xv = rows(value.reshape(num_tokens, num_heads * v_dim))
    eg = jnp.exp(rows(decay))  # [R, W, H, K]
    bb = jnp.broadcast_to(rows(beta)[..., None], (num_rows, width, num_heads, v_dim))
    wts = [jnp.swapaxes(w, 0, 1).astype(f32) for w in weights]  # [dc, dim]
    views = [jnp.swapaxes(p, 1, 2) for p in (q_pool, k_pool, v_pool)]  # [S, dc, dim]

    stream_dtypes = (query.dtype, key.dtype, value.dtype)
    pool_dtypes = (q_pool.dtype, k_pool.dtype, v_pool.dtype)
    computes = tuple(conv_compute_dtype(s, p) for s, p in zip(stream_dtypes, pool_dtypes, strict=True))
    outs = tuple(conv_output_dtype(s) for s in stream_dtypes)

    kcols, vcols = bh * head_dim, bh * v_dim
    tok_k = pl.BlockSpec((None, width, kcols), lambda r, h, *_: (r, 0, h))
    tok_v = pl.BlockSpec((None, width, vcols), lambda r, h, *_: (r, 0, h))
    eg_spec = pl.BlockSpec((None, width, bh, head_dim), lambda r, h, *_: (r, 0, h, 0))
    b_spec = pl.BlockSpec((None, width, bh, v_dim), lambda r, h, *_: (r, 0, h, 0))
    w_k = pl.BlockSpec((d_conv, kcols), lambda r, h, *_: (0, h))
    w_v = pl.BlockSpec((d_conv, vcols), lambda r, h, *_: (0, h))

    def win_map(r, h, rslot, *_):
        return (rslot[r], 0, h)

    def rec_map(r, h, rslot, *_):
        return (rslot[r], h, 0, 0)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    kernel = functools.partial(
        _short_kernel,
        width=width,
        d_conv=d_conv,
        block_h=bh,
        head_dim=head_dim,
        v_dim=v_dim,
        computes=computes,
        outs=outs,
        pool_dtypes=pool_dtypes,
        rec_dtype=rec.dtype,
        use_qk_l2norm=use_qk_l2norm,
    )
    o, nq, nk, nv, new_rec = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(num_rows, num_heads // bh),
            in_specs=[
                tok_k,
                tok_k,
                tok_v,
                eg_spec,
                b_spec,
                w_k,
                w_k,
                w_v,
                pl.BlockSpec((None, d_conv, kcols), win_map),
                pl.BlockSpec((None, d_conv, kcols), win_map),
                pl.BlockSpec((None, d_conv, vcols), win_map),
                pl.BlockSpec((None, bh, head_dim, v_dim), rec_map),
            ],
            out_specs=[b_spec, hbm, hbm, hbm, hbm],
            scratch_shapes=[
                pltpu.VMEM((bh, head_dim, v_dim), f32),
                pltpu.VMEM((bh, head_dim), f32),
                pltpu.VMEM((bh, head_dim), f32),
                pltpu.VMEM((bh, v_dim), f32),
                pltpu.VMEM((d_conv, kcols), f32),
                pltpu.VMEM((d_conv, kcols), f32),
                pltpu.VMEM((d_conv, vcols), f32),
                pltpu.VMEM((width, d_conv, kcols), q_pool.dtype),
                pltpu.VMEM((width, d_conv, kcols), k_pool.dtype),
                pltpu.VMEM((width, d_conv, vcols), v_pool.dtype),
                pltpu.VMEM((width, bh, head_dim, v_dim), rec.dtype),
                pltpu.SemaphoreType.DMA((4, width)),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_rows, width, num_heads, v_dim), f32),
            jax.ShapeDtypeStruct(views[0].shape, views[0].dtype),
            jax.ShapeDtypeStruct(views[1].shape, views[1].dtype),
            jax.ShapeDtypeStruct(views[2].shape, views[2].dtype),
            jax.ShapeDtypeStruct(rec.shape, rec.dtype),
        ],
        # operands: 5 tables, xq, xk, xv, eg, b, wq, wk, wv, cq (13), ck, cv, rec (16) -> outputs 1..4 in place.
        input_output_aliases={13: 1, 14: 2, 15: 3, 16: 4},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=_vmem_limit_bytes(),
        ),
        name=f"ejkernel_ragged_kda_short_w{width}_bh{bh}",
    )(
        jnp.clip(plan.read_slot, 0, num_slots - 1).astype(i32),
        plan.base_slot.astype(i32),
        lens,
        plan.commit.astype(i32),
        plan.init.astype(i32),
        xq,
        xk,
        xv,
        eg,
        bb,
        *wts,
        *views,
        rec,
    )
    dest = jnp.where(valid, tok, out.shape[0]).reshape(-1)
    out = out.at[dest].set(o.reshape(num_rows * width, num_heads, v_dim), mode="drop")
    nq, nk, nv = (jnp.swapaxes(p, 1, 2) for p in (nq, nk, nv))
    return out, (nq, nk, nv, new_rec)


__all__ = ("pick_block_h", "ragged_kda_short_pallas", "supported")
