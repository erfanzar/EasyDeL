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

"""High-throughput Pallas TPU kernels for ragged Gated Delta Rule (GDN).

Provides three in-place-pool Pallas TPU kernels tailored to continuous-batching
serving (eSurge):

* :func:`pallas_gdn_chunk_prefill`: Double-buffered direct-DMA chunked GDN for
  prefill and mixed prefill+decode steps. Reads compact ``mixed_qkv`` directly
  from HBM via 16-row-aligned async DMA + 4-bit VMEM barrel shift, computes
  L2-norm, lower-triangular ``cumsum(g)``, grouped ``K@K^T`` / ``Q@K^T``
  (reused across value heads per key head), and WY solve with recurrent state
  pinned in VMEM. Single-token decode rows in mixed batches are dispatched
  separately via a direct-DMA dual-core decode kernel so they never pay chunk
  padding overhead.
* :func:`pallas_gdn_decode_inplace`: Dual-core, ring-buffered in-place decode
  kernel operating directly on the recurrent-state pool via ``state_indices``.
* :func:`pallas_gdn_conv_decode`: Fused causal depthwise Conv1D + SiLU +
  L2-norm + GDN single-token decode in a single ``pallas_call``, updating both
  the conv-state pool and recurrent-state pool in place.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ejkernel.kernels._pallas.tpu.ragged_kda._pallas_impl import max_live_chunks
from ejkernel.kernels._xla.kda_chunk._xla_impl import iota_rc, unit_lower_inverse_masked

_SUPPORTED_CHUNK_SIZES = (64, 128)
_SUPPORTED_DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))


def _dot(a: jax.Array, b: jax.Array, precision: jax.lax.PrecisionLike) -> jax.Array:
    return lax.dot(a, b, precision=precision, preferred_element_type=jnp.float32)


def _vmem_shift_left_16(x_f32: jax.Array, rem: jax.Array, cs: int) -> jax.Array:
    """Shift a ``[cs + 16, D]`` fp32 VMEM tile left by ``rem`` in ``[0, 15]``."""
    cur = x_f32
    cur_len = cs + 16
    for bit in (3, 2, 1, 0):
        step = 1 << bit
        next_len = cur_len - step
        shifted = cur[step : step + next_len, :]
        stay = cur[0:next_len, :]
        take = ((rem >> bit) & 1) != 0
        cur = jnp.where(take, shifted, stay)
        cur_len = next_len
    return cur[:cs, :]


def _vmem_capacity_bytes() -> int:
    try:
        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:
        return 32 * 1024 * 1024


def _num_tensor_cores() -> int:
    try:
        return max(1, int(pltpu.get_tpu_info().num_cores))
    except Exception:
        return 1


def _round_up(x: int, m: int) -> int:
    return ((int(x) + int(m) - 1) // int(m)) * int(m)


def _tile_bytes(rows: int, cols: int, itemsize: int) -> int:
    """VMEM bytes of a ``[rows, cols]`` tile, padded to native ``(8 * packing, 128)`` tiles."""
    packing = max(1, 4 // int(itemsize))
    return _round_up(rows, 8 * packing) * _round_up(cols, 128) * int(itemsize)


def _short_vmem_estimate_bytes(
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    num_rows: int,
    in_itemsize: int,
    st_itemsize: int,
) -> int:
    """Estimate per-core VMEM bytes for :func:`_short_rows_direct_kernel`."""
    conv_dim = 2 * int(n_kq) * int(d_k) + int(n_v) * int(d_v)
    r_pad = ((max(1, int(num_rows)) + 15) // 16) * 16
    if r_pad > 16:
        r_pad = ((r_pad + 31) // 32) * 32
    cores_s = 2 if (_num_tensor_cores() >= 2 and r_pad >= 32 and r_pad % 32 == 0) else 1
    per_s = r_pad // cores_s
    qkv_buf = _tile_bytes(16, conv_dim, in_itemsize)
    gb_buf = _tile_bytes(16, 256, 4)
    st_buf = 2 * int(n_v) * _tile_bytes(d_k, d_v, st_itemsize)
    out_ref = per_s * _tile_bytes(n_v, d_v, in_itemsize)
    live_f32_state = 2 * int(n_v) * _tile_bytes(d_k, d_v, 4)
    return qkv_buf + gb_buf + st_buf + out_ref + live_f32_state


def _chunk_vmem_estimate_bytes(
    *,
    bkq: int,
    expand: int,
    cs: int,
    d_k: int,
    d_v: int,
    in_itemsize: int,
    st_itemsize: int,
) -> int:
    """Estimate per-core VMEM bytes for :func:`_gdn_chunk_v5_kernel`."""
    bv = int(bkq) * int(expand)
    cs_i = int(cs)
    d_k_i, d_v_i = int(d_k), int(d_v)
    state_scr = bv * _tile_bytes(d_k_i, d_v_i, 4)
    qk_bufs = 2 * 2 * _tile_bytes(cs_i + 16, int(bkq) * d_k_i, in_itemsize)
    v_buf = 2 * _tile_bytes(cs_i + 16, bv * d_v_i, in_itemsize)
    gb_buf = 2 * _tile_bytes(cs_i + 16, 256, 4)
    o_buf = 2 * _tile_bytes(cs_i, bv * d_v_i, in_itemsize)
    st_io_bufs = 2 * bv * _tile_bytes(d_k_i, d_v_i, st_itemsize)
    live_vmap = (8 * _tile_bytes(cs_i, cs_i, 4) + 6 * _tile_bytes(cs_i, max(d_k_i, d_v_i), 4)) * bv
    return state_scr + qk_bufs + v_buf + gb_buf + o_buf + st_io_bufs + live_vmap


def _conv_decode_vmem_estimate_bytes(
    *,
    num_tokens: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    d_conv: int,
    in_itemsize: int,
    conv_itemsize: int,
    st_itemsize: int,
    out_itemsize: int,
    ring: int = 3,
) -> int:
    """Estimate per-core VMEM bytes for :func:`_fused_conv_gdn_decode_kernel`."""
    conv_dim = 2 * int(n_kq) * int(d_k) + int(n_v) * int(d_v)
    t_pad = ((max(1, int(num_tokens)) + 15) // 16) * 16
    if _num_tensor_cores() >= 2 and t_pad >= 32:
        t_pad = ((t_pad + 31) // 32) * 32
    cores = 2 if (_num_tensor_cores() >= 2 and t_pad >= 32 and t_pad % 32 == 0) else 1
    per_core = t_pad // cores
    ring_eff = min(max(1, int(ring)), max(2, per_core))

    x_ref = _tile_bytes(per_core, conv_dim, in_itemsize)
    beg_ref = 2 * _tile_bytes(per_core, 128, 4)
    w_ref = _tile_bytes(d_conv, conv_dim, 4)
    out_ref = per_core * _tile_bytes(n_v, d_v, out_itemsize)
    sx_scr = _tile_bytes(per_core, conv_dim, 4)
    sbeg_scr = 2 * _tile_bytes(per_core, 128, 4)
    wb_scr = ring_eff * _tile_bytes(d_conv, conv_dim, conv_itemsize)
    stb_scr = ring_eff * int(n_v) * _tile_bytes(d_k, d_v, st_itemsize)
    hist_scr = _tile_bytes(d_conv, conv_dim, 4)
    qkv_scr = 2 * _tile_bytes(n_kq, d_k, 4) + _tile_bytes(n_v, d_v, 4) + 2 * _tile_bytes(n_v, d_k, 4)
    live_f32_state = 2 * int(n_v) * _tile_bytes(d_k, d_v, 4)
    return x_ref + beg_ref + w_ref + out_ref + sx_scr + sbeg_scr + wb_scr + stb_scr + hist_scr + qkv_scr + live_f32_state


def pallas_gdn_chunk_prefill_supported(
    *,
    mixed_qkv: jax.Array,
    recurrent_state: jax.Array,
    query_start_loc: jax.Array,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int,
) -> bool:
    """Return whether :func:`pallas_gdn_chunk_prefill` supports the given shapes."""
    if jax.default_backend() != "tpu":
        return False
    if int(d_k) != 128 or int(d_v) != 128:
        return False
    if int(chunk_size) not in _SUPPORTED_CHUNK_SIZES:
        return False
    if int(n_kq) <= 0 or int(n_v) <= 0 or int(n_v) > 128 or int(n_v) % int(n_kq) != 0:
        return False
    if (int(n_v) > 8 and int(n_v) % 8 != 0) or (int(n_kq) > 8 and int(n_kq) % 8 != 0):
        return False
    if mixed_qkv.ndim != 2 or recurrent_state.ndim != 4 or query_start_loc.ndim != 1:
        return False
    if int(mixed_qkv.shape[0]) <= 0 or int(query_start_loc.shape[0]) < 2:
        return False
    if int(recurrent_state.shape[0]) <= 0:
        return False
    if tuple(int(dim) for dim in recurrent_state.shape[1:]) != (int(n_v), int(d_k), int(d_v)):
        return False
    if jnp.dtype(mixed_qkv.dtype) not in _SUPPORTED_DTYPES:
        return False
    if jnp.dtype(recurrent_state.dtype) not in _SUPPORTED_DTYPES:
        return False
    expected_width = 2 * int(n_kq) * int(d_k) + int(n_v) * int(d_v)
    if int(mixed_qkv.shape[-1]) != expected_width:
        return False
    vmem_budget = int(_vmem_capacity_bytes() * 0.70)
    in_itemsize = jnp.dtype(mixed_qkv.dtype).itemsize
    st_itemsize = jnp.dtype(recurrent_state.dtype).itemsize
    short_bytes = _short_vmem_estimate_bytes(
        n_kq=int(n_kq),
        n_v=int(n_v),
        d_k=int(d_k),
        d_v=int(d_v),
        num_rows=int(query_start_loc.shape[0]) - 1,
        in_itemsize=in_itemsize,
        st_itemsize=st_itemsize,
    )
    chunk_bytes = _chunk_vmem_estimate_bytes(
        bkq=1,
        expand=int(n_v) // int(n_kq),
        cs=int(chunk_size),
        d_k=int(d_k),
        d_v=int(d_v),
        in_itemsize=in_itemsize,
        st_itemsize=st_itemsize,
    )
    if short_bytes > vmem_budget or chunk_bytes > vmem_budget:
        return False
    return True


def pallas_gdn_conv_decode_supported(
    *,
    conv_input: jax.Array,
    conv_weight: jax.Array,
    conv_pool: jax.Array,
    recurrent_pool: jax.Array,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    out_dtype: tp.Any = jnp.float32,
    conv_output_dtype: tp.Any = None,
    ring: int = 3,
    conv_pool_transposed: bool = False,
) -> bool:
    """Return whether :func:`pallas_gdn_conv_decode` supports the given shapes."""
    if jax.default_backend() != "tpu":
        return False
    if int(d_k) != 128 or int(d_v) != 128:
        return False
    if int(n_kq) <= 0 or int(n_v) <= 0 or int(n_v) > 128 or int(n_v) % int(n_kq) != 0:
        return False
    if (int(n_v) > 8 and int(n_v) % 8 != 0) or (int(n_kq) > 8 and int(n_kq) % 8 != 0):
        return False
    if conv_input.ndim != 2 or conv_weight.ndim != 2 or conv_pool.ndim != 3 or recurrent_pool.ndim != 4:
        return False
    if int(conv_input.shape[0]) <= 0 or int(recurrent_pool.shape[0]) <= 0:
        return False
    if int(conv_pool.shape[0]) != int(recurrent_pool.shape[0]):
        return False
    if tuple(int(dim) for dim in recurrent_pool.shape[1:]) != (int(n_v), int(d_k), int(d_v)):
        return False
    conv_dim = 2 * int(n_kq) * int(d_k) + int(n_v) * int(d_v)
    if int(conv_input.shape[-1]) != conv_dim or int(conv_weight.shape[0]) != conv_dim:
        return False
    d_conv = int(conv_weight.shape[-1])
    if d_conv < 2 or (d_conv > 8 and d_conv % 8 != 0):
        return False
    expected_pool_tail = (d_conv, conv_dim) if conv_pool_transposed else (conv_dim, d_conv)
    if tuple(int(dim) for dim in conv_pool.shape[1:]) != expected_pool_tail:
        return False
    c_dtype = jnp.dtype(conv_output_dtype or conv_input.dtype)
    o_dtype = jnp.dtype(out_dtype)
    if jnp.dtype(conv_input.dtype) not in _SUPPORTED_DTYPES:
        return False
    if jnp.dtype(conv_pool.dtype) not in _SUPPORTED_DTYPES:
        return False
    if jnp.dtype(recurrent_pool.dtype) not in _SUPPORTED_DTYPES:
        return False
    if c_dtype not in _SUPPORTED_DTYPES or o_dtype not in _SUPPORTED_DTYPES:
        return False
    vmem_budget = int(_vmem_capacity_bytes() * 0.70)
    est_bytes = _conv_decode_vmem_estimate_bytes(
        num_tokens=int(conv_input.shape[0]),
        n_kq=int(n_kq),
        n_v=int(n_v),
        d_k=int(d_k),
        d_v=int(d_v),
        d_conv=d_conv,
        in_itemsize=jnp.dtype(conv_input.dtype).itemsize,
        conv_itemsize=jnp.dtype(conv_pool.dtype).itemsize,
        st_itemsize=jnp.dtype(recurrent_pool.dtype).itemsize,
        out_itemsize=o_dtype.itemsize,
        ring=int(ring),
    )
    if est_bytes > vmem_budget:
        return False
    return True


# ---------------------------------------------------------------------------
# 1. Direct-DMA Short-Row (len == 1) Decode Kernel for Mixed Batches
# ---------------------------------------------------------------------------
def _short_rows_direct_kernel(
    n_short_ref,  # SMEM [1] int32
    s_tok_ref,  # SMEM [R_pad] int32
    s_slot_ref,  # SMEM [R_pad] int32
    s_init_ref,  # SMEM [R_pad] int32
    mixed_hbm,  # HBM [T_pad, conv_dim]
    gb_hbm,  # HBM [T_pad, 256] f32 ([:128]=g, [128:]=beta)
    pool_hbm,  # HBM [S, n_v, K, V]
    out_ref,  # VMEM [per_core, n_v, V]
    new_pool_hbm,  # HBM [S, n_v, K, V] (alias)
    qkv_buf,  # VMEM [16, conv_dim]
    gb_buf,  # VMEM [16, 256] f32
    st_buf,  # VMEM [2, n_v, K, V]
    rd_sems,  # DMA (2, 3)
    wr_sems,  # DMA (2,)
    *,
    per_core: int,
    n_kq: int,
    expand: int,
    d_k: int,
    d_v: int,
    precision: jax.lax.PrecisionLike,
    use_qk_l2norm: bool,
):
    n_short = n_short_ref[0]
    f32 = jnp.float32
    n_v = n_kq * expand
    kd = n_kq * d_k
    scale = 1.0 / (d_k**0.5)
    base = pl.program_id(0) * per_core
    n_mine = jnp.clip(n_short - base, 0, per_core)

    row_nv = lax.broadcasted_iota(jnp.int32, (n_v, 128), 0)
    col_nv = lax.broadcasted_iota(jnp.int32, (n_v, 128), 1)
    eye_nv_mask = row_nv == col_nv
    row_16 = lax.broadcasted_iota(jnp.int32, (16, 1), 0)

    @pl.when(n_mine > 0)
    def _run():
        def start_st_rd(k, b):
            @pl.when(s_init_ref[k] != 0)
            def _s():
                pltpu.make_async_copy(pool_hbm.at[s_slot_ref[k]], st_buf.at[b], rd_sems.at[b, 2]).start()

        def wait_st_rd(k, b):
            @pl.when(s_init_ref[k] != 0)
            def _w():
                pltpu.make_async_copy(pool_hbm.at[s_slot_ref[k]], st_buf.at[b], rd_sems.at[b, 2]).wait()

        start_st_rd(base, 0)

        def body(i, carry):
            k = base + i
            b = i & 1
            nb = 1 - b

            @pl.when((i + 1) < n_mine)
            def _pref():
                @pl.when(i >= 1)
                def _ww():
                    pltpu.make_async_copy(st_buf.at[nb], new_pool_hbm.at[s_slot_ref[k - 1]], wr_sems.at[nb]).wait()

                start_st_rd(k + 1, nb)

            t_idx = s_tok_ref[k]
            a_t = pl.multiple_of((t_idx // 16) * 16, 16)
            off_t = t_idx - a_t
            pltpu.make_async_copy(mixed_hbm.at[pl.ds(a_t, 16)], qkv_buf, rd_sems.at[b, 0]).start()
            pltpu.make_async_copy(gb_hbm.at[pl.ds(a_t, 16)], gb_buf, rd_sems.at[b, 1]).start()
            wait_st_rd(k, b)
            pltpu.make_async_copy(mixed_hbm.at[pl.ds(a_t, 16)], qkv_buf, rd_sems.at[b, 0]).wait()
            pltpu.make_async_copy(gb_hbm.at[pl.ds(a_t, 16)], gb_buf, rd_sems.at[b, 1]).wait()

            sel_mask = row_16 == off_t
            qkv_row = jnp.sum(jnp.where(sel_mask, qkv_buf[...].astype(f32), 0.0), axis=0, keepdims=True)
            gb_row = jnp.sum(jnp.where(sel_mask, gb_buf[...].astype(f32), 0.0), axis=0, keepdims=True)

            q_h = jnp.concatenate([qkv_row[:, g * d_k : (g + 1) * d_k] for g in range(n_kq)], axis=0)
            k_h = jnp.concatenate([qkv_row[:, kd + g * d_k : kd + (g + 1) * d_k] for g in range(n_kq)], axis=0)
            v_all = jnp.concatenate([qkv_row[:, 2 * kd + h * d_v : 2 * kd + (h + 1) * d_v] for h in range(n_v)], axis=0)
            if use_qk_l2norm:
                q_h = q_h * lax.rsqrt(jnp.sum(q_h * q_h, axis=-1, keepdims=True) + 1e-6)
                k_h = k_h * lax.rsqrt(jnp.sum(k_h * k_h, axis=-1, keepdims=True) + 1e-6)
            q_h = q_h * scale
            if expand > 1:
                q_all = jnp.concatenate([jnp.broadcast_to(q_h[g : g + 1], (expand, d_k)) for g in range(n_kq)], axis=0)
                k_all = jnp.concatenate([jnp.broadcast_to(k_h[g : g + 1], (expand, d_k)) for g in range(n_kq)], axis=0)
            else:
                q_all = q_h
                k_all = k_h

            g_col = jnp.sum(
                jnp.where(eye_nv_mask, jnp.broadcast_to(gb_row[:, :128], (n_v, 128)), 0.0), axis=1, keepdims=True
            )
            b_col = jnp.sum(
                jnp.where(eye_nv_mask, jnp.broadcast_to(gb_row[:, 128:], (n_v, 128)), 0.0), axis=1, keepdims=True
            )
            eg_col = jnp.exp(g_col)

            s = jnp.where(s_init_ref[k] != 0, st_buf[b].astype(f32), 0.0) * eg_col[:, :, None]
            ks = lax.dot_general(
                k_all.reshape(n_v, 1, d_k),
                s,
                (((2,), (1,)), ((0,), (0,))),
                precision=precision,
                preferred_element_type=f32,
            ).reshape(n_v, d_v)
            delta = (v_all - ks) * b_col
            s = s + k_all[:, :, None] * delta[:, None, :]
            o = lax.dot_general(
                q_all.reshape(n_v, 1, d_k),
                s,
                (((2,), (1,)), ((0,), (0,))),
                precision=precision,
                preferred_element_type=f32,
            ).reshape(n_v, d_v)
            out_ref[i] = o.astype(out_ref.dtype)
            st_buf[b] = s.astype(st_buf.dtype)
            pltpu.make_async_copy(st_buf.at[b], new_pool_hbm.at[s_slot_ref[k]], wr_sems.at[b]).start()
            return carry

        lax.fori_loop(0, n_mine, body, 0)

        @pl.when(n_mine >= 2)
        def _d2():
            i2 = n_mine - 2
            pltpu.make_async_copy(st_buf.at[i2 & 1], new_pool_hbm.at[s_slot_ref[base + i2]], wr_sems.at[i2 & 1]).wait()

        i1 = n_mine - 1
        pltpu.make_async_copy(st_buf.at[i1 & 1], new_pool_hbm.at[s_slot_ref[base + i1]], wr_sems.at[i1 & 1]).wait()


# ---------------------------------------------------------------------------
# 2. Double-Buffered Direct-DMA Chunked GDN Prefill Kernel
# ---------------------------------------------------------------------------
def _gdn_chunk_v5_kernel(
    n_live_ref,  # SMEM [1]
    slot_ref,  # SMEM [C]
    first_ref,  # SMEM [C]
    last_ref,  # SMEM [C]
    init_ref,  # SMEM [C]
    astart_ref,  # SMEM [C]
    rem_ref,  # SMEM [C]
    lo_ref,  # SMEM [C]
    hi_ref,  # SMEM [C]
    mixed_hbm,  # HBM [T_pad, 2*n_kq*K + n_v*V]
    gb_hbm,  # HBM [T_pad, 256] f32 ([:128]=g, [128:]=beta)
    pool_hbm,  # HBM [S, n_v, K, V]
    out_hbm,  # HBM [C, cs, n_v*V]
    new_pool_hbm,  # HBM [S, n_v, K, V] (alias)
    state_scr,  # VMEM [bv, K, V] f32
    q_buf,  # VMEM [2, cs + 16, bkq * K]
    k_buf,  # VMEM [2, cs + 16, bkq * K]
    v_buf,  # VMEM [2, cs + 16, bv * V]
    gb_buf,  # VMEM [2, cs + 16, 256] f32
    o_buf,  # VMEM [2, cs, bv * V]
    st_in_buf,  # VMEM [bv, K, V]
    st_out_buf,  # VMEM [bv, K, V]
    rd_sems,  # DMA (2, 4)
    wr_sems,  # DMA (2,)
    st_sems,  # DMA (2,)
    *,
    bkq: int,
    n_kq: int,
    expand: int,
    d_k: int,
    d_v: int,
    chunk_size: int,
    num_chunks: int,
    precision: jax.lax.PrecisionLike,
    inv_precision: jax.lax.PrecisionLike,
    qk_precision: jax.lax.PrecisionLike,
    use_qk_l2norm: bool,
):
    hb = pl.program_id(0)
    n_live = n_live_ref[0]
    f32 = jnp.float32
    cs = chunk_size
    bv = bkq * expand
    kd = n_kq * d_k
    scale = 1.0 / (d_k**0.5)

    q_col = pl.ds(pl.multiple_of(hb * bkq * d_k, 128), bkq * d_k)
    k_col = pl.ds(pl.multiple_of(kd + hb * bkq * d_k, 128), bkq * d_k)
    v_col = pl.ds(pl.multiple_of(2 * kd + hb * bv * d_v, 128), bv * d_v)
    o_col = pl.ds(pl.multiple_of(hb * bv * d_v, 128), bv * d_v)
    st_hb = pl.ds(hb * bv, bv)

    row, col = iota_rc(cs)
    incl = col <= row
    strict = col < row
    diag = row == col
    tril_f32 = incl.astype(f32)
    lane_i = lax.broadcasted_iota(jnp.int32, (cs, 1), 0)
    last_lane_mask = lane_i == (cs - 1)
    col_128 = lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    h_base = hb * bv

    def start_chunk_read(ci, b):
        a_s = pl.multiple_of(astart_ref[ci], 16)
        pltpu.make_async_copy(mixed_hbm.at[pl.ds(a_s, cs + 16), q_col], q_buf.at[b], rd_sems.at[b, 0]).start()
        pltpu.make_async_copy(mixed_hbm.at[pl.ds(a_s, cs + 16), k_col], k_buf.at[b], rd_sems.at[b, 1]).start()
        pltpu.make_async_copy(mixed_hbm.at[pl.ds(a_s, cs + 16), v_col], v_buf.at[b], rd_sems.at[b, 2]).start()
        pltpu.make_async_copy(gb_hbm.at[pl.ds(a_s, cs + 16)], gb_buf.at[b], rd_sems.at[b, 3]).start()

    def wait_chunk_read(ci, b):
        a_s = pl.multiple_of(astart_ref[ci], 16)
        pltpu.make_async_copy(mixed_hbm.at[pl.ds(a_s, cs + 16), q_col], q_buf.at[b], rd_sems.at[b, 0]).wait()
        pltpu.make_async_copy(mixed_hbm.at[pl.ds(a_s, cs + 16), k_col], k_buf.at[b], rd_sems.at[b, 1]).wait()
        pltpu.make_async_copy(mixed_hbm.at[pl.ds(a_s, cs + 16), v_col], v_buf.at[b], rd_sems.at[b, 2]).wait()
        pltpu.make_async_copy(gb_hbm.at[pl.ds(a_s, cs + 16)], gb_buf.at[b], rd_sems.at[b, 3]).wait()

    @pl.when(n_live > 0)
    def _run_all():
        start_chunk_read(0, 0)
        c0_keep = jnp.logical_and(first_ref[0] != 0, init_ref[0] != 0)

        @pl.when(c0_keep)
        def _st0_start():
            pltpu.make_async_copy(pool_hbm.at[slot_ref[0], st_hb], st_in_buf, st_sems.at[0]).start()

        def body(c, st_wr_pending):
            b = c & 1
            nb = 1 - b
            has_next = (c + 1) < n_live

            @pl.when(has_next)
            def _pref_next():
                start_chunk_read(c + 1, nb)

            wait_chunk_read(c, b)

            is_first = first_ref[c] != 0
            keep = init_ref[c] != 0

            @pl.when(jnp.logical_and(is_first, keep))
            def _seed():
                pltpu.make_async_copy(pool_hbm.at[slot_ref[c], st_hb], st_in_buf, st_sems.at[0]).wait()
                state_scr[...] = st_in_buf[...].astype(f32)

            @pl.when(jnp.logical_and(is_first, jnp.logical_not(keep)))
            def _fresh():
                state_scr[...] = jnp.zeros(state_scr.shape, f32)

            next_c = jnp.minimum(c + 1, num_chunks - 1)
            next_seed = jnp.logical_and(
                has_next,
                jnp.logical_and(first_ref[next_c] != 0, init_ref[next_c] != 0),
            )

            @pl.when(next_seed)
            def _pref_st():
                pltpu.make_async_copy(pool_hbm.at[slot_ref[next_c], st_hb], st_in_buf, st_sems.at[0]).start()

            rem = rem_ref[c]
            q_flat = _vmem_shift_left_16(q_buf[b].astype(f32), rem, cs)
            k_flat = _vmem_shift_left_16(k_buf[b].astype(f32), rem, cs)
            v_flat = _vmem_shift_left_16(v_buf[b].astype(f32), rem, cs)
            gb_flat = _vmem_shift_left_16(gb_buf[b].astype(f32), rem, cs)

            valid = jnp.logical_and(lane_i >= lo_ref[c], lane_i < hi_ref[c])
            # Clamp extreme negative log-decays (exp(-128) == 0.0 exactly in fp32)
            # so ``tril_f32 @ g_tile`` and ``gc_col - gc_row`` never hit ``0 * -inf``
            # or ``(-inf) - (-inf)`` NaNs.
            g_tile = jnp.maximum(jnp.where(valid, gb_flat[:, :128], 0.0), -128.0)
            b_tile = jnp.where(valid, gb_flat[:, 128:], 0.0)
            gc_tile = lax.dot(tril_f32, g_tile, precision=lax.Precision.HIGHEST, preferred_element_type=f32)

            gc_cols = jnp.stack(
                [jnp.sum(jnp.where(col_128 == (h_base + h), gc_tile, 0.0), axis=1, keepdims=True) for h in range(bv)],
                axis=0,
            )
            b_cols = jnp.stack(
                [jnp.sum(jnp.where(col_128 == (h_base + h), b_tile, 0.0), axis=1, keepdims=True) for h in range(bv)],
                axis=0,
            )

            q_heads = []
            k_heads = []
            for g_idx in range(bkq):
                qg = q_flat[:, g_idx * d_k : (g_idx + 1) * d_k]
                kg = k_flat[:, g_idx * d_k : (g_idx + 1) * d_k]
                if use_qk_l2norm:
                    qg = qg * lax.rsqrt(jnp.sum(qg * qg, axis=-1, keepdims=True) + 1e-6)
                    kg = kg * lax.rsqrt(jnp.sum(kg * kg, axis=-1, keepdims=True) + 1e-6)
                q_heads.append(jnp.where(valid, qg * scale, 0.0))
                k_heads.append(jnp.where(valid, kg, 0.0))
            q_all = jnp.stack(q_heads, axis=0)
            k_all = jnp.stack(k_heads, axis=0)
            v_all = jnp.stack([jnp.where(valid, v_flat[:, h * d_v : (h + 1) * d_v], 0.0) for h in range(bv)], axis=0)

            kk_kq = lax.dot_general(
                k_all,
                k_all,
                (((2,), (2,)), ((0,), (0,))),
                precision=qk_precision,
                preferred_element_type=f32,
            )
            qk_kq = lax.dot_general(
                q_all,
                k_all,
                (((2,), (2,)), ((0,), (0,))),
                precision=qk_precision,
                preferred_element_type=f32,
            )

            if expand > 1:
                q_exp = jnp.concatenate(
                    [jnp.broadcast_to(q_all[g_idx : g_idx + 1], (expand, cs, d_k)) for g_idx in range(bkq)], axis=0
                )
                k_exp = jnp.concatenate(
                    [jnp.broadcast_to(k_all[g_idx : g_idx + 1], (expand, cs, d_k)) for g_idx in range(bkq)], axis=0
                )
                kk_exp = jnp.concatenate(
                    [jnp.broadcast_to(kk_kq[g_idx : g_idx + 1], (expand, cs, cs)) for g_idx in range(bkq)], axis=0
                )
                qk_exp = jnp.concatenate(
                    [jnp.broadcast_to(qk_kq[g_idx : g_idx + 1], (expand, cs, cs)) for g_idx in range(bkq)], axis=0
                )
            else:
                q_exp, k_exp, kk_exp, qk_exp = q_all, k_all, kk_kq, qk_kq

            st_all = state_scr[...]

            def one_v_head(st, q_h, k_h, v_h, kk_h, qk_h, gc_col, beta_col):
                gc_row = jnp.sum(jnp.where(diag, gc_col, 0.0), axis=0, keepdims=True)
                g_last_scalar = jnp.sum(jnp.where(last_lane_mask, gc_col, 0.0))

                g_diff = jnp.where(diag, 0.0, gc_col - gc_row)
                decay_mask = jnp.where(incl, jnp.exp(jnp.where(incl, g_diff, -1e30)), 0.0)
                pair = jnp.where(strict, -(beta_col * kk_h) * decay_mask, 0.0)
                A = unit_lower_inverse_masked(pair, cs, precision=inv_precision)

                exp_gc = jnp.exp(gc_col)
                u = _dot(A, v_h * beta_col, inv_precision)
                w = _dot(A, k_h * (beta_col * exp_gc), inv_precision)

                q_g = q_h * exp_gc
                if cs == 64:
                    wq = jnp.concatenate([w, q_g], axis=0)
                    vp_ai = _dot(wq, st, precision)
                    v_prime = vp_ai[:cs]
                    attn_inter = vp_ai[cs:]
                else:
                    v_prime = _dot(w, st, precision)
                    attn_inter = _dot(q_g, st, precision)

                v_new = u - v_prime
                attn_qk = jnp.where(incl, qk_h * decay_mask, 0.0)
                core_out = attn_inter + _dot(attn_qk, v_new, precision)

                k_scaled = k_h * jnp.exp(jnp.where(gc_col == g_last_scalar, 0.0, g_last_scalar - gc_col))
                new_st = st * jnp.exp(g_last_scalar) + _dot(k_scaled.T, v_new, precision)
                return new_st, core_out

            new_st, core_out = jax.vmap(one_v_head)(st_all, q_exp, k_exp, v_all, kk_exp, qk_exp, gc_cols, b_cols)
            state_scr[...] = new_st

            @pl.when(c >= 2)
            def _wait_prev_o():
                pltpu.make_async_copy(o_buf.at[b], out_hbm.at[c - 2, :, o_col], wr_sems.at[b]).wait()

            o_buf[b] = jnp.concatenate([core_out[h] for h in range(bv)], axis=-1).astype(o_buf.dtype)
            pltpu.make_async_copy(o_buf.at[b], out_hbm.at[c, :, o_col], wr_sems.at[b]).start()

            is_last = last_ref[c] != 0

            @pl.when(is_last)
            def _store_st():
                @pl.when(st_wr_pending >= 0)
                def _wait_old_st():
                    pltpu.make_async_copy(st_out_buf, new_pool_hbm.at[st_wr_pending, st_hb], st_sems.at[1]).wait()

                st_out_buf[...] = state_scr[...].astype(st_out_buf.dtype)
                pltpu.make_async_copy(st_out_buf, new_pool_hbm.at[slot_ref[c], st_hb], st_sems.at[1]).start()

            return jnp.where(is_last, slot_ref[c], st_wr_pending)

        last_st_slot = lax.fori_loop(0, n_live, body, jnp.int32(-1))

        @pl.when(n_live >= 2)
        def _drain_o2():
            c_m2 = n_live - 2
            pltpu.make_async_copy(o_buf.at[c_m2 & 1], out_hbm.at[c_m2, :, o_col], wr_sems.at[c_m2 & 1]).wait()

        c_m1 = n_live - 1
        pltpu.make_async_copy(o_buf.at[c_m1 & 1], out_hbm.at[c_m1, :, o_col], wr_sems.at[c_m1 & 1]).wait()

        @pl.when(last_st_slot >= 0)
        def _drain_st():
            pltpu.make_async_copy(st_out_buf, new_pool_hbm.at[last_st_slot, st_hb], st_sems.at[1]).wait()


def pallas_gdn_chunk_prefill(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array | None = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = True,
    mask_initial_state: bool = False,
    precision: jax.lax.PrecisionLike | None = None,
    inv_precision: jax.lax.PrecisionLike | None = None,
    qk_precision: jax.lax.PrecisionLike | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run double-buffered direct-DMA Pallas TPU chunked GDN over a ragged batch.

    Handles both multi-token prefill rows (chunked WY recurrence with VMEM-pinned
    state) and single-token decode rows (direct-DMA in-place decode) without
    padding decode rows to ``chunk_size`` and without touching inactive or
    out-of-range pool slots.

    Precondition:
        Live rows within a single call must map to distinct ``state_indices``
        slots (parallel cores and double-buffered DMA do not serialize writes
        to duplicate live slots).

    Returns:
        ``(updated_recurrent_state, output)`` where ``output`` has shape
        ``(num_tokens, n_v * d_v)`` and dtype ``mixed_qkv.dtype``.
    """
    T = int(mixed_qkv.shape[0])
    R = int(query_start_loc.shape[0]) - 1
    S = int(recurrent_state.shape[0])
    E = int(n_v) // int(n_kq)
    cs = int(chunk_size)
    i32, f32 = jnp.int32, jnp.float32
    in_dtype = mixed_qkv.dtype
    is_f32 = jnp.dtype(in_dtype) == jnp.float32
    if precision is None:
        precision = lax.Precision.HIGHEST if is_f32 else lax.Precision.DEFAULT
    if inv_precision is None:
        inv_precision = lax.Precision.HIGHEST if is_f32 else lax.Precision.HIGH
    if qk_precision is None:
        qk_precision = lax.Precision.HIGHEST if is_f32 else lax.Precision.HIGH

    pad_base = max(cs - T, 0)
    padded_T = T + pad_base
    pad_dma = pad_base + 16
    mixed_pad = jnp.pad(mixed_qkv, ((0, pad_dma), (0, 0)))

    b_2d = b.reshape(T, n_v)
    a_2d = a.reshape(T, n_v)
    beta = jax.nn.sigmoid(b_2d.astype(f32))
    g = -jnp.exp(A_log.astype(f32)) * jax.nn.softplus(a_2d.astype(f32) + dt_bias.astype(f32))
    gb_flat = jnp.concatenate(
        [
            jnp.pad(g, ((0, pad_dma), (0, 128 - n_v))),
            jnp.pad(beta, ((0, pad_dma), (0, 128 - n_v))),
        ],
        axis=-1,
    )

    qsl_raw = query_start_loc.astype(i32)
    num_valid = jnp.clip(distribution[2].astype(i32), 0, R)
    last_valid_loc = qsl_raw[num_valid]
    eff_qsl = jnp.where(jnp.arange(R + 1, dtype=i32) <= num_valid, qsl_raw, last_valid_loc)
    r_idx = jnp.arange(R, dtype=i32)
    valid_row = r_idx < num_valid
    starts = jnp.where(valid_row, eff_qsl[:-1], 0)
    ends = jnp.where(valid_row, eff_qsl[1:], 0)
    lens = jnp.where(valid_row, jnp.maximum(ends - starts, 0), 0)
    sidx = state_indices.astype(i32).reshape(-1)[:R]
    slot_ok = (sidx >= 0) & (sidx < S)
    if has_initial_state is None or not mask_initial_state:
        init_flags = jnp.ones((R,), dtype=jnp.bool_)
    else:
        init_flags = has_initial_state.astype(jnp.bool_).reshape(-1)[:R]

    short_mask = valid_row & (lens == 1) & slot_ok
    chunk_lens = jnp.where(valid_row & (lens >= 2) & slot_ok, lens, 0)

    # Compact live 1-token rows for the direct-DMA short-row decode kernel.
    R_pad = ((R + 15) // 16) * 16
    if R_pad > 16:
        R_pad = ((R_pad + 31) // 32) * 32
    cores_s = 2 if (_num_tensor_cores() >= 2 and R_pad >= 32 and R_pad % 32 == 0) else 1
    per_s = R_pad // cores_s

    s_pos = jnp.cumsum(short_mask.astype(i32)) - 1
    n_short = jnp.sum(short_mask.astype(i32))
    s_dest = jnp.where(short_mask, s_pos, R_pad)
    s_tok = jnp.zeros((R_pad,), i32).at[s_dest].set(jnp.clip(starts, 0, T - 1), mode="drop")
    s_slot = jnp.zeros((R_pad,), i32).at[s_dest].set(jnp.clip(sidx, 0, S - 1), mode="drop")
    s_init = jnp.zeros((R_pad,), i32).at[s_dest].set(init_flags.astype(i32), mode="drop")

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    out_short_compact, pool = pl.pallas_call(
        functools.partial(
            _short_rows_direct_kernel,
            per_core=per_s,
            n_kq=n_kq,
            expand=E,
            d_k=d_k,
            d_v=d_v,
            precision=lax.Precision.HIGHEST,
            use_qk_l2norm=bool(use_qk_norm_in_gdn),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(cores_s,),
            in_specs=[hbm_spec, hbm_spec, hbm_spec],
            out_specs=[pl.BlockSpec((per_s, n_v, d_v), lambda p, *_: (p, 0, 0)), hbm_spec],
            scratch_shapes=[
                pltpu.VMEM((16, int(mixed_pad.shape[1])), in_dtype),
                pltpu.VMEM((16, 256), f32),
                pltpu.VMEM((2, n_v, d_k, d_v), recurrent_state.dtype),
                pltpu.SemaphoreType.DMA((2, 3)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((R_pad, n_v, d_v), in_dtype),
            jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
        ],
        input_output_aliases={4 + 2: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=int(_vmem_capacity_bytes() * 0.75),
        ),
        name=f"ejkernel_pallas_gdn_short_r{R_pad}",
    )(n_short.reshape(1), s_tok, s_slot, s_init, mixed_pad, gb_flat, recurrent_state)

    # Chunk tables for rows with lens >= 2.
    num_chunks = max_live_chunks(T, R, cs)
    n_chunks = jnp.where(chunk_lens > 0, (chunk_lens + cs - 1) // cs, 0).astype(i32)
    chunk_end = jnp.cumsum(n_chunks).astype(i32)
    chunk_begin = chunk_end - n_chunks
    n_live = chunk_end[-1]
    c = jnp.arange(num_chunks, dtype=i32)
    live = c < n_live
    cmap = jnp.minimum(c, jnp.maximum(n_live - 1, 0))
    row = jnp.minimum(jnp.sum(chunk_end[None, :] <= cmap[:, None], axis=1), R - 1).astype(i32)
    local = cmap - chunk_begin[row]
    start = starts[row] + local * cs
    win_start = jnp.clip(start, 0, padded_T - cs)
    astart = (win_start // 16) * 16
    rem = win_start - astart
    lo = jnp.where(live, start - win_start, 0)
    hi = jnp.where(live, jnp.minimum(start + cs, ends[row]) - win_start, 0)
    slot = jnp.clip(sidx[row], 0, S - 1).astype(i32)
    first = (live & (local == 0)).astype(i32)
    last = (live & (local == n_chunks[row] - 1)).astype(i32)
    init_c = init_flags[row].astype(i32)

    vmem_cap = _vmem_capacity_bytes()
    bkq2_bytes = _chunk_vmem_estimate_bytes(
        bkq=2,
        expand=E,
        cs=cs,
        d_k=d_k,
        d_v=d_v,
        in_itemsize=jnp.dtype(in_dtype).itemsize,
        st_itemsize=jnp.dtype(recurrent_state.dtype).itemsize,
    )
    bkq = 2 if (n_kq >= 4 and n_kq % 2 == 0 and bkq2_bytes < int(vmem_cap * 0.45)) else 1
    bv = bkq * E

    chunk_out, new_pool = pl.pallas_call(
        functools.partial(
            _gdn_chunk_v5_kernel,
            bkq=bkq,
            n_kq=n_kq,
            expand=E,
            d_k=d_k,
            d_v=d_v,
            chunk_size=cs,
            num_chunks=num_chunks,
            precision=precision,
            inv_precision=inv_precision,
            qk_precision=qk_precision,
            use_qk_l2norm=bool(use_qk_norm_in_gdn),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=9,
            grid=(n_kq // bkq,),
            in_specs=[hbm_spec, hbm_spec, hbm_spec],
            out_specs=[hbm_spec, hbm_spec],
            scratch_shapes=[
                pltpu.VMEM((bv, d_k, d_v), f32),
                pltpu.VMEM((2, cs + 16, bkq * d_k), in_dtype),
                pltpu.VMEM((2, cs + 16, bkq * d_k), in_dtype),
                pltpu.VMEM((2, cs + 16, bv * d_v), in_dtype),
                pltpu.VMEM((2, cs + 16, 256), f32),
                pltpu.VMEM((2, cs, bv * d_v), in_dtype),
                pltpu.VMEM((bv, d_k, d_v), pool.dtype),
                pltpu.VMEM((bv, d_k, d_v), pool.dtype),
                pltpu.SemaphoreType.DMA((2, 4)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_chunks, cs, n_v * d_v), in_dtype),
            jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        ],
        input_output_aliases={9 + 2: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=int(vmem_cap * 0.75),
        ),
        name=f"ejkernel_pallas_gdn_chunk_cs{cs}_bkq{bkq}",
    )(n_live.reshape(1), slot, first, last, init_c, astart, rem, lo, hi, mixed_pad, gb_flat, pool)

    # Assemble flat output stream: 1 contiguous gather + tiny scatter of live short rows.
    t = jnp.arange(T, dtype=i32)
    t_row = jnp.searchsorted(eff_qsl[1:], t, side="right", method="compare_all").astype(i32)
    t_row_c = jnp.minimum(t_row, R - 1)
    owned = (t_row < num_valid) & (t >= starts[t_row_c]) & (t < ends[t_row_c])
    in_chunk = owned & (chunk_lens[t_row_c] > 0)
    ch = jnp.clip(chunk_begin[t_row_c] + (t - starts[t_row_c]) // cs, 0, num_chunks - 1)
    ln = jnp.clip(t - win_start[ch], 0, cs - 1)
    flat_idx = ch * cs + ln
    out_flat = jnp.where(
        in_chunk[:, None],
        chunk_out.reshape(num_chunks * cs, n_v * d_v)[flat_idx],
        jnp.zeros((), in_dtype),
    )
    s_write = jnp.where(r_idx < n_short, s_tok[:R], T)
    out_flat = out_flat.at[s_write].set(out_short_compact[:R].reshape(R, n_v * d_v), mode="drop")
    return new_pool, out_flat


# ---------------------------------------------------------------------------
# 3. Fused Conv1D + SiLU + L2Norm + GDN Single-Token Decode Kernel (In-Place)
# ---------------------------------------------------------------------------
def _fused_conv_gdn_decode_kernel(
    code_ref,  # SMEM [T_pad] int32: -1 = idle, else 2*slot + init
    x_ref,  # VMEM [per_core, conv_dim]
    b_ref,  # VMEM [per_core, 128] f32 (sigmoided beta)
    eg_ref,  # VMEM [per_core, 128] f32 (exp(g))
    w_ref,  # VMEM [d_conv, conv_dim] f32
    conv_hbm,  # HBM [S, d_conv, conv_dim]
    rec_hbm,  # HBM [S, n_v, K, V]
    out_ref,  # VMEM [per_core, n_v, V]
    conv_out,  # HBM [S, d_conv, conv_dim] (alias)
    rec_out,  # HBM [S, n_v, K, V] (alias)
    sx,  # VMEM [per_core, conv_dim] f32
    sb,  # VMEM [per_core, 128] f32
    seg,  # VMEM [per_core, 128] f32
    wb,  # VMEM [ring, d_conv, conv_dim]
    stb,  # VMEM [ring, n_v, K, V]
    hist_scr,  # VMEM [d_conv, conv_dim] f32
    q_scr,  # VMEM [n_kq, K] f32
    k_scr,  # VMEM [n_kq, K] f32
    v_scr,  # VMEM [n_v, V] f32
    q_exp_scr,  # VMEM [n_v, K] f32
    k_exp_scr,  # VMEM [n_v, K] f32
    rd_sems,  # DMA (2, ring)
    wr_sems,  # DMA (2, ring)
    *,
    per_core: int,
    ring: int,
    n_kq: int,
    expand: int,
    d_k: int,
    d_v: int,
    d_conv: int,
    precision: jax.lax.PrecisionLike,
    use_qk_l2norm: bool,
    apply_silu: bool,
    conv_dtype: tp.Any,
    out_dtype: tp.Any,
):
    f32 = jnp.float32
    n_v = n_kq * expand
    key_dim = n_kq * d_k
    scale = 1.0 / (d_k**0.5)
    base = pl.program_id(0) * per_core

    sx[...] = x_ref[...].astype(f32)
    sb[...] = b_ref[...].astype(f32)
    seg[...] = eg_ref[...].astype(f32)
    wts = w_ref[...].astype(f32)
    w_taps = [wts[t : t + 1, :] for t in range(d_conv)]

    row_nv = lax.broadcasted_iota(jnp.int32, (n_v, 128), 0)
    col_nv = lax.broadcasted_iota(jnp.int32, (n_v, 128), 1)
    eye_nv_mask = row_nv == col_nv

    def decode_code(code):
        live = code >= 0
        keep = jnp.logical_and(live, jnp.bitwise_and(code, 1) == 1)
        return live, keep, jnp.right_shift(jnp.maximum(code, 0), 1)

    def start_read(j_tok, r_slot):
        _, keep, slot = decode_code(code_ref[base + j_tok])

        @pl.when(keep)
        def _sr():
            pltpu.make_async_copy(conv_hbm.at[slot], wb.at[r_slot], rd_sems.at[0, r_slot]).start()
            pltpu.make_async_copy(rec_hbm.at[slot], stb.at[r_slot], rd_sems.at[1, r_slot]).start()

    def wait_read(j_tok, r_slot):
        _, keep, slot = decode_code(code_ref[base + j_tok])

        @pl.when(keep)
        def _wr():
            pltpu.make_async_copy(conv_hbm.at[slot], wb.at[r_slot], rd_sems.at[0, r_slot]).wait()
            pltpu.make_async_copy(rec_hbm.at[slot], stb.at[r_slot], rd_sems.at[1, r_slot]).wait()

    def start_write(j_tok, r_slot):
        live, _, slot = decode_code(code_ref[base + j_tok])

        @pl.when(live)
        def _sw():
            pltpu.make_async_copy(wb.at[r_slot], conv_out.at[slot], wr_sems.at[0, r_slot]).start()
            pltpu.make_async_copy(stb.at[r_slot], rec_out.at[slot], wr_sems.at[1, r_slot]).start()

    def wait_write(j_tok, r_slot):
        live, _, slot = decode_code(code_ref[base + j_tok])

        @pl.when(live)
        def _ww():
            pltpu.make_async_copy(wb.at[r_slot], conv_out.at[slot], wr_sems.at[0, r_slot]).wait()
            pltpu.make_async_copy(stb.at[r_slot], rec_out.at[slot], wr_sems.at[1, r_slot]).wait()

    prefetch = min(ring - 1, per_core)
    for p in range(prefetch):
        start_read(p, p)

    def step(j, carry):
        cur = j % ring
        nxt = j + prefetch
        nxt_slot = nxt % ring
        prev_wr = nxt - ring

        @pl.when(jnp.logical_and(nxt < per_core, prev_wr >= 0))
        def _drain_for_nxt():
            wait_write(prev_wr, nxt_slot)

        @pl.when(nxt < per_core)
        def _prefetch_nxt():
            start_read(nxt, nxt_slot)

        live, keep, _ = decode_code(code_ref[base + j])
        wait_read(j, cur)

        @pl.when(jnp.logical_and(live, jnp.logical_not(keep)))
        def _fresh():
            wb[cur] = jnp.zeros(wb.shape[1:], wb.dtype)
            stb[cur] = jnp.zeros(stb.shape[1:], stb.dtype)

        @pl.when(live)
        def _compute():
            x_cur = sx[pl.ds(j, 1), :]
            old = wb[cur].astype(f32)
            taps = [old[t : t + 1, :] for t in range(1, d_conv)] + [x_cur]
            mixed = taps[0] * w_taps[0]
            for t in range(1, d_conv):
                mixed = mixed + taps[t] * w_taps[t]
            y = jax.nn.silu(mixed) if apply_silu else mixed
            y = y.astype(conv_dtype).astype(f32)
            for t in range(d_conv):
                hist_scr[t : t + 1, :] = taps[t]
            wb[cur] = hist_scr[...].astype(wb.dtype)

            for h in range(n_kq):
                q_scr[h : h + 1, :] = y[:, h * d_k : (h + 1) * d_k]
                k_scr[h : h + 1, :] = y[:, key_dim + h * d_k : key_dim + (h + 1) * d_k]
            for h in range(n_v):
                v_scr[h : h + 1, :] = y[:, 2 * key_dim + h * d_v : 2 * key_dim + (h + 1) * d_v]

            q_h = q_scr[...]
            k_h = k_scr[...]
            if use_qk_l2norm:
                q_h = q_h * lax.rsqrt(jnp.sum(q_h * q_h, axis=-1, keepdims=True) + 1e-6)
                k_h = k_h * lax.rsqrt(jnp.sum(k_h * k_h, axis=-1, keepdims=True) + 1e-6)
            q_h = q_h * scale

            for g_idx in range(n_kq):
                qg = q_h[g_idx : g_idx + 1, :]
                kg = k_h[g_idx : g_idx + 1, :]
                for e in range(expand):
                    q_exp_scr[g_idx * expand + e : g_idx * expand + e + 1, :] = qg
                    k_exp_scr[g_idx * expand + e : g_idx * expand + e + 1, :] = kg

            q_all = q_exp_scr[...]
            k_all = k_exp_scr[...]
            v_all = v_scr[...]

            b_row = sb[pl.ds(j, 1), :]
            eg_row = seg[pl.ds(j, 1), :]
            b_col = jnp.sum(jnp.where(eye_nv_mask, jnp.broadcast_to(b_row, (n_v, 128)), 0.0), axis=1, keepdims=True)
            eg_col = jnp.sum(jnp.where(eye_nv_mask, jnp.broadcast_to(eg_row, (n_v, 128)), 0.0), axis=1, keepdims=True)

            s = stb[cur].astype(f32) * eg_col[:, :, None]
            ks = lax.dot_general(
                k_all.reshape(n_v, 1, d_k),
                s,
                (((2,), (1,)), ((0,), (0,))),
                precision=precision,
                preferred_element_type=f32,
            ).reshape(n_v, d_v)
            delta = (v_all - ks) * b_col
            s = s + k_all[:, :, None] * delta[:, None, :]
            o = lax.dot_general(
                q_all.reshape(n_v, 1, d_k),
                s,
                (((2,), (1,)), ((0,), (0,))),
                precision=precision,
                preferred_element_type=f32,
            ).reshape(n_v, d_v)
            out_ref[j] = o.astype(out_dtype)
            stb[cur] = s.astype(stb.dtype)

        start_write(j, cur)

        @pl.when(jnp.logical_not(live))
        def _idle():
            out_ref[j] = jnp.zeros(out_ref.shape[1:], out_ref.dtype)

        return carry

    lax.fori_loop(0, per_core, step, 0)

    tail_start = max(0, per_core - ring)
    for t_j in range(tail_start, per_core):
        wait_write(t_j, t_j % ring)


def pallas_gdn_conv_decode(
    conv_input: jax.Array,
    b_raw: jax.Array,
    a_raw: jax.Array,
    conv_weight: jax.Array,
    conv_pool: jax.Array,
    recurrent_pool: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    token_slots: jax.Array,
    token_init: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    use_qk_l2norm: bool = True,
    apply_silu: bool = True,
    conv_output_dtype: tp.Any = None,
    out_dtype: tp.Any = jnp.float32,
    precision: jax.lax.PrecisionLike = lax.Precision.HIGHEST,
    ring: int = 3,
    conv_pool_transposed: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fused Pallas TPU causal Conv1D + SiLU + L2Norm + GDN single-token decode.

    Updates ``conv_pool`` and ``recurrent_pool`` in place for every live token
    whose ``token_slots[t]`` is in ``[0, num_slots)``. Idle/padding tokens
    output zeros and leave all pool slots untouched.

    Note on ``conv_pool`` layout:
        Mosaic TPU requires the 128-aligned ``conv_dim`` axis to be minor (lane
        dimension) and ``d_conv`` (typically 4) to be the sublane dimension, so
        the kernel operates on ``[S, d_conv, conv_dim]`` via ``pl.ANY`` HBM
        aliasing (matching :mod:`ragged_kda_conv_decode`). On TPU, XLA stores
        ``[S, conv_dim, d_conv]`` with physical minor-to-major ``{1, 2, 0}`` so
        ``swapaxes(1, 2)`` is a zero-copy metadata bitcast; callers that already
        keep ``conv_pool`` in ``[S, d_conv, conv_dim]`` may pass
        ``conv_pool_transposed=True`` to skip ``swapaxes`` entirely.

    Args:
        conv_input: Pre-conv flat tokens ``[T, 2*n_kq*d_k + n_v*d_v]``.
        b_raw: Pre-sigmoid beta gate ``[T, n_v]``.
        a_raw: Pre-softplus alpha gate ``[T, n_v]``.
        conv_weight: Depthwise conv kernel ``[conv_dim, d_conv]``.
        conv_pool: Rolling conv-state pool ``[S, conv_dim, d_conv]`` (or
            ``[S, d_conv, conv_dim]`` when ``conv_pool_transposed=True``).
        recurrent_pool: GDN state pool ``[S, n_v, d_k, d_v]``.
        A_log: Per-value-head log-decay ``[n_v]``.
        dt_bias: Per-value-head dt bias ``[n_v]``.
        token_slots: Pool slot per token ``[T]`` (``< 0`` or ``>= S`` = idle).
            Live tokens within a single call must map to distinct pool slots.
        token_init: Bool per token ``[T]`` (``False`` = fresh zero window/state).
        conv_pool_transposed: When ``True``, ``conv_pool`` is already in
            ``[S, d_conv, conv_dim]`` layout and returned in the same layout.

    Returns:
        ``(out [T, n_v, d_v], updated_conv_pool, updated_recurrent_pool)``.
    """
    T, conv_dim = int(conv_input.shape[0]), int(conv_input.shape[1])
    S = int(recurrent_pool.shape[0])
    d_conv = int(conv_weight.shape[-1])
    E = int(n_v) // int(n_kq)
    i32, f32 = jnp.int32, jnp.float32
    c_dtype = jnp.dtype(conv_output_dtype or conv_input.dtype)
    o_dtype = jnp.dtype(out_dtype)

    # Pad T up to a Mosaic-aligned count so dual-core splits cleanly on (16, 128) tiles.
    T_pad = ((T + 15) // 16) * 16
    if _num_tensor_cores() >= 2 and T_pad >= 32:
        T_pad = ((T_pad + 31) // 32) * 32
    pad_t = T_pad - T

    x_in = jnp.pad(conv_input, ((0, pad_t), (0, 0))) if pad_t else conv_input
    b_2d = b_raw.reshape(T, n_v)
    a_2d = a_raw.reshape(T, n_v)
    beta = jax.nn.sigmoid(b_2d.astype(f32))
    g = -jnp.exp(A_log.astype(f32)) * jax.nn.softplus(a_2d.astype(f32) + dt_bias.astype(f32))
    exp_g = jnp.exp(g)
    b_pad = jnp.pad(beta, ((0, pad_t), (0, 128 - n_v)))
    eg_pad = jnp.pad(exp_g, ((0, pad_t), (0, 128 - n_v)))

    slots = token_slots.astype(i32).reshape(-1)[:T]
    init_i = token_init.astype(jnp.bool_).astype(i32).reshape(-1)[:T]
    live = (slots >= 0) & (slots < S)
    code = jnp.where(live, slots * 2 + init_i, -1)
    if pad_t:
        code = jnp.pad(code, ((0, pad_t),), constant_values=-1)

    cores = 2 if (_num_tensor_cores() >= 2 and T_pad >= 32 and T_pad % 32 == 0) else 1
    per_core = T_pad // cores
    ring_eff = min(max(1, int(ring)), max(2, per_core))

    # [S, d_conv, conv_dim] matches the pools' TPU physical layout ({1, 2, 0})
    # so swapaxes lowers to a zero-copy layout bitcast when aliased via pl.ANY.
    conv_view = conv_pool if conv_pool_transposed else jnp.swapaxes(conv_pool, 1, 2)
    w_t = jnp.swapaxes(conv_weight, 0, 1).astype(f32)  # [d_conv, conv_dim]

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    kernel = functools.partial(
        _fused_conv_gdn_decode_kernel,
        per_core=per_core,
        ring=ring_eff,
        n_kq=n_kq,
        expand=E,
        d_k=d_k,
        d_v=d_v,
        d_conv=d_conv,
        precision=precision,
        use_qk_l2norm=bool(use_qk_l2norm),
        apply_silu=bool(apply_silu),
        conv_dtype=c_dtype,
        out_dtype=o_dtype,
    )
    out, new_conv_view, new_rec = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(cores,),
            in_specs=[
                pl.BlockSpec((per_core, conv_dim), lambda p, *_: (p, 0)),
                pl.BlockSpec((per_core, 128), lambda p, *_: (p, 0)),
                pl.BlockSpec((per_core, 128), lambda p, *_: (p, 0)),
                pl.BlockSpec((d_conv, conv_dim), lambda p, *_: (0, 0)),
                any_spec,
                any_spec,
            ],
            out_specs=[
                pl.BlockSpec((per_core, n_v, d_v), lambda p, *_: (p, 0, 0)),
                any_spec,
                any_spec,
            ],
            scratch_shapes=[
                pltpu.VMEM((per_core, conv_dim), f32),
                pltpu.VMEM((per_core, 128), f32),
                pltpu.VMEM((per_core, 128), f32),
                pltpu.VMEM((ring_eff, d_conv, conv_dim), conv_pool.dtype),
                pltpu.VMEM((ring_eff, n_v, d_k, d_v), recurrent_pool.dtype),
                pltpu.VMEM((d_conv, conv_dim), f32),
                pltpu.VMEM((n_kq, d_k), f32),
                pltpu.VMEM((n_kq, d_k), f32),
                pltpu.VMEM((n_v, d_v), f32),
                pltpu.VMEM((n_v, d_k), f32),
                pltpu.VMEM((n_v, d_k), f32),
                pltpu.SemaphoreType.DMA((2, ring_eff)),
                pltpu.SemaphoreType.DMA((2, ring_eff)),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((T_pad, n_v, d_v), o_dtype),
            jax.ShapeDtypeStruct(conv_view.shape, conv_view.dtype),
            jax.ShapeDtypeStruct(recurrent_pool.shape, recurrent_pool.dtype),
        ],
        input_output_aliases={1 + 4: 1, 1 + 5: 2},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=int(_vmem_capacity_bytes() * 0.75),
        ),
        name=f"ejkernel_pallas_gdn_conv_decode_c{cores}_t{T_pad}_r{ring_eff}",
    )(code, x_in, b_pad, eg_pad, w_t, conv_view, recurrent_pool)

    updated_conv_pool = new_conv_view if conv_pool_transposed else jnp.swapaxes(new_conv_view, 1, 2)
    return out[:T], updated_conv_pool, new_rec


# ---------------------------------------------------------------------------
# 4. In-Place Direct-DMA Decode-Only Wrapper (Post-Conv QKV)
# ---------------------------------------------------------------------------
def pallas_gdn_decode_inplace(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array | None = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    use_qk_norm_in_gdn: bool = True,
    mask_initial_state: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Dual-core direct-DMA in-place decode over post-conv ``mixed_qkv``.

    Handles arbitrary pool size ``S >= 1``, empty rows, out-of-range slots, and
    ``has_initial_state`` without gathering or scattering the recurrent pool.
    Live rows within a single call must map to distinct ``state_indices`` slots.
    """
    T = int(mixed_qkv.shape[0])
    R = int(query_start_loc.shape[0]) - 1
    S = int(recurrent_state.shape[0])
    E = int(n_v) // int(n_kq)
    i32, f32 = jnp.int32, jnp.float32
    in_dtype = mixed_qkv.dtype

    pad_dma = 16
    mixed_pad = jnp.pad(mixed_qkv, ((0, pad_dma), (0, 0)))

    b_2d = b.reshape(T, n_v)
    a_2d = a.reshape(T, n_v)
    beta = jax.nn.sigmoid(b_2d.astype(f32))
    g = -jnp.exp(A_log.astype(f32)) * jax.nn.softplus(a_2d.astype(f32) + dt_bias.astype(f32))
    gb_flat = jnp.concatenate(
        [
            jnp.pad(g, ((0, pad_dma), (0, 128 - n_v))),
            jnp.pad(beta, ((0, pad_dma), (0, 128 - n_v))),
        ],
        axis=-1,
    )

    qsl_raw = query_start_loc.astype(i32)
    num_valid = jnp.clip(distribution[2].astype(i32), 0, R)
    last_valid_loc = qsl_raw[num_valid]
    eff_qsl = jnp.where(jnp.arange(R + 1, dtype=i32) <= num_valid, qsl_raw, last_valid_loc)
    r_idx = jnp.arange(R, dtype=i32)
    valid_row = r_idx < num_valid
    starts = jnp.where(valid_row, eff_qsl[:-1], 0)
    ends = jnp.where(valid_row, eff_qsl[1:], 0)
    lens = jnp.where(valid_row, jnp.maximum(ends - starts, 0), 0)
    sidx = state_indices.astype(i32).reshape(-1)[:R]
    live = valid_row & (lens > 0) & (sidx >= 0) & (sidx < S)
    if has_initial_state is None or not mask_initial_state:
        init_flags = jnp.ones((R,), dtype=jnp.bool_)
    else:
        init_flags = has_initial_state.astype(jnp.bool_).reshape(-1)[:R]

    R_pad = ((R + 15) // 16) * 16
    if R_pad > 16:
        R_pad = ((R_pad + 31) // 32) * 32
    cores_s = 2 if (_num_tensor_cores() >= 2 and R_pad >= 32 and R_pad % 32 == 0) else 1
    per_s = R_pad // cores_s

    s_pos = jnp.cumsum(live.astype(i32)) - 1
    n_short = jnp.sum(live.astype(i32))
    s_dest = jnp.where(live, s_pos, R_pad)
    s_tok = jnp.zeros((R_pad,), i32).at[s_dest].set(jnp.clip(starts, 0, T - 1), mode="drop")
    s_slot = jnp.zeros((R_pad,), i32).at[s_dest].set(jnp.clip(sidx, 0, S - 1), mode="drop")
    s_init = jnp.zeros((R_pad,), i32).at[s_dest].set(init_flags.astype(i32), mode="drop")

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    out_compact, new_pool = pl.pallas_call(
        functools.partial(
            _short_rows_direct_kernel,
            per_core=per_s,
            n_kq=n_kq,
            expand=E,
            d_k=d_k,
            d_v=d_v,
            precision=lax.Precision.HIGHEST,
            use_qk_l2norm=bool(use_qk_norm_in_gdn),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(cores_s,),
            in_specs=[hbm_spec, hbm_spec, hbm_spec],
            out_specs=[pl.BlockSpec((per_s, n_v, d_v), lambda p, *_: (p, 0, 0)), hbm_spec],
            scratch_shapes=[
                pltpu.VMEM((16, int(mixed_pad.shape[1])), in_dtype),
                pltpu.VMEM((16, 256), f32),
                pltpu.VMEM((2, n_v, d_k, d_v), recurrent_state.dtype),
                pltpu.SemaphoreType.DMA((2, 3)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((R_pad, n_v, d_v), in_dtype),
            jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
        ],
        input_output_aliases={4 + 2: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=int(_vmem_capacity_bytes() * 0.75),
        ),
        name=f"ejkernel_pallas_gdn_decode_inplace_r{R_pad}",
    )(n_short.reshape(1), s_tok, s_slot, s_init, mixed_pad, gb_flat, recurrent_state)

    out_flat = jnp.zeros((T, n_v * d_v), in_dtype)
    s_write = jnp.where(r_idx < n_short, s_tok[:R], T)
    out_flat = out_flat.at[s_write].set(out_compact[:R].reshape(R, n_v * d_v), mode="drop")
    return new_pool, out_flat


__all__ = (
    "pallas_gdn_chunk_prefill",
    "pallas_gdn_chunk_prefill_supported",
    "pallas_gdn_conv_decode",
    "pallas_gdn_conv_decode_supported",
    "pallas_gdn_decode_inplace",
)
