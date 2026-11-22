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

"""Forward Pallas kernels of the compressed-window attention (online softmax, sink-seeded).

Two launch structures share one per-tile update:

* :func:`_fwd_kernel` (default): grid ``(batch, head_block, q_block)``; the KV / bias tiles are streamed by an explicit
  ``pltpu.make_async_copy`` double buffer over a static, unrolled scan of ``n_tok_iter + n_entry_tiles`` positions
  (band-anchored token tiles, then every entry tile). Token tiles of an implicit mask build their bias from
  positions instead of streaming it.
* :func:`_fwd_grid_kernel` (``causal`` tilings): grid ``(batch, head_block, q_block, scan position)``; KV / bias tiles
  arrive through BlockSpecs whose index maps clamp the token position at the q-block's diagonal, so steps past the
  cut re-target the resident tile (no DMA) and ``pl.when`` skips their compute. Measured on a v5p (S=32k, H=16,
  D=512, tiles 128 x 512), a manual-DMA loop with ``pl.when``-guarded steps saved only 9% for half the tiles
  (76.8 vs 84.5 ms) while this grid structure skips like the flash backward does.

Both emit the attention output plus the per-row running max ``m`` and ``log`` of the normaliser, kept separate
because a fully masked row has ``m`` at the ``MASK_MIN`` sentinel whose ulp (~2e31) would swallow ``log(ell)``.
Per-row statistics stay ``[H, block_q, 1]`` (keepdims reductions, sinks pre-shaped ``[H, 1, 1]``): a 2-D
``[H, block_q]`` statistic puts the query axis on lanes, a sublane<->lane relayout TPU v4 does not support.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ._layout import MASK_MIN, Tiling, compiler_params, n_visit, tile_lo, tile_of, token_mask_bias


def matmul_precision(dtype) -> jax.lax.Precision:
    """``HIGHEST`` (multi-pass MXU) for fp32 operands, native single pass for bf16."""
    return jax.lax.Precision.HIGHEST if dtype == jnp.float32 else jax.lax.Precision.DEFAULT


def _seed_stats(sink_ref, use_sink: bool, block_h: int, block_q: int):
    """Initial ``(m, ell)`` of the online softmax: seeded at the sink column, or empty."""
    if use_sink:
        sink = sink_ref[...].astype(jnp.float32)  # [block_h, 1, 1]
        return jnp.broadcast_to(sink, (block_h, block_q, 1)), jnp.ones((block_h, block_q, 1), jnp.float32)
    return jnp.full((block_h, block_q, 1), MASK_MIN, jnp.float32), jnp.zeros((block_h, block_q, 1), jnp.float32)


def _online_update(q, kv_t, bias_t, m, ell, acc, softmax_scale: float):
    """Fold one KV tile (``K == V``) into the online-softmax state ``(m, ell, acc)``."""
    prec = matmul_precision(q.dtype)
    logits = jax.lax.dot_general(q, kv_t, (((2,), (1,)), ((), ())), preferred_element_type=jnp.float32, precision=prec)
    logits = logits * softmax_scale + bias_t[None, :, :]  # [block_h, block_q, block_kv]
    m_new = jnp.maximum(m, jnp.max(logits, axis=-1, keepdims=True))
    corr = jnp.exp(m - m_new)  # finite operands only -> no NaN
    p = jnp.exp(logits - m_new)
    ell = ell * corr + jnp.sum(p, axis=-1, keepdims=True)
    pv = jax.lax.dot_general(
        p.astype(q.dtype), kv_t, (((2,), (0,)), ((), ())), preferred_element_type=jnp.float32, precision=prec
    )
    return m_new, ell, acc * corr + pv


def _write_outputs(o_ref, m_ref, logl_ref, m, ell, acc):
    """Normalise and store the output and the per-row statistics."""
    ell = jnp.where(ell == 0.0, 1.0, ell)
    o_ref[...] = (acc / ell).astype(o_ref.dtype)
    m_ref[...] = m
    logl_ref[...] = jnp.log(ell)


def _fwd_kernel(
    q_ref,
    kv_hbm_ref,
    bias_hbm_ref,
    sink_ref,
    tv_ref,
    o_ref,
    m_ref,
    logl_ref,
    kv_x2,
    bias_x2,
    kv_sem,
    bias_sem,
    *,
    softmax_scale: float,
    use_sink: bool,
    t: Tiling,
):
    """Manual double-buffered scan over the visited KV tiles of one ``(batch, head_block, q_block)`` program.

    Args:
        q_ref: Query block ``[block_h, block_q, d_pad]``.
        kv_hbm_ref: Padded KV in HBM ``[batch, kv_pad, d_pad]``.
        bias_hbm_ref: Padded bias in HBM (all columns, or only the entry columns for an implicit mask).
        sink_ref: Sink logits ``[block_h, 1, 1]`` (fp32).
        tv_ref: Per-key token validity ``[1, token_kv_pad]``; read only for an implicit mask.
        o_ref: Output block ``[block_h, block_q, d_pad]``.
        m_ref: Running-max output ``[block_h, block_q, 1]``.
        logl_ref: ``log`` normaliser output ``[block_h, block_q, 1]``.
        kv_x2: KV double buffer ``[2, block_kv, d_pad]``.
        bias_x2: Bias double buffer ``[2, block_q, block_kv]``.
        kv_sem: KV DMA semaphores ``[2]``.
        bias_sem: Bias DMA semaphores ``[2]``.
        softmax_scale: Logit scale.
        use_sink: Seed the softmax with the per-head sink.
        t: Static tiling.
    """
    b = pl.program_id(0)
    qb = pl.program_id(2)
    q = q_ref[...]
    block_h, block_q, _ = q.shape
    lo = tile_lo(qb, t)
    total = t.n_tok_iter + t.n_entry_tiles

    def _tile(p: int):
        return lo + p if p < t.n_tok_iter else t.n_token_tiles + (p - t.n_tok_iter)

    def _bias_col(p: int):
        if t.implicit:
            return None if p < t.n_tok_iter else p - t.n_tok_iter
        return _tile(p)

    def _copies(p: int, buf: int):
        copies = [
            pltpu.make_async_copy(
                kv_hbm_ref.at[b, pl.ds(_tile(p) * t.block_kv, t.block_kv), :], kv_x2.at[buf], kv_sem.at[buf]
            )
        ]
        col = _bias_col(p)
        if col is not None:
            copies.append(
                pltpu.make_async_copy(
                    bias_hbm_ref.at[b, pl.ds(qb * t.block_q, t.block_q), pl.ds(col * t.block_kv, t.block_kv)],
                    bias_x2.at[buf],
                    bias_sem.at[buf],
                )
            )
        return copies

    m, ell = _seed_stats(sink_ref, use_sink, block_h, block_q)
    acc = jnp.zeros(q.shape, jnp.float32)
    for copy in _copies(0, 0):
        copy.start()
    for p in range(total):
        cur = p % 2
        for copy in _copies(p, cur):
            copy.wait()
        if p + 1 < total:
            for copy in _copies(p + 1, (p + 1) % 2):
                copy.start()
        if t.implicit and p < t.n_tok_iter:
            bias_t = token_mask_bias(tv_ref, qb, _tile(p), t)
        else:
            bias_t = bias_x2[cur].astype(jnp.float32)
        m, ell, acc = _online_update(q, kv_x2[cur], bias_t, m, ell, acc, softmax_scale)
    _write_outputs(o_ref, m_ref, logl_ref, m, ell, acc)


def _fwd_grid_kernel(
    q_ref,
    kv_ref,
    bias_ref,
    sink_ref,
    o_ref,
    m_ref,
    logl_ref,
    m_scr,
    ell_scr,
    acc_scr,
    *,
    softmax_scale: float,
    use_sink: bool,
    t: Tiling,
):
    """One grid step per (batch, head block, q block, scan position) with the causal tile skip (dense bias).

    The online-softmax state lives in VMEM scratch across the scan axis; token steps past this q-block's diagonal
    are inert (their BlockSpecs re-target the last visited tile and ``pl.when`` skips the compute).
    """
    qb = pl.program_id(2)
    step = pl.program_id(3)
    total = t.n_tok_iter + t.n_entry_tiles
    block_h, block_q = m_scr.shape[0], m_scr.shape[1]

    @pl.when(step == 0)
    def _init():
        m0, ell0 = _seed_stats(sink_ref, use_sink, block_h, block_q)
        m_scr[...] = m0
        ell_scr[...] = ell0
        acc_scr[...] = jnp.zeros(acc_scr.shape, jnp.float32)

    is_tok = step < t.n_tok_iter

    @pl.when(jnp.logical_or(step < n_visit(qb, t), jnp.logical_not(is_tok)))
    def _update():
        m, ell, acc = _online_update(
            q_ref[...],
            kv_ref[...],
            bias_ref[...].astype(jnp.float32),
            m_scr[...],
            ell_scr[...],
            acc_scr[...],
            softmax_scale,
        )
        m_scr[...] = m
        ell_scr[...] = ell
        acc_scr[...] = acc

    @pl.when(step == total - 1)
    def _finish():
        _write_outputs(o_ref, m_ref, logl_ref, m_scr[...], ell_scr[...], acc_scr[...])


def flash_forward(
    q_p: Array,
    kv_p: Array,
    bias_p: Array,
    sink_p: Array,
    tv_p: Array,
    *,
    softmax_scale: float,
    use_sink: bool,
    t: Tiling,
) -> tuple[Array, Array, Array]:
    """Launch the forward kernel over padded operands (see :func:`._layout.pad_operands`).

    Args:
        q_p: ``[B, H, q_pad, d_pad]``.
        kv_p: ``[B, kv_pad, d_pad]``.
        bias_p: Padded bias.
        sink_p: ``[H, 1, 1]`` fp32 sinks.
        tv_p: ``[B, 1, tv_len]`` fp32 token validity (placeholder for dense biases).
        softmax_scale: Logit scale.
        use_sink: Include the sink column.
        t: Static tiling.

    Returns:
        ``(out [B, H, q_pad, d_pad], m [B, H, q_pad, 1], logl [B, H, q_pad, 1])``.
    """
    batch, num_heads, q_pad, d_pad = q_p.shape
    bq, bkv = t.block_q, t.block_kv
    interpret = jax.default_backend() != "tpu"
    out_shape = [
        jax.ShapeDtypeStruct((batch, num_heads, q_pad, d_pad), kv_p.dtype),
        jax.ShapeDtypeStruct((batch, num_heads, q_pad, 1), jnp.float32),
        jax.ShapeDtypeStruct((batch, num_heads, q_pad, 1), jnp.float32),
    ]
    if t.causal:
        total = t.n_tok_iter + t.n_entry_tiles

        def row(b, hb, qb, p):
            return (b, hb, qb, 0)

        return pl.pallas_call(
            functools.partial(_fwd_grid_kernel, softmax_scale=softmax_scale, use_sink=use_sink, t=t),
            grid=(batch, 1, t.n_qb, total),
            in_specs=[
                pl.BlockSpec((None, num_heads, bq, d_pad), row),
                pl.BlockSpec((None, bkv, d_pad), lambda b, hb, qb, p: (b, tile_of(qb, p, t), 0)),
                pl.BlockSpec((None, bq, bkv), lambda b, hb, qb, p: (b, qb, tile_of(qb, p, t))),
                pl.BlockSpec((num_heads, 1, 1), lambda b, hb, qb, p: (0, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((None, num_heads, bq, d_pad), row),
                pl.BlockSpec((None, num_heads, bq, 1), row),
                pl.BlockSpec((None, num_heads, bq, 1), row),
            ],
            out_shape=out_shape,
            scratch_shapes=[
                pltpu.VMEM((num_heads, bq, 1), jnp.float32),
                pltpu.VMEM((num_heads, bq, 1), jnp.float32),
                pltpu.VMEM((num_heads, bq, d_pad), jnp.float32),
            ],
            compiler_params=compiler_params(t, ("parallel", "parallel", "parallel", "arbitrary")),
            interpret=interpret,
        )(q_p, kv_p, bias_p, sink_p)

    def row3(b, hb, qb):
        return (b, hb, qb, 0)

    return pl.pallas_call(
        functools.partial(_fwd_kernel, softmax_scale=softmax_scale, use_sink=use_sink, t=t),
        grid=(batch, 1, t.n_qb),
        in_specs=[
            pl.BlockSpec((None, num_heads, bq, d_pad), row3),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec((num_heads, 1, 1), lambda b, hb, qb: (0, 0, 0)),
            pl.BlockSpec((None, 1, tv_p.shape[-1]), lambda b, hb, qb: (b, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((None, num_heads, bq, d_pad), row3),
            pl.BlockSpec((None, num_heads, bq, 1), row3),
            pl.BlockSpec((None, num_heads, bq, 1), row3),
        ],
        out_shape=out_shape,
        scratch_shapes=[
            pltpu.VMEM((2, bkv, d_pad), kv_p.dtype),
            pltpu.VMEM((2, bq, bkv), bias_p.dtype),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=compiler_params(t, ("parallel", "parallel", "parallel")),
        interpret=interpret,
    )(q_p, kv_p, bias_p, sink_p, tv_p)
