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

"""Two-pass Pallas TPU backward of the pool-bitmap block-sparse DSA attention over the forward's active tables.

Both passes recompute the transposed probabilities from the forward's saved statistics
``P^T = where(open, exp(scale * K q^T - m - log l), 0)`` and ``dS^T = P^T (dP^T - delta)`` with ``dP^T = K dO^T``
-- keys on sublanes, queries on lanes, exactly the forward's tile layout; ``K q^T`` and ``K dO^T`` are NT matmuls.

* **dQ + d_pool pass** (``grid = (B, nQB, nKB)``, q-block major, the forward's static pipeline): per active kv-block
  and head ``dQ_h^T += K^T dS^T`` (fp32 scratch ``[H, D, bq]``), and the bias cotangent ``sum_h dS^T [bk, bq]``
  pool-summed over groups of ``kpool`` keys into the block ``[bk/kpool, bq]`` of ``d_pool^T [B, P, S]`` whose index
  follows the active kv-block. Each active pair writes its block once; the output is aliased to a zero buffer
  (``input_output_aliases``) so never-visited pairs read 0. A q-block with no active kv-block writes zeros to its
  (clamped) resident block.
* **dKV pass** (``grid = (B, nKB, nQB)``, kv-block major over the q-blocks that visited the kv-block, from
  ``build_q_tables``): ``d_latent[kb] = sum over visiting q-blocks and heads of (P^T dO_h + scale * dS^T q_h)``
  (``K == V``: value and key roles add) in a ``[bk, D]`` fp32 scratch, written once -- no scatter, no atomics.

``delta = rowsum(dO * O)`` per (head, query) comes from XLA as ``[B, H, 8, S]`` (row 0 used) like ``m`` / ``log l``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ejkernel.kernels._xla.dsa_sparse_attention._tables import POOL_SUMS, WORD

from ._pallas_impl_fwd import (
    NT_DIMS,
    STAT_ROWS,
    check_tables_interpret,
    check_tile_alignment,
    kv_block_index,
    open_mask,
    q_block_index,
)

DEFAULT_BWD_VMEM_LIMIT = 63 * 2**20


def recompute(k, q_h, do_h, m_row, logl_row, delta_row, open_, softmax_scale: float, prec, lean: bool = False):
    """``(P^T, dS^T)`` ``[bk, bq]`` fp32 for one head.

    Args:
        k: ``[bk, D]`` keys of the block.
        q_h: ``[bq, D]`` queries of the head.
        do_h: ``[bq, D]`` output cotangent of the head.
        m_row: ``[1, bq]`` saved running max.
        logl_row: ``[1, bq]`` saved log-normaliser.
        delta_row: ``[1, bq]`` ``rowsum(dO * O)``.
        open_: ``[bk, bq]`` open mask.
        softmax_scale: Logit scale.
        prec: Matmul precision.
        lean: Fold ``m + log l`` once per head (an fp32 reassociation, not bitwise).

    Returns:
        ``(P^T, dS^T)``.
    """
    st = lax.dot_general(k, q_h, NT_DIMS, preferred_element_type=jnp.float32, precision=prec) * softmax_scale
    if lean:
        p = jnp.where(open_, jnp.exp(st - (m_row + logl_row)), 0.0)
    else:
        p = jnp.where(open_, jnp.exp(st - m_row - logl_row), 0.0)
    dp = lax.dot_general(k, do_h, NT_DIMS, preferred_element_type=jnp.float32, precision=prec)
    return p, p * (dp - delta_row)


def pool_sum(ds_sum, kpool: int, how: str):
    """``[bk, bq] -> [bk/kpool, bq]`` fp32: the bias cotangent summed over each pool's ``kpool`` consecutive keys.

    ``"reshape"``: a sublane-group reduce; ``"matmul"``: a constant 0/1 fp32 matrix on the MXU at HIGHEST precision
    (fp32-exact sums of ``kpool`` terms; a ~1e-7 relative reassociation vs the reshape).
    """
    block_kv, block_q = ds_sum.shape
    if how == "reshape":
        return jnp.sum(ds_sum.reshape(block_kv // kpool, kpool, block_q), axis=1)
    rows = lax.broadcasted_iota(jnp.int32, (block_kv // kpool, block_kv), 0)
    cols = lax.div(lax.broadcasted_iota(jnp.int32, (block_kv // kpool, block_kv), 1), kpool)
    pmat = (rows == cols).astype(jnp.float32)
    return lax.dot(pmat, ds_sum, preferred_element_type=jnp.float32, precision=lax.Precision.HIGHEST)


def _dq_kernel(
    count_ref,  # SMEM [B, nQB]
    kv_list_ref,  # SMEM [B, nQB, nKB]
    q_ref,  # [H, bq, D]
    k_ref,  # [bk, D]
    bits_ref,  # [W_pad, bq]
    tails_ref,  # [8, bq]
    do_ref,  # [H, bq, D]
    m_ref,  # [H, 8, bq]
    logl_ref,  # [H, 8, bq]
    delta_ref,  # [H, 8, bq]
    dpool_init_ref,  # ANY: the zero-initialised d_pool^T aliased to the output; never read
    dq_ref,  # [H, bq, D]
    dpool_ref,  # [bk/kpool, bq] fp32
    dq_scr,  # [H, D, bq] fp32
    *,
    softmax_scale: float,
    kpool: int,
    block_kv: int,
    n_kb: int,
    n_tails: int,
    lean: bool,
    pool_sum_how: str,
):
    del dpool_init_ref
    b, qb, i = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    num_heads, block_q, _ = q_ref.shape
    mm_dtype = q_ref.dtype
    prec = lax.Precision.HIGHEST if mm_dtype == jnp.float32 else lax.Precision.DEFAULT
    count = count_ref[b, qb]

    @pl.when(i == 0)
    def _init():
        dq_scr[...] = jnp.zeros(dq_scr.shape, dq_scr.dtype)

    @pl.when((i == 0) & (count == 0))
    def _no_block():
        dpool_ref[...] = jnp.zeros(dpool_ref.shape, dpool_ref.dtype)

    @pl.when(i < count)
    def _step():
        kb = kv_list_ref[b, qb, i]
        k = k_ref[...]
        kt = k.T
        open_ = open_mask(bits_ref, tails_ref, kb, block_kv, block_q, kpool, n_tails, lean=lean)
        ds_sum = jnp.zeros((block_kv, block_q), jnp.float32)
        for h in range(num_heads):
            _, ds = recompute(
                k,
                q_ref[h],
                do_ref[h],
                m_ref[h, pl.ds(0, 1), :],
                logl_ref[h, pl.ds(0, 1), :],
                delta_ref[h, pl.ds(0, 1), :],
                open_,
                softmax_scale,
                prec,
                lean=lean,
            )
            ds_sum = ds_sum + ds
            dq_scr[h] = dq_scr[h] + lax.dot(kt, ds.astype(mm_dtype), preferred_element_type=jnp.float32, precision=prec)
        dpool_ref[...] = pool_sum(ds_sum, kpool, pool_sum_how)

    @pl.when(i == n_kb - 1)
    def _finish():
        for h in range(num_heads):
            dq_ref[h] = (dq_scr[h] * softmax_scale).T.astype(dq_ref.dtype)


def _dkv_kernel(
    q_count_ref,  # SMEM [B, nKB]
    q_list_ref,  # SMEM [B, nKB, nQB]
    q_ref,  # [H, bq, D]
    k_ref,  # [bk, D]
    bits_ref,  # [W_pad, bq]
    tails_ref,  # [8, bq]
    do_ref,  # [H, bq, D]
    m_ref,
    logl_ref,
    delta_ref,
    dkv_ref,  # [bk, D]
    dkv_scr,  # [bk, D] fp32
    *rest,  # lean: dkv2_scr [bk, D] fp32 (the dS^T q sum, scaled once at the end)
    softmax_scale: float,
    kpool: int,
    block_kv: int,
    n_qb: int,
    n_tails: int,
    lean: bool,
):
    b, kb, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    num_heads, block_q, _ = q_ref.shape
    mm_dtype = q_ref.dtype
    prec = lax.Precision.HIGHEST if mm_dtype == jnp.float32 else lax.Precision.DEFAULT
    dkv2_scr = rest[0] if lean else None

    @pl.when(j == 0)
    def _init():
        dkv_scr[...] = jnp.zeros(dkv_scr.shape, dkv_scr.dtype)
        if lean:
            dkv2_scr[...] = jnp.zeros(dkv2_scr.shape, dkv2_scr.dtype)

    count = q_count_ref[b, kb]

    @pl.when(j < count)
    def _step():
        k = k_ref[...]
        open_ = open_mask(bits_ref, tails_ref, kb, block_kv, block_q, kpool, n_tails, lean=lean)
        acc = dkv_scr[...]
        acc2 = dkv2_scr[...] if lean else None
        for h in range(num_heads):
            q_h = q_ref[h]
            do_h = do_ref[h]
            p, ds = recompute(
                k,
                q_h,
                do_h,
                m_ref[h, pl.ds(0, 1), :],
                logl_ref[h, pl.ds(0, 1), :],
                delta_ref[h, pl.ds(0, 1), :],
                open_,
                softmax_scale,
                prec,
                lean=lean,
            )
            acc = acc + lax.dot(p.astype(mm_dtype), do_h, preferred_element_type=jnp.float32, precision=prec)
            dsq = lax.dot(ds.astype(mm_dtype), q_h, preferred_element_type=jnp.float32, precision=prec)
            if lean:
                acc2 = acc2 + dsq
            else:
                acc = acc + softmax_scale * dsq
        dkv_scr[...] = acc
        if lean:
            dkv2_scr[...] = acc2

    @pl.when(j == n_qb - 1)
    def _finish():
        if lean:
            dkv_ref[...] = (dkv_scr[...] + softmax_scale * dkv2_scr[...]).astype(dkv_ref.dtype)
        else:
            dkv_ref[...] = dkv_scr[...].astype(dkv_ref.dtype)


def bwd_block_layout(num_heads: int, block_q: int, block_kv: int, head_dim: int, w_pad: int, kpool: int) -> dict:
    """Block and scratch shapes of both backward kernels (``None`` = squeezed batch dim)."""
    return {
        "q": (None, num_heads, block_q, head_dim),
        "latent": (None, block_kv, head_dim),
        "bits_t": (None, w_pad, block_q),
        "tails_t": (None, STAT_ROWS, block_q),
        "dO": (None, num_heads, block_q, head_dim),
        "stats": (None, num_heads, STAT_ROWS, block_q),
        "dq": (None, num_heads, block_q, head_dim),
        "d_pool_t": (None, block_kv // kpool, block_q),
        "d_latent": (None, block_kv, head_dim),
        "scratch:dq": (num_heads, head_dim, block_q),
        "scratch:dkv": (block_kv, head_dim),
    }


def dsa_sparse_attention_pallas_bwd(
    query: Array,
    latent: Array,
    bits_t: Array,
    tails_t: Array,
    count: Array,
    kv_list: Array,
    q_count: Array,
    q_list: Array,
    m: Array,
    logl: Array,
    delta: Array,
    grad: Array,
    *,
    kpool: int,
    block_q: int,
    block_kv: int,
    softmax_scale: float,
    n_tails: int = 3,
    interpret: bool | None = None,
    check_alignment: bool | None = None,
    vmem_limit_bytes: int | None = None,
    lean: bool = False,
    pool_sum_how: str = "reshape",
    which: str = "both",
) -> tuple[Array | None, Array | None, Array | None]:
    """Both backward passes (``which="both"``), or one of them for per-pass timing.

    Args:
        query: ``[B, H, S, D]``.
        latent: ``[B, S, D]``.
        bits_t: ``[B, W_pad, S]`` (``build_tables``).
        tails_t: ``[B, 8, S]``.
        count: ``[B, nQB]``.
        kv_list: ``[B, nQB, nKB]``.
        q_count: ``[B, nKB]`` (``build_q_tables``).
        q_list: ``[B, nKB, nQB]``.
        m: ``[B, H, 8, S]`` fp32 forward running max (row 0 used).
        logl: ``[B, H, 8, S]`` fp32 forward log-normaliser.
        delta: ``[B, H, 8, S]`` fp32 ``rowsum(dO * O)``.
        grad: ``[B, H, S, D]`` output cotangent.
        kpool: Keys per pool.
        block_q: Query block.
        block_kv: Key block.
        softmax_scale: Logit scale.
        n_tails: Tail rows in use.
        interpret: Pallas interpret mode (``None``: off-TPU only).
        check_alignment: Check the (8, 128) layout rule.
        vmem_limit_bytes: Scoped VMEM request (default 63 MiB).
        lean: Fold ``m + log l`` once per head; dKV keeps two scratches and scales once (fp32 reassociations).
        pool_sum_how: ``"reshape"`` or ``"matmul"``.
        which: ``"both"``, ``"dq"`` (returns ``d_latent=None``) or ``"dkv"`` (returns ``dq=d_pool_t=None``).

    Returns:
        ``(dq [B, H, S, D], d_latent [B, S, D], d_pool_t [B, S // kpool, S] fp32)``; ``d_pool_t[b, p, s]`` is the
        cotangent of pool ``p``'s bias for query ``s`` (0 for pools of never-visited block pairs).
    """
    batch, num_heads, seq, head_dim = query.shape
    if seq % block_q or seq % block_kv:
        raise ValueError(f"seq {seq} must be a multiple of block_q {block_q} and block_kv {block_kv}")
    if block_kv % (WORD * kpool):
        raise ValueError(f"block_kv {block_kv} must be a multiple of {WORD * kpool}")
    if pool_sum_how not in POOL_SUMS:
        raise ValueError(f"pool_sum={pool_sum_how!r}: expected one of {POOL_SUMS}")
    if which not in ("both", "dq", "dkv"):
        raise ValueError(f"which={which!r}: expected both, dq or dkv")
    n_qb, n_kb = seq // block_q, seq // block_kv
    w_pad = bits_t.shape[1]
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    if check_alignment is None:
        check_alignment = not interpret
    lay = bwd_block_layout(num_heads, block_q, block_kv, head_dim, w_pad, kpool)
    if check_alignment:
        check_tile_alignment(lay)
    if interpret:
        check_tables_interpret(count, kv_list, n_kb, "backward kv_list")
        check_tables_interpret(q_count, q_list, n_qb, "backward q_list")
    lean = bool(lean)
    cparams = pltpu.CompilerParams(
        dimension_semantics=("parallel", "parallel", "arbitrary"),
        vmem_limit_bytes=int(vmem_limit_bytes or DEFAULT_BWD_VMEM_LIMIT),
    )
    f32 = jnp.float32
    dq = d_pool_t = d_latent = None

    def kv_of(b, qb, i, c, kl):
        return kv_block_index(c, kl, b, qb, i, n_kb)

    if which in ("both", "dq"):
        dpool_init = jnp.zeros((batch, seq // kpool, seq), f32)
        dq, d_pool_t = pl.pallas_call(
            functools.partial(
                _dq_kernel,
                softmax_scale=float(softmax_scale),
                kpool=kpool,
                block_kv=block_kv,
                n_kb=n_kb,
                n_tails=n_tails,
                lean=lean,
                pool_sum_how=pool_sum_how,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=(batch, n_qb, n_kb),
                in_specs=[
                    pl.BlockSpec(lay["q"], lambda b, qb, i, c, kl: (b, 0, qb, 0)),
                    pl.BlockSpec(lay["latent"], lambda b, qb, i, c, kl: (b, kv_of(b, qb, i, c, kl), 0)),
                    pl.BlockSpec(lay["bits_t"], lambda b, qb, i, c, kl: (b, 0, qb)),
                    pl.BlockSpec(lay["tails_t"], lambda b, qb, i, c, kl: (b, 0, qb)),
                    pl.BlockSpec(lay["dO"], lambda b, qb, i, c, kl: (b, 0, qb, 0)),
                    pl.BlockSpec(lay["stats"], lambda b, qb, i, c, kl: (b, 0, 0, qb)),
                    pl.BlockSpec(lay["stats"], lambda b, qb, i, c, kl: (b, 0, 0, qb)),
                    pl.BlockSpec(lay["stats"], lambda b, qb, i, c, kl: (b, 0, 0, qb)),
                    pl.BlockSpec(memory_space=pl.ANY),
                ],
                out_specs=[
                    pl.BlockSpec(lay["dq"], lambda b, qb, i, c, kl: (b, 0, qb, 0)),
                    pl.BlockSpec(lay["d_pool_t"], lambda b, qb, i, c, kl: (b, kv_of(b, qb, i, c, kl), qb)),
                ],
                scratch_shapes=[pltpu.VMEM(lay["scratch:dq"], f32)],
            ),
            out_shape=[
                jax.ShapeDtypeStruct((batch, num_heads, seq, head_dim), query.dtype),
                jax.ShapeDtypeStruct((batch, seq // kpool, seq), f32),
            ],
            input_output_aliases={10: 1},  # operand 10 (counting the 2 scalar-prefetch tables) -> output 1
            compiler_params=cparams,
            interpret=interpret,
        )(count, kv_list, query, latent, bits_t, tails_t, grad, m, logl, delta, dpool_init)
    if which == "dq":
        return dq, None, d_pool_t

    def qb_of(b, kb, j, qc, ql):
        return q_block_index(qc, ql, b, kb, j, n_qb)

    dkv_scratch = [pltpu.VMEM(lay["scratch:dkv"], f32)] * (2 if lean else 1)
    d_latent = pl.pallas_call(
        functools.partial(
            _dkv_kernel,
            softmax_scale=float(softmax_scale),
            kpool=kpool,
            block_kv=block_kv,
            n_qb=n_qb,
            n_tails=n_tails,
            lean=lean,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(batch, n_kb, n_qb),
            in_specs=[
                pl.BlockSpec(lay["q"], lambda b, kb, j, qc, ql: (b, 0, qb_of(b, kb, j, qc, ql), 0)),
                pl.BlockSpec(lay["latent"], lambda b, kb, j, qc, ql: (b, kb, 0)),
                pl.BlockSpec(lay["bits_t"], lambda b, kb, j, qc, ql: (b, 0, qb_of(b, kb, j, qc, ql))),
                pl.BlockSpec(lay["tails_t"], lambda b, kb, j, qc, ql: (b, 0, qb_of(b, kb, j, qc, ql))),
                pl.BlockSpec(lay["dO"], lambda b, kb, j, qc, ql: (b, 0, qb_of(b, kb, j, qc, ql), 0)),
                pl.BlockSpec(lay["stats"], lambda b, kb, j, qc, ql: (b, 0, 0, qb_of(b, kb, j, qc, ql))),
                pl.BlockSpec(lay["stats"], lambda b, kb, j, qc, ql: (b, 0, 0, qb_of(b, kb, j, qc, ql))),
                pl.BlockSpec(lay["stats"], lambda b, kb, j, qc, ql: (b, 0, 0, qb_of(b, kb, j, qc, ql))),
            ],
            out_specs=pl.BlockSpec(lay["d_latent"], lambda b, kb, j, qc, ql: (b, kb, 0)),
            scratch_shapes=dkv_scratch,
        ),
        out_shape=jax.ShapeDtypeStruct((batch, seq, head_dim), latent.dtype),
        compiler_params=cparams,
        interpret=interpret,
    )(q_count, q_list, query, latent, bits_t, tails_t, grad, m, logl, delta)
    return dq, d_latent, d_pool_t
