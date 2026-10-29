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

"""Fused single-pass Pallas TPU backward of the pool-bitmap block-sparse DSA attention.

ONE pass over the active pairs recomputes ``P^T`` / ``dS^T`` once per (pair, head) and accumulates dQ, d_latent and
the pool cotangent together -- 5 matmuls per (pair, head) instead of the two-pass backward's 3 + 4. Measured on one
v5p chip (256 x 512 tiles, H = 16, D = 512, S = 32k, bf16): 133.7 ms vs 209.1 ms for the two passes at block density
1.0 (-36%), 60.9 vs 78.6 ms at density 0.4 (-22%).

Structure::

    grid = (G, B, nQB / G, nKB)       ("parallel", "arbitrary", "arbitrary", "arbitrary")
    q-block of step (g, b, j, i) = j * G + g   (interleaved halves balance the causal pair counts of two cores)
    per active pair (the forward's static pipeline: latent block index folded from kv_list, clamped; pl.when):
        dkv_vmem <- DMA dkv_partial[g, b, kb-block]          (explicit copy + wait: the read-modify-write is ordered)
        per head h:  P^T, dS^T recomputed from (m + log l, delta)
                     dQ_h^T   += K^T dS^T                                (VMEM scratch, written at the last step)
                     dkv_vmem += [P^T | scale dS^T] @ [dO_h ; q_h]      (ONE matmul, K == V: value and key roles add)
                     ds_sum   += dS^T
        dkv_partial[g, b, kb-block] <- DMA dkv_vmem
        d_pool^T block (kb, qb) = pool-sum of ds_sum

The TensorCores split ONLY the leading ``G`` axis (the one "parallel" dimension), so a core owns its
``dkv_partial[g]`` buffer outright and within a core the DMA waits serialise consecutive pairs (a q-block boundary
can revisit a kv-block). ``d_latent = sum_g dkv_partial[g]`` in XLA. ``G = 2`` when ``nQB`` is even, else 1.
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

from ._pallas_impl_bwd import DEFAULT_BWD_VMEM_LIMIT, bwd_block_layout, pool_sum, recompute
from ._pallas_impl_fwd import check_tables_interpret, check_tile_alignment, kv_block_index, open_mask


def _fused_kernel(
    count_ref,  # SMEM [B, nQB]
    kv_list_ref,  # SMEM [B, nQB, nKB]
    q_ref,  # [H, bq, D]
    k_ref,  # [bk, D]
    bits_ref,  # [W_pad, bq]
    tails_ref,  # [8, bq]
    do_ref,  # [H, bq, D]
    m_ref,  # [H, 8, bq]
    logl_ref,
    delta_ref,
    *rest,
    softmax_scale: float,
    kpool: int,
    block_kv: int,
    n_kb: int,
    n_tails: int,
    n_groups: int,
    pool_sum_how: str,
    with_pool: bool = True,
):
    # with_pool:    dpool_init (ANY, aliased; never read), dkv_init (ANY, aliased; never read),
    #               dq [H, bq, D], dpool [bk/kpool, bq] fp32, dkv_hbm (ANY) [G, B, S, D] fp32,
    #               dq_scr [H, D, bq] fp32, dkv_vmem [bk, D] fp32, sem (2,)
    # without pool: the same minus dpool_init / dpool (no pool cotangent is accumulated or written).
    if with_pool:
        _dpool_init_ref, _dkv_init_ref, dq_ref, dpool_ref, dkv_hbm_ref, dq_scr, dkv_vmem, sem = rest
    else:
        _dkv_init_ref, dq_ref, dkv_hbm_ref, dq_scr, dkv_vmem, sem = rest
        dpool_ref = None
    g, b, j, i = pl.program_id(0), pl.program_id(1), pl.program_id(2), pl.program_id(3)
    qb = j * n_groups + g
    num_heads, block_q, _ = q_ref.shape
    mm_dtype = q_ref.dtype
    prec = lax.Precision.HIGHEST if mm_dtype == jnp.float32 else lax.Precision.DEFAULT
    count = count_ref[b, qb]

    @pl.when(i == 0)
    def _init():
        dq_scr[...] = jnp.zeros(dq_scr.shape, dq_scr.dtype)

    if with_pool:

        @pl.when((i == 0) & (count == 0))
        def _no_block():
            dpool_ref[...] = jnp.zeros(dpool_ref.shape, dpool_ref.dtype)

    @pl.when(i < count)
    def _step():
        kb = kv_list_ref[b, qb, i]
        k = k_ref[...]
        kt = k.T
        open_ = open_mask(bits_ref, tails_ref, kb, block_kv, block_q, kpool, n_tails, lean=True)
        rows = pl.ds(pl.multiple_of(kb * block_kv, block_kv), block_kv)
        fetch = pltpu.make_async_copy(dkv_hbm_ref.at[g, b, rows], dkv_vmem, sem.at[0])
        fetch.start()
        fetch.wait()
        acc = dkv_vmem[...]
        ds_sum = jnp.zeros((block_kv, block_q), jnp.float32) if with_pool else None
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
                lean=True,
            )
            if with_pool:
                ds_sum = ds_sum + ds
            dq_scr[h] = dq_scr[h] + lax.dot(kt, ds.astype(mm_dtype), preferred_element_type=jnp.float32, precision=prec)
            lhs = jnp.concatenate([p.astype(mm_dtype), (ds * softmax_scale).astype(mm_dtype)], axis=1)  # [bk, 2bq]
            rhs = jnp.concatenate([do_h, q_h], axis=0)  # [2bq, D]
            acc = acc + lax.dot(lhs, rhs, preferred_element_type=jnp.float32, precision=prec)
        dkv_vmem[...] = acc
        store = pltpu.make_async_copy(dkv_vmem, dkv_hbm_ref.at[g, b, rows], sem.at[1])
        store.start()
        store.wait()
        if with_pool:
            dpool_ref[...] = pool_sum(ds_sum, kpool, pool_sum_how)

    @pl.when(i == n_kb - 1)
    def _finish():
        for h in range(num_heads):
            dq_ref[h] = (dq_scr[h] * softmax_scale).T.astype(dq_ref.dtype)


def fused_groups_for(n_qb: int, requested: int | None = None) -> int:
    """Core-group count of the fused backward.

    Args:
        n_qb: Number of q-blocks.
        requested: ``None`` (automatic: 2 when ``n_qb`` is even, else 1), ``1`` (single-core form) or ``2``
            (falls back to 1 when ``n_qb`` is odd).

    Returns:
        1 or 2.

    Raises:
        ValueError: On any other ``requested`` value.
    """
    if requested is None:
        return 2 if n_qb % 2 == 0 else 1
    if requested not in (1, 2):
        raise ValueError(f"dsa_sparse_attention: fused_groups must be None, 1 or 2, got {requested}")
    return 1 if requested == 2 and n_qb % 2 else int(requested)


def dsa_sparse_attention_pallas_bwd_fused(
    query: Array,
    latent: Array,
    bits_t: Array,
    tails_t: Array,
    count: Array,
    kv_list: Array,
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
    pool_sum_how: str = "reshape",
    fused_groups: int | None = None,
    with_pool: bool = True,
) -> tuple[Array, Array, Array | None]:
    """The fused backward (see the module docstring); same arguments as the two-pass backward minus the q-tables.

    Args:
        query: ``[B, H, S, D]``.
        latent: ``[B, S, D]``.
        bits_t: ``[B, W_pad, S]``.
        tails_t: ``[B, 8, S]``.
        count: ``[B, nQB]``.
        kv_list: ``[B, nQB, nKB]``.
        m: ``[B, H, 8, S]`` fp32.
        logl: ``[B, H, 8, S]`` fp32.
        delta: ``[B, H, 8, S]`` fp32.
        grad: ``[B, H, S, D]``.
        kpool: Keys per pool.
        block_q: Query block.
        block_kv: Key block.
        softmax_scale: Logit scale.
        n_tails: Tail rows in use.
        interpret: Pallas interpret mode (``None``: off-TPU only).
        check_alignment: Check the (8, 128) layout rule.
        vmem_limit_bytes: Scoped VMEM request (default 63 MiB).
        pool_sum_how: ``"reshape"`` or ``"matmul"``.
        fused_groups: Core groups (see :func:`fused_groups_for`).
        with_pool: Accumulate and return the pool cotangent. ``False`` skips the ``[B, S // kpool, S]`` fp32
            output (and its zero-initialised alias) entirely; ``dq`` and ``d_latent`` are unchanged.

    Returns:
        ``(dq [B, H, S, D], d_latent [B, S, D], d_pool_t [B, S // kpool, S] fp32)``; ``d_pool_t`` is ``None``
        when ``with_pool`` is ``False``.
    """
    batch, num_heads, seq, head_dim = query.shape
    if seq % block_q or seq % block_kv:
        raise ValueError(f"seq {seq} must be a multiple of block_q {block_q} and block_kv {block_kv}")
    if block_kv % (WORD * kpool):
        raise ValueError(f"block_kv {block_kv} must be a multiple of {WORD * kpool}")
    if pool_sum_how not in POOL_SUMS:
        raise ValueError(f"pool_sum={pool_sum_how!r}: expected one of {POOL_SUMS}")
    n_qb, n_kb = seq // block_q, seq // block_kv
    n_groups = fused_groups_for(n_qb, fused_groups)
    w_pad = bits_t.shape[1]
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    if check_alignment is None:
        check_alignment = not interpret
    lay = bwd_block_layout(num_heads, block_q, block_kv, head_dim, w_pad, kpool)
    if check_alignment:
        check_tile_alignment(lay)
    if interpret:
        check_tables_interpret(count, kv_list, n_kb, "fused backward kv_list")
    f32 = jnp.float32

    def qb_of(g, j):
        return j * n_groups + g

    def kv_of(g, b, j, i, c, kl):
        return kv_block_index(c, kl, b, qb_of(g, j), i, n_kb)

    dkv_init = jnp.zeros((n_groups, batch, seq, head_dim), f32)
    in_specs = [
        pl.BlockSpec(lay["q"], lambda g, b, j, i, c, kl: (b, 0, qb_of(g, j), 0)),
        pl.BlockSpec(lay["latent"], lambda g, b, j, i, c, kl: (b, kv_of(g, b, j, i, c, kl), 0)),
        pl.BlockSpec(lay["bits_t"], lambda g, b, j, i, c, kl: (b, 0, qb_of(g, j))),
        pl.BlockSpec(lay["tails_t"], lambda g, b, j, i, c, kl: (b, 0, qb_of(g, j))),
        pl.BlockSpec(lay["dO"], lambda g, b, j, i, c, kl: (b, 0, qb_of(g, j), 0)),
        pl.BlockSpec(lay["stats"], lambda g, b, j, i, c, kl: (b, 0, 0, qb_of(g, j))),
        pl.BlockSpec(lay["stats"], lambda g, b, j, i, c, kl: (b, 0, 0, qb_of(g, j))),
        pl.BlockSpec(lay["stats"], lambda g, b, j, i, c, kl: (b, 0, 0, qb_of(g, j))),
    ]
    dq_spec = pl.BlockSpec(lay["dq"], lambda g, b, j, i, c, kl: (b, 0, qb_of(g, j), 0))
    dq_shape = jax.ShapeDtypeStruct((batch, num_heads, seq, head_dim), query.dtype)
    dkv_shape = jax.ShapeDtypeStruct((n_groups, batch, seq, head_dim), f32)
    operands = [count, kv_list, query, latent, bits_t, tails_t, grad, m, logl, delta]
    if with_pool:
        in_specs += [pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pl.ANY)]
        out_specs = [
            dq_spec,
            pl.BlockSpec(lay["d_pool_t"], lambda g, b, j, i, c, kl: (b, kv_of(g, b, j, i, c, kl), qb_of(g, j))),
            pl.BlockSpec(memory_space=pl.ANY),
        ]
        out_shape = [dq_shape, jax.ShapeDtypeStruct((batch, seq // kpool, seq), f32), dkv_shape]
        aliases = {10: 1, 11: 2}  # operand indices include the 2 scalar-prefetch tables
        operands += [jnp.zeros((batch, seq // kpool, seq), f32), dkv_init]
    else:
        in_specs += [pl.BlockSpec(memory_space=pl.ANY)]
        out_specs = [dq_spec, pl.BlockSpec(memory_space=pl.ANY)]
        out_shape = [dq_shape, dkv_shape]
        aliases = {10: 1}
        operands += [dkv_init]
    outs = pl.pallas_call(
        functools.partial(
            _fused_kernel,
            softmax_scale=float(softmax_scale),
            kpool=kpool,
            block_kv=block_kv,
            n_kb=n_kb,
            n_tails=n_tails,
            n_groups=n_groups,
            pool_sum_how=pool_sum_how,
            with_pool=with_pool,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(n_groups, batch, n_qb // n_groups, n_kb),
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=[
                pltpu.VMEM(lay["scratch:dq"], f32),
                pltpu.VMEM(lay["scratch:dkv"], f32),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        out_shape=out_shape,
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary", "arbitrary"),
            vmem_limit_bytes=int(vmem_limit_bytes or DEFAULT_BWD_VMEM_LIMIT),
        ),
        interpret=interpret,
    )(*operands)
    if with_pool:
        dq, d_pool_t, dkv_partial = outs
    else:
        (dq, dkv_partial), d_pool_t = outs, None
    d_latent = jnp.sum(dkv_partial, axis=0).astype(latent.dtype)
    return dq, d_latent, d_pool_t
