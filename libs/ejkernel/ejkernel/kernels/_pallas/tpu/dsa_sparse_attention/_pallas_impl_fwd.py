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

"""Pallas TPU forward of the pool-bitmap block-sparse DSA attention (``K == V`` latent, pool selection).

Structure -- a fully STATIC pipeline (a dynamic-trip-count ``fori_loop`` with dynamic double-buffer slots measured
much slower in the dense compressed-window kernel)::

    grid = (batch, n_qb, n_kb)                      # ("parallel", "parallel", "arbitrary")
    scalar prefetch: count [B, nQB], kv_list [B, nQB, nKB]   (build_tables)
    latent BlockSpec: (None, block_kv, D) at kv_block_index(...)  -- always in [0, n_kb)
        -> steps past the active count re-target the resident block: no DMA; pl.when(i < count) skips compute
    per (b, qb):  q block [H, block_q, D] (fetched once), bits_t [W_pad, block_q], tails_t [8, block_q]
        i == 0:      q^T per head -> VMEM scratch [H, D, block_q]; acc^T [H, D, block_q] fp32 = 0;
                     m, l [H, 8, block_q] fp32 (row 0 used) = MASK_MIN, 0
        active step: K = latent block [block_kv, D]; K^T once
                     mask^T [block_kv, block_q] = ((bits row (word of key) >> bit(key)) & 1) | (key == tail[t])
                     per head h: s^T = K @ q_h^T (fp32)          -- keys on sublanes, queries on lanes
                                 online softmax over the column (axis 0), acc_h += K^T @ p
        i == n_kb-1: out_h = (acc_h / l_h)^T -> [block_q, D]; m, log(l) emitted [H, 8, block_q] (row 0)

The transposed tiles put the query axis on lanes, so the per-query bitmap word is a ``[1, block_q]`` ROW read
from the ``bits_t`` block with a dynamic sublane index, the tail compare is a row broadcast, the online-softmax
statistics are column reductions, and no mask is ever transposed.

Numerics: fp32 logits and softmax, bf16 MXU operands for bf16 inputs; the closed-key sentinel is
``finfo(float32).min`` and closed keys contribute nothing. A query with no open key at all yields 0 and FINITE
statistics (``m = 0, log l = 0``) so a recompute-based backward gets ``P = 0`` there. Every block and scratch is
(8, 128)-aligned.

A q-block with ``count == 0`` (a fully padded row, the padded tail of a packed row) has no active block: its list
holds only the ``n_kb`` pad sentinel. The latent index map clamps it into ``[0, n_kb)``; without the clamp the
BlockSpec DMAs block ``n_kb``, one past the latent buffer, which real TPUs report as a ``BoundsCheck`` fault on
``dma.hbm_to_vmem`` while interpret mode clamps silently.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ejkernel.kernels._xla.dsa_sparse_attention._tables import WORD, dma_block_indices

MASK_MIN = float(jnp.finfo(jnp.float32).min)
M_INIT_LEAN = -1e30
NT_DIMS = (((1,), (1,)), ((), ()))  # contract the last dims: a @ b.T
STAT_ROWS = 8
DEFAULT_FWD_VMEM_LIMIT = 48 * 2**20


def open_mask(bits_ref, tails_ref, kb, block_kv: int, block_q: int, kpool: int, n_tails: int, lean: bool = False):
    """Transposed open mask of kv-block ``kb``.

    Args:
        bits_ref: ``[W_pad, bq]`` int32 bitmap block (words on sublanes).
        tails_ref: ``[8, bq]`` int32 tail-key block.
        kb: Kv-block id (traced scalar).
        block_kv: Key block.
        block_q: Query block.
        kpool: Keys per pool.
        n_tails: Number of used tail rows (static).
        lean: Build the mask per bitmap-word slab (one static sublane concat) instead of a select chain over the
            full tile -- the same set with fewer full-tile passes.

    Returns:
        ``[bk, bq]`` bool: key ``r`` of the block is open for query ``c``.
    """
    keys_per_word = WORD * kpool
    words_per_block = block_kv // keys_per_word
    key_g = lax.broadcasted_iota(jnp.int32, (block_kv, block_q), 0) + kb * block_kv
    if lean:
        bit_r = lax.div(lax.broadcasted_iota(jnp.int32, (keys_per_word, block_q), 0), kpool)
        slabs = []
        for w in range(words_per_block):
            row = jnp.broadcast_to(bits_ref[pl.ds(kb * words_per_block + w, 1), :], (keys_per_word, block_q))
            slabs.append(lax.shift_right_logical(row, bit_r) & 1)
        bits = slabs[0] if words_per_block == 1 else jnp.concatenate(slabs, axis=0)
        open_ = bits == 1
    else:
        key_r = lax.broadcasted_iota(jnp.int32, (block_kv, block_q), 0)
        word_of_key = lax.div(key_r, keys_per_word)
        bit_of_key = lax.div(lax.rem(key_r, keys_per_word), kpool)
        word_rows = None
        for w in range(words_per_block):
            row = jnp.broadcast_to(bits_ref[pl.ds(kb * words_per_block + w, 1), :], (block_kv, block_q))
            word_rows = row if word_rows is None else jnp.where(word_of_key == w, row, word_rows)
        open_ = (lax.shift_right_logical(word_rows, bit_of_key) & 1) == 1
    for t in range(n_tails):
        open_ = open_ | (key_g == tails_ref[pl.ds(t, 1), :])
    return open_


def kv_block_index(count_ref, kv_list_ref, b, qb, i, n_kb: int):
    """The kv-block the latent BlockSpec DMAs at grid step ``(b, qb, i)``, ALWAYS in ``[0, n_kb)``.

    Steps past the q-block's ``count`` re-target the last active block (no DMA); a ``count == 0`` q-block holds only
    the ``n_kb`` pad sentinel, so the entry is clamped. Shared by the forward, the dQ pass and the fused backward.
    """
    entry = kv_list_ref[b, qb, jnp.minimum(i, jnp.maximum(count_ref[b, qb] - 1, 0))]
    return jnp.clip(entry, 0, n_kb - 1)


def q_block_index(q_count_ref, q_list_ref, b, kb, j, n_qb: int):
    """The q-block the dKV pass DMAs at step ``(b, kb, j)``, ALWAYS in ``[0, n_qb)`` (see :func:`kv_block_index`)."""
    entry = q_list_ref[b, kb, jnp.minimum(j, jnp.maximum(q_count_ref[b, kb] - 1, 0))]
    return jnp.clip(entry, 0, n_qb - 1)


def assert_tables_in_range(count, lists, n_blocks: int, name: str) -> None:
    """Host-side check that every active list entry and every DMA'd block index lies in ``[0, n_blocks)``.

    Interpret mode clamps an out-of-range BlockSpec index silently; this makes an interpret run see the fault a TPU
    reports as a DMA bounds check.

    Raises:
        ValueError: On an out-of-range active entry or DMA index.
    """
    count, lists = np.asarray(count), np.asarray(lists)
    active = np.arange(lists.shape[-1])[None, None, :] < count[..., None]
    entries = lists[active]
    if entries.size and (entries.min() < 0 or entries.max() >= n_blocks):
        raise ValueError(
            f"dsa_sparse_attention {name}: active list entry out of [0, {n_blocks}): {entries.min()}..{entries.max()}"
        )
    dma = np.asarray(dma_block_indices(jnp.asarray(count), jnp.asarray(lists), n_blocks))
    if dma.min() < 0 or dma.max() >= n_blocks:
        raise ValueError(
            f"dsa_sparse_attention {name}: DMA block index out of [0, {n_blocks}): {dma.min()}..{dma.max()}"
        )


def check_tables_interpret(count, lists, n_blocks: int, name: str) -> None:
    """Emit :func:`assert_tables_in_range` as a host callback (interpret mode only, never on a TPU program)."""
    jax.debug.callback(functools.partial(assert_tables_in_range, n_blocks=n_blocks, name=name), count, lists)


def block_layout(num_heads: int, block_q: int, block_kv: int, head_dim: int, w_pad: int) -> dict:
    """Block and scratch shapes of the forward (``None`` = squeezed batch dim)."""
    return {
        "q": (None, num_heads, block_q, head_dim),
        "latent": (None, block_kv, head_dim),
        "bits_t": (None, w_pad, block_q),
        "tails_t": (None, STAT_ROWS, block_q),
        "out": (None, num_heads, block_q, head_dim),
        "m": (None, num_heads, STAT_ROWS, block_q),
        "logl": (None, num_heads, STAT_ROWS, block_q),
        "scratch:qT": (num_heads, head_dim, block_q),
        "scratch:acc": (num_heads, head_dim, block_q),
        "scratch:m": (num_heads, STAT_ROWS, block_q),
        "scratch:l": (num_heads, STAT_ROWS, block_q),
    }


def check_tile_alignment(layout: dict) -> None:
    """Raise ``ValueError`` unless every block / scratch has (8, 128)-aligned trailing dims."""
    bad = []
    for name, shape in layout.items():
        dims = [d for d in shape if d is not None]
        if len(dims) < 2 or dims[-2] % 8 or dims[-1] % 128:
            bad.append(f"{name}: {shape}")
    if bad:
        raise ValueError("dsa_sparse_attention: unaligned block/scratch shapes:\n  " + "\n  ".join(bad))


def _kernel(
    count_ref,  # SMEM [B, nQB]
    kv_list_ref,  # SMEM [B, nQB, nKB]
    q_ref,  # [H, bq, D]
    k_ref,  # [bk, D]  (the active kv block)
    bits_ref,  # [W_pad, bq] int32
    tails_ref,  # [8, bq] int32
    o_ref,  # [H, bq, D]
    m_ref,  # [H, 8, bq] fp32
    logl_ref,  # [H, 8, bq] fp32
    qt_scr,  # [H, D, bq] mm dtype (stub under NT dots)
    acc_scr,  # [H, D, bq] fp32
    m_scr,  # [H, 8, bq] fp32
    l_scr,  # [H, 8, bq] fp32
    *,
    softmax_scale: float,
    kpool: int,
    block_kv: int,
    n_kb: int,
    n_tails: int,
    nt_dots: bool,
    lean: bool,
    stale_tau: float | None,
):
    b = pl.program_id(0)
    qb = pl.program_id(1)
    i = pl.program_id(2)
    num_heads, block_q, _ = q_ref.shape
    mm_dtype = q_ref.dtype
    prec = lax.Precision.HIGHEST if mm_dtype == jnp.float32 else lax.Precision.DEFAULT
    lean = lean or stale_tau is not None  # the stale-max pass needs the finite -1e30 seed

    @pl.when(i == 0)
    def _init():
        if not nt_dots:
            for h in range(num_heads):
                qt_scr[h] = q_ref[h].T
        acc_scr[...] = jnp.zeros(acc_scr.shape, acc_scr.dtype)
        m_scr[...] = jnp.full(m_scr.shape, M_INIT_LEAN if lean else MASK_MIN, m_scr.dtype)
        l_scr[...] = jnp.zeros(l_scr.shape, l_scr.dtype)

    count = count_ref[b, qb]

    @pl.when(i < count)
    def _step():
        kb = kv_list_ref[b, qb, i]
        k = k_ref[...]
        kt = k.T
        open_ = open_mask(bits_ref, tails_ref, kb, block_kv, block_q, kpool, n_tails, lean=lean)

        def _logits(h):
            if nt_dots:
                st = lax.dot_general(k, q_ref[h], NT_DIMS, preferred_element_type=jnp.float32, precision=prec)
            else:
                st = lax.dot(k, qt_scr[h], preferred_element_type=jnp.float32, precision=prec)
            return jnp.where(open_, st * softmax_scale, MASK_MIN)

        def _exact_update(h, st, m_prev, l_prev, select: bool):
            m_new = jnp.maximum(m_prev, jnp.max(st, axis=0, keepdims=True))
            p = jnp.exp(st - m_new)
            if select:  # closed keys contribute nothing, even when m is still MASK_MIN
                p = jnp.where(open_, p, 0.0)
            corr = jnp.exp(m_prev - m_new)
            l_new = l_prev * corr + jnp.sum(p, axis=0, keepdims=True)
            pv = lax.dot(kt, p.astype(mm_dtype), preferred_element_type=jnp.float32, precision=prec)
            acc_scr[h] = acc_scr[h] * corr + pv
            m_scr[h] = jnp.broadcast_to(m_new, (STAT_ROWS, block_q))
            l_scr[h] = jnp.broadcast_to(l_new, (STAT_ROWS, block_q))

        for h in range(num_heads):
            st = _logits(h)
            m_prev = m_scr[h, pl.ds(0, 1), :]
            l_prev = l_scr[h, pl.ds(0, 1), :]
            if stale_tau is None:
                _exact_update(h, st, m_prev, l_prev, select=not lean)
                continue
            # Speculative stale-max pass: p = exp(st - m_prev) in the same sweep as the column max; the block is
            # redone exactly only if some column's max grew by more than tau (p <= e^tau stays finite). m stays
            # stale and l accumulates against it, so (m, log l) remain a valid LSE split for the backward.
            colmax = jnp.max(st, axis=0, keepdims=True)
            p_fast = jnp.exp(st - m_prev)
            redo = jnp.max(colmax - m_prev) > stale_tau

            @pl.when(jnp.logical_not(redo))
            def _fast(h=h, p_fast=p_fast, l_prev=l_prev):
                l_scr[h] = jnp.broadcast_to(l_prev + jnp.sum(p_fast, axis=0, keepdims=True), (STAT_ROWS, block_q))
                acc_scr[h] = acc_scr[h] + lax.dot(
                    kt, p_fast.astype(mm_dtype), preferred_element_type=jnp.float32, precision=prec
                )

            @pl.when(redo)
            def _redo(h=h, m_prev=m_prev, l_prev=l_prev):
                _exact_update(h, _logits(h), m_prev, l_prev, select=False)

    @pl.when(i == n_kb - 1)
    def _finish():
        for h in range(num_heads):
            l_h = l_scr[h, pl.ds(0, 1), :]
            inv = 1.0 / jnp.where(l_h == 0.0, 1.0, l_h)
            o_ref[h] = (acc_scr[h] * inv).T.astype(o_ref.dtype)
            no_key = l_scr[h] == 0.0
            m_ref[h] = jnp.where(no_key, 0.0, m_scr[h])
            logl_ref[h] = jnp.log(jnp.where(no_key, 1.0, l_scr[h]))


def dsa_sparse_attention_pallas_fwd(
    query: Array,
    latent: Array,
    bits_t: Array,
    tails_t: Array,
    count: Array,
    kv_list: Array,
    *,
    kpool: int,
    block_q: int,
    block_kv: int,
    softmax_scale: float,
    n_tails: int = 3,
    interpret: bool | None = None,
    check_alignment: bool | None = None,
    vmem_limit_bytes: int | None = None,
    nt_dots: bool = False,
    lean: bool = False,
    stale_tau: float | None = None,
) -> tuple[Array, Array, Array]:
    """Launch the sparse forward over the tables of ``build_tables``.

    Args:
        query: ``[B, H, S, D]`` with ``S % block_q == S % block_kv == 0``.
        latent: ``[B, S, D]`` shared ``K == V``.
        bits_t: ``[B, W_pad, S]`` int32 pool bitmap.
        tails_t: ``[B, 8, S]`` int32 tail keys.
        count: ``[B, nQB]`` active kv-block counts.
        kv_list: ``[B, nQB, nKB]`` active kv-block ids.
        kpool: Keys per pool.
        block_q: Query block.
        block_kv: Key block (multiple of ``32 * kpool``).
        softmax_scale: Logit scale.
        n_tails: Tail rows in use.
        interpret: Pallas interpret mode (``None``: off-TPU only).
        check_alignment: Check the (8, 128) layout rule (``None``: when not interpreting).
        vmem_limit_bytes: Scoped VMEM request (default 48 MiB).
        nt_dots: ``K @ q_h^T`` as an NT ``dot_general`` on the resident q block instead of a per-q-block transposed
            copy of q (bit-identical output).
        lean: Running max seeded at ``-1e30`` (drops the post-exp select) and slab-built masks; bitwise the base.
        stale_tau: Speculative stale-max softmax threshold in nats (``None`` = off). Output within fp32 rounding of
            the base; measured slower on v5p at d = 1, so never the default.

    Returns:
        ``(out [B, H, S, D] in latent's dtype, m [B, H, 8, S] fp32, logl [B, H, 8, S] fp32)``; row 0 of the
        statistics is the value (the 8-row form keeps the blocks tile-aligned).
    """
    batch, num_heads, seq, head_dim = query.shape
    if seq % block_q or seq % block_kv:
        raise ValueError(f"seq {seq} must be a multiple of block_q {block_q} and block_kv {block_kv}")
    if block_kv % (WORD * kpool):
        raise ValueError(f"block_kv {block_kv} must be a multiple of {WORD * kpool}")
    n_qb, n_kb = seq // block_q, seq // block_kv
    w_pad = bits_t.shape[1]
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    if check_alignment is None:
        check_alignment = not interpret
    lay = block_layout(num_heads, block_q, block_kv, head_dim, w_pad)
    if nt_dots:
        lay["scratch:qT"] = (STAT_ROWS, 128)  # unused under NT dots; the smallest tile-aligned stub
    if check_alignment:
        check_tile_alignment(lay)
    if interpret:
        check_tables_interpret(count, kv_list, n_kb, "forward kv_list")

    def kv_block(b, qb, i, count_ref, kv_list_ref):
        return kv_block_index(count_ref, kv_list_ref, b, qb, i, n_kb)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(batch, n_qb, n_kb),
        in_specs=[
            pl.BlockSpec(lay["q"], lambda b, qb, i, c, kl: (b, 0, qb, 0)),
            pl.BlockSpec(lay["latent"], lambda b, qb, i, c, kl: (b, kv_block(b, qb, i, c, kl), 0)),
            pl.BlockSpec(lay["bits_t"], lambda b, qb, i, c, kl: (b, 0, qb)),
            pl.BlockSpec(lay["tails_t"], lambda b, qb, i, c, kl: (b, 0, qb)),
        ],
        out_specs=[
            pl.BlockSpec(lay["out"], lambda b, qb, i, c, kl: (b, 0, qb, 0)),
            pl.BlockSpec(lay["m"], lambda b, qb, i, c, kl: (b, 0, 0, qb)),
            pl.BlockSpec(lay["logl"], lambda b, qb, i, c, kl: (b, 0, 0, qb)),
        ],
        scratch_shapes=[
            pltpu.VMEM(lay["scratch:qT"], query.dtype),
            pltpu.VMEM(lay["scratch:acc"], jnp.float32),
            pltpu.VMEM(lay["scratch:m"], jnp.float32),
            pltpu.VMEM(lay["scratch:l"], jnp.float32),
        ],
    )
    kernel = functools.partial(
        _kernel,
        softmax_scale=float(softmax_scale),
        kpool=kpool,
        block_kv=block_kv,
        n_kb=n_kb,
        n_tails=n_tails,
        nt_dots=bool(nt_dots),
        lean=bool(lean),
        stale_tau=None if stale_tau is None else float(stale_tau),
    )
    out, m, logl = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((batch, num_heads, seq, head_dim), latent.dtype),
            jax.ShapeDtypeStruct((batch, num_heads, STAT_ROWS, seq), jnp.float32),
            jax.ShapeDtypeStruct((batch, num_heads, STAT_ROWS, seq), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=int(vmem_limit_bytes or DEFAULT_FWD_VMEM_LIMIT),
        ),
        interpret=interpret,
    )(count, kv_list, query, latent, bits_t, tails_t)
    return out, m, logl
