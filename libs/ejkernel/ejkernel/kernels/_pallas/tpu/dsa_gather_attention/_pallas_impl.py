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

"""Per-query DMA-gather DSA attention kernels (Pallas TPU).

Each query attends to exactly its open pools (``_xla.dsa_gather_attention._layout``), so the work is
``O(N * K * kpool * H * D)`` and independent of the sequence length and of how the selection is spread. The block-sparse
bitmap kernel instead visits every (q-block, kv-block) tile a selection touches, which at long context with a spread
selection is the whole causal triangle.

Metadata (XLA, one SMEM row per query): ``[count, tail_len, slot_0 .. slot_{ng*G-1}, pad]``. Slot 0 is the query's own
(incomplete) pool when ``tail_len > 0``; slots ``1 .. count`` are the open selected pools in ascending order; every other
slot holds ``P`` (= number of pools), a trash row: latent reads clamp it to ``P - 1`` and its scores are masked, gradient
writes land in an extra accumulator row that is dropped.

The latent is viewed as 32-bit words ``[B, S / pack, pack, D]`` (``pack`` keys per word: 2 for bf16, 1 for fp32) so one
pool is one DMA at an untiled leading offset. Pools are fetched ``G`` at a time into an ``nbuf``-deep VMEM ring.

Forward: grid over blocks of ``qb`` queries; online softmax per query; emits ``out`` and ``lse``.

Backward: grid ``(cores, blocks per core)``; per query and group, ``dq`` accumulates in VMEM and the group's ``dK``
rows are read-modify-written into an fp32 HBM accumulator ``[cores, B, P + 1, kpool, D]`` (one copy per TensorCore,
summed in XLA). Pools inside one query are distinct, and the accumulator reads of query ``i + 1`` start only after all
writes of query ``i`` completed, so no update is lost.

Fused backward (:func:`gather_bwd_fused`, one pool group per query): the backward above is bound by DMA issue, with
three DMAs per (query, pool) (key read, accumulator read, accumulator write). Each pool instead owns one fp32 HBM row
``[2, kpool, D]`` = (dK accumulator, key), so one DMA reads both and one writes the accumulator back. A query's pools are
read as :data:`FUSED_SPLIT` sub-groups on separate semaphores, so the second sub-group's reads overlap the first one's
compute. dlatent is bit-identical to the three-DMA kernel's; dq is within one bf16 ulp (it sums two partial products
instead of one).
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NEG = -1e30
F32 = jnp.float32
META_HEAD = 2  # [count, tail_len] precede the slots
DEFAULT_VMEM_LIMIT = 48 * 2**20


def dtype_packing(dtype) -> int:
    """Keys per 32-bit HBM word (2 for 2-byte dtypes, 1 for 4-byte); 0 when unsupported."""
    size = jnp.dtype(dtype).itemsize
    return 4 // size if size in (2, 4) else 0


def _precision(dtype):
    """``HIGHEST`` for fp32 operands, so they do not drop to a single bf16 MXU pass; ``None`` (native) otherwise."""
    return lax.Precision.HIGHEST if jnp.dtype(dtype) == jnp.float32 else None


def _nt(a, b, precision):
    """``a [M, K] @ b [N, K]^T -> [M, N]`` in fp32."""
    return lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=F32, precision=precision)


def _tn(a, b, precision):
    """``a [K, M]^T @ b [K, N] -> [M, N]`` in fp32."""
    return lax.dot_general(a, b, (((0,), (0,)), ((), ())), preferred_element_type=F32, precision=precision)


def _open_mask(g, count, tail_len, *, g_sz: int, kpool: int):
    """``[1, G * kpool]`` open keys of slot group ``g``: selected slots ``1..count``, slot 0 up to ``tail_len``."""
    lanes = lax.broadcasted_iota(jnp.int32, (1, g_sz * kpool), 1)
    j = g * g_sz + lax.div(lanes, kpool)
    rr = lax.rem(lanes, kpool)
    return ((j >= 1) & (j <= count)) | ((j == 0) & (rr < tail_len))


def _load_keys(kbuf, slot, *, g_sz: int, wp: int, kpool: int, d: int):
    """The ``[G * kpool, D]`` keys of ring slot ``slot`` (32-bit words reinterpreted in the latent dtype)."""
    words = kbuf.bitcast(jnp.uint32).at[slot].reshape(g_sz * wp, d)
    return pltpu.bitcast(words[...], kbuf.dtype).reshape(g_sz * kpool, d)


def build_meta(
    indices: jax.Array,
    selected_ok: jax.Array,
    tail_len: jax.Array,
    own: jax.Array,
    *,
    g_sz: int,
    ng: int,
    num_pools: int,
) -> jax.Array:
    """``[B, N, M]`` int32 metadata rows (see the module docstring); ``M`` is a multiple of 128."""
    ok = selected_ok & (indices < num_pools)
    count = jnp.sum(ok, axis=-1, dtype=jnp.int32)
    ids = jnp.sort(jnp.where(ok, indices, num_pools).astype(jnp.int32), axis=-1)  # open ids first, ascending
    tail = jnp.where(own < num_pools, tail_len, 0).astype(jnp.int32)
    own_slot = jnp.where(tail > 0, own, num_pools).astype(jnp.int32)
    slots = jnp.concatenate([own_slot[..., None], ids], axis=-1)
    slots = jnp.pad(slots, ((0, 0), (0, 0), (0, ng * g_sz - slots.shape[-1])), constant_values=num_pools)
    meta = jnp.concatenate([count[..., None], tail[..., None], slots], axis=-1)
    width = -(-meta.shape[-1] // 128) * 128
    return jnp.pad(meta, ((0, 0), (0, 0), (0, width - meta.shape[-1])), constant_values=num_pools).astype(jnp.int32)


# ----------------------------------------------------------------------------------------------------------- forward


def _fwd_kernel(
    meta_ref, q_ref, lat_hbm, o_ref, lse_ref, kbuf, sem, acc_ref, lse_scr,
    *, qb, g_sz, ng, kpool, pack, scale, ns, nbuf, num_pools,
):  # fmt: skip
    b = lax.div(pl.program_id(0), ns)
    wp = kpool // pack
    n_items = qb * ng
    h, d = q_ref.shape[1], q_ref.shape[2]
    prec = _precision(q_ref.dtype)

    def start(t, slot):
        i = lax.div(t, ng)
        base = META_HEAD + lax.rem(t, ng) * g_sz
        for j in range(g_sz):
            p = jnp.minimum(meta_ref[i, base + j], num_pools - 1)
            pltpu.make_async_copy(
                lat_hbm.at[b, pl.ds(p * wp, wp)], kbuf.at[slot, pl.ds(j * wp, wp)], sem.at[slot]
            ).start()

    def wait(slot):
        # One descriptor covering the whole slot: its byte count equals the G per-pool copies signalled.
        pltpu.make_async_copy(lat_hbm.at[b, pl.ds(0, g_sz * wp)], kbuf.at[slot], sem.at[slot]).wait()

    for t0 in range(min(nbuf - 1, n_items)):
        start(jnp.int32(t0), t0)
    lane_q = lax.broadcasted_iota(jnp.int32, (h, qb), 1)
    lse_scr[...] = jnp.full(lse_scr.shape, NEG, F32)

    def per_query(i, carry):
        q = q_ref[i]
        count = meta_ref[i, 0]
        tail_len = meta_ref[i, 1]
        acc_ref[...] = jnp.zeros(acc_ref.shape, F32)

        def group(g, ml):
            m_prev, l_prev = ml
            t = i * ng + g
            slot = lax.rem(t, nbuf)

            @pl.when(t + (nbuf - 1) < n_items)
            def _():
                start(t + (nbuf - 1), lax.rem(t + (nbuf - 1), nbuf))

            wait(slot)
            k = _load_keys(kbuf, slot, g_sz=g_sz, wp=wp, kpool=kpool, d=d)
            ok = _open_mask(g, count, tail_len, g_sz=g_sz, kpool=kpool)
            sc = jnp.where(ok, _nt(q, k, prec) * scale, NEG)
            m_new = jnp.maximum(m_prev, jnp.max(sc, axis=1, keepdims=True))
            p = jnp.where(ok, jnp.exp(sc - m_new), 0.0)
            alpha = jnp.exp(m_prev - m_new)
            pv = jnp.dot(p.astype(k.dtype), k, preferred_element_type=F32, precision=prec)
            acc_ref[...] = alpha * acc_ref[...] + pv
            return m_new, alpha * l_prev + jnp.sum(p, axis=1, keepdims=True)

        m, l = lax.fori_loop(0, ng, group, (jnp.full((h, 1), NEG, F32), jnp.zeros((h, 1), F32)))
        safe = jnp.where(l > 0, l, 1.0)
        o_ref[i] = jnp.where(l > 0, acc_ref[...] / safe, 0.0).astype(o_ref.dtype)
        lse_col = jnp.where(l > 0, m + jnp.log(safe), NEG)
        lse_scr[...] = jnp.where(lane_q == i, jnp.broadcast_to(lse_col, (h, qb)), lse_scr[...])
        return carry

    lax.fori_loop(0, qb, per_query, 0)
    lse_ref[...] = lse_scr[...]


def gather_fwd(
    query: jax.Array,
    latw: jax.Array,
    meta: jax.Array,
    *,
    kpool: int,
    scale: float,
    qb: int,
    g_sz: int,
    ng: int,
    nbuf: int,
    num_pools: int,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Forward kernel call.

    Args:
        query: ``[B, N, H, D]`` with ``N % qb == 0``.
        latw: Latent as words ``[B, S / pack, pack, D]``.
        meta: ``[B, N, M]`` rows from :func:`build_meta`.

    Returns:
        ``(out [B, N, H, D], lse [B, N / qb, H, qb] fp32)`` (the statistics stay in the kernel layout).
    """
    bsz, n, h, d = query.shape
    pack = latw.shape[2]
    ns = n // qb
    wp = kpool // pack
    m_w = meta.shape[-1]
    kernel = functools.partial(
        _fwd_kernel, qb=qb, g_sz=g_sz, ng=ng, kpool=kpool, pack=pack, scale=float(scale), ns=ns, nbuf=nbuf,
        num_pools=num_pools,
    )  # fmt: skip

    def qmap(t):
        return (t // ns, t % ns, 0, 0)

    out, lse = pl.pallas_call(
        kernel,
        grid=(bsz * ns,),
        in_specs=[
            pl.BlockSpec((qb, m_w), lambda t: (t, 0), memory_space=pltpu.SMEM),
            pl.BlockSpec((None, qb, h, d), qmap),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=[pl.BlockSpec((None, qb, h, d), qmap), pl.BlockSpec((None, None, h, qb), qmap)],
        out_shape=[jax.ShapeDtypeStruct((bsz, n, h, d), latw.dtype), jax.ShapeDtypeStruct((bsz, ns, h, qb), F32)],
        scratch_shapes=[
            pltpu.VMEM((nbuf, g_sz * wp, pack, d), latw.dtype),
            pltpu.SemaphoreType.DMA((nbuf,)),
            pltpu.VMEM((h, d), F32),
            pltpu.VMEM((h, qb), F32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=int(vmem_limit_bytes or DEFAULT_VMEM_LIMIT)
        ),
        interpret=interpret,
        name=f"dsa_gather_fwd_q{qb}_g{g_sz}_b{nbuf}",
    )(meta.reshape(bsz * n, m_w), query, latw)
    return out, lse


# ---------------------------------------------------------------------------------------------------------- backward


def _bwd_kernel(
    meta_ref, q_ref, do_ref, lse_ref, dlt_ref, lat_hbm, acc_alias, dq_ref, acc_hbm,
    kbuf, abuf, ksem, asem, wsem, dq_scr,
    *, qb, g_sz, ng, kpool, pack, scale, ns, nb, nbuf, num_pools,
):  # fmt: skip
    del acc_alias
    c = pl.program_id(0)
    b = lax.div(c * nb + pl.program_id(1), ns)
    wp = kpool // pack
    n_items = qb * ng
    h, d = q_ref.shape[1], q_ref.shape[2]
    prec = _precision(q_ref.dtype)

    def pool(i, g, j):
        return meta_ref[i, META_HEAD + g * g_sz + j]

    def start_k(t, slot):
        i = lax.div(t, ng)
        g = lax.rem(t, ng)
        for j in range(g_sz):
            p = jnp.minimum(pool(i, g, j), num_pools - 1)
            pltpu.make_async_copy(
                lat_hbm.at[b, pl.ds(p * wp, wp)], kbuf.at[slot, pl.ds(j * wp, wp)], ksem.at[slot]
            ).start()

    def wait_k(slot):
        pltpu.make_async_copy(lat_hbm.at[b, pl.ds(0, g_sz * wp)], kbuf.at[slot], ksem.at[slot]).wait()

    def start_a(i, g, slot):
        for j in range(g_sz):
            pltpu.make_async_copy(acc_hbm.at[c, b, pool(i, g, j)], abuf.at[slot, j], asem.at[slot]).start()

    def wait_a(slot):
        pltpu.make_async_copy(acc_hbm.at[c, b, pl.ds(0, g_sz)], abuf.at[slot], asem.at[slot]).wait()

    def start_w(i, g, slot):
        for j in range(g_sz):
            pltpu.make_async_copy(abuf.at[slot, j], acc_hbm.at[c, b, pool(i, g, j)], wsem.at[slot]).start()

    def wait_w(slot):
        pltpu.make_async_copy(abuf.at[slot], acc_hbm.at[c, b, pl.ds(0, g_sz)], wsem.at[slot]).wait()

    for t0 in range(min(nbuf - 1, n_items)):
        start_k(jnp.int32(t0), t0)
    lane_q = lax.broadcasted_iota(jnp.int32, (h, qb), 1)
    lse_blk = lse_ref[...]
    dlt_blk = dlt_ref[...]

    def per_query(i, carry):
        q = q_ref[i]
        do = do_ref[i]
        qd = jnp.concatenate([q, do], axis=0)
        sel = lane_q == i
        lse_i = jnp.sum(jnp.where(sel, lse_blk, 0.0), axis=1, keepdims=True)
        dlt_i = jnp.sum(jnp.where(sel, dlt_blk, 0.0), axis=1, keepdims=True)
        count = meta_ref[i, 0]
        tail_len = meta_ref[i, 1]
        for g0 in range(min(nbuf - 1, ng)):
            start_a(i, g0, g0)
        dq_scr[...] = jnp.zeros(dq_scr.shape, F32)

        def group(g, carry2):
            t = i * ng + g
            ks = lax.rem(t, nbuf)

            @pl.when(t + (nbuf - 1) < n_items)
            def _():
                start_k(t + (nbuf - 1), lax.rem(t + (nbuf - 1), nbuf))

            @pl.when(g + (nbuf - 1) < ng)
            def _():
                nslot = lax.rem(g + (nbuf - 1), nbuf)

                @pl.when(g >= 1)
                def _():
                    wait_w(nslot)  # group g - 1 wrote from this slot

                start_a(i, g + (nbuf - 1), nslot)

            wait_k(ks)
            k = _load_keys(kbuf, ks, g_sz=g_sz, wp=wp, kpool=kpool, d=d)
            ok = _open_mask(g, count, tail_len, g_sz=g_sz, kpool=kpool)
            p = jnp.where(ok, jnp.exp(_nt(q, k, prec) * scale - lse_i), 0.0)
            ds = p * (_nt(do, k, prec) - dlt_i)
            dq_scr[...] += jnp.dot(ds.astype(k.dtype), k, preferred_element_type=F32, precision=prec)
            a = jnp.concatenate([(ds * scale).astype(k.dtype), p.astype(k.dtype)], axis=0)
            dk = _tn(a, qd, prec)  # [G * kpool, D] = scale * dS^T q + P^T dO
            asl = lax.rem(g, nbuf)
            wait_a(asl)
            for jj in range(g_sz):
                abuf[asl, jj] = abuf[asl, jj] + dk[jj * kpool : (jj + 1) * kpool]
            start_w(i, g, asl)
            return carry2

        lax.fori_loop(0, ng, group, 0)
        for gl in range(max(0, ng - nbuf), ng):
            wait_w(gl % nbuf)
        dq_ref[i] = (dq_scr[...] * scale).astype(dq_ref.dtype)
        return carry

    lax.fori_loop(0, qb, per_query, 0)


def gather_bwd(
    query: jax.Array,
    latw: jax.Array,
    meta: jax.Array,
    do: jax.Array,
    lse: jax.Array,
    delta: jax.Array,
    *,
    kpool: int,
    scale: float,
    qb: int,
    g_sz: int,
    ng: int,
    nbuf: int,
    num_pools: int,
    cores: int,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Backward kernel call.

    Args:
        query, latw, meta: As in :func:`gather_fwd`.
        do: ``[B, N, H, D]`` output cotangent.
        lse, delta: ``[B, N / qb, H, qb]`` fp32 log-sum-exp and ``rowsum(dO * O)`` in the kernel layout.
        cores: Accumulator copies / leading parallel grid axis (``(B * N / qb) % cores == 0``).

    Returns:
        ``(dq [B, N, H, D], dlatent [B, num_pools * kpool, D] fp32)``.
    """
    bsz, n, h, d = query.shape
    pack = latw.shape[2]
    ns = n // qb
    wp = kpool // pack
    m_w = meta.shape[-1]
    nblk = bsz * ns
    nb = nblk // cores
    acc0 = jnp.zeros((cores, bsz, num_pools + 1, kpool, d), F32)
    kernel = functools.partial(
        _bwd_kernel, qb=qb, g_sz=g_sz, ng=ng, kpool=kpool, pack=pack, scale=float(scale), ns=ns, nb=nb, nbuf=nbuf,
        num_pools=num_pools,
    )  # fmt: skip

    def qmap(c, j):
        t = c * nb + j
        return (t // ns, t % ns, 0, 0)

    dq, acc = pl.pallas_call(
        kernel,
        grid=(cores, nb),
        in_specs=[
            pl.BlockSpec((qb, m_w), lambda c, j: (c * nb + j, 0), memory_space=pltpu.SMEM),
            pl.BlockSpec((None, qb, h, d), qmap),
            pl.BlockSpec((None, qb, h, d), qmap),
            pl.BlockSpec((None, None, h, qb), qmap),
            pl.BlockSpec((None, None, h, qb), qmap),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=[pl.BlockSpec((None, qb, h, d), qmap), pl.BlockSpec(memory_space=pltpu.HBM)],
        out_shape=[jax.ShapeDtypeStruct((bsz, n, h, d), query.dtype), jax.ShapeDtypeStruct(acc0.shape, F32)],
        scratch_shapes=[
            pltpu.VMEM((nbuf, g_sz * wp, pack, d), latw.dtype),
            pltpu.VMEM((nbuf, g_sz, kpool, d), F32),
            pltpu.SemaphoreType.DMA((nbuf,)),
            pltpu.SemaphoreType.DMA((nbuf,)),
            pltpu.SemaphoreType.DMA((nbuf,)),
            pltpu.VMEM((h, d), F32),
        ],
        input_output_aliases={6: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=int(vmem_limit_bytes or DEFAULT_VMEM_LIMIT),
        ),
        interpret=interpret,
        name=f"dsa_gather_bwd_q{qb}_g{g_sz}_b{nbuf}_c{cores}",
    )(meta.reshape(bsz * n, m_w), query, do, lse, delta, latw, acc0)
    dlat = jnp.sum(acc, axis=0)[:, :num_pools].reshape(bsz, num_pools * kpool, d)
    return dq, dlat


# ---------------------------------------------------------------------------------------------------- fused backward

FUSED_SPLIT = 2
"""Read sub-groups per query in the fused backward (2 measured best: 295.7 ms vs 321.6 for 1, 308.2 for 4)."""
FUSED_ALIGN = 2 * FUSED_SPLIT // math.gcd(2, FUSED_SPLIT)
"""Slots per query are rounded up to this: whole sub-groups, and an even count for the one write-wait descriptor."""
FUSED_VMEM_LIMIT = 96 * 2**20
"""Scoped-VMEM limit of the fused backward (the measured setting), capped at 90% of the chip's VMEM."""


def fused_bwd_vmem_limit() -> int:
    """:data:`FUSED_VMEM_LIMIT` capped at 90% of the local TPU's VMEM; :data:`DEFAULT_VMEM_LIMIT` when unknown."""
    try:
        capacity = int(getattr(pltpu.get_tpu_info(), "vmem_capacity_bytes", 0) or 0)
    except Exception:  # off TPU or an older runtime
        capacity = 0
    return min(FUSED_VMEM_LIMIT, int(0.9 * capacity)) if capacity > 0 else DEFAULT_VMEM_LIMIT


def fused_bwd_vmem_bytes(g_sz: int, *, kpool: int, head_dim: int, num_heads: int, itemsize: int, block_q: int) -> int:
    """Working-set estimate of one fused-backward grid step (``g_sz`` already aligned to :data:`FUSED_ALIGN`)."""
    fixed = 6 * block_q * num_heads * head_dim * itemsize + 4 * num_heads * block_q * 4  # q/dO/dq + lse/delta blocks
    rows = g_sz * 2 * kpool * head_dim * 4  # (accumulator, key) rows of every slot
    keys = (g_sz // FUSED_SPLIT) * kpool
    live = keys * head_dim * (itemsize + 8) + num_heads * keys * (16 + 2 * itemsize) + 2 * num_heads * head_dim * 4
    return fixed + rows + live


def fused_rows(latent: jax.Array, *, kpool: int, cores: int) -> jax.Array:
    """``[cores, B, P + 1, 2, kpool, D]`` fp32 rows (zero accumulator, key) per pool; row ``P`` is the zero trash row."""
    bsz, s, d = latent.shape
    keys = latent.astype(F32).reshape(bsz, s // kpool, kpool, d)
    keys = jnp.pad(keys, ((0, 0), (0, 1), (0, 0), (0, 0)))
    one = jnp.stack([jnp.zeros_like(keys), keys], axis=2)
    return jnp.broadcast_to(one[None], (cores, *one.shape))


def _bwd_fused_kernel(
    meta_ref, q_ref, do_ref, lse_ref, dlt_ref, rows_alias, dq_ref, rows_hbm, rbuf, rsem, wsem,
    *, qb, g_sz, kpool, scale, ns, nb,
):  # fmt: skip
    del rows_alias
    c = pl.program_id(0)
    b = lax.div(c * nb + pl.program_id(1), ns)
    h, d = q_ref.shape[1], q_ref.shape[2]
    prec = _precision(q_ref.dtype)
    lane_q = lax.broadcasted_iota(jnp.int32, (h, qb), 1)
    lse_blk = lse_ref[...]
    dlt_blk = dlt_ref[...]
    gs = g_sz // FUSED_SPLIT

    def pool(i, j):
        return meta_ref[i, META_HEAD + j]

    def per_query(i, carry):
        # Every slot's (accumulator, key) row, sub-group s signalling rsem[s]. Slots of one query are distinct pools
        # except padding (trash row P, whose masked keys add exactly 0), and the previous query's writes completed.
        for j in range(g_sz):
            pltpu.make_async_copy(rows_hbm.at[c, b, pool(i, j)], rbuf.at[j], rsem.at[j // gs]).start()
        q = q_ref[i]
        do = do_ref[i]
        qd = jnp.concatenate([q, do], axis=0)
        sel = lane_q == i
        lse_i = jnp.sum(jnp.where(sel, lse_blk, 0.0), axis=1, keepdims=True)
        dlt_i = jnp.sum(jnp.where(sel, dlt_blk, 0.0), axis=1, keepdims=True)
        count = meta_ref[i, 0]
        tail_len = meta_ref[i, 1]
        dq = jnp.zeros((h, d), F32)
        for s in range(FUSED_SPLIT):
            lo = s * gs
            pltpu.make_async_copy(rows_hbm.at[c, b, pl.ds(0, gs)], rbuf.at[pl.ds(lo, gs)], rsem.at[s]).wait()
            k = rbuf[lo : lo + gs, 1].reshape(gs * kpool, d).astype(q.dtype)
            ok = _open_mask(s, count, tail_len, g_sz=gs, kpool=kpool)
            p = jnp.where(ok, jnp.exp(_nt(q, k, prec) * scale - lse_i), 0.0)
            ds = p * (_nt(do, k, prec) - dlt_i)
            dq = dq + jnp.dot(ds.astype(k.dtype), k, preferred_element_type=F32, precision=prec)
            a = jnp.concatenate([(ds * scale).astype(k.dtype), p.astype(k.dtype)], axis=0)
            dk = _tn(a, qd, prec)  # [gs * kpool, D] = scale * dS^T q + P^T dO
            for jj in range(gs):
                rbuf[lo + jj, 0] = rbuf[lo + jj, 0] + dk[jj * kpool : (jj + 1) * kpool]
            for jj in range(gs):
                pltpu.make_async_copy(rbuf.at[lo + jj, 0], rows_hbm.at[c, b, pool(i, lo + jj), 0], wsem).start()
        dq_ref[i] = (dq * scale).astype(dq_ref.dtype)
        # g_sz accumulator halves are g_sz / 2 whole rows of bytes; the next query's reads must see these writes.
        pltpu.make_async_copy(rbuf.at[pl.ds(0, g_sz // 2)], rows_hbm.at[c, b, pl.ds(0, g_sz // 2)], wsem).wait()
        return carry

    lax.fori_loop(0, qb, per_query, 0)


def gather_bwd_fused(
    query: jax.Array,
    rows: jax.Array,
    meta: jax.Array,
    do: jax.Array,
    lse: jax.Array,
    delta: jax.Array,
    *,
    kpool: int,
    scale: float,
    qb: int,
    g_sz: int,
    cores: int,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused-row backward kernel call (one pool group of ``g_sz`` slots per query, ``g_sz % FUSED_ALIGN == 0``).

    Args:
        query, meta, do, lse, delta: As in :func:`gather_bwd` (``meta`` built with ``g_sz`` slots, one group).
        rows: :func:`fused_rows` of the latent, ``[cores, B, P + 1, 2, kpool, D]`` fp32 (donated: aliased output).
        cores: Accumulator copies / leading parallel grid axis (``(B * N / qb) % cores == 0``).

    Returns:
        ``(dq [B, N, H, D], rows)`` with ``rows[..., 0, :, :]`` holding the per-core dK accumulators.

    Raises:
        ValueError: When ``g_sz`` is not a multiple of :data:`FUSED_ALIGN`.
    """
    if g_sz % FUSED_ALIGN:
        raise ValueError(f"gather_bwd_fused: g_sz={g_sz} must be a multiple of {FUSED_ALIGN}")
    bsz, n, h, d = query.shape
    ns = n // qb
    m_w = meta.shape[-1]
    nb = (bsz * ns) // cores
    kernel = functools.partial(_bwd_fused_kernel, qb=qb, g_sz=g_sz, kpool=kpool, scale=float(scale), ns=ns, nb=nb)

    def qmap(c, j):
        t = c * nb + j
        return (t // ns, t % ns, 0, 0)

    return pl.pallas_call(
        kernel,
        grid=(cores, nb),
        in_specs=[
            pl.BlockSpec((qb, m_w), lambda c, j: (c * nb + j, 0), memory_space=pltpu.SMEM),
            pl.BlockSpec((None, qb, h, d), qmap),
            pl.BlockSpec((None, qb, h, d), qmap),
            pl.BlockSpec((None, None, h, qb), qmap),
            pl.BlockSpec((None, None, h, qb), qmap),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=[pl.BlockSpec((None, qb, h, d), qmap), pl.BlockSpec(memory_space=pltpu.HBM)],
        out_shape=[jax.ShapeDtypeStruct((bsz, n, h, d), query.dtype), jax.ShapeDtypeStruct(rows.shape, F32)],
        scratch_shapes=[
            pltpu.VMEM((g_sz, 2, kpool, d), F32),
            pltpu.SemaphoreType.DMA((FUSED_SPLIT,)),
            pltpu.SemaphoreType.DMA(()),
        ],
        input_output_aliases={5: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=int(vmem_limit_bytes or DEFAULT_VMEM_LIMIT),
        ),
        interpret=interpret,
        name=f"dsa_gather_bwd_fused_q{qb}_g{g_sz}_s{FUSED_SPLIT}_c{cores}",
    )(meta.reshape(bsz * n, m_w), query, do, lse, delta, rows)


__all__ = (
    "DEFAULT_VMEM_LIMIT",
    "FUSED_ALIGN",
    "FUSED_SPLIT",
    "build_meta",
    "dtype_packing",
    "fused_bwd_vmem_bytes",
    "fused_bwd_vmem_limit",
    "fused_rows",
    "gather_bwd",
    "gather_bwd_fused",
    "gather_fwd",
)
