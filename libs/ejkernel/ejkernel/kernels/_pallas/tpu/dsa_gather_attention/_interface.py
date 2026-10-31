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

"""Differentiable per-query DMA-gather DSA attention (Pallas TPU).

The wrapper resolves the open pools (``_xla.dsa_gather_attention._layout``), pads the queries to the query block and
the latent to whole pools, builds the per-query metadata once outside the ``custom_vjp`` core and runs the forward /
backward kernels of ``_pallas_impl``. The forward residuals carry the same ``checkpoint_name`` tags as the
block-sparse ``dsa_sparse_attention`` kernel, so one remat policy that saves or offloads those names covers both.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int

from ejkernel.kernels._xla.dsa_gather_attention._layout import default_positions, selection_layout, validate

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl import (
    DEFAULT_VMEM_LIMIT,
    FUSED_ALIGN,
    build_meta,
    dtype_packing,
    fused_bwd_vmem_bytes,
    fused_bwd_vmem_limit,
    fused_rows,
    gather_bwd,
    gather_bwd_fused,
    gather_fwd,
)

#: ``checkpoint_name`` of the forward residuals the backward reads (output and log-sum-exp).
RESIDUAL_NAME = "dsa_sparse_attention_residuals"
#: ``checkpoint_name`` of the per-query gather metadata.
TABLES_NAME = "dsa_sparse_attention_tables"

F32 = jnp.float32


def _round_up(x: int, m: int) -> int:
    return -(-int(x) // int(m)) * int(m)


def dsa_gather_group_plan(
    num_slots: int,
    *,
    kpool: int,
    head_dim: int,
    num_heads: int,
    itemsize: int,
    block_q: int = 16,
    vmem_budget: int | None = None,
) -> tuple[int, int, int]:
    """Automatic ``(pools_per_step, num_groups, num_buffers)`` for ``num_slots`` pool slots per query.

    The kernels pay per DMA group as well as per DMA (one semaphore wait, one online-softmax rescale and one
    accumulator round trip per group), so the plan takes the fewest groups whose backward working set fits
    ``vmem_budget`` and splits the slots evenly over them (``ceil(num_slots / num_groups)`` each). One group needs
    only a two-deep ring (the next query's group is prefetched); several groups use four, and deeper rings gain
    little once groups are large.

    Args:
        num_slots: Pool slots per query (selected pools plus the own-pool tail).
        kpool: Keys per pool.
        head_dim: Query / latent head dim.
        num_heads: Query heads.
        itemsize: Bytes per latent element.
        block_q: Queries per grid step.
        vmem_budget: Bytes available to the kernels' working set; ``None`` = 85% of the default scoped VMEM limit.

    Returns:
        ``(pools_per_step, num_groups, num_buffers)`` with ``pools_per_step * num_groups >= num_slots``.
    """
    num_slots = max(1, int(num_slots))
    kpool, head_dim, num_heads, itemsize = int(kpool), int(head_dim), int(num_heads), int(itemsize)
    budget = int(vmem_budget) if vmem_budget else int(0.85 * DEFAULT_VMEM_LIMIT)
    # q, dO and dq blocks double-buffered, plus the fp32 dq scratch.
    fixed = 6 * int(block_q) * num_heads * head_dim * itemsize + num_heads * head_dim * 4

    def group_bytes(g: int, nbuf: int) -> int:
        keys = g * kpool
        ring = nbuf * keys * head_dim * (itemsize + 4)  # latent ring + fp32 dlatent accumulator ring
        live = keys * head_dim * (itemsize + 8) + num_heads * keys * (16 + 2 * itemsize)  # keys, dk, p / ds, a
        return ring + live

    for ng in range(1, num_slots + 1):
        g = -(-num_slots // ng)
        nbuf = 2 if ng == 1 else 4
        if fixed + group_bytes(g, nbuf) <= budget:
            return g, -(-num_slots // g), nbuf
    return 1, num_slots, 4


@dataclasses.dataclass(frozen=True)
class _Static:
    kpool: int
    scale: float
    qb: int
    g_sz: int
    ng: int
    nbuf: int
    num_pools: int
    cores: int
    interpret: bool
    fused: bool = False
    bwd_vmem: int | None = None


def _fwd(query, latw, meta, st: _Static):
    return gather_fwd(
        query,
        latw,
        meta,
        kpool=st.kpool,
        scale=st.scale,
        qb=st.qb,
        g_sz=st.g_sz,
        ng=st.ng,
        nbuf=st.nbuf,
        num_pools=st.num_pools,
        interpret=st.interpret,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _core(query, latw, meta, st: _Static):
    return _fwd(query, latw, meta, st)[0]


def _core_fwd(query, latw, meta, st: _Static):
    out, lse = _fwd(query, latw, meta, st)
    # Remat names (identity at runtime). The primal output is the named value too, so a policy that keeps the
    # residual never has to re-run this kernel for downstream backward rules that read the output.
    out, lse = checkpoint_name((out, lse), RESIDUAL_NAME)
    meta = checkpoint_name(meta, TABLES_NAME)
    return out, (query, latw, meta, out, lse)


def _core_bwd(st: _Static, res, do):
    query, latw, meta, out, lse = res
    bsz, n, h, _ = query.shape
    delta = jnp.sum(do.astype(F32) * out.astype(F32), axis=-1)  # [B, N, H]
    delta = jnp.swapaxes(delta.reshape(bsz, n // st.qb, st.qb, h), 2, 3)  # kernel layout [B, ns, H, qb]
    if st.fused:
        rows = fused_rows(latw.reshape(bsz, -1, latw.shape[-1]), kpool=st.kpool, cores=st.cores)
        dq, rows = gather_bwd_fused(
            query,
            rows,
            meta,
            do.astype(query.dtype),
            lse,
            delta,
            kpool=st.kpool,
            scale=st.scale,
            qb=st.qb,
            g_sz=st.g_sz,
            cores=st.cores,
            vmem_limit_bytes=st.bwd_vmem,
            interpret=st.interpret,
        )
        dlat = jnp.sum(rows[:, :, : st.num_pools, 0], axis=0)  # [B, P, kpool, D]
        return dq, dlat.astype(latw.dtype).reshape(latw.shape), None
    dq, dlat = gather_bwd(
        query,
        latw,
        meta,
        do.astype(query.dtype),
        lse,
        delta,
        kpool=st.kpool,
        scale=st.scale,
        qb=st.qb,
        g_sz=st.g_sz,
        ng=st.ng,
        nbuf=st.nbuf,
        num_pools=st.num_pools,
        cores=st.cores,
        interpret=st.interpret,
    )
    return dq, dlat.astype(latw.dtype).reshape(latw.shape), None


_core.defvjp(_core_fwd, _core_bwd)


def dsa_gather_attention_pallas(
    query: Array,
    latent: Array,
    indices: Array,
    positions: Array | None = None,
    *,
    kpool: int,
    softmax_scale: float | None = None,
    include_tail: bool = True,
    block_q: int = 16,
    pools_per_step: int | None = None,
    num_buffers: int | None = None,
    num_cores: int | None = None,
    fused_backward: bool | None = None,
    interpret: bool | None = None,
) -> Array:
    """Unregistered entry point of the Pallas path (see :func:`dsa_gather_attention` for the arguments).

    Raises:
        ValueError: On unsupported dtypes / head dims (message contains "not supported"), bad options, or
            ``fused_backward=True`` where the fused backward cannot run (several pool groups per query, or its
            working set exceeds VMEM).
    """
    kpool = int(kpool)
    validate(query.shape, latent.shape, indices.shape, None if positions is None else positions.shape, kpool)
    bsz, n, _, d = query.shape
    s_len = latent.shape[1]
    k_sel = indices.shape[-1]
    pack = dtype_packing(latent.dtype)
    if pack == 0 or kpool % pack:
        raise ValueError(
            f"dsa_gather_attention: latent dtype {latent.dtype} with kpool={kpool} is not supported "
            f"(2- or 4-byte dtypes; kpool a multiple of the {pack or '?'} keys per 32-bit word)"
        )
    if d % 128:
        raise ValueError(f"dsa_gather_attention: head_dim {d} is not supported (multiple of 128)")
    qb = int(block_q)
    if (
        qb < 1
        or (pools_per_step is not None and int(pools_per_step) < 1)
        or (num_buffers is not None and int(num_buffers) < 2)
    ):
        raise ValueError(
            f"dsa_gather_attention: block_q ({qb}) and pools_per_step ({pools_per_step}) must be positive and "
            f"num_buffers ({num_buffers}) at least 2"
        )
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    scale = d**-0.5 if softmax_scale is None else float(softmax_scale)
    if positions is None:
        positions = default_positions(bsz, n)
    selected_ok, tail_len, own = selection_layout(indices, positions, kpool=kpool, include_tail=bool(include_tail))

    n_pad = _round_up(n, qb)
    s_pad = _round_up(s_len, kpool)
    num_pools = s_pad // kpool
    query = query.astype(latent.dtype)
    if n_pad != n:
        dn = n_pad - n
        query = jnp.pad(query, ((0, 0), (0, dn), (0, 0), (0, 0)))
        indices = jnp.pad(indices, ((0, 0), (0, dn), (0, 0)), constant_values=-1)
        selected_ok = jnp.pad(selected_ok, ((0, 0), (0, dn), (0, 0)))
        tail_len = jnp.pad(tail_len, ((0, 0), (0, dn)))
        own = jnp.pad(own, ((0, 0), (0, dn)))
    if s_pad != s_len:
        latent = jnp.pad(latent, ((0, 0), (0, s_pad - s_len), (0, 0)))

    num_slots = k_sel + 1  # selected pools plus the own-pool tail
    if pools_per_step is None:
        g_sz, _, _ = dsa_gather_group_plan(
            num_slots,
            kpool=kpool,
            head_dim=d,
            num_heads=query.shape[2],
            itemsize=jnp.dtype(latent.dtype).itemsize,
            block_q=qb,
        )
    else:
        lane_g = max(1, 128 // kpool)  # pools per 128 score lanes
        g_sz = max(1, min(int(pools_per_step), _round_up(num_slots, lane_g)))
    ng = -(-num_slots // g_sz)
    nbuf = int(num_buffers) if num_buffers is not None else (2 if ng == 1 else 4)
    fused, bwd_vmem = False, None
    if fused_backward is None or fused_backward:
        g_fused = _round_up(g_sz, FUSED_ALIGN)
        bwd_vmem = fused_bwd_vmem_limit()
        need = fused_bwd_vmem_bytes(
            g_fused,
            kpool=kpool,
            head_dim=d,
            num_heads=query.shape[2],
            itemsize=jnp.dtype(latent.dtype).itemsize,
            block_q=qb,
        )
        if ng != 1:
            reason = f"{ng} pool groups per query (the fused backward needs one)"
        elif need > 0.85 * bwd_vmem:
            reason = f"its working set (~{need >> 20} MiB) exceeds 85% of the {bwd_vmem >> 20} MiB VMEM limit"
        else:
            reason = None
        if reason is None:
            fused, g_sz = True, g_fused  # padding slots point at the trash pool: masked, they add exactly 0
        elif fused_backward:
            raise ValueError(f"dsa_gather_attention: fused_backward=True is not supported here: {reason}")
        else:
            bwd_vmem = None
    meta = build_meta(indices, selected_ok, tail_len, own, g_sz=g_sz, ng=ng, num_pools=num_pools)
    nblk = bsz * (n_pad // qb)
    cores = int(num_cores) if num_cores else (2 if nblk % 2 == 0 else 1)
    if cores < 1 or nblk % cores:
        raise ValueError(f"dsa_gather_attention: num_cores={cores} must divide the {nblk} query blocks")
    st = _Static(
        kpool=kpool,
        scale=float(scale),
        qb=qb,
        g_sz=g_sz,
        ng=ng,
        nbuf=nbuf,
        num_pools=num_pools,
        cores=cores,
        interpret=bool(interpret),
        fused=fused,
        bwd_vmem=bwd_vmem,
    )
    latw = latent.reshape(bsz, s_pad // pack, pack, d)
    out = _core(query, latw, meta, st)
    return out[:, :n]


@kernel_registry.register("dsa_gather_attention", Platform.PALLAS, Backend.TPU, priority=10)
@jaxtyping.jaxtyped(typechecker=beartype)
def dsa_gather_attention(
    query: Float[Array, "batch seq num_heads head_dim"],
    latent: Float[Array, "batch kv_seq head_dim"],
    indices: Int[Array, "batch seq num_selected"],
    positions: Int[Array, "batch seq"] | None = None,
    *,
    kpool: int,
    softmax_scale: float | None = None,
    include_tail: bool = True,
    block_q: int = 16,
    pools_per_step: int | None = None,
    num_buffers: int | None = None,
    num_cores: int | None = None,
    fused_backward: bool | None = None,
    interpret: bool | None = None,
) -> Float[Array, "batch seq num_heads head_dim"]:
    """Attention of each query over its selected pools plus its own-pool tail (Pallas TPU, differentiable).

    Registered under ``"dsa_gather_attention"`` for ``Platform.PALLAS`` / ``Backend.TPU`` (priority 10); the XLA
    implementation is the value and gradient reference. Each query DMA-gathers only its open pools, so the cost is
    ``O(N * K * kpool * H * D)`` regardless of the sequence length or how the selection is spread. ``jax.grad`` flows
    to ``query`` and ``latent``.

    Args:
        query: ``[B, N, H, D]`` queries (absorbed-MLA queries for GLM-5-style DSA), query-major. ``D % 128 == 0``.
        latent: ``[B, S, D]`` shared ``K == V`` latent (2- or 4-byte float dtype).
        indices: ``[B, N, K]`` selected pool ids (``-1`` = none); only complete causal pools are attended; valid
            ids must be distinct.
        positions: ``[B, N]`` absolute query positions (negative = invalid query, output 0); ``None`` = ``0..N-1``.
        kpool: Keys per pool (a multiple of the keys per 32-bit word: even for 2-byte latents).
        softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
        include_tail: Open each query's own incomplete pool ``kpool * (pos // kpool) .. pos``.
        block_q: Queries per grid step.
        pools_per_step: Pools fetched per DMA group; ``None`` (default) = :func:`dsa_gather_group_plan` (the fewest
            evenly split groups whose working set fits VMEM, usually one group per query).
        num_buffers: DMA ring depth (``num_buffers - 1`` groups in flight); ``None`` = 2 for one group per query,
            else 4.
        num_cores: Backward TensorCore split (``None``: 2 when the query-block count is even, else 1).
        fused_backward: Backward with one fused fp32 (dK accumulator, key) HBM row per pool, so two DMAs per
            (query, pool) instead of three (dlatent bit-identical to the three-DMA backward, dq within one bf16 ulp).
            ``None`` (default) = on when a query's pools form one group and the working set fits VMEM, ``True`` =
            required (``ValueError`` otherwise), ``False`` = the three-DMA backward.
        interpret: Pallas interpret mode (``None``: off-TPU only).

    Returns:
        ``[B, N, H, D]`` in ``latent``'s dtype; queries without any open key give exactly 0.
    """
    return dsa_gather_attention_pallas(
        query,
        latent,
        indices,
        positions,
        kpool=kpool,
        softmax_scale=softmax_scale,
        include_tail=include_tail,
        block_q=block_q,
        pools_per_step=pools_per_step,
        num_buffers=num_buffers,
        num_cores=num_cores,
        fused_backward=fused_backward,
        interpret=interpret,
    )
