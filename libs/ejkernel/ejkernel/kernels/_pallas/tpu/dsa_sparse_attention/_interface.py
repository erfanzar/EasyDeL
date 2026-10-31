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

"""Differentiable pool-bitmap block-sparse DSA attention (Pallas TPU): sparse forward + sparse backward.

Forward: :func:`dsa_sparse_attention_pallas_fwd` over the tables of ``build_tables`` (no ``O(S^2)`` tensor).

Backward:

* ``"fused"`` (default, measured fastest): one pass over the visited pairs (``_pallas_impl_bwd_fused``).
* ``"two_pass"``: a q-block-major dQ + pool-cotangent pass and a kv-block-major dKV pass over the transposed lists of
  ``build_q_tables`` (``_pallas_impl_bwd``).

Both consume the forward's saved ``(m, log l)`` and ``delta = rowsum(dO * O)``; no per-key tensor exists at any
point. The pool cotangent comes back as ``d_pool^T [B, P_pad, S]`` fp32 and is transposed once.

``ste_pool`` is the indexer's straight-through pool term: a ``[B, S, P]`` fp32 primal the forward never reads (its
value is exactly 0 in production) whose cotangent is ``d_score`` for the selected pools.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from ejkernel.kernels._xla.dsa_sparse_attention._tables import (
    WORD,
    as_pool_bits,
    build_q_tables,
    build_tables,
    validate_blocks,
    validate_options,
)

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_bwd import dsa_sparse_attention_pallas_bwd
from ._pallas_impl_bwd_fused import dsa_sparse_attention_pallas_bwd_fused
from ._pallas_impl_fwd import STAT_ROWS, dsa_sparse_attention_pallas_fwd

#: ``checkpoint_name`` of the forward residuals the backward reads: the output and the (m, log l) statistics.
RESIDUAL_NAME = "dsa_sparse_attention_residuals"
#: ``checkpoint_name`` of the block tables (packed bits, tails, counts, kv lists) the kernels stream.
TABLES_NAME = "dsa_sparse_attention_tables"


@dataclasses.dataclass(frozen=True)
class _Static:
    """Hashable static options of one call (the ``custom_vjp`` non-differentiable argument)."""

    kpool: int
    block_q: int
    block_kv: int
    softmax_scale: float
    n_tails: int
    interpret: bool | None
    mode: str
    backward: str
    fwd_nt: bool
    fwd_lean: bool
    bwd_lean: bool
    pool_sum: str
    stale_tau: float | None
    fused_groups: int | None
    with_pool: bool = True
    num_pools: int = 0


def _forward(query, latent, bits_t, tails_t, count, kv_list, st: _Static):
    return dsa_sparse_attention_pallas_fwd(
        query,
        latent,
        bits_t,
        tails_t,
        count,
        kv_list,
        kpool=st.kpool,
        block_q=st.block_q,
        block_kv=st.block_kv,
        softmax_scale=st.softmax_scale,
        n_tails=st.n_tails,
        interpret=st.interpret,
        nt_dots=st.fwd_nt,
        lean=st.fwd_lean,
        stale_tau=st.stale_tau,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def _core(query, latent, bits_t, tails_t, count, kv_list, ste_pool, st: _Static):
    out, _, _ = _forward(query, latent, bits_t, tails_t, count, kv_list, st)
    _ = ste_pool  # value is identically 0 in the forward; only its cotangent matters
    return out


def _core_fwd(query, latent, bits_t, tails_t, count, kv_list, ste_pool, st: _Static):
    out, m, logl = _forward(query, latent, bits_t, tails_t, count, kv_list, st)
    _ = ste_pool
    # The residuals are the kernels' own tables (packed words, not a [B, S, P] mask), so the backward neither keeps
    # a per-pool tensor alive nor rebuilds the tables. Only row 0 of the 8-row statistics carries information:
    # save [B, H, S] and re-broadcast in the backward.
    # Remat names (identity at runtime): a checkpoint policy that saves or offloads RESIDUAL_NAME / TABLES_NAME lets
    # the backward skip re-running this forward kernel and, for the tables, the selection that produced them.
    res_out, res_m, res_l = checkpoint_name((out, m[:, :, 0, :], logl[:, :, 0, :]), RESIDUAL_NAME)
    tables = checkpoint_name((bits_t, tails_t, count, kv_list), TABLES_NAME)
    # The primal output is the NAMED value too: downstream ops' backward (e.g. the W_v projection) needs it, and an
    # unnamed copy would force the remat to re-run this kernel even when the policy keeps the residual.
    return res_out, (query, latent, *tables, res_out, res_m, res_l)


def _stat_rows(x):
    return jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], STAT_ROWS, x.shape[-1]))


def _core_bwd(st: _Static, res, grad):
    query, latent, bits_t, tails_t, count, kv_list, out, m, logl = res
    delta = _stat_rows(jnp.sum(grad.astype(jnp.float32) * out.astype(jnp.float32), axis=-1))
    m, logl = _stat_rows(m), _stat_rows(logl)
    common = dict(
        kpool=st.kpool,
        block_q=st.block_q,
        block_kv=st.block_kv,
        softmax_scale=st.softmax_scale,
        n_tails=st.n_tails,
        interpret=st.interpret,
        pool_sum_how=st.pool_sum,
    )
    grad = grad.astype(query.dtype)
    if st.backward == "fused":
        dq, d_latent, d_pool_t = dsa_sparse_attention_pallas_bwd_fused(
            query, latent, bits_t, tails_t, count, kv_list, m, logl, delta, grad, fused_groups=st.fused_groups,
            with_pool=st.with_pool, **common,
        )  # fmt: skip
    else:
        q_count, q_list = build_q_tables(count, kv_list)
        dq, d_latent, d_pool_t = dsa_sparse_attention_pallas_bwd(
            query, latent, bits_t, tails_t, count, kv_list, q_count, q_list, m, logl, delta, grad, lean=st.bwd_lean,
            **common,
        )  # fmt: skip
    if not st.with_pool:
        # No straight-through operand (ste_pool=None): no pool cotangent is returned.
        return dq, d_latent, None, None, None, None, None
    d_ste_pool = jnp.transpose(d_pool_t[:, : st.num_pools, :], (0, 2, 1))
    return dq, d_latent, None, None, None, None, d_ste_pool


_core.defvjp(_core_fwd, _core_bwd)


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def dsa_sparse_attention_pallas(
    query: Array,
    latent: Array,
    pool_mask: Array,
    tails: Array | None = None,
    ste_pool: Array | None = None,
    *,
    kpool: int,
    softmax_scale: float | None = None,
    mode: str = "sparse",
    block_q: int = 256,
    block_kv: int = 512,
    backward: str = "fused",
    fwd_nt: bool = False,
    fwd_lean: bool = False,
    bwd_lean: bool = False,
    pool_sum: str = "reshape",
    stale_tau: float | None = None,
    fused_groups: int | None = None,
    interpret: bool | None = None,
) -> Array:
    """Unregistered entry point of the Pallas path (see :func:`dsa_sparse_attention` for the arguments).

    Packs a bool selection into words once, pads the sequence to the block grid (padded queries select nothing:
    output 0, finite statistics; padded keys are never selected) and the word / pool axes to ``S_pad / kpool`` pools,
    builds the block tables outside the ``custom_vjp`` core (its residuals are then the tables, never a ``[B, S, P]``
    mask), runs the core and slices the output.
    """
    mode = validate_options(
        mode=mode, backward=backward, pool_sum=pool_sum, stale_tau=stale_tau, fused_groups=fused_groups
    )
    block_q, block_kv, kpool = int(block_q), int(block_kv), int(kpool)
    validate_blocks(block_q, block_kv, kpool)
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    n_tails = 0 if tails is None else int(tails.shape[-1])
    if tails is None:
        tails = jnp.full((*pool_mask.shape[:2], 1), -1, jnp.int32)
    tails = tails.astype(jnp.int32)
    bits = as_pool_bits(pool_mask)  # [B, S, W] int32: the only form of the selection the kernels read
    # ste_pool=None means "no indexer gradient": skip the [B, S, P] fp32 operand and the [B, P, S] fp32 pool
    # cotangent of the backward entirely (both are O(S^2 / kpool) and dominate memory at long context).
    with_pool = ste_pool is not None
    seq = query.shape[2]
    step = _lcm(block_q, block_kv)
    seq_pad = -(-seq // step) * step
    dq, dk = seq_pad - seq, seq_pad - latent.shape[1]
    if dq or dk:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, dq), (0, 0)))
        latent = jnp.pad(latent, ((0, 0), (0, dk), (0, 0)))
        tails = jnp.pad(tails, ((0, 0), (0, dq), (0, 0)), constant_values=-1)
    words_pad = max(bits.shape[-1], seq_pad // (WORD * kpool))  # exact: block_kv % (32 * kpool) == 0
    if dq or words_pad != bits.shape[-1]:
        bits = jnp.pad(bits, ((0, 0), (0, dq), (0, words_pad - bits.shape[-1])))
    num_pools = 0
    if with_pool:
        ste_pool = ste_pool.astype(jnp.float32)
        pools_pad = max(ste_pool.shape[-1], seq_pad // kpool)
        if dq or pools_pad != ste_pool.shape[-1]:
            ste_pool = jnp.pad(ste_pool, ((0, 0), (0, dq), (0, pools_pad - ste_pool.shape[-1])))
        num_pools = pools_pad
    bits_t, tails_t, count, kv_list = build_tables(
        bits, tails, kpool=kpool, block_q=block_q, block_kv=block_kv, mode=mode
    )
    st = _Static(
        kpool=kpool,
        block_q=block_q,
        block_kv=block_kv,
        softmax_scale=scale,
        n_tails=n_tails,
        interpret=interpret,
        mode=mode,
        backward=backward,
        fwd_nt=bool(fwd_nt),
        fwd_lean=bool(fwd_lean),
        bwd_lean=bool(bwd_lean),
        pool_sum=pool_sum,
        stale_tau=None if stale_tau is None else float(stale_tau),
        fused_groups=fused_groups,
        with_pool=with_pool,
        num_pools=num_pools,
    )
    out = _core(query, latent, bits_t, tails_t, count, kv_list, ste_pool if with_pool else None, st)
    return out[:, :, :seq, :]


@kernel_registry.register("dsa_sparse_attention", Platform.PALLAS, Backend.TPU, priority=10)
@jaxtyping.jaxtyped(typechecker=beartype)
def dsa_sparse_attention(
    query: Float[Array, "batch num_heads seq head_dim"],
    latent: Float[Array, "batch seq head_dim"],
    pool_mask: Bool[Array, "batch seq num_pools"] | Int[Array, "batch seq num_words"],
    tails: Int[Array, "batch seq num_tails"] | None = None,
    ste_pool: Float[Array, "batch seq num_pools"] | None = None,
    *,
    kpool: int,
    softmax_scale: float | None = None,
    mode: str = "sparse",
    block_q: int = 256,
    block_kv: int = 512,
    backward: str = "fused",
    fwd_nt: bool = False,
    fwd_lean: bool = False,
    bwd_lean: bool = False,
    pool_sum: str = "reshape",
    stale_tau: float | None = None,
    fused_groups: int | None = None,
    interpret: bool | None = None,
) -> Float[Array, "batch num_heads seq head_dim"]:
    """Block-sparse DSA attention over the indexer's pool selection (Pallas TPU, differentiable).

    Registered under ``"dsa_sparse_attention"`` for ``Platform.PALLAS`` / ``Backend.TPU`` (priority 10); the XLA
    implementation is the value and gradient reference. ``jax.grad`` flows to ``query``, ``latent`` and ``ste_pool``.

    Args:
        query: ``[B, H, S, D]`` queries (absorbed-MLA queries for GLM-5-style DSA).
        latent: ``[B, S, D]`` shared ``K == V`` latent.
        pool_mask: ``[B, S, P]`` bool selected pools (causal and same-document by the indexer's construction; pool
            ``p`` = keys ``p*kpool .. p*kpool + kpool - 1``), or the same selection as ``[B, S, W]`` int32 packed
            words (bit ``p % 32`` of word ``p // 32`` = pool ``p``). Packed words are what the kernels read; passing
            them keeps every ``[B, S, P]`` tensor out of the graph.
        tails: ``[B, S, T]`` int32 tail keys (``-1`` = none), ``T <= 8``, or ``None``.
        ste_pool: ``[B, S, P]`` fp32 pooled straight-through term (value 0); ``None`` = no indexer gradient.
        kpool: Keys per pool.
        softmax_scale: Logit scale; ``None`` = ``D ** -0.5``.
        mode: Visited (q-block, kv-block) pairs: ``"sparse"`` (the active pairs), ``"dense"`` (every causal pair:
            constant work, bitwise ``"sparse"``) or ``"budget:B"`` (the ``B`` heaviest kv-blocks per q-block, the
            rest dropped: a numerics change).
        block_q: Query tile (multiple of 128). Default 256.
        block_kv: Key tile (multiple of 128 and ``32 * kpool``). Default 512: bitwise the 256 tiling and ~8% faster
            on v5p.
        backward: ``"fused"`` (single pass, default; -22% .. -36% vs two passes on v5p) or ``"two_pass"``.
        fwd_nt: Forward ``K @ q^T`` as NT dots without the q^T scratch (bit-identical; default off).
        fwd_lean: Forward with the ``-1e30`` running-max seed and slab-built masks (bitwise the base forward).
        bwd_lean: Two-pass backward with ``m + log l`` folded once and the dKV scale applied once (fp32
            reassociation).
        pool_sum: Pool-cotangent reduction, ``"reshape"`` (default) or ``"matmul"`` (0/1 fp32 matmul).
        stale_tau: Speculative stale-max forward threshold in nats; ``None`` (default) = off. Measured 1.7x slower on
            v5p at block density 1, so it stays opt-in.
        fused_groups: Fused-backward core groups, ``None`` (2 when the q-block count is even, else 1), 1 or 2.
        interpret: Pallas interpret mode (``None``: off-TPU only).

    Returns:
        ``[B, H, S, D]`` in ``latent``'s dtype; rows without any open key are exactly 0.
    """
    return dsa_sparse_attention_pallas(
        query,
        latent,
        pool_mask,
        tails,
        ste_pool,
        kpool=kpool,
        softmax_scale=softmax_scale,
        mode=mode,
        block_q=block_q,
        block_kv=block_kv,
        backward=backward,
        fwd_nt=fwd_nt,
        fwd_lean=fwd_lean,
        bwd_lean=bwd_lean,
        pool_sum=pool_sum,
        stale_tau=stale_tau,
        fused_groups=fused_groups,
        interpret=interpret,
    )
