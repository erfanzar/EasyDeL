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
from jaxtyping import Array, Bool, Float, Int

from ejkernel.kernels._xla.dsa_sparse_attention._tables import (
    build_q_tables,
    build_tables,
    validate_blocks,
    validate_options,
)

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_bwd import dsa_sparse_attention_pallas_bwd
from ._pallas_impl_bwd_fused import dsa_sparse_attention_pallas_bwd_fused
from ._pallas_impl_fwd import STAT_ROWS, dsa_sparse_attention_pallas_fwd


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


def _tables(pool_mask, tails, st: _Static):
    return build_tables(pool_mask, tails, kpool=st.kpool, block_q=st.block_q, block_kv=st.block_kv, mode=st.mode)


def _forward(query, latent, pool_mask, tails, st: _Static):
    bits_t, tails_t, count, kv_list = _tables(pool_mask, tails, st)
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


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _core(query, latent, pool_mask, tails, ste_pool, st: _Static):
    out, _, _ = _forward(query, latent, pool_mask, tails, st)
    _ = ste_pool  # value is identically 0 in the forward; only its cotangent matters
    return out


def _core_fwd(query, latent, pool_mask, tails, ste_pool, st: _Static):
    out, m, logl = _forward(query, latent, pool_mask, tails, st)
    _ = ste_pool
    # Only row 0 of the 8-row statistics carries information: save [B, H, S] and re-broadcast in the backward.
    return out, (query, latent, pool_mask, tails, out, m[:, :, 0, :], logl[:, :, 0, :])


def _stat_rows(x):
    return jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], STAT_ROWS, x.shape[-1]))


def _core_bwd(st: _Static, res, grad):
    query, latent, pool_mask, tails, out, m, logl = res
    num_pools = pool_mask.shape[-1]
    bits_t, tails_t, count, kv_list = _tables(pool_mask, tails, st)
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
            **common,
        )  # fmt: skip
    else:
        q_count, q_list = build_q_tables(count, kv_list)
        dq, d_latent, d_pool_t = dsa_sparse_attention_pallas_bwd(
            query, latent, bits_t, tails_t, count, kv_list, q_count, q_list, m, logl, delta, grad, lean=st.bwd_lean,
            **common,
        )  # fmt: skip
    d_ste_pool = jnp.transpose(d_pool_t[:, :num_pools, :], (0, 2, 1))
    return dq, d_latent, None, None, d_ste_pool


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

    Pads the sequence to the block grid (padded queries select nothing: output 0, finite statistics; padded keys
    are never selected) and the pool axis to ``S_pad / kpool``, runs the ``custom_vjp`` core and slices the output.
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
    pool_mask = pool_mask.astype(jnp.bool_)
    if ste_pool is None:
        ste_pool = jnp.zeros(pool_mask.shape, jnp.float32)
    seq = query.shape[2]
    step = _lcm(block_q, block_kv)
    seq_pad = -(-seq // step) * step
    num_pools = pool_mask.shape[-1]
    pools_pad = max(num_pools, seq_pad // kpool)
    if seq_pad != seq or pools_pad != num_pools:
        dq, dk, dp = seq_pad - seq, seq_pad - latent.shape[1], pools_pad - num_pools
        query = jnp.pad(query, ((0, 0), (0, 0), (0, dq), (0, 0)))
        latent = jnp.pad(latent, ((0, 0), (0, dk), (0, 0)))
        pool_mask = jnp.pad(pool_mask, ((0, 0), (0, dq), (0, dp)))
        ste_pool = jnp.pad(ste_pool, ((0, 0), (0, dq), (0, dp)))
        tails = jnp.pad(tails, ((0, 0), (0, dq), (0, 0)), constant_values=-1)
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
    )
    out = _core(query, latent, pool_mask, tails, ste_pool.astype(jnp.float32), st)
    return out[:, :, :seq, :]


@kernel_registry.register("dsa_sparse_attention", Platform.PALLAS, Backend.TPU, priority=10)
@jaxtyping.jaxtyped(typechecker=beartype)
def dsa_sparse_attention(
    query: Float[Array, "batch num_heads seq head_dim"],
    latent: Float[Array, "batch seq head_dim"],
    pool_mask: Bool[Array, "batch seq num_pools"],
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
            ``p`` = keys ``p*kpool .. p*kpool + kpool - 1``).
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
