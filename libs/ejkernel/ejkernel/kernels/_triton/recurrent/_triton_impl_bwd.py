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


"""Backward pass Triton kernels for recurrent linear attention.

This module provides gradient computation for the recurrent linear attention
forward pass. The backward kernel processes sequences in the reverse direction
of the forward pass and computes gradients for queries, keys, values, and
optional gating parameters.

Algorithm Overview:
------------------
The backward pass consists of two sequential phases within a single kernel:

Phase 1 - Compute dQ:
    Replay the forward hidden state evolution, using the chain rule to
    compute dq_t = sum(h_t * do_t, axis=value_dim) * scale

Phase 2 - Compute dK, dV:
    Process in reverse order (opposite to phase 1), accumulating a backward
    hidden state dh that flows from the output gradients back through the
    gated recurrence to produce dk and dv.

Gate Gradients:
    Gradients for gating parameters (g, gk, gv) are computed via cumulative
    sums over the dq*q - dk*k products, leveraging the structure of the
    exponential gating mechanism.

Functions:
- bwd_kernel: Triton JIT kernel for combined dQ/dK/dV backward computation
- bwd_triton_impl: Python wrapper that launches the kernel and computes gate gradients
"""

import jax
import triton
import triton.language as tl
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ejkernel.callib import cdiv, triton_call
from ejkernel.xla_utils.cumsum import chunk_global_cumsum


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] != 1,
        "STORE_FINAL_STATE": lambda args: args["ht"] != 1,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] != 1,
        "IS_VARLEN": lambda args: args["cu_seqlens"] != 1,
    }
)
@triton.jit
def bwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    h0,
    do,
    dht,
    cu_seqlens,
    softmax_scale,
    ht,
    dq,
    dk,
    dv,
    dh0,
    SEQUENCE: tl.constexpr,
    Z: tl.constexpr,
    HEAD: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    BLOCKSIZE_V: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Triton kernel for backward pass of recurrent linear attention.

    Computes gradients for queries, keys, values, and optional gating
    parameters by processing the sequence in reverse order from the
    forward pass.

    Args:
        q, k, v: Input tensor pointers from forward pass
        g, g_gamma, gk, gv: Optional gating tensor pointers
        h0: Initial hidden state pointer
        do: Gradient of output tensor
        dht: Gradient of final hidden state
        cu_seqlens: Cumulative sequence lengths for variable-length mode
        softmax_scale: Query scaling factor
        ht: Final hidden state from forward pass
        dq, dk, dv: Output gradient pointers for q, k, v
        dh0: Output gradient pointer for initial state
        SEQUENCE, Z, HEAD: Dimensions (sequence, batch, heads)
        DIM_K, DIM_V: Key and value dimensions
        BLOCKSIZE_K, BLOCKSIZE_V: Block sizes for tiling
        REVERSE: Whether forward pass was reversed
        USE_G, USE_G_GAMMA, USE_GK, USE_GV: Gating configuration
        USE_INITIAL_STATE: Whether initial state was used
        STORE_FINAL_STATE: Whether to store final state gradient
        STORE_INITIAL_STATE_GRADIENT: Whether to compute initial state gradient
        USE_FINAL_STATE_GRADIENT: Whether final state gradient is provided
        IS_VARLEN: Variable-length sequence mode
    """
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // HEAD, i_nh % HEAD
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        scope = SEQUENCE
        SEQUENCE = eos - bos
    else:
        bos, eos = i_n * SEQUENCE, i_n * SEQUENCE + SEQUENCE
        scope = Z * SEQUENCE
    o_k = i_k * BLOCKSIZE_K + tl.arange(0, BLOCKSIZE_K)
    o_v = i_v * BLOCKSIZE_V + tl.arange(0, BLOCKSIZE_V)

    p_k = k + (bos + ((SEQUENCE - 1) if REVERSE else 0)) * HEAD * DIM_K + i_h * DIM_K + o_k
    p_v = v + (bos + ((SEQUENCE - 1) if REVERSE else 0)) * HEAD * DIM_V + i_h * DIM_V + o_v
    p_do = do + (bos + ((SEQUENCE - 1) if REVERSE else 0)) * HEAD * DIM_V + i_h * DIM_V + o_v
    p_dq = dq + ((i_v * scope + bos) + ((SEQUENCE - 1) if REVERSE else 0)) * HEAD * DIM_K + i_h * DIM_K + o_k
    if USE_G:
        p_g = g + (bos + ((SEQUENCE - 1) if REVERSE else 0)) * HEAD + i_h
    if USE_GK:
        p_gk = gk + (bos + ((SEQUENCE - 1) if REVERSE else 0)) * HEAD * DIM_K + i_h * DIM_K + o_k
    if USE_GV:
        p_gv = gv + (bos + ((SEQUENCE - 1) if REVERSE else 0)) * HEAD * DIM_V + i_h * DIM_V + o_v
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)

    mask_k = o_k < DIM_K
    mask_v = o_v < DIM_V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BLOCKSIZE_K, BLOCKSIZE_V], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * DIM_K * DIM_V + o_k[:, None] * DIM_V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, SEQUENCE):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * tl.exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * tl.exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_h = b_h * tl.exp(b_gv[None, :])

        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        b_dq = tl.sum(b_dq, axis=1) * softmax_scale

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += (-1 if REVERSE else 1) * HEAD * DIM_K
        p_v += (-1 if REVERSE else 1) * HEAD * DIM_V
        p_do += (-1 if REVERSE else 1) * HEAD * DIM_V
        p_dq += (-1 if REVERSE else 1) * HEAD * DIM_K
        if USE_G:
            p_g += (-1 if REVERSE else 1) * HEAD
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * HEAD * DIM_K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * HEAD * DIM_V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * DIM_K * DIM_V + o_k[:, None] * DIM_V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

    tl.debug_barrier()

    p_q = q + (bos + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD * DIM_K + i_h * DIM_K + o_k
    p_k = k + (bos + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD * DIM_K + i_h * DIM_K + o_k
    p_v = v + (bos + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD * DIM_V + i_h * DIM_V + o_v
    p_do = do + (bos + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD * DIM_V + i_h * DIM_V + o_v
    p_dk = dk + ((i_v * scope + bos) + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD * DIM_K + i_h * DIM_K + o_k
    p_dv = dv + ((i_k * scope + bos) + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD * DIM_V + i_h * DIM_V + o_v
    if USE_G:
        p_g = g + (bos + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD + i_h
    if USE_GK:
        p_gk = gk + (bos + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD * DIM_K + i_h * DIM_K + o_k
    if USE_GV:
        p_gv = gv + (bos + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD * DIM_V + i_h * DIM_V + o_v

    b_dh = tl.zeros([BLOCKSIZE_K, BLOCKSIZE_V], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * DIM_K * DIM_V + o_k[:, None] * DIM_V + o_v[None, :]
        b_dh += tl.load(p_dht, mask=mask_h, other=0).to(tl.float32)

    for _ in range(SEQUENCE):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * softmax_scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_dh *= tl.exp(b_g)
        if USE_G_GAMMA:
            b_dh *= tl.exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_dh *= tl.exp(b_gk)[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_dh *= tl.exp(b_gv)[None, :]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)

        p_q += (1 if REVERSE else -1) * HEAD * DIM_K
        p_k += (1 if REVERSE else -1) * HEAD * DIM_K
        p_v += (1 if REVERSE else -1) * HEAD * DIM_V
        p_do += (1 if REVERSE else -1) * HEAD * DIM_V
        p_dk += (1 if REVERSE else -1) * HEAD * DIM_K
        p_dv += (1 if REVERSE else -1) * HEAD * DIM_V
        if USE_G:
            p_g += (1 if REVERSE else -1) * HEAD
        if USE_GK:
            p_gk += (1 if REVERSE else -1) * HEAD * DIM_K
        if USE_GV:
            p_gv += (1 if REVERSE else -1) * HEAD * DIM_V

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = dh0 + i_nh * DIM_K * DIM_V + o_k[:, None] * DIM_V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)


def bwd_triton_impl(
    q: Float[Array, "batch seq_len num_heads head_dim"],
    k: Float[Array, "batch seq_len num_heads head_dim"],
    v: Float[Array, "batch seq_len num_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "batch num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    o: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    do: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    dht: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch num_heads head_dim head_dim"] | None,
]:
    """Execute recurrent linear attention backward pass using Triton kernel.

    Computes gradients for queries, keys, values, and optional gating
    parameters. Uses cumulative sum operations for gate gradients when
    applicable. Processes sequence in reverse order from forward pass.

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads, head_dim]
        v: Value tensor [batch, seq_len, num_heads, head_dim]
        g: Optional GLA-style scalar gate per head. The kernel accesses this as
            ``[batch, seq_len, num_heads]`` (stride H per timestep). See note in
            ``fwd_triton_impl`` regarding the type-annotation discrepancy.
        g_gamma: Optional Lightning-style decay factor [batch, num_heads]
        gk: Optional per-key gate [batch, seq_len, num_heads, head_dim]
        gv: Optional per-value gate [batch, seq_len, num_heads, head_dim]
        o: Output from forward pass (needed for gv gradient computation)
        do: Gradient of output tensor
        dht: Optional gradient of final hidden state
        softmax_scale: Query scaling factor (default: 1.0)
        initial_state: Initial hidden state from forward pass
        reverse: Whether forward pass was reversed
        cu_seqlens: Cumulative sequence lengths for variable-length mode
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        tuple: (dq, dk, dv, dg, dgk, dgv, dh0) gradients for each input.
            Gate gradients (dg, dgk, dgv) are computed using cumulative sums
            over the sequence. dh0 is the initial state gradient.
    """
    Z, SEQUENCE, HEAD, DIM_K, DIM_V = *k.shape, v.shape[-1]
    N = Z if cu_seqlens is None else len(cu_seqlens) - 1

    BLOCKSIZE_K, BLOCKSIZE_V = min(DIM_K, int(block_k)), min(DIM_V, int(block_v))
    NumKBlocks, NumVBlocks = cdiv(DIM_K, BLOCKSIZE_K), cdiv(DIM_V, BLOCKSIZE_V)

    h0 = initial_state
    ht_shape = (N, HEAD, DIM_K, DIM_V) if (g is not None or gk is not None or gv is not None) else None
    dq_shape = (NumVBlocks, *q.shape)
    dk_shape = (NumVBlocks, *k.shape)
    dv_shape = (NumKBlocks, *v.shape)
    dh0_shape = h0.shape if h0 is not None else (1,)

    USE_INITIAL_STATE = ht_shape is not None
    STORE_INITIAL_STATE_GRADIENT = h0 is not None
    if ht_shape is None:
        ht_shape = (1,)
    dg, dgk, dgv = None, None, None

    if g is not None:
        dg = jnp.empty((NumKBlocks * NumVBlocks, *g.shape), dtype=jnp.float32)
    if gk is not None:
        dgk = jnp.empty((NumVBlocks, *gk.shape), dtype=jnp.float32)
    if gv is not None:
        dgv = jnp.empty((NumKBlocks, *gv.shape), dtype=jnp.float32)

    grid = (NumVBlocks, NumKBlocks, N * HEAD)
    metaparams = dict(
        Z=Z,
        SEQUENCE=SEQUENCE,
        HEAD=HEAD,
        DIM_K=DIM_K,
        DIM_V=DIM_V,
        BLOCKSIZE_K=BLOCKSIZE_K,
        BLOCKSIZE_V=BLOCKSIZE_V,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
        STORE_INITIAL_STATE_GRADIENT=STORE_INITIAL_STATE_GRADIENT,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
    )
    ht, dq, dk, dv, dh0 = triton_call(
        q,
        k,
        v,
        g if g is not None else 1,
        g_gamma if g_gamma is not None else 1,
        gk if gk is not None else 1,
        gv if gv is not None else 1,
        h0 if h0 is not None else 1,
        do if do is not None else 1,
        dht if dht is not None else 1,
        cu_seqlens if cu_seqlens is not None else 1,
        softmax_scale if softmax_scale is not None else 1,
        kernel=bwd_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(ht_shape, jnp.float32),
            jax.ShapeDtypeStruct(dq_shape, jnp.float32),
            jax.ShapeDtypeStruct(dk_shape, jnp.float32),
            jax.ShapeDtypeStruct(dv_shape, jnp.float32),
            jax.ShapeDtypeStruct(dh0_shape, jnp.float32),
        ],
        name="ejkernel::triton::recurrent_bwd",
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )

    dq = jnp.sum(dq, axis=0)
    dk = jnp.sum(dk, axis=0)
    dv = jnp.sum(dv, axis=0)
    if g is not None:
        dg = chunk_global_cumsum(
            (dq * q.astype("f4") - dk * k.astype("f4")).sum(-1),
            reverse=not reverse,
            cu_seqlens=cu_seqlens,
        )
        if dht is not None:
            dg += jnp.expand_dims((ht * dht).sum((2, 3)), 1)
    if gk is not None:
        dgk = chunk_global_cumsum(
            dq * q.astype("f4") - dk * k.astype("f4"),
            reverse=not reverse,
            cu_seqlens=cu_seqlens,
        )
        if dht is not None:
            dgk += jnp.expand_dims((ht * dht).sum(3), 1)
    if gv is not None:
        dgv = chunk_global_cumsum(
            do.astype("f4") * o.astype("f4") - dv * v.astype("f4"),
            reverse=not reverse,
            cu_seqlens=cu_seqlens,
        )
        if dht is not None:
            dgv += jnp.expand_dims((ht * dht).sum(2), 1)
    return dq, dk, dv, dg, dgk, dgv, dh0
