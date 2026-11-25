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
    HEAD_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    BLOCKSIZE_V: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    G_IS_4D: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    G_GAMMA_PER_SEQ: tl.constexpr,
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
    parameters by replaying the forward recurrence for ``dq`` and sweeping
    in reverse for ``dk``/``dv``/``dh0``.

    Args:
        q, k, v: Input tensor pointers from forward pass.
        g, g_gamma, gk, gv: Optional gating tensor pointers.
        h0: Initial hidden state pointer.
        do: Gradient of output tensor.
        dht: Gradient of final hidden state.
        cu_seqlens: Cumulative sequence lengths for variable-length mode.
        softmax_scale: Query scaling factor.
        ht: Final hidden state output buffer (written only when ``STORE_FINAL_STATE``).
        dq, dk, dv: Output gradient pointers for q, k, v (per query head).
        dh0: Output gradient pointer for initial state.
        SEQUENCE, Z, HEAD, HEAD_KV, GROUP_SIZE: Batch, sequence, and head dims.
        DIM_K, DIM_V: Key and value dimensions.
        BLOCKSIZE_K, BLOCKSIZE_V: Power-of-2 block sizes for tiling.
        REVERSE: Whether forward pass was reversed.
        USE_G, G_IS_4D, USE_G_GAMMA, G_GAMMA_PER_SEQ, USE_GK, USE_GV: Gating flags.
        USE_INITIAL_STATE: Whether initial state was used.
        STORE_FINAL_STATE: Whether to store final state into ``ht``.
        STORE_INITIAL_STATE_GRADIENT: Whether to store initial state gradient.
        USE_FINAL_STATE_GRADIENT: Whether final state gradient is provided.
        IS_VARLEN: Variable-length sequence mode.
    """
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // HEAD, i_nh % HEAD
    i_kv_h = i_h // GROUP_SIZE
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        scope = SEQUENCE
        SEQUENCE = eos - bos
    else:
        bos, eos = i_n * SEQUENCE, i_n * SEQUENCE + SEQUENCE
        scope = Z * SEQUENCE
    o_k = i_k * BLOCKSIZE_K + tl.arange(0, BLOCKSIZE_K)
    o_v = i_v * BLOCKSIZE_V + tl.arange(0, BLOCKSIZE_V)

    t_fwd_start = bos + ((SEQUENCE - 1) if REVERSE else 0)
    p_k = k + t_fwd_start * HEAD_KV * DIM_K + i_kv_h * DIM_K + o_k
    p_v = v + t_fwd_start * HEAD_KV * DIM_V + i_kv_h * DIM_V + o_v
    p_do = do + t_fwd_start * HEAD * DIM_V + i_h * DIM_V + o_v
    p_dq = dq + ((i_v * scope + bos) + ((SEQUENCE - 1) if REVERSE else 0)) * HEAD * DIM_K + i_h * DIM_K + o_k
    if USE_G:
        if G_IS_4D:
            p_g = g + t_fwd_start * HEAD * DIM_K + i_h * DIM_K + o_k
        else:
            p_g = g + t_fwd_start * HEAD + i_h
    if USE_GK:
        p_gk = gk + t_fwd_start * HEAD * DIM_K + i_h * DIM_K + o_k
    if USE_GV:
        p_gv = gv + t_fwd_start * HEAD * DIM_V + i_h * DIM_V + o_v
    if USE_G_GAMMA:
        g_gamma_offset = i_nh if G_GAMMA_PER_SEQ else i_h
        b_g_gamma_decay = tl.exp(tl.load(g_gamma + g_gamma_offset).to(tl.float32))

    mask_k = o_k < DIM_K
    mask_v = o_v < DIM_V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BLOCKSIZE_K, BLOCKSIZE_V], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * DIM_K * DIM_V + o_k[:, None] * DIM_V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    fwd_step = -1 if REVERSE else 1
    for _ in range(0, SEQUENCE):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if USE_G:
            if G_IS_4D:
                b_g = tl.load(p_g, mask=mask_k, other=0).to(tl.float32)
                b_h = b_h * tl.exp(b_g[:, None])
            else:
                b_g = tl.load(p_g).to(tl.float32)
                b_h = b_h * tl.exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * b_g_gamma_decay
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

        p_k += fwd_step * HEAD_KV * DIM_K
        p_v += fwd_step * HEAD_KV * DIM_V
        p_do += fwd_step * HEAD * DIM_V
        p_dq += fwd_step * HEAD * DIM_K
        if USE_G:
            p_g += fwd_step * (HEAD * DIM_K if G_IS_4D else HEAD)
        if USE_GK:
            p_gk += fwd_step * HEAD * DIM_K
        if USE_GV:
            p_gv += fwd_step * HEAD * DIM_V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * DIM_K * DIM_V + o_k[:, None] * DIM_V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

    t_bwd_start = bos + ((SEQUENCE - 1) if not REVERSE else 0)
    p_q = q + t_bwd_start * HEAD * DIM_K + i_h * DIM_K + o_k
    p_k = k + t_bwd_start * HEAD_KV * DIM_K + i_kv_h * DIM_K + o_k
    p_v = v + t_bwd_start * HEAD_KV * DIM_V + i_kv_h * DIM_V + o_v
    p_do = do + t_bwd_start * HEAD * DIM_V + i_h * DIM_V + o_v
    p_dk = dk + ((i_v * scope + bos) + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD * DIM_K + i_h * DIM_K + o_k
    p_dv = dv + ((i_k * scope + bos) + ((SEQUENCE - 1) if not REVERSE else 0)) * HEAD * DIM_V + i_h * DIM_V + o_v
    if USE_G:
        if G_IS_4D:
            p_g = g + t_bwd_start * HEAD * DIM_K + i_h * DIM_K + o_k
        else:
            p_g = g + t_bwd_start * HEAD + i_h
    if USE_GK:
        p_gk = gk + t_bwd_start * HEAD * DIM_K + i_h * DIM_K + o_k
    if USE_GV:
        p_gv = gv + t_bwd_start * HEAD * DIM_V + i_h * DIM_V + o_v

    b_dh = tl.zeros([BLOCKSIZE_K, BLOCKSIZE_V], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * DIM_K * DIM_V + o_k[:, None] * DIM_V + o_v[None, :]
        b_dh += tl.load(p_dht, mask=mask_h, other=0).to(tl.float32)

    bwd_step = 1 if REVERSE else -1
    for _ in range(SEQUENCE):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * softmax_scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)
        if USE_G:
            if G_IS_4D:
                b_g = tl.load(p_g, mask=mask_k, other=0).to(tl.float32)
                b_dh *= tl.exp(b_g)[:, None]
            else:
                b_g = tl.load(p_g).to(tl.float32)
                b_dh *= tl.exp(b_g)
        if USE_G_GAMMA:
            b_dh *= b_g_gamma_decay
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_dh *= tl.exp(b_gk)[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            b_dh *= tl.exp(b_gv)[None, :]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)

        p_q += bwd_step * HEAD * DIM_K
        p_k += bwd_step * HEAD_KV * DIM_K
        p_v += bwd_step * HEAD_KV * DIM_V
        p_do += bwd_step * HEAD * DIM_V
        p_dk += bwd_step * HEAD * DIM_K
        p_dv += bwd_step * HEAD * DIM_V
        if USE_G:
            p_g += bwd_step * (HEAD * DIM_K if G_IS_4D else HEAD)
        if USE_GK:
            p_gk += bwd_step * HEAD * DIM_K
        if USE_GV:
            p_gv += bwd_step * HEAD * DIM_V

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = dh0 + i_nh * DIM_K * DIM_V + o_k[:, None] * DIM_V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)


def _resolve_g_gamma_per_seq(g_gamma: Array | None, num_seqs: int, num_heads: int) -> bool:
    """Validate ``g_gamma`` shape and return True when indexed per sequence ``[N, H]``."""
    if g_gamma is None:
        return False
    if g_gamma.ndim == 1:
        if g_gamma.shape != (num_heads,):
            raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({num_heads},) or ({num_seqs}, {num_heads}).")
        return False
    if g_gamma.ndim == 2:
        if g_gamma.shape[1] != num_heads:
            raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({num_heads},) or ({num_seqs}, {num_heads}).")
        if g_gamma.shape[0] == 1:
            return False
        if g_gamma.shape[0] == num_seqs:
            return True
        raise ValueError(f"g_gamma.shape={g_gamma.shape} must be ({num_heads},) or ({num_seqs}, {num_heads}).")
    raise ValueError(f"g_gamma.ndim={g_gamma.ndim} must be 1 or 2.")


def bwd_triton_impl(
    q: Float[Array, "batch seq_len num_heads head_dim"],
    k: Float[Array, "batch seq_len num_kv_heads head_dim"],
    v: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads ..."] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None,
    o: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None,
    do: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None,
    dht: Float[Array, "... num_heads head_dim v_head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_kv_heads head_dim"] | None,
    Float[Array, "batch seq_len num_kv_heads v_head_dim"] | None,
    Float[Array, "batch seq_len num_heads ..."] | None,
    Float[Array, "batch seq_len num_heads head_dim"] | None,
    Float[Array, "batch seq_len num_heads v_head_dim"] | None,
    Float[Array, "... num_heads head_dim v_head_dim"] | None,
]:
    """Execute recurrent linear attention backward pass using Triton kernel.

    Computes gradients for queries, keys, values, and optional gating
    parameters. Uses cumulative sum operations for gate gradients when
    applicable. Processes sequence in reverse order from forward pass.

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim].
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim].
        v: Value tensor [batch, seq_len, num_kv_heads, v_head_dim].
        g: Optional gate tensor, either 3-D ``[batch, seq_len, num_heads]`` or
            4-D ``[batch, seq_len, num_heads, head_dim]``.
        g_gamma: Optional Lightning-style decay factor of shape ``[num_heads]``,
            ``[1, num_heads]``, or ``[batch_or_num_seqs, num_heads]``.
        gk: Optional per-key gate [batch, seq_len, num_heads, head_dim].
        gv: Optional per-value gate [batch, seq_len, num_heads, v_head_dim].
        o: Output from forward pass (needed for gv gradient computation).
        do: Gradient of output tensor.
        dht: Optional gradient of final hidden state.
        softmax_scale: Query scaling factor (defaults to ``1 / sqrt(head_dim)``).
        initial_state: Initial hidden state from forward pass.
        reverse: Whether forward pass was reversed.
        cu_seqlens: Cumulative sequence lengths for variable-length mode.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        tuple: (dq, dk, dv, dg, dgk, dgv, dh0) gradients cast to their input
        dtypes, with ``dh0=None`` when ``initial_state is None``.
    """
    Z, SEQUENCE, HEAD, DIM_K = q.shape
    HEAD_KV = k.shape[2]
    DIM_V = v.shape[-1]
    if HEAD % HEAD_KV != 0 or v.shape[2] != HEAD_KV:
        raise ValueError(
            f"Incompatible GQA head counts: query has {HEAD} heads, key has {HEAD_KV}, value has {v.shape[2]}."
        )
    group_size = HEAD // HEAD_KV
    N = Z if cu_seqlens is None else int(cu_seqlens.shape[0]) - 1
    if g is not None and g.ndim not in (3, 4):
        raise ValueError(f"g must be 3-D [B, T, H] or 4-D [B, T, H, K], got shape {g.shape}.")
    g_is_4d = (g is not None) and (g.ndim == 4)
    g_gamma_per_seq = _resolve_g_gamma_per_seq(g_gamma, N, HEAD)
    if softmax_scale is None:
        softmax_scale = float(DIM_K**-0.5)

    BLOCKSIZE_K = triton.next_power_of_2(min(DIM_K, int(block_k)))
    BLOCKSIZE_V = triton.next_power_of_2(min(DIM_V, int(block_v)))
    NumKBlocks, NumVBlocks = cdiv(DIM_K, BLOCKSIZE_K), cdiv(DIM_V, BLOCKSIZE_V)

    h0 = initial_state
    store_final_state = (dht is not None) and (g is not None or gk is not None or gv is not None)
    use_initial_state = h0 is not None
    store_initial_state_gradient = h0 is not None
    ht_shape = (N, HEAD, DIM_K, DIM_V) if store_final_state else (1,)
    dq_shape = (NumVBlocks, Z, SEQUENCE, HEAD, DIM_K)
    dk_shape = (NumVBlocks, Z, SEQUENCE, HEAD, DIM_K)
    dv_shape = (NumKBlocks, Z, SEQUENCE, HEAD, DIM_V)
    dh0_shape = h0.shape if h0 is not None else (1,)

    dg, dgk, dgv = None, None, None

    grid = (NumVBlocks, NumKBlocks, N * HEAD)
    metaparams = dict(
        Z=Z,
        SEQUENCE=SEQUENCE,
        HEAD=HEAD,
        HEAD_KV=HEAD_KV,
        GROUP_SIZE=group_size,
        DIM_K=DIM_K,
        DIM_V=DIM_V,
        BLOCKSIZE_K=BLOCKSIZE_K,
        BLOCKSIZE_V=BLOCKSIZE_V,
        USE_G=g is not None,
        G_IS_4D=g_is_4d,
        USE_G_GAMMA=g_gamma is not None,
        G_GAMMA_PER_SEQ=g_gamma_per_seq,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        REVERSE=reverse,
        USE_INITIAL_STATE=use_initial_state,
        STORE_FINAL_STATE=store_final_state,
        STORE_INITIAL_STATE_GRADIENT=store_initial_state_gradient,
        USE_FINAL_STATE_GRADIENT=dht is not None,
        IS_VARLEN=cu_seqlens is not None,
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
        float(softmax_scale),
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

    dq_f32 = jnp.sum(dq, axis=0)
    dk_per_head = jnp.sum(dk, axis=0)
    dv_per_head = jnp.sum(dv, axis=0)

    def _broadcast_ht_dht_term(term: Array) -> Array:
        if cu_seqlens is None:
            return jnp.expand_dims(term, 1)
        pos = jnp.arange(SEQUENCE, dtype=cu_seqlens.dtype)
        seq_ids = jnp.clip(jnp.searchsorted(cu_seqlens[1:], pos, side="right"), 0, max(N - 1, 0))
        valid = (pos >= cu_seqlens[0]) & (pos < cu_seqlens[-1])
        gathered = term[seq_ids]
        valid_mask = valid.reshape((SEQUENCE,) + (1,) * (gathered.ndim - 1))
        return jnp.expand_dims(jnp.where(valid_mask, gathered, 0), 0)

    k_for_gate = jnp.repeat(k, group_size, axis=2) if group_size > 1 else k
    v_for_gate = jnp.repeat(v, group_size, axis=2) if group_size > 1 else v
    dht_f32 = dht.astype(jnp.float32) if dht is not None else None

    if g is not None:
        if g_is_4d:
            dg_f32 = chunk_global_cumsum(
                dq_f32 * q.astype(jnp.float32) - dk_per_head * k_for_gate.astype(jnp.float32),
                reverse=not reverse,
                cu_seqlens=cu_seqlens,
            )
            if dht_f32 is not None:
                dg_f32 = dg_f32 + _broadcast_ht_dht_term((ht * dht_f32).sum(3))
        else:
            dg_f32 = chunk_global_cumsum(
                (dq_f32 * q.astype(jnp.float32) - dk_per_head * k_for_gate.astype(jnp.float32)).sum(-1),
                reverse=not reverse,
                cu_seqlens=cu_seqlens,
            )
            if dht_f32 is not None:
                dg_f32 = dg_f32 + _broadcast_ht_dht_term((ht * dht_f32).sum((2, 3)))
        dg = dg_f32.astype(g.dtype)
    if gk is not None:
        dgk_f32 = chunk_global_cumsum(
            dq_f32 * q.astype(jnp.float32) - dk_per_head * k_for_gate.astype(jnp.float32),
            reverse=not reverse,
            cu_seqlens=cu_seqlens,
        )
        if dht_f32 is not None:
            dgk_f32 = dgk_f32 + _broadcast_ht_dht_term((ht * dht_f32).sum(3))
        dgk = dgk_f32.astype(gk.dtype)
    if gv is not None:
        dgv_f32 = chunk_global_cumsum(
            do.astype(jnp.float32) * o.astype(jnp.float32) - dv_per_head * v_for_gate.astype(jnp.float32),
            reverse=not reverse,
            cu_seqlens=cu_seqlens,
        )
        if dht_f32 is not None:
            dgv_f32 = dgv_f32 + _broadcast_ht_dht_term((ht * dht_f32).sum(2))
        dgv = dgv_f32.astype(gv.dtype)

    if group_size > 1:
        dk_f32 = dk_per_head.reshape(Z, SEQUENCE, HEAD_KV, group_size, DIM_K).sum(axis=3)
        dv_f32 = dv_per_head.reshape(Z, SEQUENCE, HEAD_KV, group_size, DIM_V).sum(axis=3)
    else:
        dk_f32 = dk_per_head
        dv_f32 = dv_per_head

    dq_out = dq_f32.astype(q.dtype)
    dk_out = dk_f32.astype(k.dtype)
    dv_out = dv_f32.astype(v.dtype)
    dh0_out = dh0.astype(h0.dtype) if h0 is not None else None
    return dq_out, dk_out, dv_out, dg, dgk, dgv, dh0_out
