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


"""Forward pass Triton kernels for recurrent linear attention.

This module provides the GPU-accelerated forward pass for recurrent linear
attention mechanisms. The kernel processes sequences step-by-step with O(N)
complexity, maintaining a hidden state matrix that accumulates key-value
outer products with optional gating.

Algorithm Overview:
------------------
For each timestep t:
    1. Apply gating: h *= exp(gate)  (if gating is enabled)
    2. Update state: h += k_t * v_t^T  (outer product)
    3. Compute output: o_t = sum(h * q_t, axis=key_dim)

The hidden state h has shape [key_dim, value_dim] and represents the
compressed history of the sequence up to the current timestep.

Gating Modes:
- g (scalar gate per head): h *= exp(g_t), uniform decay across dimensions
- g_gamma (fixed decay per head): h *= exp(gamma), constant decay rate
- gk (per-key-dim gate): h *= exp(gk_t)[:, None], key-selective decay
- gv (per-value-dim gate): h *= exp(gv_t)[None, :], value-selective decay

Memory Layout:
- Inputs: [batch, sequence, heads, dim] format
- Hidden state: [batch*heads, key_dim, value_dim]
- Output is tiled across key and value dimensions for parallelism

Functions:
- fwd_kernel: Triton JIT kernel for recurrent forward computation
- fwd_triton_impl: Python wrapper that launches the kernel
"""

import jax
import triton
import triton.language as tl
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ejkernel.callib import cdiv, triton_call


@triton.jit
def fwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    gk,
    gv,
    h0,
    cu_seqlens,
    softmax_scale,
    o,
    ht,
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    H_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    key_chunk_size: tl.constexpr,
    blocksize_v: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_G: tl.constexpr,
    G_IS_4D: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    G_GAMMA_PER_SEQ: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Triton kernel for forward pass of recurrent linear attention.

    Processes sequences step-by-step with O(N) complexity, maintaining a
    hidden state that accumulates key-value information. Supports various
    gating mechanisms, grouped-query attention (GQA), and both forward/reverse
    processing.

    Args:
        q, k, v: Query, key, value tensor pointers.
        g: Optional gate tensor (3-D scalar per head or 4-D per-key-dim).
        g_gamma: Optional per-head decay factor (1-D ``[H]`` or 2-D ``[N, H]``).
        gk, gv: Optional gates applied to keys and values.
        h0: Initial hidden state pointer.
        cu_seqlens: Cumulative sequence lengths for variable-length mode.
        softmax_scale: Query scaling factor.
        o: Output tensor pointer.
        ht: Final hidden state pointer.
        T, B, H, H_KV, GROUP_SIZE, K, V: Tensor dimensions.
        key_chunk_size, blocksize_v: Power-of-2 block sizes for tiling.
        REVERSE: Process sequence in reverse order.
        USE_G, G_IS_4D, USE_G_GAMMA, G_GAMMA_PER_SEQ, USE_GK, USE_GV: Gating flags.
        USE_INITIAL_STATE: Whether to use initial hidden state.
        STORE_FINAL_STATE: Whether to store final hidden state.
        IS_VARLEN: Variable-length sequence mode.
    """
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    i_kv_h = i_h // GROUP_SIZE
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        scope = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        scope = B * T

    o_k = i_k * key_chunk_size + tl.arange(0, key_chunk_size)
    o_v = i_v * blocksize_v + tl.arange(0, blocksize_v)
    t_start = bos + ((T - 1) if REVERSE else 0)
    p_q = q + t_start * H * K + i_h * K + o_k
    p_k = k + t_start * H_KV * K + i_kv_h * K + o_k
    p_v = v + t_start * H_KV * V + i_kv_h * V + o_v
    p_o = o + ((i_k * scope + bos) + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    if USE_G:
        if G_IS_4D:
            p_g = g + t_start * H * K + i_h * K + o_k
        else:
            p_g = g + t_start * H + i_h
    if USE_GK:
        p_gk = gk + t_start * H * K + i_h * K + o_k
    if USE_GV:
        p_gv = gv + t_start * H * V + i_h * V + o_v
    if USE_G_GAMMA:
        g_gamma_offset = i_nh if G_GAMMA_PER_SEQ else i_h
        b_g_gamma_decay = tl.exp(tl.load(g_gamma + g_gamma_offset).to(tl.float32))

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([key_chunk_size, blocksize_v], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    step = -1 if REVERSE else 1
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * softmax_scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
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
        b_o = b_h * b_q[:, None]
        b_o = tl.sum(b_o, axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += step * H * K
        p_k += step * H_KV * K
        p_v += step * H_KV * V
        p_o += step * H * V
        if USE_G:
            p_g += step * (H * K if G_IS_4D else H)
        if USE_GK:
            p_gk += step * H * K
        if USE_GV:
            p_gv += step * H * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


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


def fwd_triton_impl(
    q: Float[Array, "batch seq_len num_heads head_dim"],
    k: Float[Array, "batch seq_len num_kv_heads head_dim"],
    v: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads ..."] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads head_dim v_head_dim"]]:
    """Execute recurrent linear attention forward pass using Triton kernel.

    Implements efficient O(N) recurrent attention by maintaining a hidden state
    that accumulates key-value information. Supports various gating mechanisms
    including 3-D scalar or 4-D per-key GLA gates, Lightning-style decay,
    per-dimension key/value gates, and grouped-query attention (GQA).

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim].
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim].
        v: Value tensor [batch, seq_len, num_kv_heads, v_head_dim].
        g: Optional gate tensor, either 3-D ``[batch, seq_len, num_heads]``
            (scalar per head) or 4-D ``[batch, seq_len, num_heads, head_dim]``
            (per-key-dim GLA gate).
        g_gamma: Optional Lightning-style decay factor of shape ``[num_heads]``,
            ``[1, num_heads]``, or ``[batch_or_num_seqs, num_heads]``.
        gk: Optional per-key gate [batch, seq_len, num_heads, head_dim].
        gv: Optional per-value gate [batch, seq_len, num_heads, v_head_dim].
        softmax_scale: Query scaling factor (defaults to ``1 / sqrt(head_dim)``).
        initial_state: Initial hidden state [batch_or_num_seqs, num_heads, head_dim, v_head_dim].
        reverse: Process sequence in reverse order.
        cu_seqlens: Cumulative sequence lengths for variable-length mode.
        block_k: Key-dimension tile size selected by the operation config.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        tuple: (output, final_state) where output is attention result and
        final_state is the hidden state after processing.
    """
    B, T, H, K = q.shape
    H_KV = k.shape[2]
    V = v.shape[-1]
    if H % H_KV != 0 or v.shape[2] != H_KV:
        raise ValueError(
            f"Incompatible GQA head counts: query has {H} heads, key has {H_KV}, value has {v.shape[2]}."
        )
    group_size = H // H_KV
    N = B if cu_seqlens is None else int(cu_seqlens.shape[0]) - 1
    if g is not None and g.ndim not in (3, 4):
        raise ValueError(f"g must be 3-D [B, T, H] or 4-D [B, T, H, K], got shape {g.shape}.")
    g_is_4d = (g is not None) and (g.ndim == 4)
    g_gamma_per_seq = _resolve_g_gamma_per_seq(g_gamma, N, H)
    if softmax_scale is None:
        softmax_scale = float(K**-0.5)

    key_chunk_size = triton.next_power_of_2(min(K, int(block_k)))
    blocksize_v = triton.next_power_of_2(min(V, int(block_v)))
    NumKBlocks, NumVBlocks = cdiv(K, key_chunk_size), cdiv(V, blocksize_v)

    h0 = initial_state
    ht_shape = (N, H, K, V)
    out_shape = (NumKBlocks, B, T, H, V)
    out_buf_dtype = q.dtype if NumKBlocks == 1 else jnp.float32
    grid = (NumVBlocks, NumKBlocks, N * H)
    metaparams = dict(
        T=T,
        B=B,
        H=H,
        H_KV=H_KV,
        GROUP_SIZE=group_size,
        K=K,
        V=V,
        key_chunk_size=key_chunk_size,
        blocksize_v=blocksize_v,
        USE_G=g is not None,
        G_IS_4D=g_is_4d,
        USE_G_GAMMA=g_gamma is not None,
        G_GAMMA_PER_SEQ=g_gamma_per_seq,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        USE_INITIAL_STATE=h0 is not None,
        STORE_FINAL_STATE=True,
        IS_VARLEN=cu_seqlens is not None,
        REVERSE=reverse,
    )
    out, ht = triton_call(
        q,
        k,
        v,
        g if g is not None else 1,
        g_gamma if g_gamma is not None else 1,
        gk if gk is not None else 1,
        gv if gv is not None else 1,
        h0 if h0 is not None else 1,
        cu_seqlens if cu_seqlens is not None else 1,
        float(softmax_scale),
        kernel=fwd_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(out_shape, out_buf_dtype),
            jax.ShapeDtypeStruct(ht_shape, jnp.float32),
        ],
        name="ejkernel::triton::recurrent_fwd",
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )
    out = out[0] if NumKBlocks == 1 else jnp.sum(out, axis=0).astype(q.dtype)
    return out, ht
