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

"""RWKV-7 forward pass Triton kernel implementation.

This module provides the Triton GPU kernel for RWKV-7 DPLR (Diagonal + Low-Rank)
recurrence. It uses explicit launcher-supplied block sizes and supports
variable-length sequences (packed mode) and reverse processing.
"""

from __future__ import annotations

import jax
import triton
import triton.language as tl
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ejkernel.callib import cdiv, triton_call


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] != 1,
        "IS_VARLEN": lambda args: args["cu_seqlens"] != 1,
    }
)
@triton.jit
def _rwkv7_fwd_kernel(
    r,
    w,
    k,
    v,
    a,
    b,
    h0,
    cu_seqlens,
    scale,
    o,
    ht,
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """RWKV-7 forward pass Triton kernel with DPLR state transitions.

    Processes the RWKV-7 recurrence with Diagonal Plus Low-Rank state updates.
    Each program instance handles one (value_block, batch*head) tuple and
    iterates through all timesteps sequentially.

    The DPLR recurrence computes:
        hb = sum(b_t[:, None] * h, axis=0)  (project state via b)
        h = exp(w_t)[:, None] * h + a_t[:, None] * hb[None, :]  (DPLR update)
        h += k_t[:, None] * v_t[None, :]  (rank-1 KV update)
        o_t = sum(h * r_t[:, None], axis=0)  (read from state via receptance)

    Unlike RWKV-6, the full key dimension is processed in a single block (BK),
    while the value dimension is autotuned, since the low-rank (a, b) updates
    require the full key dimension to be available simultaneously.

    Args:
        r: Receptance tensor pointer, shape (B, T, H, K).
        w: Log-decay tensor pointer, shape (B, T, H, K).
        k: Key tensor pointer, shape (B, T, H, K).
        v: Value tensor pointer, shape (B, T, H, V).
        a: Low-rank update vector pointer, shape (B, T, H, K).
        b: Low-rank projection vector pointer, shape (B, T, H, K).
        h0: Initial hidden state pointer, shape (N, H, K, V).
        cu_seqlens: Cumulative sequence lengths pointer.
        scale: Receptance scaling factor.
        o: Output tensor pointer, shape (B, T, H, V).
        ht: Final state output pointer, shape (N, H, K, V).
        T, B, H, K, V: Tensor dimensions.
        BK, BV: Block sizes for key and value dimensions.
        REVERSE: Whether to process sequence in reverse.
        USE_INITIAL_STATE: Whether to load initial hidden state.
        IS_VARLEN: Whether using variable-length sequences.
    """
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T
        eos = i_n * T + T
        tl.multiple_of(bos, 1)
        tl.multiple_of(eos, 1)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_r = r + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_w = w + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_k = k + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_a = a + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_b = b + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_o = o + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)

        hb = tl.sum(b_b[:, None] * b_h, axis=0)
        b_h = tl.exp(b_w)[:, None] * b_h + b_a[:, None] * hb[None, :]
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_r[:, None], axis=0)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_r += (-1 if REVERSE else 1) * H * K
        p_w += (-1 if REVERSE else 1) * H * K
        p_k += (-1 if REVERSE else 1) * H * K
        p_v += (-1 if REVERSE else 1) * H * V
        p_a += (-1 if REVERSE else 1) * H * K
        p_b += (-1 if REVERSE else 1) * H * K
        p_o += (-1 if REVERSE else 1) * H * V

    p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fwd_triton_impl(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    *,
    softmax_scale: float,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """Execute RWKV-7 DPLR forward pass on GPU via Triton.

    Args:
        r: Receptance tensor `[B, T, H, K]`.
        w: Log decay tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        a: Low-rank update vector `[B, T, H, K]`.
        b: Low-rank projection vector `[B, T, H, K]`.
        softmax_scale: Scale factor for receptance.
        initial_state: Optional initial state.
        reverse: Whether to process in reverse order.
        cu_seqlens: Cumulative sequence lengths for packed mode.
        block_v: Value-dimension tile size selected by the operation config.
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        Tuple of (output, final_state) both in appropriate shapes.
    """
    B, T, H, K = r.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else int(cu_seqlens.shape[0] - 1)

    BK = triton.next_power_of_2(K)
    BV = min(V, int(block_v))

    out_shape = jax.ShapeDtypeStruct(v.shape, v.dtype)
    ht_shape = jax.ShapeDtypeStruct((N, H, K, V), jnp.float32)

    def grid(meta):
        return (cdiv(V, meta["BV"]), N * H)

    metaparams = dict(
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
    )

    out, ht = triton_call(
        r,
        w,
        k,
        v,
        a,
        b,
        initial_state if initial_state is not None else 1,
        cu_seqlens if cu_seqlens is not None else 1,
        softmax_scale,
        kernel=_rwkv7_fwd_kernel,
        out_shape=[out_shape, ht_shape],
        name="ejkernel::triton::rwkv7_fwd",
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        **metaparams,
    )
    return out, ht
