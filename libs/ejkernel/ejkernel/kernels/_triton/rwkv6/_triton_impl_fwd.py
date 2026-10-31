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

"""RWKV-6 forward pass Triton kernel implementation.

This module provides the Triton GPU kernel for RWKV-6 linear attention recurrence.
Supports multi-head attention, variable-length sequences (packed mode), and
optional reverse processing.
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
def _rwkv6_fwd_kernel(
    r,
    k,
    v,
    w,
    u,
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
    """RWKV-6 forward pass Triton kernel with multi-head recurrence.

    Processes the RWKV-6 recurrence with per-timestep log-decay and bonus
    term. Each program instance handles one (value_block, key_block, batch*head)
    tuple and iterates through all timesteps sequentially.

    The recurrence computes:
        kv_t = k_t[:, None] * v_t[None, :]  (outer product)
        o_t = sum((h + kv_t * u[:, None]) * r_t[:, None], axis=0)
        h = h * exp(w_t)[:, None] + kv_t

    Output is tiled across key blocks and later summed to produce the final
    output, since each key block contributes a partial sum to the output.

    Args:
        r: Receptance tensor pointer, shape (B, T, H, K).
        k: Key tensor pointer, shape (B, T, H, K).
        v: Value tensor pointer, shape (B, T, H, V).
        w: Log-decay tensor pointer, shape (B, T, H, K).
        u: Bonus tensor pointer, shape (H, K).
        h0: Initial hidden state pointer, shape (N, H, K, V).
        cu_seqlens: Cumulative sequence lengths pointer.
        scale: Receptance scaling factor.
        o: Output tensor pointer, shape (NK, B, T, H, V).
        ht: Final state output pointer, shape (N, H, K, V).
        T, B, H, K, V: Tensor dimensions.
        BK, BV: Block sizes for key and value dimensions.
        REVERSE: Whether to process sequence in reverse.
        USE_INITIAL_STATE: Whether to load initial hidden state.
        IS_VARLEN: Whether using variable-length sequences.
    """
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        scope = T
        T = eos - bos
    else:
        bos = i_n * T
        eos = i_n * T + T
        scope = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_r = r + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_k = k + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_w = w + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
    p_o = o + ((i_k * scope + bos) + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
    p_u = u + i_h * K + o_k

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_u = tl.load(p_u, mask=mask_k, other=0).to(tl.float32)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)

        b_kv = b_k[:, None] * b_v[None, :]
        b_o = tl.sum((b_h + b_kv * b_u[:, None]) * b_r[:, None], axis=0)
        b_h = b_h * tl.exp(b_w)[:, None] + b_kv

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_r += (-1 if REVERSE else 1) * H * K
        p_k += (-1 if REVERSE else 1) * H * K
        p_v += (-1 if REVERSE else 1) * H * V
        p_w += (-1 if REVERSE else 1) * H * K
        p_o += (-1 if REVERSE else 1) * H * V

    p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fwd_triton_impl(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    *,
    softmax_scale: float,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """Execute RWKV-6 forward pass on GPU via Triton.

    Args:
        r: Receptance tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        w: Log decay tensor `[B, T, H, K]`.
        u: Bonus tensor `[H, K]`.
        softmax_scale: Scale factor for receptance.
        initial_state: Optional initial state.
        reverse: Whether to process in reverse order.
        cu_seqlens: Cumulative sequence lengths for packed mode.

    Returns:
        Tuple of (output, final_state) both in appropriate shapes.
    """
    B, T, H, K = r.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else int(cu_seqlens.shape[0] - 1)

    BK = min(triton.next_power_of_2(K), 32)
    BV = min(triton.next_power_of_2(V), 32)
    NK = cdiv(K, BK)
    NV = cdiv(V, BV)

    out_partial_shape = jax.ShapeDtypeStruct((NK, *v.shape), jnp.float32)
    ht_shape = jax.ShapeDtypeStruct((N, H, K, V), jnp.float32)

    grid = (NV, NK, N * H)
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

    out_partial, ht = triton_call(
        r,
        k,
        v,
        w,
        u,
        initial_state if initial_state is not None else 1,
        cu_seqlens if cu_seqlens is not None else 1,
        softmax_scale,
        kernel=_rwkv6_fwd_kernel,
        out_shape=[out_partial_shape, ht_shape],
        name="ejkernel::triton::rwkv6_fwd",
        grid=grid,
        **metaparams,
    )

    out = jnp.sum(out_partial, axis=0).astype(v.dtype)
    return out, ht
