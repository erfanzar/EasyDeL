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

"""RWKV-4 forward pass Triton kernel implementation.

This module provides the Triton GPU kernel for RWKV-4 time-mix recurrence.
The kernel processes sequences in a numerically-stable manner using the
(alpha, beta, eps) state formulation.
"""

from __future__ import annotations

import jax
import triton
import triton.language as tl
from jax import numpy as jnp
from jaxtyping import Array, Float

from ejkernel.callib import cdiv, triton_call


@triton.jit
def _rwkv4_fwd_kernel(
    w_ptr,
    u_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    o_ptr,
    state_out_ptr,
    T: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """RWKV-4 forward pass Triton kernel with numerically stable recurrence.

    Processes the WKV recurrence for a block of channels using the (alpha,
    beta, eps) formulation for numerical stability. Each program instance
    handles one (batch, channel_block) pair and iterates through all timesteps.

    The computation uses a log-space max trick:
        tau = max(u+k_t, eps)
        wkv_t = (exp(eps-tau)*alpha + exp(u+k_t-tau)*v_t) /
                (exp(eps-tau)*beta + exp(u+k_t-tau))
        alpha_{t+1} = exp(w+eps-eps_next)*alpha + exp(k_t-eps_next)*v_t
        beta_{t+1}  = exp(w+eps-eps_next)*beta  + exp(k_t-eps_next)

    Args:
        w_ptr: Negated exponentiated time decay [-exp(w)], shape (C,).
        u_ptr: Time-mix bias, shape (C,).
        k_ptr: Key tensor pointer, shape (B, T, C).
        v_ptr: Value tensor pointer, shape (B, T, C).
        state_ptr: Initial state (alpha, beta, eps), shape (B, 3, C).
        o_ptr: Output tensor pointer, shape (B, T, C).
        state_out_ptr: Final state output pointer, shape (B, 3, C).
        T: Sequence length (compile-time constant).
        C: Number of channels (compile-time constant).
        BLOCK_C: Channel block size (compile-time constant).
    """
    b = tl.program_id(0)
    c_blk = tl.program_id(1)

    cs = c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cs < C

    w = tl.load(w_ptr + cs, mask=cmask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cs, mask=cmask, other=0.0).to(tl.float32)

    base_state = (b * 3) * C
    alpha = tl.load(state_ptr + base_state + 0 * C + cs, mask=cmask, other=0.0).to(tl.float32)
    beta = tl.load(state_ptr + base_state + 1 * C + cs, mask=cmask, other=0.0).to(tl.float32)
    eps = tl.load(state_ptr + base_state + 2 * C + cs, mask=cmask, other=-1e30).to(tl.float32)

    base_seq = b * T * C
    for t in range(0, T):
        off = base_seq + t * C + cs
        kt = tl.load(k_ptr + off, mask=cmask, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + off, mask=cmask, other=0.0).to(tl.float32)

        ukt = u + kt
        tau = tl.maximum(ukt, eps)
        e1a = tl.exp(eps - tau)
        e2a = tl.exp(ukt - tau)
        wkv = (e1a * alpha + e2a * vt) / (e1a * beta + e2a)
        tl.store(o_ptr + off, wkv.to(o_ptr.dtype.element_ty), mask=cmask)

        w_eps = w + eps
        eps_next = tl.maximum(w_eps, kt)
        e1b = tl.exp(w_eps - eps_next)
        e2b = tl.exp(kt - eps_next)
        alpha = e1b * alpha + e2b * vt
        beta = e1b * beta + e2b
        eps = eps_next

    base_state_out = (b * 3) * C
    tl.store(state_out_ptr + base_state_out + 0 * C + cs, alpha.to(tl.float32), mask=cmask)
    tl.store(state_out_ptr + base_state_out + 1 * C + cs, beta.to(tl.float32), mask=cmask)
    tl.store(state_out_ptr + base_state_out + 2 * C + cs, eps.to(tl.float32), mask=cmask)


@triton.jit
def _rwkv4_fwd_kernel_with_hist(
    w_ptr,
    u_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    o_ptr,
    state_out_ptr,
    state_hist_ptr,
    T: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """RWKV-4 forward pass kernel that additionally stores per-step state history."""
    b = tl.program_id(0).to(tl.int64)
    c_blk = tl.program_id(1)

    cs = c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cs < C

    w = tl.load(w_ptr + cs, mask=cmask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cs, mask=cmask, other=0.0).to(tl.float32)

    base_state = (b * 3) * C
    alpha = tl.load(state_ptr + base_state + 0 * C + cs, mask=cmask, other=0.0).to(tl.float32)
    beta = tl.load(state_ptr + base_state + 1 * C + cs, mask=cmask, other=0.0).to(tl.float32)
    eps = tl.load(state_ptr + base_state + 2 * C + cs, mask=cmask, other=-1e30).to(tl.float32)

    base_seq = b * T * C

    base_hist0 = ((b * (T + 1)) * 3) * C
    tl.store(state_hist_ptr + base_hist0 + 0 * C + cs, alpha.to(tl.float32), mask=cmask)
    tl.store(state_hist_ptr + base_hist0 + 1 * C + cs, beta.to(tl.float32), mask=cmask)
    tl.store(state_hist_ptr + base_hist0 + 2 * C + cs, eps.to(tl.float32), mask=cmask)

    for t in range(0, T):
        off = base_seq + t * C + cs
        kt = tl.load(k_ptr + off, mask=cmask, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + off, mask=cmask, other=0.0).to(tl.float32)

        ukt = u + kt
        tau = tl.maximum(ukt, eps)
        e1a = tl.exp(eps - tau)
        e2a = tl.exp(ukt - tau)
        wkv = (e1a * alpha + e2a * vt) / (e1a * beta + e2a)
        tl.store(o_ptr + off, wkv.to(o_ptr.dtype.element_ty), mask=cmask)

        w_eps = w + eps
        eps_next = tl.maximum(w_eps, kt)
        e1b = tl.exp(w_eps - eps_next)
        e2b = tl.exp(kt - eps_next)
        alpha = e1b * alpha + e2b * vt
        beta = e1b * beta + e2b
        eps = eps_next

        base_hist_t = ((b * (T + 1) + (t + 1)) * 3) * C
        tl.store(state_hist_ptr + base_hist_t + 0 * C + cs, alpha.to(tl.float32), mask=cmask)
        tl.store(state_hist_ptr + base_hist_t + 1 * C + cs, beta.to(tl.float32), mask=cmask)
        tl.store(state_hist_ptr + base_hist_t + 2 * C + cs, eps.to(tl.float32), mask=cmask)

    base_state_out = (b * 3) * C
    tl.store(state_out_ptr + base_state_out + 0 * C + cs, alpha.to(tl.float32), mask=cmask)
    tl.store(state_out_ptr + base_state_out + 1 * C + cs, beta.to(tl.float32), mask=cmask)
    tl.store(state_out_ptr + base_state_out + 2 * C + cs, eps.to(tl.float32), mask=cmask)


def fwd_triton_impl(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"],
) -> tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]:
    """Execute RWKV-4 forward pass on GPU via Triton.

    Args:
        w: Negative exponentiated time-decay `[C]` (already `-exp(w_raw)`).
        u: Time-mix bias `[C]`.
        k: Key tensor `[B, T, C]`.
        v: Value tensor `[B, T, C]`.
        state: Initial state `[B, 3, C]` (alpha, beta, eps).

    Returns:
        Tuple of (output `[B, T, C]`, final_state `[B, 3, C]`).
    """
    B, T, C = k.shape
    out_shape = jax.ShapeDtypeStruct(k.shape, v.dtype)
    state_shape = jax.ShapeDtypeStruct((B, 3, C), jnp.float32)

    BLOCK_C = 128 if C >= 128 else 64 if C >= 64 else 32
    grid = (B, cdiv(C, BLOCK_C))

    o, state_out = triton_call(
        w,
        u,
        k,
        v,
        state,
        kernel=_rwkv4_fwd_kernel,
        out_shape=[out_shape, state_shape],
        name="ejkernel::triton::rwkv4_fwd",
        grid=grid,
        T=T,
        C=C,
        BLOCK_C=BLOCK_C,
    )
    return o, state_out


def fwd_triton_impl_with_history(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"],
) -> tuple[
    Float[Array, "batch seq_len chans"],
    Float[Array, "batch three chans"],
    Float[Array, "batch seq_plus_one three chans"],
]:
    """Execute RWKV-4 forward pass and return the per-step state history.

    Identical to ``fwd_triton_impl`` but also records the recurrent state
    ``(alpha, beta, eps)`` at every timestep (including the initial state) into
    a history tensor of shape ``(B, T+1, 3, C)``. The history is required by the
    backward pass (``bwd_triton_impl``) to avoid recomputation of intermediate
    states during gradient computation.

    Launches ``_rwkv4_fwd_kernel_with_hist`` on a ``(B, ceil(C / BLOCK_C))`` grid.

    Args:
        w: Negated exponentiated time-decay ``-exp(w_raw)``, shape ``(C,)``.
        u: Time-mix bias, shape ``(C,)``.
        k: Key tensor, shape ``(B, T, C)``.
        v: Value tensor, shape ``(B, T, C)``.
        state: Initial recurrent state ``(alpha, beta, eps)``, shape ``(B, 3, C)``.
            Must be float32.

    Returns:
        Tuple ``(output, final_state, state_hist)`` where:
            - ``output``: WKV result, shape ``(B, T, C)``.
            - ``final_state``: State after the last token, shape ``(B, 3, C)``, float32.
            - ``state_hist``: Per-step state snapshot, shape ``(B, T+1, 3, C)``, float32.
              ``state_hist[:, 0]`` is the initial state; ``state_hist[:, t+1]`` is the
              state after processing token ``t``.
    """
    B, T, C = k.shape
    out_shape = jax.ShapeDtypeStruct(k.shape, v.dtype)
    state_shape = jax.ShapeDtypeStruct((B, 3, C), jnp.float32)
    state_hist_shape = jax.ShapeDtypeStruct((B, T + 1, 3, C), jnp.float32)

    BLOCK_C = 128 if C >= 128 else 64 if C >= 64 else 32
    grid = (B, cdiv(C, BLOCK_C))

    o, state_out, state_hist = triton_call(
        w,
        u,
        k,
        v,
        state,
        kernel=_rwkv4_fwd_kernel_with_hist,
        out_shape=[out_shape, state_shape, state_hist_shape],
        name="ejkernel::triton::rwkv4_fwd_with_hist",
        grid=grid,
        T=T,
        C=C,
        BLOCK_C=BLOCK_C,
    )
    return o, state_out, state_hist
