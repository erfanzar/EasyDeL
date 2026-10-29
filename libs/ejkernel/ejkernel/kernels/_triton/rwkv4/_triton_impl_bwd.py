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

"""RWKV-4 Triton backward implementation."""

from __future__ import annotations

import jax
import triton
import triton.language as tl
from jax import numpy as jnp
from jaxtyping import Array, Float

from ejkernel.callib import cdiv, triton_call


@triton.jit
def _rwkv4_bwd_kernel(
    w_ptr,
    u_ptr,
    k_ptr,
    v_ptr,
    state_hist_ptr,
    do_ptr,
    dstate_ptr,
    dk_ptr,
    dv_ptr,
    du_bt_ptr,
    dw_bt_ptr,
    dstate0_ptr,
    T: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compute RWKV-4 backward pass for one (batch, channel-block) program."""
    b = tl.program_id(0).to(tl.int64)
    c_blk = tl.program_id(1)

    cs = c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cs < C

    w = tl.load(w_ptr + cs, mask=cmask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + cs, mask=cmask, other=0.0).to(tl.float32)

    base_seq = b * T * C
    base_state = (b * 3) * C

    d_alpha_next = tl.load(dstate_ptr + base_state + 0 * C + cs, mask=cmask, other=0.0).to(tl.float32)
    d_beta_next = tl.load(dstate_ptr + base_state + 1 * C + cs, mask=cmask, other=0.0).to(tl.float32)
    d_eps_next = tl.load(dstate_ptr + base_state + 2 * C + cs, mask=cmask, other=0.0).to(tl.float32)

    du_acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    dw_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for t in range(T - 1, -1, -1):
        off = base_seq + t * C + cs
        kt = tl.load(k_ptr + off, mask=cmask, other=0.0).to(tl.float32)
        vt = tl.load(v_ptr + off, mask=cmask, other=0.0).to(tl.float32)
        dot = tl.load(do_ptr + off, mask=cmask, other=0.0).to(tl.float32)

        base_hist = ((b * (T + 1) + t) * 3) * C
        alpha = tl.load(state_hist_ptr + base_hist + 0 * C + cs, mask=cmask, other=0.0).to(tl.float32)
        beta = tl.load(state_hist_ptr + base_hist + 1 * C + cs, mask=cmask, other=0.0).to(tl.float32)
        eps = tl.load(state_hist_ptr + base_hist + 2 * C + cs, mask=cmask, other=-1e30).to(tl.float32)

        ukt = u + kt
        tau = tl.maximum(ukt, eps)
        e1a = tl.exp(eps - tau)
        e2a = tl.exp(ukt - tau)
        num = e1a * alpha + e2a * vt
        den = e1a * beta + e2a

        w_eps = w + eps
        eps_next = tl.maximum(w_eps, kt)
        e1b = tl.exp(w_eps - eps_next)
        e2b = tl.exp(kt - eps_next)

        d_alpha = d_alpha_next * e1b
        d_beta = d_beta_next * e1b
        d_v = d_alpha_next * e2b

        de1b = d_alpha_next * alpha + d_beta_next * beta
        de2b = d_alpha_next * vt + d_beta_next
        g1b = de1b * e1b
        g2b = de2b * e2b

        d_w_eps = g1b
        d_eps_next_total = d_eps_next - g1b - g2b
        d_k = g2b

        gt_b = w_eps > kt
        lt_b = w_eps < kt
        eq_b = w_eps == kt
        split_w = tl.where(gt_b, 1.0, tl.where(eq_b, 0.5, 0.0))
        split_k = tl.where(lt_b, 1.0, tl.where(eq_b, 0.5, 0.0))

        d_w_eps += d_eps_next_total * split_w
        d_k += d_eps_next_total * split_k

        dw_acc += d_w_eps
        d_eps = d_w_eps

        d_num = dot / den
        d_den = -dot * num / (den * den)

        d_alpha += d_num * e1a
        d_beta += d_den * e1a
        d_v += d_num * e2a

        de1a = d_num * alpha + d_den * beta
        de2a = d_num * vt + d_den
        g1a = de1a * e1a
        g2a = de2a * e2a

        d_eps += g1a
        d_ukt = g2a
        d_tau = -(g1a + g2a)

        gt_a = ukt > eps
        lt_a = ukt < eps
        eq_a = ukt == eps
        split_ukt = tl.where(gt_a, 1.0, tl.where(eq_a, 0.5, 0.0))
        split_eps = tl.where(lt_a, 1.0, tl.where(eq_a, 0.5, 0.0))

        d_ukt += d_tau * split_ukt
        d_eps += d_tau * split_eps

        du_acc += d_ukt
        d_k += d_ukt

        tl.store(dk_ptr + off, d_k.to(tl.float32), mask=cmask)
        tl.store(dv_ptr + off, d_v.to(tl.float32), mask=cmask)

        d_alpha_next = d_alpha
        d_beta_next = d_beta
        d_eps_next = d_eps

    base_bc = b * C
    tl.store(du_bt_ptr + base_bc + cs, du_acc.to(tl.float32), mask=cmask)
    tl.store(dw_bt_ptr + base_bc + cs, dw_acc.to(tl.float32), mask=cmask)

    tl.store(dstate0_ptr + base_state + 0 * C + cs, d_alpha_next.to(tl.float32), mask=cmask)
    tl.store(dstate0_ptr + base_state + 1 * C + cs, d_beta_next.to(tl.float32), mask=cmask)
    tl.store(dstate0_ptr + base_state + 2 * C + cs, d_eps_next.to(tl.float32), mask=cmask)


def bwd_triton_impl(
    w_neg: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state_hist: Float[Array, "batch seq_plus_one three chans"],
    do: Float[Array, "batch seq_len chans"],
    dstate: Float[Array, "batch three chans"],
) -> tuple[
    Float[Array, "chans"],
    Float[Array, "chans"],
    Float[Array, "batch seq_len chans"],
    Float[Array, "batch seq_len chans"],
    Float[Array, "batch three chans"],
]:
    """Execute RWKV-4 backward pass using Triton and reduce per-batch parameter gradients.

    Launches ``_rwkv4_bwd_kernel`` on a ``(B, ceil(C / BLOCK_C))`` grid. The
    kernel computes per-batch-element gradients for ``u`` and ``w_neg``; this
    function then sums them over the batch dimension to produce the final
    channel-wise gradients.

    The state history ``state_hist`` of shape ``(B, T+1, 3, C)`` must have been
    produced by ``fwd_triton_impl_with_history``. Each entry ``state_hist[:, t]``
    contains the recurrent state ``(alpha, beta, eps)`` **before** processing
    token ``t``, enabling exact gradient computation without recomputation.

    Args:
        w_neg: Negated exponentiated time-decay ``-exp(w_raw)``, shape ``(C,)``.
        u: Time-mix bias, shape ``(C,)``.
        k: Key tensor, shape ``(B, T, C)``.
        v: Value tensor, shape ``(B, T, C)``.
        state_hist: Per-step state history from forward pass, shape ``(B, T+1, 3, C)``.
            Axis 2 indexes the three state components: alpha (0), beta (1), eps (2).
        do: Gradient of the output WKV tensor, shape ``(B, T, C)``.
        dstate: Gradient of the final state, shape ``(B, 3, C)``.

    Returns:
        Tuple ``(dw_neg, du, dk, dv, dstate0)`` where:
            - ``dw_neg``: Gradient w.r.t. ``w_neg``, shape ``(C,)`` (batch-reduced).
            - ``du``: Gradient w.r.t. ``u``, shape ``(C,)`` (batch-reduced).
            - ``dk``: Gradient w.r.t. ``k``, shape ``(B, T, C)``.
            - ``dv``: Gradient w.r.t. ``v``, shape ``(B, T, C)``.
            - ``dstate0``: Gradient w.r.t. the initial state, shape ``(B, 3, C)``.
    """
    B, T, C = k.shape
    BLOCK_C = 128 if C >= 128 else 64 if C >= 64 else 32
    grid = (B, cdiv(C, BLOCK_C))

    dk_shape = jax.ShapeDtypeStruct(k.shape, jnp.float32)
    dv_shape = jax.ShapeDtypeStruct(v.shape, jnp.float32)
    du_bt_shape = jax.ShapeDtypeStruct((B, C), jnp.float32)
    dw_bt_shape = jax.ShapeDtypeStruct((B, C), jnp.float32)
    dstate0_shape = jax.ShapeDtypeStruct((B, 3, C), jnp.float32)

    dk, dv, du_bt, dw_bt, dstate0 = triton_call(
        w_neg,
        u,
        k,
        v,
        state_hist,
        do,
        dstate,
        kernel=_rwkv4_bwd_kernel,
        out_shape=[dk_shape, dv_shape, du_bt_shape, dw_bt_shape, dstate0_shape],
        name="ejkernel::triton::rwkv4_bwd",
        grid=grid,
        T=T,
        C=C,
        BLOCK_C=BLOCK_C,
    )

    du = jnp.sum(du_bt, axis=0)
    dw_neg = jnp.sum(dw_bt, axis=0)
    return dw_neg, du, dk, dv, dstate0
