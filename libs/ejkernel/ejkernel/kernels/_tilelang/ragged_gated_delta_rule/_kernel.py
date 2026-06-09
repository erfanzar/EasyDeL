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

"""tile-lang Ragged Gated Delta Rule forward.

Implements the GDR step from the XLA reference:

    k_state = k @ state
    v_diff = v - exp(g) * k_state
    v_new = beta * v_diff
    q_state = (q * scale) @ state
    q_k = sum(q * scale * k)
    out = exp(g) * q_state + q_k * v_new
    new_state = state * exp(g) + k outer v_new

Per-token recurrence. v0 supports the **decode-only** case (one token per
request, i.e. ``num_tokens == num_requests``); the grid maps each token to
its own request slot. Prefill (multi-token-per-request) will land later as
a chunked variant. The grid is ``(num_tokens, num_heads)``: each CTA loads
its own state slot, applies the single step, and writes both ``out`` and
the updated state back into the pool.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Convert a JAX/NumPy floating-point dtype to its TileLang string name.

    Args:
        dtype: Any dtype accepted by ``jnp.dtype()``.

    Returns:
        One of ``"float16"``, ``"bfloat16"``, or ``"float32"``.

    Raises:
        TypeError: if ``dtype`` is not one of the three supported types.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for ragged GDR: {dtype}")
    return mapping[canonical]


def make_decode_step_prim_func(
    *,
    num_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    num_slots: int,
    use_decay: bool,
    use_qk_l2norm: bool,
    dtype,
    threads: int = 128,
):
    """Build the decode-only (one token per request) GDR step ``@T.prim_func``.

    Grid: ``T.Kernel(num_tokens, num_heads)``.  Each CTA loads the state slot
    for its token, applies the single GDR update step, then writes the output
    activation and the updated state back.

    Accumulation dtype is always ``float32`` regardless of the input dtype.

    Prim_func buffer layout::

        Q, K:        (num_tokens, num_heads, qk_head_dim)       — compute dtype
        V:           (num_tokens, num_heads, v_head_dim)         — compute dtype
        Beta:        (num_tokens, num_heads)                     — compute dtype
        Decay:       (num_tokens, num_heads)                     — compute dtype
            Gating value ``g`` in log-space; passed as a zero-valued buffer when
            ``use_decay=False``.
        State0:      (num_slots, num_heads, qk_head_dim, v_head_dim)  — float32
        StateIndices:(num_tokens,)                               — int32
        O:           (num_tokens, num_heads, v_head_dim)         — compute dtype
        StateF:      (num_slots, num_heads, qk_head_dim, v_head_dim)  — float32
        HScan:       (num_tokens, NT+1+(NT%2), num_heads, qk_head_dim, v_head_dim) — float32
            Stores the state before (index 0) and after (index 1) the single
            token's update; used by the backward kernel.

    Args:
        num_tokens: Total token count ``NT`` (= number of requests for decode).
        num_heads: Number of heads ``H``.
        qk_head_dim: Q/K head dimension ``Dq``.
        v_head_dim: V head dimension ``Dv``.
        num_slots: State pool size ``NS``.
        use_decay: If ``False`` the kernel skips ``exp(Decay)`` and uses 1.0.
        use_qk_l2norm: If ``True`` Q and K are L2-normalised before the inner product.
        dtype: Compute dtype for all non-state tensors.
        threads: Threads per CUDA CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` (``rgdr_step``).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    NT, H, Dq, Dv, NS = num_tokens, num_heads, qk_head_dim, v_head_dim, num_slots
    HS = NT + 1 + (NT % 2)
    scale = 1.0 / math.sqrt(Dq)

    @T.prim_func
    def rgdr_step(
        Q: T.Tensor((NT, H, Dq), ts),
        K: T.Tensor((NT, H, Dq), ts),
        V: T.Tensor((NT, H, Dv), ts),
        Beta: T.Tensor((NT, H), ts),
        Decay: T.Tensor((NT, H), ts),
        State0: T.Tensor((NS, H, Dq, Dv), accum),
        StateIndices: T.Tensor((NT,), "int32"),
        O: T.Tensor((NT, H, Dv), ts),
        StateF: T.Tensor((NS, H, Dq, Dv), accum),
        HScan: T.Tensor((NT, HS, H, Dq, Dv), accum),
    ):
        with T.Kernel(NT, H, threads=threads) as (tx, hx):
            _ns_pin = T.alloc_fragment((NS,), accum)
            _nt_pin = T.alloc_fragment((1,), accum)
            _hs_pin = T.alloc_fragment((1,), accum)
            _ = _ns_pin
            _ = _nt_pin
            _hs_pin[0] = HS
            slot_var = T.alloc_fragment((1,), "int32")
            slot_var[0] = StateIndices[tx]

            q_loc = T.alloc_fragment((Dq,), accum)
            k_loc = T.alloc_fragment((Dq,), accum)
            v_loc = T.alloc_fragment((Dv,), accum)
            beta_val = T.alloc_fragment((1,), accum)
            g_exp = T.alloc_fragment((1,), accum)
            qk_norm = T.alloc_fragment((1,), accum)
            kk_norm = T.alloc_fragment((1,), accum)

            for i in T.Parallel(Dq):
                q_loc[i] = T.Cast(accum, Q[tx, hx, i])
                k_loc[i] = T.Cast(accum, K[tx, hx, i])
            for j in T.Parallel(Dv):
                v_loc[j] = T.Cast(accum, V[tx, hx, j])

            beta_val[0] = T.Cast(accum, Beta[tx, hx])
            if use_decay:
                g_exp[0] = T.exp(T.Cast(accum, Decay[tx, hx]))
            else:
                g_exp[0] = 1.0

            if use_qk_l2norm:
                q_sq = T.alloc_fragment((Dq,), accum)
                k_sq = T.alloc_fragment((Dq,), accum)
                for i in T.Parallel(Dq):
                    q_sq[i] = q_loc[i] * q_loc[i]
                    k_sq[i] = k_loc[i] * k_loc[i]
                T.reduce_sum(q_sq, qk_norm, dim=0, clear=True)
                T.reduce_sum(k_sq, kk_norm, dim=0, clear=True)
                inv_q = T.alloc_fragment((1,), accum)
                inv_k = T.alloc_fragment((1,), accum)
                inv_q[0] = 1.0 / T.sqrt(T.max(qk_norm[0], 1e-12))
                inv_k[0] = 1.0 / T.sqrt(T.max(kk_norm[0], 1e-12))
                for i in T.Parallel(Dq):
                    q_loc[i] = q_loc[i] * inv_q[0]
                    k_loc[i] = k_loc[i] * inv_k[0]

            for i in T.Parallel(Dq):
                q_loc[i] = q_loc[i] * scale

            state = T.alloc_fragment((Dq, Dv), accum)
            for i, j in T.Parallel(Dq, Dv):
                state[i, j] = State0[slot_var[0], hx, i, j]
                HScan[tx, 0, hx, i, j] = state[i, j]

            k_state_prod = T.alloc_fragment((Dq, Dv), accum)
            k_state = T.alloc_fragment((Dv,), accum)
            for i, j in T.Parallel(Dq, Dv):
                k_state_prod[i, j] = k_loc[i] * state[i, j]
            T.reduce_sum(k_state_prod, k_state, dim=0, clear=True)

            v_new = T.alloc_fragment((Dv,), accum)
            for j in T.Parallel(Dv):
                v_new[j] = beta_val[0] * (v_loc[j] - g_exp[0] * k_state[j])

            q_state_prod = T.alloc_fragment((Dq, Dv), accum)
            q_state = T.alloc_fragment((Dv,), accum)
            for i, j in T.Parallel(Dq, Dv):
                q_state_prod[i, j] = q_loc[i] * state[i, j]
            T.reduce_sum(q_state_prod, q_state, dim=0, clear=True)

            qk_prod = T.alloc_fragment((Dq,), accum)
            qk = T.alloc_fragment((1,), accum)
            for i in T.Parallel(Dq):
                qk_prod[i] = q_loc[i] * k_loc[i]
            T.reduce_sum(qk_prod, qk, dim=0, clear=True)

            for j in T.Parallel(Dv):
                O[tx, hx, j] = T.Cast(ts, g_exp[0] * q_state[j] + qk[0] * v_new[j])

            for i, j in T.Parallel(Dq, Dv):
                next_state = state[i, j] * g_exp[0] + k_loc[i] * v_new[j]
                StateF[slot_var[0], hx, i, j] = next_state
                HScan[tx, 1, hx, i, j] = next_state

    return rgdr_step


def make_decode_bwd_prim_func(
    *,
    num_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    num_slots: int,
    use_decay: bool,
    use_qk_l2norm: bool,
    dtype,
    threads: int = 128,
):
    """Build the decode-path GDR backward ``@T.prim_func``.

    Grid: ``T.Kernel(num_slots, num_heads)``.  Each CTA finds the token that
    maps to its slot (by scanning ``StateIndices``), re-computes intermediate
    activations from ``HScan``, and accumulates gradients for ``Q``, ``K``,
    ``V``, ``Beta``, ``Decay``, and ``State0``.

    Buffer layout mirrors the forward kernel plus gradient outputs::

        Q, K, V, Beta, Decay:  same as forward
        HScan:  (num_tokens, NT+1+(NT%2), num_heads, Dq, Dv)  — float32
            Pre-update state at index 0, post-update at index 1.
        StateIndices: (num_tokens,) int32
        dO:     (num_tokens, num_heads, v_head_dim)  — compute dtype
        dStateF:(num_slots, num_heads, Dq, Dv)       — float32
        dQ, dK: (num_tokens, num_heads, Dq)          — compute dtype
        dV:     (num_tokens, num_heads, Dv)           — compute dtype
        dBeta:  (num_tokens, num_heads)               — compute dtype
        dDecay: (num_tokens, num_heads)               — compute dtype
        dState0:(num_slots, num_heads, Dq, Dv)        — float32

    Args:
        num_tokens: ``NT`` (decode: equals num_requests).
        num_heads: ``H``.
        qk_head_dim: ``Dq``.
        v_head_dim: ``Dv``.
        num_slots: ``NS``.
        use_decay: Whether to differentiate through the decay gate.
        use_qk_l2norm: Whether Q/K were L2-normalised in the forward pass.
        dtype: Compute dtype.
        threads: Threads per CUDA CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` (``rgdr_step_bwd``).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    NT, H, Dq, Dv, NS = num_tokens, num_heads, qk_head_dim, v_head_dim, num_slots
    HS = NT + 1 + (NT % 2)
    scale = 1.0 / math.sqrt(Dq)

    @T.prim_func
    def rgdr_step_bwd(
        Q: T.Tensor((NT, H, Dq), ts),
        K: T.Tensor((NT, H, Dq), ts),
        V: T.Tensor((NT, H, Dv), ts),
        Beta: T.Tensor((NT, H), ts),
        Decay: T.Tensor((NT, H), ts),
        HScan: T.Tensor((NT, HS, H, Dq, Dv), accum),
        StateIndices: T.Tensor((NT,), "int32"),
        dO: T.Tensor((NT, H, Dv), ts),
        dStateF: T.Tensor((NS, H, Dq, Dv), accum),
        dQ: T.Tensor((NT, H, Dq), ts),
        dK: T.Tensor((NT, H, Dq), ts),
        dV: T.Tensor((NT, H, Dv), ts),
        dBeta: T.Tensor((NT, H), ts),
        dDecay: T.Tensor((NT, H), ts),
        dState0: T.Tensor((NS, H, Dq, Dv), accum),
    ):
        with T.Kernel(NS, H, threads=threads) as (sx, hx):
            _nt_pin = T.alloc_fragment((NT,), accum)
            _hs_pin = T.alloc_fragment((1,), accum)
            _ = _nt_pin
            _hs_pin[0] = HS

            token_var = T.alloc_fragment((1,), "int32")
            token_var[0] = -1
            for tok in T.serial(NT):
                if StateIndices[tok] == sx:
                    token_var[0] = tok
            active_slot = token_var[0] >= 0
            tx = T.if_then_else(active_slot, token_var[0], 0)

            dh = T.alloc_fragment((Dq, Dv), accum)
            dh_pre = T.alloc_fragment((Dq, Dv), accum)
            h_prev = T.alloc_fragment((Dq, Dv), accum)
            h_pre = T.alloc_fragment((Dq, Dv), accum)
            h_new = T.alloc_fragment((Dq, Dv), accum)
            q_raw = T.alloc_fragment((Dq,), accum)
            k_raw = T.alloc_fragment((Dq,), accum)
            q_loc = T.alloc_fragment((Dq,), accum)
            k_loc = T.alloc_fragment((Dq,), accum)
            v_loc = T.alloc_fragment((Dv,), accum)
            do_loc = T.alloc_fragment((Dv,), accum)
            q_sq = T.alloc_fragment((Dq,), accum)
            k_sq = T.alloc_fragment((Dq,), accum)
            q_norm = T.alloc_fragment((1,), accum)
            k_norm = T.alloc_fragment((1,), accum)
            inv_q = T.alloc_fragment((1,), accum)
            inv_k = T.alloc_fragment((1,), accum)
            beta_val = T.alloc_fragment((1,), accum)
            g_exp = T.alloc_fragment((1,), accum)
            kv_prod = T.alloc_fragment((Dq, Dv), accum)
            kv_mem = T.alloc_fragment((Dv,), accum)
            delta = T.alloc_fragment((Dv,), accum)
            dq_prod = T.alloc_fragment((Dq, Dv), accum)
            dq_scaled = T.alloc_fragment((Dq,), accum)
            ddelta_prod = T.alloc_fragment((Dq, Dv), accum)
            ddelta = T.alloc_fragment((Dv,), accum)
            dk_prod = T.alloc_fragment((Dq, Dv), accum)
            dk_loc = T.alloc_fragment((Dq,), accum)
            dkv = T.alloc_fragment((Dv,), accum)
            dv_loc = T.alloc_fragment((Dv,), accum)
            dbeta_prod = T.alloc_fragment((Dv,), accum)
            dbeta_acc = T.alloc_fragment((1,), accum)
            dg_prod = T.alloc_fragment((Dq, Dv), accum)
            dg_rows = T.alloc_fragment((Dq,), accum)
            dg_acc = T.alloc_fragment((1,), accum)
            dq_normed = T.alloc_fragment((Dq,), accum)
            dk_normed = T.alloc_fragment((Dq,), accum)
            dotq_prod = T.alloc_fragment((Dq,), accum)
            dotk_prod = T.alloc_fragment((Dq,), accum)
            dotq = T.alloc_fragment((1,), accum)
            dotk = T.alloc_fragment((1,), accum)

            for i, j in T.Parallel(Dq, Dv):
                dState0[sx, hx, i, j] = dStateF[sx, hx, i, j]

            for i in T.Parallel(Dq):
                q_raw[i] = T.Cast(accum, Q[tx, hx, i])
                k_raw[i] = T.Cast(accum, K[tx, hx, i])
                q_loc[i] = q_raw[i]
                k_loc[i] = k_raw[i]
            for j in T.Parallel(Dv):
                v_loc[j] = T.Cast(accum, V[tx, hx, j])
                do_loc[j] = T.Cast(accum, dO[tx, hx, j])

            if use_qk_l2norm:
                for i in T.Parallel(Dq):
                    q_sq[i] = q_raw[i] * q_raw[i]
                    k_sq[i] = k_raw[i] * k_raw[i]
                T.reduce_sum(q_sq, q_norm, dim=0, clear=True)
                T.reduce_sum(k_sq, k_norm, dim=0, clear=True)
                inv_q[0] = 1.0 / T.sqrt(T.max(q_norm[0], 1e-12))
                inv_k[0] = 1.0 / T.sqrt(T.max(k_norm[0], 1e-12))
                for i in T.Parallel(Dq):
                    q_loc[i] = q_raw[i] * inv_q[0]
                    k_loc[i] = k_raw[i] * inv_k[0]
            else:
                inv_q[0] = 1.0
                inv_k[0] = 1.0

            beta_val[0] = T.Cast(accum, Beta[tx, hx])
            if use_decay:
                g_exp[0] = T.exp(T.Cast(accum, Decay[tx, hx]))
            else:
                g_exp[0] = 1.0

            for i, j in T.Parallel(Dq, Dv):
                h_prev[i, j] = HScan[tx, 0, hx, i, j]
                h_new[i, j] = HScan[tx, 1, hx, i, j]
                h_pre[i, j] = h_prev[i, j] * g_exp[0]
                kv_prod[i, j] = h_pre[i, j] * k_loc[i]
                dh[i, j] = dStateF[sx, hx, i, j]
            T.reduce_sum(kv_prod, kv_mem, dim=0, clear=True)

            for j in T.Parallel(Dv):
                delta[j] = beta_val[0] * (v_loc[j] - kv_mem[j])

            for i, j in T.Parallel(Dq, Dv):
                dq_prod[i, j] = h_new[i, j] * do_loc[j]
            T.reduce_sum(dq_prod, dq_scaled, dim=1, clear=True)

            for i in T.Parallel(Dq):
                dq_normed[i] = dq_scaled[i] * scale
                dk_loc[i] = 0.0

            for i, j in T.Parallel(Dq, Dv):
                dh[i, j] = dh[i, j] + q_loc[i] * scale * do_loc[j]
                dh_pre[i, j] = dh[i, j]
                dk_prod[i, j] = dh[i, j] * delta[j]
                ddelta_prod[i, j] = dh[i, j] * k_loc[i]
            T.reduce_sum(dk_prod, dk_loc, dim=1, clear=True)
            T.reduce_sum(ddelta_prod, ddelta, dim=0, clear=True)

            for j in T.Parallel(Dv):
                dv_loc[j] = ddelta[j] * beta_val[0]
                dkv[j] = -ddelta[j] * beta_val[0]
                dbeta_prod[j] = ddelta[j] * (v_loc[j] - kv_mem[j])
            T.reduce_sum(dbeta_prod, dbeta_acc, dim=0, clear=True)

            for i, j in T.Parallel(Dq, Dv):
                dh_pre[i, j] = dh_pre[i, j] + dkv[j] * k_loc[i]
                dk_prod[i, j] = dkv[j] * h_pre[i, j]
            T.reduce_sum(dk_prod, dk_normed, dim=1, clear=True)

            for i in T.Parallel(Dq):
                dk_loc[i] = dk_loc[i] + dk_normed[i]

            for i, j in T.Parallel(Dq, Dv):
                dg_prod[i, j] = dh_pre[i, j] * h_prev[i, j]
            T.reduce_sum(dg_prod, dg_rows, dim=1, clear=True)
            T.reduce_sum(dg_rows, dg_acc, dim=0, clear=True)

            if use_qk_l2norm:
                for i in T.Parallel(Dq):
                    dotq_prod[i] = dq_normed[i] * q_loc[i]
                    dotk_prod[i] = dk_loc[i] * k_loc[i]
                T.reduce_sum(dotq_prod, dotq, dim=0, clear=True)
                T.reduce_sum(dotk_prod, dotk, dim=0, clear=True)
                for i in T.Parallel(Dq):
                    if active_slot:
                        dQ[tx, hx, i] = T.Cast(ts, inv_q[0] * (dq_normed[i] - q_loc[i] * dotq[0]))
                        dK[tx, hx, i] = T.Cast(ts, inv_k[0] * (dk_loc[i] - k_loc[i] * dotk[0]))
            else:
                for i in T.Parallel(Dq):
                    if active_slot:
                        dQ[tx, hx, i] = T.Cast(ts, dq_normed[i])
                        dK[tx, hx, i] = T.Cast(ts, dk_loc[i])

            for j in T.Parallel(Dv):
                if active_slot:
                    dV[tx, hx, j] = T.Cast(ts, dv_loc[j])
            if active_slot:
                dBeta[tx, hx] = T.Cast(ts, dbeta_acc[0])
                if use_decay:
                    dDecay[tx, hx] = T.Cast(ts, dg_acc[0] * g_exp[0])
                else:
                    dDecay[tx, hx] = T.Cast(ts, 0.0)

            for i, j in T.Parallel(Dq, Dv):
                if active_slot:
                    dState0[sx, hx, i, j] = dh_pre[i, j] * g_exp[0]

    return rgdr_step_bwd


def make_ragged_prim_func(
    *,
    num_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    num_slots: int,
    num_requests: int,
    use_decay: bool,
    use_qk_l2norm: bool,
    dtype,
    threads: int = 128,
):
    """Build the general ragged GDR forward ``@T.prim_func`` (prefill + decode mix).

    Grid: ``T.Kernel(num_requests, num_heads)``.  Each CTA:

    1. Reads ``slot = StateIndices[rx]`` and loads ``State0[slot, hx]``.
    2. Walks all ``num_tokens`` positions via ``T.serial(NT)``; applies the
       GDR step only when ``t ∈ [QueryStartLoc[rx], QueryStartLoc[rx+1])``.
    3. Writes the accumulated output to ``O``, the updated state to
       ``StateF[slot, hx]``, and per-position snapshots to ``HScan``.

    Because each CTA applies the predicate ``active = (t >= tok_start) & (t <
    tok_end)`` to every token in the batch, the inner loop runs ``num_tokens``
    iterations regardless of request length.  This is efficient when
    ``num_requests`` is small relative to ``num_tokens`` (typical prefill case).

    Buffer layout::

        Q, K:          (num_tokens, num_heads, qk_head_dim)       — compute dtype
        V:             (num_tokens, num_heads, v_head_dim)         — compute dtype
        Beta:          (num_tokens, num_heads)                     — compute dtype
        Decay:         (num_tokens, num_heads)                     — compute dtype
        State0:        (num_slots, num_heads, Dq, Dv)             — float32
        QueryStartLoc: (num_requests + 1,)                        — int32
        StateIndices:  (num_requests,)                            — int32
        O:             (num_tokens, num_heads, v_head_dim)         — compute dtype
        StateF:        (num_slots, num_heads, Dq, Dv)             — float32
        HScan:         (num_requests, NT+1+(NT%2), num_heads, Dq, Dv) — float32

    Args:
        num_tokens: Total tokens ``NT`` across all requests.
        num_heads: ``H``.
        qk_head_dim: ``Dq``.
        v_head_dim: ``Dv``.
        num_slots: State pool size ``NS``.
        num_requests: Number of requests ``NR``.
        use_decay: Whether to apply ``exp(Decay)`` per token.
        use_qk_l2norm: Whether to L2-normalise Q and K before the inner product.
        dtype: Compute dtype.
        threads: Threads per CUDA CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` (``rgdr_ragged``).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    NT, H, Dq, Dv = num_tokens, num_heads, qk_head_dim, v_head_dim
    NS, NR = num_slots, num_requests
    HS = NT + 1 + (NT % 2)
    scale = 1.0 / math.sqrt(Dq)

    @T.prim_func
    def rgdr_ragged(
        Q: T.Tensor((NT, H, Dq), ts),
        K: T.Tensor((NT, H, Dq), ts),
        V: T.Tensor((NT, H, Dv), ts),
        Beta: T.Tensor((NT, H), ts),
        Decay: T.Tensor((NT, H), ts),
        State0: T.Tensor((NS, H, Dq, Dv), accum),
        QueryStartLoc: T.Tensor((NR + 1,), "int32"),
        StateIndices: T.Tensor((NR,), "int32"),
        O: T.Tensor((NT, H, Dv), ts),
        StateF: T.Tensor((NS, H, Dq, Dv), accum),
        HScan: T.Tensor((NR, HS, H, Dq, Dv), accum),
    ):
        with T.Kernel(NR, H, threads=threads) as (rx, hx):
            _ns_pin = T.alloc_fragment((NS,), accum)
            _nt_pin = T.alloc_fragment((1,), accum)
            _hs_pin = T.alloc_fragment((1,), accum)
            _ = _ns_pin
            _ = _nt_pin
            _hs_pin[0] = HS

            slot_var = T.alloc_fragment((1,), "int32")
            tok_start = T.alloc_fragment((1,), "int32")
            tok_end = T.alloc_fragment((1,), "int32")
            local_pos = T.alloc_fragment((1,), "int32")
            slot_var[0] = StateIndices[rx]
            tok_start[0] = QueryStartLoc[rx]
            tok_end[0] = QueryStartLoc[rx + 1]
            local_pos[0] = 0

            state = T.alloc_fragment((Dq, Dv), accum)
            q_loc = T.alloc_fragment((Dq,), accum)
            k_loc = T.alloc_fragment((Dq,), accum)
            v_loc = T.alloc_fragment((Dv,), accum)
            beta_val = T.alloc_fragment((1,), accum)
            g_exp = T.alloc_fragment((1,), accum)
            q_sq = T.alloc_fragment((Dq,), accum)
            k_sq = T.alloc_fragment((Dq,), accum)
            qk_norm = T.alloc_fragment((1,), accum)
            kk_norm = T.alloc_fragment((1,), accum)
            inv_q = T.alloc_fragment((1,), accum)
            inv_k = T.alloc_fragment((1,), accum)
            k_state_prod = T.alloc_fragment((Dq, Dv), accum)
            k_state = T.alloc_fragment((Dv,), accum)
            v_new = T.alloc_fragment((Dv,), accum)
            q_state_prod = T.alloc_fragment((Dq, Dv), accum)
            q_state = T.alloc_fragment((Dv,), accum)
            qk_prod = T.alloc_fragment((Dq,), accum)
            qk = T.alloc_fragment((1,), accum)

            for i, j in T.Parallel(Dq, Dv):
                state[i, j] = State0[slot_var[0], hx, i, j]
                HScan[rx, 0, hx, i, j] = state[i, j]

            for t in T.serial(NT):
                active = (t >= tok_start[0]) & (t < tok_end[0])

                for i in T.Parallel(Dq):
                    q_loc[i] = T.Cast(accum, Q[t, hx, i])
                    k_loc[i] = T.Cast(accum, K[t, hx, i])
                for j in T.Parallel(Dv):
                    v_loc[j] = T.Cast(accum, V[t, hx, j])
                beta_val[0] = T.Cast(accum, Beta[t, hx])
                if use_decay:
                    g_exp[0] = T.exp(T.Cast(accum, Decay[t, hx]))
                else:
                    g_exp[0] = 1.0

                if use_qk_l2norm:
                    for i in T.Parallel(Dq):
                        q_sq[i] = q_loc[i] * q_loc[i]
                        k_sq[i] = k_loc[i] * k_loc[i]
                    T.reduce_sum(q_sq, qk_norm, dim=0, clear=True)
                    T.reduce_sum(k_sq, kk_norm, dim=0, clear=True)
                    inv_q[0] = 1.0 / T.sqrt(T.max(qk_norm[0], 1e-12))
                    inv_k[0] = 1.0 / T.sqrt(T.max(kk_norm[0], 1e-12))
                    for i in T.Parallel(Dq):
                        q_loc[i] = q_loc[i] * inv_q[0]
                        k_loc[i] = k_loc[i] * inv_k[0]

                for i in T.Parallel(Dq):
                    q_loc[i] = q_loc[i] * scale

                for i, j in T.Parallel(Dq, Dv):
                    k_state_prod[i, j] = k_loc[i] * state[i, j]
                T.reduce_sum(k_state_prod, k_state, dim=0, clear=True)
                for j in T.Parallel(Dv):
                    v_new[j] = beta_val[0] * (v_loc[j] - g_exp[0] * k_state[j])

                for i, j in T.Parallel(Dq, Dv):
                    q_state_prod[i, j] = q_loc[i] * state[i, j]
                T.reduce_sum(q_state_prod, q_state, dim=0, clear=True)

                for i in T.Parallel(Dq):
                    qk_prod[i] = q_loc[i] * k_loc[i]
                T.reduce_sum(qk_prod, qk, dim=0, clear=True)

                for j in T.Parallel(Dv):
                    out_val = g_exp[0] * q_state[j] + qk[0] * v_new[j]
                    if active:
                        O[t, hx, j] = T.Cast(ts, out_val)

                for i, j in T.Parallel(Dq, Dv):
                    new_s = state[i, j] * g_exp[0] + k_loc[i] * v_new[j]
                    state[i, j] = T.if_then_else(active, new_s, state[i, j])
                    if active:
                        HScan[rx, local_pos[0] + 1, hx, i, j] = state[i, j]

                if active:
                    local_pos[0] = local_pos[0] + 1

            for i, j in T.Parallel(Dq, Dv):
                StateF[slot_var[0], hx, i, j] = state[i, j]

    return rgdr_ragged


def make_ragged_bwd_prim_func(
    *,
    num_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    num_slots: int,
    num_requests: int,
    use_decay: bool,
    use_qk_l2norm: bool,
    dtype,
    threads: int = 128,
):
    """Build the reverse-time adjoint for the ragged GDR scan (older variant).

    Note: this function is defined but **not used** by the current VJP wiring in
    ``_impl.py``, which uses :func:`make_ragged_bwd_simple_prim_func` instead.
    It is retained for reference / experimentation.

    Grid: ``T.Kernel(num_requests, num_heads)``.  Each CTA iterates the request's
    tokens in reverse order using the stored ``HScan`` snapshots.

    Args:
        num_tokens: ``NT``.
        num_heads: ``H``.
        qk_head_dim: ``Dq``.
        v_head_dim: ``Dv``.
        num_slots: ``NS``.
        num_requests: ``NR``.
        use_decay: Whether to differentiate through the decay gate.
        use_qk_l2norm: Whether Q/K were L2-normalised in the forward pass.
        dtype: Compute dtype.
        threads: Threads per CUDA CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` (``rgdr_bwd``).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    NT, H, Dq, Dv = num_tokens, num_heads, qk_head_dim, v_head_dim
    NS, NR = num_slots, num_requests
    HS = NT + 1 + (NT % 2)
    scale = 1.0 / math.sqrt(Dq)

    @T.prim_func
    def rgdr_bwd(
        Q: T.Tensor((NT, H, Dq), ts),
        K: T.Tensor((NT, H, Dq), ts),
        V: T.Tensor((NT, H, Dv), ts),
        Beta: T.Tensor((NT, H), ts),
        Decay: T.Tensor((NT, H), ts),
        HScan: T.Tensor((NR, HS, H, Dq, Dv), accum),
        QueryStartLoc: T.Tensor((NR + 1,), "int32"),
        StateIndices: T.Tensor((NR,), "int32"),
        dO: T.Tensor((NT, H, Dv), ts),
        dStateF: T.Tensor((NS, H, Dq, Dv), accum),
        dQ: T.Tensor((NT, H, Dq), ts),
        dK: T.Tensor((NT, H, Dq), ts),
        dV: T.Tensor((NT, H, Dv), ts),
        dBeta: T.Tensor((NT, H), ts),
        dDecay: T.Tensor((NT, H), ts),
        dState0: T.Tensor((NS, H, Dq, Dv), accum),
    ):
        with T.Kernel(NR, H, threads=threads) as (rx, hx):
            _ns_pin = T.alloc_fragment((NS,), accum)
            _hs_pin = T.alloc_fragment((1,), accum)
            _ = _ns_pin
            _hs_pin[0] = HS

            slot_var = T.alloc_fragment((1,), "int32")
            tok_start = T.alloc_fragment((1,), "int32")
            tok_end = T.alloc_fragment((1,), "int32")
            seq_len = T.alloc_fragment((1,), "int32")
            slot_var[0] = StateIndices[rx]
            tok_start[0] = QueryStartLoc[rx]
            tok_end[0] = QueryStartLoc[rx + 1]
            seq_len[0] = tok_end[0] - tok_start[0]

            dh = T.alloc_fragment((Dq, Dv), accum)
            dh_pre = T.alloc_fragment((Dq, Dv), accum)
            h_prev = T.alloc_fragment((Dq, Dv), accum)
            h_pre = T.alloc_fragment((Dq, Dv), accum)
            h_new = T.alloc_fragment((Dq, Dv), accum)
            q_raw = T.alloc_fragment((Dq,), accum)
            k_raw = T.alloc_fragment((Dq,), accum)
            q_loc = T.alloc_fragment((Dq,), accum)
            k_loc = T.alloc_fragment((Dq,), accum)
            v_loc = T.alloc_fragment((Dv,), accum)
            do_loc = T.alloc_fragment((Dv,), accum)
            q_sq = T.alloc_fragment((Dq,), accum)
            k_sq = T.alloc_fragment((Dq,), accum)
            q_norm = T.alloc_fragment((1,), accum)
            k_norm = T.alloc_fragment((1,), accum)
            inv_q = T.alloc_fragment((1,), accum)
            inv_k = T.alloc_fragment((1,), accum)
            beta_val = T.alloc_fragment((1,), accum)
            g_exp = T.alloc_fragment((1,), accum)
            kv_prod = T.alloc_fragment((Dq, Dv), accum)
            kv_mem = T.alloc_fragment((Dv,), accum)
            delta = T.alloc_fragment((Dv,), accum)
            dq_prod = T.alloc_fragment((Dq, Dv), accum)
            dq_scaled = T.alloc_fragment((Dq,), accum)
            ddelta_prod = T.alloc_fragment((Dq, Dv), accum)
            ddelta = T.alloc_fragment((Dv,), accum)
            dk_prod = T.alloc_fragment((Dq, Dv), accum)
            dk_loc = T.alloc_fragment((Dq,), accum)
            dkv = T.alloc_fragment((Dv,), accum)
            dv_loc = T.alloc_fragment((Dv,), accum)
            dbeta_prod = T.alloc_fragment((Dv,), accum)
            dbeta_acc = T.alloc_fragment((1,), accum)
            dg_prod = T.alloc_fragment((Dq, Dv), accum)
            dg_rows = T.alloc_fragment((Dq,), accum)
            dg_acc = T.alloc_fragment((1,), accum)
            dq_normed = T.alloc_fragment((Dq,), accum)
            dk_normed = T.alloc_fragment((Dq,), accum)
            dotq_prod = T.alloc_fragment((Dq,), accum)
            dotk_prod = T.alloc_fragment((Dq,), accum)
            dotq = T.alloc_fragment((1,), accum)
            dotk = T.alloc_fragment((1,), accum)

            for i, j in T.Parallel(Dq, Dv):
                dh[i, j] = dStateF[slot_var[0], hx, i, j]

            for local_iter in T.serial(NT):
                active = local_iter < seq_len[0]
                local_pos = seq_len[0] - 1 - local_iter
                t = T.if_then_else(active, tok_start[0] + local_pos, 0)
                scan_pos = T.if_then_else(active, local_pos + 1, 0)

                for i in T.Parallel(Dq):
                    q_raw[i] = T.Cast(accum, Q[t, hx, i])
                    k_raw[i] = T.Cast(accum, K[t, hx, i])
                    q_loc[i] = q_raw[i]
                    k_loc[i] = k_raw[i]
                for j in T.Parallel(Dv):
                    v_loc[j] = T.Cast(accum, V[t, hx, j])
                    do_loc[j] = T.Cast(accum, dO[t, hx, j])

                if use_qk_l2norm:
                    for i in T.Parallel(Dq):
                        q_sq[i] = q_raw[i] * q_raw[i]
                        k_sq[i] = k_raw[i] * k_raw[i]
                    T.reduce_sum(q_sq, q_norm, dim=0, clear=True)
                    T.reduce_sum(k_sq, k_norm, dim=0, clear=True)
                    inv_q[0] = 1.0 / T.sqrt(T.max(q_norm[0], 1e-12))
                    inv_k[0] = 1.0 / T.sqrt(T.max(k_norm[0], 1e-12))
                    for i in T.Parallel(Dq):
                        q_loc[i] = q_raw[i] * inv_q[0]
                        k_loc[i] = k_raw[i] * inv_k[0]
                else:
                    inv_q[0] = 1.0
                    inv_k[0] = 1.0

                beta_val[0] = T.Cast(accum, Beta[t, hx])
                if use_decay:
                    g_exp[0] = T.exp(T.Cast(accum, Decay[t, hx]))
                else:
                    g_exp[0] = 1.0

                for i, j in T.Parallel(Dq, Dv):
                    h_prev[i, j] = HScan[rx, scan_pos - 1, hx, i, j]
                    h_new[i, j] = HScan[rx, scan_pos, hx, i, j]
                    h_pre[i, j] = h_prev[i, j] * g_exp[0]
                    kv_prod[i, j] = h_pre[i, j] * k_loc[i]
                T.reduce_sum(kv_prod, kv_mem, dim=0, clear=True)

                for j in T.Parallel(Dv):
                    delta[j] = beta_val[0] * (v_loc[j] - kv_mem[j])

                for i, j in T.Parallel(Dq, Dv):
                    dq_prod[i, j] = h_new[i, j] * do_loc[j]
                T.reduce_sum(dq_prod, dq_scaled, dim=1, clear=True)

                for i in T.Parallel(Dq):
                    dq_normed[i] = dq_scaled[i] * scale
                    dk_loc[i] = 0.0

                for i, j in T.Parallel(Dq, Dv):
                    dh[i, j] = dh[i, j] + q_loc[i] * scale * do_loc[j]
                    dh_pre[i, j] = dh[i, j]
                    dk_prod[i, j] = dh[i, j] * delta[j]
                    ddelta_prod[i, j] = dh[i, j] * k_loc[i]
                T.reduce_sum(dk_prod, dk_loc, dim=1, clear=True)
                T.reduce_sum(ddelta_prod, ddelta, dim=0, clear=True)

                for j in T.Parallel(Dv):
                    dv_loc[j] = ddelta[j] * beta_val[0]
                    dkv[j] = -ddelta[j] * beta_val[0]
                    dbeta_prod[j] = ddelta[j] * (v_loc[j] - kv_mem[j])
                T.reduce_sum(dbeta_prod, dbeta_acc, dim=0, clear=True)

                for i, j in T.Parallel(Dq, Dv):
                    dh_pre[i, j] = dh_pre[i, j] + dkv[j] * k_loc[i]
                    dk_prod[i, j] = dkv[j] * h_pre[i, j]
                T.reduce_sum(dk_prod, dk_normed, dim=1, clear=True)

                for i in T.Parallel(Dq):
                    dk_loc[i] = dk_loc[i] + dk_normed[i]

                for i, j in T.Parallel(Dq, Dv):
                    dg_prod[i, j] = dh_pre[i, j] * h_prev[i, j]
                T.reduce_sum(dg_prod, dg_rows, dim=1, clear=True)
                T.reduce_sum(dg_rows, dg_acc, dim=0, clear=True)

                if use_qk_l2norm:
                    for i in T.Parallel(Dq):
                        dotq_prod[i] = dq_normed[i] * q_loc[i]
                        dotk_prod[i] = dk_loc[i] * k_loc[i]
                    T.reduce_sum(dotq_prod, dotq, dim=0, clear=True)
                    T.reduce_sum(dotk_prod, dotk, dim=0, clear=True)
                    for i in T.Parallel(Dq):
                        if active:
                            dQ[t, hx, i] = T.Cast(ts, inv_q[0] * (dq_normed[i] - q_loc[i] * dotq[0]))
                            dK[t, hx, i] = T.Cast(ts, inv_k[0] * (dk_loc[i] - k_loc[i] * dotk[0]))
                else:
                    for i in T.Parallel(Dq):
                        if active:
                            dQ[t, hx, i] = T.Cast(ts, dq_normed[i])
                            dK[t, hx, i] = T.Cast(ts, dk_loc[i])

                for j in T.Parallel(Dv):
                    if active:
                        dV[t, hx, j] = T.Cast(ts, dv_loc[j])
                if active:
                    dBeta[t, hx] = T.Cast(ts, dbeta_acc[0])
                    if use_decay:
                        dDecay[t, hx] = T.Cast(ts, dg_acc[0] * g_exp[0])
                    else:
                        dDecay[t, hx] = T.Cast(ts, 0.0)

                for i, j in T.Parallel(Dq, Dv):
                    if active:
                        dh[i, j] = dh_pre[i, j] * g_exp[0]

            for i, j in T.Parallel(Dq, Dv):
                dState0[slot_var[0], hx, i, j] = dh[i, j]

    return rgdr_bwd


def make_ragged_bwd_simple_prim_func(
    *,
    num_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    num_slots: int,
    num_requests: int,
    use_decay: bool,
    use_qk_l2norm: bool,
    dtype,
    threads: int = 128,
):
    """Build a layout-conservative reverse-time adjoint for ragged prefill.

    This is the **active** backward variant used by the VJP in ``_impl.py``.
    Unlike :func:`make_ragged_bwd_prim_func` it uses slot-indexed CTAs
    (``T.Kernel(num_slots, num_heads)``) rather than request-indexed CTAs,
    which avoids potential aliasing when multiple requests map to the same slot.

    Each CTA:

    1. Scans ``StateIndices`` to find which request (if any) maps to its slot.
    2. If active, iterates that request's tokens in reverse and accumulates
       ``dQ``, ``dK``, ``dV``, ``dBeta``, ``dDecay``, and ``dState0``.
    3. If inactive (slot not used by any request in this batch), the output
       gradient tensors are left at their initialised-zero values.

    Buffer layout::

        Same as :func:`make_ragged_bwd_prim_func` except:
        BlockTables axis-0 is ``num_slots`` (not ``num_requests``).

    Args:
        num_tokens: ``NT``.
        num_heads: ``H``.
        qk_head_dim: ``Dq``.
        v_head_dim: ``Dv``.
        num_slots: ``NS``.
        num_requests: ``NR``.
        use_decay: Whether to differentiate through the decay gate.
        use_qk_l2norm: Whether Q/K were L2-normalised in the forward pass.
        dtype: Compute dtype.
        threads: Threads per CUDA CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` (``rgdr_bwd_simple``).
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    NT, H, Dq, Dv = num_tokens, num_heads, qk_head_dim, v_head_dim
    NS, NR = num_slots, num_requests
    HS = NT + 1 + (NT % 2)
    scale = 1.0 / math.sqrt(Dq)

    @T.prim_func
    def rgdr_bwd_simple(
        Q: T.Tensor((NT, H, Dq), ts),
        K: T.Tensor((NT, H, Dq), ts),
        V: T.Tensor((NT, H, Dv), ts),
        Beta: T.Tensor((NT, H), ts),
        Decay: T.Tensor((NT, H), ts),
        HScan: T.Tensor((NR, HS, H, Dq, Dv), accum),
        QueryStartLoc: T.Tensor((NR + 1,), "int32"),
        StateIndices: T.Tensor((NR,), "int32"),
        dO: T.Tensor((NT, H, Dv), ts),
        dStateF: T.Tensor((NS, H, Dq, Dv), accum),
        dQ: T.Tensor((NT, H, Dq), ts),
        dK: T.Tensor((NT, H, Dq), ts),
        dV: T.Tensor((NT, H, Dv), ts),
        dBeta: T.Tensor((NT, H), ts),
        dDecay: T.Tensor((NT, H), ts),
        dState0: T.Tensor((NS, H, Dq, Dv), accum),
    ):
        with T.Kernel(NS, H, threads=threads) as (sx, hx):
            _nr_pin = T.alloc_fragment((NR,), accum)
            _hs_pin = T.alloc_fragment((1,), accum)
            _ = _nr_pin
            _hs_pin[0] = HS

            req_var = T.alloc_fragment((1,), "int32")
            tok_start = T.alloc_fragment((1,), "int32")
            tok_end = T.alloc_fragment((1,), "int32")
            seq_len = T.alloc_fragment((1,), "int32")
            req_var[0] = -1
            for req in T.serial(NR):
                if StateIndices[req] == sx:
                    req_var[0] = req
            active_req = req_var[0] >= 0
            rx = T.if_then_else(active_req, req_var[0], 0)
            tok_start[0] = T.if_then_else(active_req, QueryStartLoc[rx], 0)
            tok_end[0] = T.if_then_else(active_req, QueryStartLoc[rx + 1], 0)
            seq_len[0] = T.if_then_else(active_req, tok_end[0] - tok_start[0], 0)

            dh = T.alloc_fragment((Dq, Dv), accum)
            dh_pre = T.alloc_fragment((Dq, Dv), accum)
            h_prev = T.alloc_fragment((Dq, Dv), accum)
            h_new = T.alloc_fragment((Dq, Dv), accum)
            q_raw = T.alloc_fragment((Dq,), accum)
            k_raw = T.alloc_fragment((Dq,), accum)
            q_loc = T.alloc_fragment((Dq,), accum)
            k_loc = T.alloc_fragment((Dq,), accum)
            v_loc = T.alloc_fragment((Dv,), accum)
            do_loc = T.alloc_fragment((Dv,), accum)
            kv_mem = T.alloc_fragment((Dv,), accum)
            delta = T.alloc_fragment((Dv,), accum)
            ddelta = T.alloc_fragment((Dv,), accum)
            dk_loc = T.alloc_fragment((Dq,), accum)
            dv_loc = T.alloc_fragment((Dv,), accum)
            dq_loc = T.alloc_fragment((Dq,), accum)
            beta_val = T.alloc_fragment((1,), accum)
            g_exp = T.alloc_fragment((1,), accum)
            q_norm = T.alloc_fragment((1,), accum)
            k_norm = T.alloc_fragment((1,), accum)
            inv_q = T.alloc_fragment((1,), accum)
            inv_k = T.alloc_fragment((1,), accum)
            dot_q = T.alloc_fragment((1,), accum)
            dot_k = T.alloc_fragment((1,), accum)
            dbeta_acc = T.alloc_fragment((1,), accum)
            ddecay_acc = T.alloc_fragment((1,), accum)

            for i, j in T.Parallel(Dq, Dv):
                dh[i, j] = dStateF[sx, hx, i, j]

            for local_iter in T.serial(NT):
                active = active_req & (local_iter < seq_len[0])
                local_pos = seq_len[0] - 1 - local_iter
                t = T.if_then_else(active, tok_start[0] + local_pos, 0)
                scan_pos = T.if_then_else(active, local_pos + 1, 1)

                if active:
                    for i in T.Parallel(Dq):
                        q_raw[i] = T.Cast(accum, Q[t, hx, i])
                        k_raw[i] = T.Cast(accum, K[t, hx, i])
                        q_loc[i] = q_raw[i]
                        k_loc[i] = k_raw[i]
                    for j in T.Parallel(Dv):
                        v_loc[j] = T.Cast(accum, V[t, hx, j])
                        do_loc[j] = T.Cast(accum, dO[t, hx, j])

                    if use_qk_l2norm:
                        q_norm[0] = 0.0
                        k_norm[0] = 0.0
                        for i in T.serial(Dq):
                            q_norm[0] = q_norm[0] + q_raw[i] * q_raw[i]
                            k_norm[0] = k_norm[0] + k_raw[i] * k_raw[i]
                        inv_q[0] = 1.0 / T.sqrt(T.max(q_norm[0], 1e-12))
                        inv_k[0] = 1.0 / T.sqrt(T.max(k_norm[0], 1e-12))
                        for i in T.Parallel(Dq):
                            q_loc[i] = q_raw[i] * inv_q[0]
                            k_loc[i] = k_raw[i] * inv_k[0]
                    else:
                        inv_q[0] = 1.0
                        inv_k[0] = 1.0

                    beta_val[0] = T.Cast(accum, Beta[t, hx])
                    if use_decay:
                        g_exp[0] = T.exp(T.Cast(accum, Decay[t, hx]))
                    else:
                        g_exp[0] = 1.0

                    for i, j in T.Parallel(Dq, Dv):
                        h_prev[i, j] = HScan[rx, scan_pos - 1, hx, i, j]
                        h_new[i, j] = HScan[rx, scan_pos, hx, i, j]

                    for j in T.Parallel(Dv):
                        kv_mem[j] = 0.0
                        ddelta[j] = 0.0
                        dv_loc[j] = 0.0
                    for i in T.Parallel(Dq):
                        dk_loc[i] = 0.0
                        dq_loc[i] = 0.0

                    for j in T.serial(Dv):
                        for i in T.serial(Dq):
                            kv_mem[j] = kv_mem[j] + h_prev[i, j] * g_exp[0] * k_loc[i]
                        delta[j] = beta_val[0] * (v_loc[j] - kv_mem[j])

                    for i in T.serial(Dq):
                        for j in T.serial(Dv):
                            dq_loc[i] = dq_loc[i] + h_new[i, j] * do_loc[j]
                        dq_loc[i] = dq_loc[i] * scale

                    for i, j in T.Parallel(Dq, Dv):
                        dh[i, j] = dh[i, j] + q_loc[i] * scale * do_loc[j]
                        dh_pre[i, j] = dh[i, j]

                    for i in T.serial(Dq):
                        for j in T.serial(Dv):
                            dk_loc[i] = dk_loc[i] + dh[i, j] * delta[j]
                            ddelta[j] = ddelta[j] + dh[i, j] * k_loc[i]

                    dbeta_acc[0] = 0.0
                    for j in T.serial(Dv):
                        dv_loc[j] = ddelta[j] * beta_val[0]
                        dbeta_acc[0] = dbeta_acc[0] + ddelta[j] * (v_loc[j] - kv_mem[j])
                        ddelta[j] = -ddelta[j] * beta_val[0]

                    for i in T.serial(Dq):
                        for j in T.serial(Dv):
                            dh_pre[i, j] = dh_pre[i, j] + ddelta[j] * k_loc[i]
                            dk_loc[i] = dk_loc[i] + ddelta[j] * h_prev[i, j] * g_exp[0]

                    ddecay_acc[0] = 0.0
                    for i in T.serial(Dq):
                        for j in T.serial(Dv):
                            ddecay_acc[0] = ddecay_acc[0] + dh_pre[i, j] * h_prev[i, j]

                    if use_qk_l2norm:
                        dot_q[0] = 0.0
                        dot_k[0] = 0.0
                        for i in T.serial(Dq):
                            dot_q[0] = dot_q[0] + dq_loc[i] * q_loc[i]
                            dot_k[0] = dot_k[0] + dk_loc[i] * k_loc[i]
                        for i in T.Parallel(Dq):
                            dQ[t, hx, i] = T.Cast(ts, inv_q[0] * (dq_loc[i] - q_loc[i] * dot_q[0]))
                            dK[t, hx, i] = T.Cast(ts, inv_k[0] * (dk_loc[i] - k_loc[i] * dot_k[0]))
                    else:
                        for i in T.Parallel(Dq):
                            dQ[t, hx, i] = T.Cast(ts, dq_loc[i])
                            dK[t, hx, i] = T.Cast(ts, dk_loc[i])

                    for j in T.Parallel(Dv):
                        dV[t, hx, j] = T.Cast(ts, dv_loc[j])
                    dBeta[t, hx] = T.Cast(ts, dbeta_acc[0])
                    if use_decay:
                        dDecay[t, hx] = T.Cast(ts, ddecay_acc[0] * g_exp[0])
                    else:
                        dDecay[t, hx] = T.Cast(ts, 0.0)

                    for i, j in T.Parallel(Dq, Dv):
                        dh[i, j] = dh_pre[i, j] * g_exp[0]

            for i, j in T.Parallel(Dq, Dv):
                dState0[sx, hx, i, j] = dh[i, j]

    return rgdr_bwd_simple
