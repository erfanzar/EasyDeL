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

"""TileLang prim_funcs for the padded GDR/KDA recurrent scan.

Three factory functions are provided:

* :func:`make_init_state_prim_func` — allocates a zero float32 hidden state.
* :func:`make_fwd_states_prim_func` — forward recurrent scan; saves the full
  per-timestep hidden-state history (``HScan``) needed by the backward pass.
* :func:`make_bwd_prim_func` — backward pass; reverses the scan in time,
  propagating cotangents through the recurrent update.

All three kernels use a single-block-per-(head, batch) grid layout with
``threads=128`` threads per block.  Internal accumulation is always in
float32 (``accum = "float32"``); inputs and outputs are in the dtype of
``query`` (``ts``).

Padding convention:
  The ``HScan`` buffer has length ``S + 1 + (S % 2)`` along the time axis
  to guarantee 64-bit alignment of each row even when ``S`` is odd.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    """Convert a NumPy/JAX dtype to the TileLang string identifier.

    Args:
        dtype: A dtype understood by ``jnp.dtype``.

    Returns:
        One of ``"float16"``, ``"bfloat16"``, or ``"float32"``.

    Raises:
        TypeError: If ``dtype`` is not one of the three supported types.
    """
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for tile-lang gated_delta_rule: {dtype}")
    return mapping[canonical]


def make_init_state_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    threads: int = 128,
):
    """Build a TileLang prim_func that returns a zero float32 hidden state.

    The generated kernel fills ``H0[b, h, i, j] = 0`` for all ``b, h, i, j``
    using a 2-D parallel loop over ``(Dq, Dv)``.

    Grid layout:
        ``Kernel(H, B)`` — one block per (head, batch) pair.

    Args:
        batch: Batch dimension ``B``.
        seq_len: Sequence length ``S`` (only used for the type-reference scalar
            that forces TileLang to specialise on the input dtype).
        num_heads: Number of attention heads ``H``.
        qk_head_dim: Query/key head dimension ``Dq``.
        v_head_dim: Value head dimension ``Dv``.
        dtype: Input dtype of ``Q``; determines the TileLang type string.
        threads: Number of CUDA threads per block (default 128).

    Returns:
        A ``T.prim_func`` with signature
        ``(Q: [B, S, H, Dq] <ts>, H0: [B, H, Dq, Dv] <float32>)``.
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, Dq, Dv = batch, seq_len, num_heads, qk_head_dim, v_head_dim

    @T.prim_func
    def gdr_init_state(
        Q: T.Tensor((B, S, H, Dq), ts),
        H0: T.Tensor((B, H, Dq, Dv), accum),
    ):
        """TileLang kernel: fill ``H0`` with zeros (shape ``[B, H, Dq, Dv]``)."""
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            _ts_ref = T.alloc_fragment((1,), ts)
            _seq_ref = T.alloc_fragment((1,), accum)
            _ts_ref[0] = Q[0, 0, 0, 0]
            _seq_ref[0] = S
            for i, j in T.Parallel(Dq, Dv):
                H0[bx, hx, i, j] = 0.0

    return gdr_init_state


def make_fwd_states_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    softmax_scale: float,
    use_decay: bool,
    use_qk_l2norm: bool,
    dtype,
    threads: int = 128,
):
    """Build the TileLang forward prim_func for the GDR recurrent scan.

    The generated kernel iterates over all ``S`` timesteps in serial order,
    performing the following update at each step ``t``::

        # optional L2 normalisation
        q_t = q_t / ||q_t||   (if use_qk_l2norm)
        k_t = k_t / ||k_t||   (if use_qk_l2norm)

        q_t = q_t * softmax_scale
        g   = exp(Decay[t])   (if use_decay, else g = 1.0)

        h_t     = g * h_{t-1}
        kh      = h_t^T k_t                    # Dv-vector
        delta   = beta_t * (v_t - kh)
        h_t     = h_t + k_t ⊗ delta            # outer product update
        o_t     = h_t^T q_t                    # Dv-vector output

    The full hidden-state history is stored in ``HScan[b, 0..S, h, :, :]``
    (index 0 = initial state, index t+1 = state after timestep t).

    Grid layout:
        ``Kernel(H, B)`` — one block per (head, batch) pair.

    Shared memory / fragment allocation per block:
        ``h_state (Dq, Dv)``, plus ~12 smaller ``Dq`` or ``Dv`` fragment
        buffers, all in float32.

    Args:
        batch: Batch size ``B``.
        seq_len: Sequence length ``S``.
        num_heads: Number of attention heads ``H``.
        qk_head_dim: Query/key head dimension ``Dq``.
        v_head_dim: Value head dimension ``Dv``.
        softmax_scale: Scalar applied to every query vector.
        use_decay: Compile-time flag; if ``True`` the kernel reads
            ``exp(Decay[b, t, h])`` as a per-step forgetting factor.
        use_qk_l2norm: Compile-time flag; if ``True`` query and key vectors
            are L2-normalised (with epsilon 1e-6) inside the kernel.
        dtype: Element dtype of input/output tensors.
        threads: Number of CUDA threads per block (default 128).

    Returns:
        A ``T.prim_func`` with signature::

            gdr_fwd_states(
                Q     : [B, S, H, Dq]   <ts>
                K     : [B, S, H, Dq]   <ts>
                V     : [B, S, H, Dv]   <ts>
                Beta  : [B, S, H]       <ts>
                Decay : [B, S, H]       <ts>   (read only when use_decay=True)
                H0    : [B, H, Dq, Dv]  <float32>
                O     : [B, S, H, Dv]   <ts>          (output)
                Hf    : [B, H, Dq, Dv]  <float32>     (output: final state)
                HScan : [B, S+1+(S%2), H, Dq, Dv] <float32>  (output: history)
            )
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, Dq, Dv = batch, seq_len, num_heads, qk_head_dim, v_head_dim
    HS = S + 1 + (S % 2)
    scale = float(softmax_scale)

    @T.prim_func
    def gdr_fwd_states(
        Q: T.Tensor((B, S, H, Dq), ts),
        K: T.Tensor((B, S, H, Dq), ts),
        V: T.Tensor((B, S, H, Dv), ts),
        Beta: T.Tensor((B, S, H), ts),
        Decay: T.Tensor((B, S, H), ts),
        H0: T.Tensor((B, H, Dq, Dv), accum),
        O: T.Tensor((B, S, H, Dv), ts),
        Hf: T.Tensor((B, H, Dq, Dv), accum),
        HScan: T.Tensor((B, HS, H, Dq, Dv), accum),
    ):
        """TileLang kernel: GDR forward recurrent scan.

        Writes output ``O``, final state ``Hf``, and full scan history
        ``HScan``.  See :func:`make_fwd_states_prim_func` for the algorithm.
        """
        with T.Kernel(H, B, threads=threads) as (hx, bx):
            h_state = T.alloc_fragment((Dq, Dv), accum)
            q_loc = T.alloc_fragment((Dq,), accum)
            k_loc = T.alloc_fragment((Dq,), accum)
            v_loc = T.alloc_fragment((Dv,), accum)
            q_sq = T.alloc_fragment((Dq,), accum)
            k_sq = T.alloc_fragment((Dq,), accum)
            q_norm = T.alloc_fragment((1,), accum)
            k_norm = T.alloc_fragment((1,), accum)
            k_state_prod = T.alloc_fragment((Dq, Dv), accum)
            k_state = T.alloc_fragment((Dv,), accum)
            delta = T.alloc_fragment((Dv,), accum)
            out_prod = T.alloc_fragment((Dq, Dv), accum)
            out_acc = T.alloc_fragment((Dv,), accum)
            beta_val = T.alloc_fragment((1,), accum)
            g_exp = T.alloc_fragment((1,), accum)
            _ts_ref = T.alloc_fragment((1,), ts)
            _hs_ref = T.alloc_fragment((1,), accum)
            _hs_ref[0] = HS

            for i, j in T.Parallel(Dq, Dv):
                h_state[i, j] = H0[bx, hx, i, j]
                HScan[bx, 0, hx, i, j] = h_state[i, j]

            for t in T.serial(S):
                for i in T.Parallel(Dq):
                    q_loc[i] = T.Cast(accum, Q[bx, t, hx, i])
                    k_loc[i] = T.Cast(accum, K[bx, t, hx, i])
                for j in T.Parallel(Dv):
                    v_loc[j] = T.Cast(accum, V[bx, t, hx, j])

                if use_qk_l2norm:
                    for i in T.Parallel(Dq):
                        q_sq[i] = q_loc[i] * q_loc[i]
                        k_sq[i] = k_loc[i] * k_loc[i]
                    T.reduce_sum(q_sq, q_norm, dim=0, clear=True)
                    T.reduce_sum(k_sq, k_norm, dim=0, clear=True)
                    inv_q = T.alloc_fragment((1,), accum)
                    inv_k = T.alloc_fragment((1,), accum)
                    inv_q[0] = 1.0 / T.sqrt(q_norm[0] + 1e-6)
                    inv_k[0] = 1.0 / T.sqrt(k_norm[0] + 1e-6)
                    for i in T.Parallel(Dq):
                        q_loc[i] = q_loc[i] * inv_q[0]
                        k_loc[i] = k_loc[i] * inv_k[0]

                for i in T.Parallel(Dq):
                    q_loc[i] = q_loc[i] * scale

                beta_val[0] = T.Cast(accum, Beta[bx, t, hx])
                if use_decay:
                    g_exp[0] = T.exp(T.Cast(accum, Decay[bx, t, hx]))
                else:
                    g_exp[0] = 1.0

                for i, j in T.Parallel(Dq, Dv):
                    h_state[i, j] = h_state[i, j] * g_exp[0]
                    k_state_prod[i, j] = h_state[i, j] * k_loc[i]
                T.reduce_sum(k_state_prod, k_state, dim=0, clear=True)

                for j in T.Parallel(Dv):
                    delta[j] = beta_val[0] * (v_loc[j] - k_state[j])

                for i, j in T.Parallel(Dq, Dv):
                    h_state[i, j] = h_state[i, j] + k_loc[i] * delta[j]
                    out_prod[i, j] = h_state[i, j] * q_loc[i]
                T.reduce_sum(out_prod, out_acc, dim=0, clear=True)

                for j in T.Parallel(Dv):
                    O[bx, t, hx, j] = T.Cast(ts, out_acc[j])
                for i, j in T.Parallel(Dq, Dv):
                    HScan[bx, t + 1, hx, i, j] = h_state[i, j]

            for i, j in T.Parallel(Dq, Dv):
                Hf[bx, hx, i, j] = h_state[i, j]

    return gdr_fwd_states


def make_bwd_prim_func(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    softmax_scale: float,
    use_decay: bool,
    use_qk_l2norm: bool,
    dtype,
    threads: int = 128,
):
    """Build the TileLang backward prim_func for the GDR recurrent scan.

    The generated kernel traverses time in reverse (``t = S-1 .. 0``),
    reading the pre-saved ``HScan`` states from the forward pass to
    reconstruct intermediate quantities without re-running the forward scan.

    For each timestep the backward kernel propagates cotangents through:
    - the output projection ``o_t = h_t^T q_t``
    - the hidden-state update ``h_t = g * h_{t-1} + k_t ⊗ delta``
    - the delta computation ``delta = beta_t * (v_t - k_t^T h_{t-1})``
    - the optional L2 normalisation of ``q`` and ``k``
    - the optional decay gate ``g = exp(Decay[t])``

    Grid layout:
        ``Kernel(H, B)`` — one block per (head, batch) pair.

    Fragment allocation per block:
        ~25 fragment buffers in float32 (``Dq``- and ``Dv``-sized vectors
        plus ``(Dq, Dv)`` matrices for outer-product temporaries).

    Args:
        batch: Batch size ``B``.
        seq_len: Sequence length ``S``.
        num_heads: Number of attention heads ``H``.
        qk_head_dim: Query/key head dimension ``Dq``.
        v_head_dim: Value head dimension ``Dv``.
        softmax_scale: Scalar applied to query vectors (must match forward).
        use_decay: Compile-time flag — must match the forward kernel.
        use_qk_l2norm: Compile-time flag — must match the forward kernel.
        dtype: Element dtype of input/output tensors.
        threads: Number of CUDA threads per block (default 128).

    Returns:
        A ``T.prim_func`` with signature::

            gdr_bwd(
                Q       : [B, S, H, Dq]             <ts>
                K       : [B, S, H, Dq]             <ts>
                V       : [B, S, H, Dv]             <ts>
                Beta    : [B, S, H]                 <ts>
                Decay   : [B, S, H]                 <ts>
                HScan   : [B, S+1+(S%2), H, Dq, Dv] <float32>
                dO      : [B, S, H, Dv]             <ts>
                dH_final: [B, H, Dq, Dv]            <float32>
                dQ      : [B, S, H, Dq]             <ts>      (output)
                dK      : [B, S, H, Dq]             <ts>      (output)
                dV      : [B, S, H, Dv]             <ts>      (output)
                dBeta   : [B, S, H]                 <ts>      (output)
                dDecay  : [B, S, H]                 <ts>      (output; zero if
                                                               use_decay=False)
                dH0     : [B, H, Dq, Dv]            <float32> (output)
            )
    """
    ts = _dtype_str(dtype)
    accum = "float32"
    B, S, H, Dq, Dv = batch, seq_len, num_heads, qk_head_dim, v_head_dim
    HS = S + 1 + (S % 2)
    scale = float(softmax_scale)

    @T.prim_func
    def gdr_bwd(
        Q: T.Tensor((B, S, H, Dq), ts),
        K: T.Tensor((B, S, H, Dq), ts),
        V: T.Tensor((B, S, H, Dv), ts),
        Beta: T.Tensor((B, S, H), ts),
        Decay: T.Tensor((B, S, H), ts),
        HScan: T.Tensor((B, HS, H, Dq, Dv), accum),
        dO: T.Tensor((B, S, H, Dv), ts),
        dH_final: T.Tensor((B, H, Dq, Dv), accum),
        dQ: T.Tensor((B, S, H, Dq), ts),
        dK: T.Tensor((B, S, H, Dq), ts),
        dV: T.Tensor((B, S, H, Dv), ts),
        dBeta: T.Tensor((B, S, H), ts),
        dDecay: T.Tensor((B, S, H), ts),
        dH0: T.Tensor((B, H, Dq, Dv), accum),
    ):
        """TileLang kernel: GDR backward pass (reverse-time scan).

        Reads ``HScan`` saved during the forward pass and writes gradients
        ``dQ``, ``dK``, ``dV``, ``dBeta``, ``dDecay``, ``dH0``.
        See :func:`make_bwd_prim_func` for the algorithm.
        """
        with T.Kernel(H, B, threads=threads) as (hx, bx):
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
            _ts_ref = T.alloc_fragment((1,), ts)
            _hs_ref = T.alloc_fragment((1,), accum)
            _hs_ref[0] = HS

            for i, j in T.Parallel(Dq, Dv):
                dh[i, j] = dH_final[bx, hx, i, j]

            for t_iter in T.serial(S):
                t = S - 1 - t_iter
                for i in T.Parallel(Dq):
                    q_raw[i] = T.Cast(accum, Q[bx, t, hx, i])
                    k_raw[i] = T.Cast(accum, K[bx, t, hx, i])
                    q_loc[i] = q_raw[i]
                    k_loc[i] = k_raw[i]
                for j in T.Parallel(Dv):
                    v_loc[j] = T.Cast(accum, V[bx, t, hx, j])
                    do_loc[j] = T.Cast(accum, dO[bx, t, hx, j])

                if use_qk_l2norm:
                    for i in T.Parallel(Dq):
                        q_sq[i] = q_raw[i] * q_raw[i]
                        k_sq[i] = k_raw[i] * k_raw[i]
                    T.reduce_sum(q_sq, q_norm, dim=0, clear=True)
                    T.reduce_sum(k_sq, k_norm, dim=0, clear=True)
                    inv_q[0] = 1.0 / T.sqrt(q_norm[0] + 1e-6)
                    inv_k[0] = 1.0 / T.sqrt(k_norm[0] + 1e-6)
                    for i in T.Parallel(Dq):
                        q_loc[i] = q_raw[i] * inv_q[0]
                        k_loc[i] = k_raw[i] * inv_k[0]
                else:
                    inv_q[0] = 1.0
                    inv_k[0] = 1.0

                beta_val[0] = T.Cast(accum, Beta[bx, t, hx])
                if use_decay:
                    g_exp[0] = T.exp(T.Cast(accum, Decay[bx, t, hx]))
                else:
                    g_exp[0] = 1.0

                for i, j in T.Parallel(Dq, Dv):
                    h_prev[i, j] = HScan[bx, t, hx, i, j]
                    h_new[i, j] = HScan[bx, t + 1, hx, i, j]
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
                        dQ[bx, t, hx, i] = T.Cast(ts, inv_q[0] * (dq_normed[i] - q_loc[i] * dotq[0]))
                        dK[bx, t, hx, i] = T.Cast(ts, inv_k[0] * (dk_loc[i] - k_loc[i] * dotk[0]))
                else:
                    for i in T.Parallel(Dq):
                        dQ[bx, t, hx, i] = T.Cast(ts, dq_normed[i])
                        dK[bx, t, hx, i] = T.Cast(ts, dk_loc[i])

                for j in T.Parallel(Dv):
                    dV[bx, t, hx, j] = T.Cast(ts, dv_loc[j])
                dBeta[bx, t, hx] = T.Cast(ts, dbeta_acc[0])
                if use_decay:
                    dDecay[bx, t, hx] = T.Cast(ts, dg_acc[0] * g_exp[0])
                else:
                    dDecay[bx, t, hx] = T.Cast(ts, 0.0)

                for i, j in T.Parallel(Dq, Dv):
                    dh[i, j] = dh_pre[i, j] * g_exp[0]

            for i, j in T.Parallel(Dq, Dv):
                dH0[bx, hx, i, j] = dh[i, j]

    return gdr_bwd
