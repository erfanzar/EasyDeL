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

"""Backward TPU Pallas kernels for Gated Delta Rule (GDR).

Pure Pallas backward with a single reverse-scan kernel that recomputes
all forward intermediates (Neumann series, decay masks) from raw inputs
+ saved per-chunk states, then computes all gradients in one fused pass.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float

from ...._xla.gated_delta_rule._xla_impl_fwd import _l2norm_bwd
from ._pallas_impl_fwd import _N_FUSE, _chunk_blockspec, _dot, _neumann_inv


def _bwd_one_chunk(q, k, v, beta, decay, d_out, state_pre, d_state_next, C):
    """Compute gradients for a single GDR chunk via full re-materialisation.

    Recomputes all forward intermediates (Neumann inverse, decay masks,
    scaled keys/queries, etc.) from the raw inputs and the saved pre-chunk
    state, then applies the chain rule to obtain per-token gradients.
    All intermediate tensors are sanitized with ``nan_to_num``.

    Args:
        q: Query slice [C, K], float32.
        k: Key slice [C, K], float32.
        v: Value slice [C, V], float32.
        beta: Gate slice [C], float32.
        decay: Log-decay slice [C], float32.
        d_out: Gradient of output w.r.t. this chunk [C, V], float32.
        state_pre: Recurrent state before this chunk [K, V], float32.
        d_state_next: Gradient of state arriving from the next chunk [K, V].
        C: Chunk size.

    Returns:
        Six-element tuple:
        ``(d_state_prev [K, V], d_q [C, K], d_k [C, K], d_v [C, V],
        d_beta [C, 1], d_decay [C, 1])``.
    """
    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
    strict_lower = lower_mask - jnp.eye(C, dtype=jnp.float32)
    upper_mask = jnp.triu(jnp.ones((C, C), dtype=jnp.float32))

    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]

    g_cumsum = jnp.sum(lower_mask * decay[None, :], axis=1, keepdims=True)
    g_diff = g_cumsum - g_cumsum.T
    decay_mask = jnp.exp(jnp.clip(g_diff * lower_mask, -20.0, 20.0)) * lower_mask

    attn_neg = -(_dot(k_beta, k.T) * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn = _neumann_inv(attn_neg, C, strict_lower=strict_lower, lower_mask=lower_mask)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))
    g_end = g_cumsum[C - 1 : C, :]
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0))
    g_diff_state_exp = jnp.exp(jnp.clip(g_end - g_cumsum, -20.0, 20.0))

    def _s(x):
        return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    k_beta_scaled = k_beta * g_cumsum_exp
    value_local = _s(_dot(attn, v_beta))
    k_cumdecay = _s(_dot(attn, k_beta_scaled))

    attn_qk_base = _dot(q, k.T)
    attn_qk = _s(attn_qk_base * decay_mask)
    q_scaled = q * g_cumsum_exp
    v_prime = _s(_dot(k_cumdecay, state_pre))
    v_new = _s(value_local - v_prime)
    k_scaled = k * g_diff_state_exp

    d_state = _s(d_state_next * g_end_exp)
    d_g_end = jnp.sum(d_state_next * state_pre)

    d_k_scaled = _s(_dot(v_new, d_state_next.T))
    d_v_new = _s(_dot(k_scaled, d_state_next))
    d_attn_qk = _s(_dot(d_out, v_new.T))
    d_v_new = _s(d_v_new + _dot(attn_qk.T, d_out))

    d_value_local = d_v_new
    d_v_prime = -d_v_new

    d_k_cumdecay = _s(_dot(d_v_prime, state_pre.T))
    d_state = _s(d_state + _dot(k_cumdecay.T, d_v_prime))
    d_q_scaled = _s(_dot(d_out, state_pre.T))
    d_state = _s(d_state + _dot(q_scaled.T, d_out))

    d_q = d_q_scaled * g_cumsum_exp
    d_g_exp = jnp.sum(d_q_scaled * q, axis=-1, keepdims=True)
    d_k = d_k_scaled * g_diff_state_exp
    d_g_diff = jnp.sum(d_k_scaled * k, axis=-1, keepdims=True)

    d_attn_qk_base = _s(d_attn_qk * decay_mask)
    d_decay_mask_from_qk = d_attn_qk * attn_qk_base
    d_q = d_q + _dot(d_attn_qk_base, k)
    d_k = d_k + _dot(d_attn_qk_base.T, q)

    d_attn = _s(_dot(d_value_local, v_beta.T) + _dot(d_k_cumdecay, k_beta_scaled.T))
    d_value_beta = _s(_dot(attn.T, d_value_local))
    d_key_beta_scaled = _s(_dot(attn.T, d_k_cumdecay))

    d_key_beta = d_key_beta_scaled * g_cumsum_exp
    d_g_exp = d_g_exp + jnp.sum(d_key_beta_scaled * k_beta, axis=-1, keepdims=True)

    tmp = _dot(attn.T, d_attn)
    d_k_attn = _s(-_dot(tmp, attn.T))
    d_k_attn = d_k_attn * strict_lower

    kk = _dot(k_beta, k.T)
    d_kk = d_k_attn * decay_mask
    d_decay_mask = _s((d_decay_mask_from_qk + d_k_attn * kk) * lower_mask)

    d_key_beta = d_key_beta + _dot(d_kk, k)
    d_k = d_k + _dot(d_kk.T, k_beta)

    d_v = d_value_beta * beta[:, None]
    d_beta = jnp.sum(d_value_beta * v, axis=-1, keepdims=True)
    d_k = d_k + d_key_beta * beta[:, None]
    d_beta = d_beta + jnp.sum(d_key_beta * k, axis=-1, keepdims=True)

    d_decay_f = _s(d_decay_mask * decay_mask)
    d_g_row = jnp.sum(d_decay_f, axis=-1, keepdims=True)
    d_g_col = jnp.sum(d_decay_f.T, axis=1, keepdims=True)
    d_g = d_g_row - d_g_col
    d_g = d_g + d_g_exp * g_cumsum_exp

    d_g_diff_term = d_g_diff * g_diff_state_exp
    d_g_end_total = jnp.sum(d_g_diff_term) + d_g_end * g_end_exp
    d_g = d_g - d_g_diff_term
    d_decay_final = _dot(upper_mask, d_g) + d_g_end_total

    d_state = _s(d_state)
    return d_state, d_q, d_k, d_v, d_beta, d_decay_final


def _gdr_bwd_grad_kernel(
    state_pre_ref,
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    d_out_ref,
    d_state_next_ref,
    d_state_ref,
    d_q_ref,
    d_k_ref,
    d_v_ref,
    d_beta_ref,
    d_decay_ref,
):
    """Pallas kernel for one reverse-scan step, processing ``_N_FUSE`` chunks.

    Iterates ``_N_FUSE`` chunks in reverse order.  For chunk ``i > 0`` it
    re-runs the forward pass for chunks ``0 … i-1`` to recover the
    intermediate state (since only the state *before* the fused group is
    saved).

    Grid: ``(batch, num_heads)``; both axes are "parallel".

    BlockSpec shape: ``(1, 1, total_rows, dim)`` for Q/K/V/dO;
    ``(1, 1, K, V)`` for states; ``(1, 1, 1, total_beta)`` for beta/decay.

    Args:
        state_pre_ref: [1,1,K,V] — recurrent state before the fused group.
        q_ref: [1,1,N_FUSE*C,K].
        k_ref: [1,1,N_FUSE*C,K].
        v_ref: [1,1,N_FUSE*C,V].
        beta_ref: [1,1,1,N_FUSE*C].
        decay_ref: [1,1,1,N_FUSE*C].
        d_out_ref: [1,1,N_FUSE*C,V] — upstream gradient.
        d_state_next_ref: [1,1,K,V] — gradient arriving from the next group.

    Outputs written:
        d_state_ref: [1,1,K,V] — gradient for state before this group.
        d_q_ref, d_k_ref: [1,1,N_FUSE*C,K].
        d_v_ref: [1,1,N_FUSE*C,V].
        d_beta_ref: [1,1,N_FUSE*C,1].
        d_decay_ref: [1,1,N_FUSE*C,1].
    """
    total_rows = q_ref.shape[2]
    C = total_rows // _N_FUSE

    d_state_next = d_state_next_ref[0, 0].astype(jnp.float32)
    d_state_next = jnp.nan_to_num(d_state_next, nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(_N_FUSE - 1, -1, -1):
        s = i * C
        q = q_ref[0, 0, s : s + C].astype(jnp.float32)
        k = k_ref[0, 0, s : s + C].astype(jnp.float32)
        v = v_ref[0, 0, s : s + C].astype(jnp.float32)
        beta = beta_ref[0, 0, 0, i * C : (i + 1) * C]
        decay = decay_ref[0, 0, 0, i * C : (i + 1) * C]
        d_out = d_out_ref[0, 0, s : s + C].astype(jnp.float32)
        if i == 0:
            state_pre = state_pre_ref[0, 0].astype(jnp.float32)
            state_pre = jnp.nan_to_num(state_pre, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            from ._pallas_impl_fwd import _process_one_chunk

            sp = state_pre_ref[0, 0].astype(jnp.float32)
            sp = jnp.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0)
            for j in range(i):
                sj = j * C
                qj = q_ref[0, 0, sj : sj + C].astype(jnp.float32)
                kj = k_ref[0, 0, sj : sj + C].astype(jnp.float32)
                vj = v_ref[0, 0, sj : sj + C].astype(jnp.float32)
                bj = beta_ref[0, 0, 0, j * C : (j + 1) * C]
                dj = decay_ref[0, 0, 0, j * C : (j + 1) * C]
                _, sp = _process_one_chunk(qj, kj, vj, bj, dj, sp, C)
            state_pre = sp

        d_st, d_q_c, d_k_c, d_v_c, d_beta_c, d_decay_c = _bwd_one_chunk(
            q,
            k,
            v,
            beta,
            decay,
            d_out,
            state_pre,
            d_state_next,
            C,
        )
        d_state_next = jnp.nan_to_num(d_st, nan=0.0, posinf=0.0, neginf=0.0)

        d_q_ref[0, 0, s : s + C] = jnp.nan_to_num(d_q_c, nan=0.0, posinf=0.0, neginf=0.0).astype(d_q_ref.dtype)
        d_k_ref[0, 0, s : s + C] = jnp.nan_to_num(d_k_c, nan=0.0, posinf=0.0, neginf=0.0).astype(d_k_ref.dtype)
        d_v_ref[0, 0, s : s + C] = jnp.nan_to_num(d_v_c, nan=0.0, posinf=0.0, neginf=0.0).astype(d_v_ref.dtype)
        d_beta_ref[0, 0, s : s + C, :] = jnp.nan_to_num(d_beta_c, nan=0.0, posinf=0.0, neginf=0.0).astype(
            d_beta_ref.dtype
        )
        d_decay_ref[0, 0, s : s + C, :] = jnp.nan_to_num(d_decay_c, nan=0.0, posinf=0.0, neginf=0.0).astype(
            d_decay_ref.dtype
        )

    d_state_ref[0, 0] = d_state_next.astype(d_state_ref.dtype)


def _run_bwd_grad_step(
    state_pre,
    q_i,
    k_i,
    v_i,
    beta_i,
    decay_i,
    d_out_i,
    d_state_next,
):
    """Launch the backward gradient Pallas kernel for one fused chunk group.

    Args:
        state_pre: [B, H, K, V] — recurrent state before this fused group.
        q_i: [B, H, N_FUSE*C, K].
        k_i: [B, H, N_FUSE*C, K].
        v_i: [B, H, N_FUSE*C, V].
        beta_i: [B, H, 1, N_FUSE*C] float32.
        decay_i: [B, H, 1, N_FUSE*C] float32.
        d_out_i: [B, H, N_FUSE*C, V] — upstream gradient.
        d_state_next: [B, H, K, V] — gradient from the following group.

    Returns:
        Six-element tuple: ``(d_state_prev, d_q, d_k, d_v, d_beta, d_decay)``
        all as float32 arrays with matching leading dimensions.
    """
    bsz, num_heads, total_rows, qk_dim = q_i.shape
    v_dim = v_i.shape[-1]
    total_beta = beta_i.shape[-1]

    call = pl.pallas_call(
        _gdr_bwd_grad_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
                _chunk_blockspec((1, 1, total_rows, qk_dim)),
                _chunk_blockspec((1, 1, total_rows, qk_dim)),
                _chunk_blockspec((1, 1, total_rows, v_dim)),
                _chunk_blockspec((1, 1, 1, total_beta)),
                _chunk_blockspec((1, 1, 1, total_beta)),
                _chunk_blockspec((1, 1, total_rows, v_dim)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            out_specs=[
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
                _chunk_blockspec((1, 1, total_rows, qk_dim)),
                _chunk_blockspec((1, 1, total_rows, qk_dim)),
                _chunk_blockspec((1, 1, total_rows, v_dim)),
                _chunk_blockspec((1, 1, total_rows, 1)),
                _chunk_blockspec((1, 1, total_rows, 1)),
            ],
            grid=(bsz, num_heads),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((bsz, num_heads, qk_dim, v_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, total_rows, qk_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, total_rows, qk_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, total_rows, v_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, total_rows, 1), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, total_rows, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )
    return call(
        state_pre,
        q_i,
        k_i,
        v_i,
        beta_i,
        decay_i,
        d_out_i,
        d_state_next,
    )


def _cast_grad(x, dtype):
    """Cast a gradient array to ``dtype``, passing ``None`` through unchanged."""
    if x is None:
        return None
    return x.astype(dtype) if x.dtype != dtype else x


def _chunk_gdr_bwd(
    chunk_size: int,
    use_qk_l2norm: bool,
    res: tuple,
    g: tuple[Float[Array, "..."], Float[Array, "..."]],
) -> tuple:
    """Compute gradients for the chunked GDR forward pass via reverse scan.

    Runs a ``lax.scan`` in reverse order over all chunk groups, calling
    ``_run_bwd_grad_step`` (which launches a Pallas kernel) for each group.

    Residual tuple layout (14 elements as packed by ``_chunk_gdr_fwd_core``)::

        (query_c, key_c, value_c, beta, decay, state_pre_all,
         initial_state, q_inv_norm, k_inv_norm,
         seq_len, pad_size, decay_was_none, initial_state_was_none,
         effective_chunk_size)

    Args:
        chunk_size: Non-diff arg from custom_vjp (overridden by
            ``effective_chunk_size`` stored in ``res``).
        use_qk_l2norm: Whether Q/K were L2-normalised in the forward pass.
        res: Residual tuple saved by ``_chunk_gdr_fwd_rule``.
        g: Upstream gradients ``(d_output, d_final_state)``.

    Returns:
        Six-element gradient tuple:
        ``(d_query, d_key, d_value, d_beta, d_decay, d_initial_state)``
        cast to the input dtype.  ``d_decay`` is ``None`` if ``decay`` was
        ``None`` in the forward call; ``d_initial_state`` is ``None`` if no
        initial state was provided.
    """
    (
        query,
        key,
        value,
        beta,
        decay,
        state_pre_all,
        _initial_state,
        q_inv_norm,
        k_inv_norm,
        seq_len,
        pad_size,
        decay_was_none,
        initial_state_was_none,
        effective_chunk_size,
    ) = res
    chunk_size = effective_chunk_size
    d_out, d_final_state = g
    input_dtype = query.dtype
    B, H, num_chunks, _C, K_dim = query.shape
    V_dim = value.shape[-1]
    scale = 1.0 / math.sqrt(K_dim)

    if pad_size > 0:
        d_out = jnp.pad(d_out, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    d_out = d_out.reshape(B, H, num_chunks, chunk_size, V_dim)

    beta_k = beta[:, :, :, None, :]
    decay_k = decay[:, :, :, None, :]

    q_tm = query.transpose(2, 0, 1, 3, 4)
    k_tm = key.transpose(2, 0, 1, 3, 4)
    v_tm = value.transpose(2, 0, 1, 3, 4)
    beta_tm = beta_k.transpose(2, 0, 1, 3, 4)
    decay_tm = decay_k.transpose(2, 0, 1, 3, 4)
    state_pre_tm = state_pre_all.transpose(2, 0, 1, 3, 4)
    d_out_tm = d_out.transpose(2, 0, 1, 3, 4)

    d_final_state = d_final_state.astype(jnp.float32)

    def grad_step(d_state_next, inputs):
        sp_i, q_i, k_i, v_i, b_i, dc_i, do_i = inputs
        d_state_i, d_q_i, d_k_i, d_v_i, d_beta_i, d_decay_i = _run_bwd_grad_step(
            sp_i,
            q_i,
            k_i,
            v_i,
            b_i,
            dc_i,
            do_i,
            d_state_next,
        )
        return d_state_i, (d_q_i, d_k_i, d_v_i, d_beta_i, d_decay_i)

    d_initial_state, grads_tm = lax.scan(
        grad_step,
        d_final_state,
        (state_pre_tm, q_tm, k_tm, v_tm, beta_tm, decay_tm, d_out_tm),
        reverse=True,
    )
    d_q_tm, d_k_tm, d_v_tm, d_beta_tm, d_decay_tm = grads_tm

    total_len = seq_len + pad_size
    d_query = d_q_tm.transpose(1, 2, 0, 3, 4).reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
    d_key = d_k_tm.transpose(1, 2, 0, 3, 4).reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
    d_value = d_v_tm.transpose(1, 2, 0, 3, 4).reshape(B, H, total_len, V_dim)[:, :, :seq_len, :]
    d_beta = d_beta_tm.transpose(1, 2, 0, 3, 4).squeeze(-1).reshape(B, H, total_len)[:, :, :seq_len]
    d_decay = d_decay_tm.transpose(1, 2, 0, 3, 4).squeeze(-1).reshape(B, H, total_len)[:, :, :seq_len]

    d_query = d_query * scale
    if use_qk_l2norm:
        q_norm = query.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :] / scale
        k_norm = key.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
        d_query = _l2norm_bwd(d_query, q_norm, q_inv_norm.astype(jnp.float32))
        d_key = _l2norm_bwd(d_key, k_norm, k_inv_norm.astype(jnp.float32))

    if decay_was_none:
        d_decay = None
    if initial_state_was_none:
        d_initial_state = None

    def _safe_cast(x, dtype):
        if x is None:
            return None
        x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if dtype == jnp.bfloat16 or dtype == jnp.float16:
            x = jnp.clip(x, -65000.0, 65000.0)
        return x.astype(dtype) if x.dtype != dtype else x

    return (
        _safe_cast(d_query, input_dtype),
        _safe_cast(d_key, input_dtype),
        _safe_cast(d_value, input_dtype),
        _safe_cast(d_beta, input_dtype),
        _safe_cast(d_decay, input_dtype),
        _safe_cast(d_initial_state, input_dtype),
    )


def _gdr_single_step_bwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    state_prev_ref,
    d_out_ref,
    d_state_next_ref,
    d_q_ref,
    d_k_ref,
    d_v_ref,
    d_beta_ref,
    d_decay_ref,
    d_state_ref,
):
    """Pallas kernel for single-step GDR backward pass.

    Recomputes the forward pass inline to recover ``state_decayed``,
    ``kv_mem``, ``delta``, and ``state_new``, then applies the chain rule.

    Grid: ``(batch, num_heads)``; both axes are "parallel".

    Inputs (BlockSpec ``(1, 1, 1, dim)`` or ``(1, 1, K, V)``):
        q_ref, k_ref: [B, H, 1, 1, K].
        v_ref: [B, H, 1, 1, V].
        beta_ref: [B, H, 1, 1] — scalar gate.
        decay_ref: [B, H, 1, 1] — scalar log decay.
        state_prev_ref: [B, H, 1, K, V] — state before the step.
        d_out_ref: [B, H, 1, 1, V] — output gradient.
        d_state_next_ref: [B, H, 1, K, V] — state gradient from downstream.

    Outputs written:
        d_q_ref, d_k_ref: [B, H, 1, 1, K].
        d_v_ref: [B, H, 1, 1, V].
        d_beta_ref: [B, H, 1, 1] scalar.
        d_decay_ref: [B, H, 1, 1] scalar.
        d_state_ref: [B, H, 1, K, V] — gradient w.r.t. ``state_prev``.
    """
    q_t = q_ref[0, 0, 0].astype(jnp.float32)
    k_t = k_ref[0, 0, 0].astype(jnp.float32)
    v_t = v_ref[0, 0, 0].astype(jnp.float32)
    beta_t = beta_ref[0, 0].reshape(())
    decay_t = decay_ref[0, 0].reshape(())
    state_prev = state_prev_ref[0, 0].astype(jnp.float32)
    d_out_t = d_out_ref[0, 0, 0].astype(jnp.float32)
    d_state_next = d_state_next_ref[0, 0].astype(jnp.float32)

    g_exp = jnp.exp(jnp.clip(decay_t, -20.0, 20.0))
    state_decayed = state_prev * g_exp
    kv_mem = jnp.sum(state_decayed * k_t[:, None], axis=0)
    delta_raw = v_t - kv_mem
    delta = delta_raw * beta_t
    state = state_decayed + k_t[:, None] * delta[None, :]

    d_s = d_state_next + q_t[:, None] * d_out_t[None, :]
    d_q = jnp.sum(state * d_out_t[None, :], axis=-1)
    d_k = jnp.sum(d_s * delta[None, :], axis=-1)
    d_delta = jnp.sum(d_s * k_t[:, None], axis=0)
    d_beta = jnp.sum(d_delta * delta_raw)
    d_v = d_delta * beta_t
    d_kv_mem = -d_delta * beta_t
    d_state_decayed = d_s + k_t[:, None] * d_kv_mem[None, :]
    d_k = d_k + jnp.sum(state_decayed * d_kv_mem[None, :], axis=-1)
    d_state = d_state_decayed * g_exp
    d_decay = jnp.sum(d_state_decayed * state_prev) * g_exp

    d_q_ref[0, 0, 0] = d_q
    d_k_ref[0, 0, 0] = d_k
    d_v_ref[0, 0, 0] = d_v
    d_beta_ref[0, 0] = d_beta.reshape(1, 1)
    d_decay_ref[0, 0] = d_decay.reshape(1, 1)
    d_state_ref[0, 0] = d_state


def _gdr_single_step_bwd_dma_kernel(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    state_prev_ref,
    d_out_ref,
    d_state_next_ref,
    d_q_ref,
    d_k_ref,
    d_v_ref,
    d_beta_ref,
    d_decay_ref,
    d_state_ref,
    state_tile_ref,
    d_state_next_tile_ref,
    dma_sem_ref,
):
    """DMA-backed Pallas kernel for single-step GDR backward.

    DMA is used only for the state matrices that can amortize the semaphore
    cost. The per-token vectors and scalar beta/decay inputs stay as direct
    loads because they are consumed once and are small relative to the state.
    """
    qk_dim = q_ref.shape[3]
    v_dim = v_ref.shape[3]
    copies = (
        pltpu.make_async_copy(
            src_ref=state_prev_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, qk_dim), pl.ds(0, v_dim)],
            dst_ref=state_tile_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, qk_dim), pl.ds(0, v_dim)],
            sem=dma_sem_ref.at[0],
        ),
        pltpu.make_async_copy(
            src_ref=d_state_next_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, qk_dim), pl.ds(0, v_dim)],
            dst_ref=d_state_next_tile_ref.at[pl.ds(0, 1), pl.ds(0, 1), pl.ds(0, qk_dim), pl.ds(0, v_dim)],
            sem=dma_sem_ref.at[1],
        ),
    )
    for copy in copies:
        copy.start()

    k_t = k_ref[0, 0, 0].astype(jnp.float32)
    v_t = v_ref[0, 0, 0].astype(jnp.float32)
    beta_t = beta_ref[0, 0, 0, 0]
    decay_t = decay_ref[0, 0, 0, 0]
    for copy in copies:
        copy.wait()
    state_prev = state_tile_ref[0, 0].astype(jnp.float32)
    d_state_next = d_state_next_tile_ref[0, 0].astype(jnp.float32)

    g_exp = jnp.exp(jnp.clip(decay_t, -20.0, 20.0))
    state_decayed = state_prev * g_exp
    kv_mem = jnp.sum(state_decayed * k_t[:, None], axis=0)
    delta_raw = v_t - kv_mem
    delta = delta_raw * beta_t
    state = state_decayed + k_t[:, None] * delta[None, :]

    q_t = q_ref[0, 0, 0].astype(jnp.float32)
    d_out_t = d_out_ref[0, 0, 0].astype(jnp.float32)

    d_s = d_state_next + q_t[:, None] * d_out_t[None, :]
    d_q = jnp.sum(state * d_out_t[None, :], axis=-1)
    d_k = jnp.sum(d_s * delta[None, :], axis=-1)
    d_delta = jnp.sum(d_s * k_t[:, None], axis=0)
    d_beta = jnp.sum(d_delta * delta_raw)
    d_v = d_delta * beta_t
    d_kv_mem = -d_delta * beta_t
    d_state_decayed = d_s + k_t[:, None] * d_kv_mem[None, :]
    d_k = d_k + jnp.sum(state_decayed * d_kv_mem[None, :], axis=-1)
    d_state = d_state_decayed * g_exp
    d_decay = jnp.sum(d_state_decayed * state_prev) * g_exp

    d_q_ref[0, 0, 0] = d_q
    d_k_ref[0, 0, 0] = d_k
    d_v_ref[0, 0, 0] = d_v
    d_beta_ref[0, 0] = d_beta.reshape(1, 1)
    d_decay_ref[0, 0] = d_decay.reshape(1, 1)
    d_state_ref[0, 0] = d_state


def _run_single_step_backward(
    query,
    key,
    value,
    beta,
    decay,
    recurrent_state,
    d_out,
    d_state_next,
):
    """Launch the single-step backward Pallas kernel.

    Args:
        query: [B, H, 1, K] already normalised/scaled (same as forward input).
        key: [B, H, 1, K].
        value: [B, H, 1, V].
        beta: [B, H, 1, 1] float32 gate.
        decay: [B, H, 1, 1] float32 log decay.
        recurrent_state: [B, H, K, V] state before the step.
        d_out: [B, H, 1, V] output gradient, float32.
        d_state_next: [B, H, K, V] state gradient from downstream, float32.

    Returns:
        Six-element tuple ``(d_q, d_k, d_v, d_beta, d_decay, d_state)``
        all as float32 arrays.
    """
    bsz, num_heads, _, qk_dim = query.shape
    v_dim = value.shape[-1]
    beta = beta[..., None].astype(jnp.float32)
    decay = decay[..., None].astype(jnp.float32)
    call = pl.pallas_call(
        _gdr_single_step_bwd_dma_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, v_dim)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
                _chunk_blockspec((1, 1, 1, v_dim)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            out_specs=[
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, v_dim)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            scratch_shapes=[
                pltpu.VMEM((1, 1, qk_dim, v_dim), recurrent_state.dtype),
                pltpu.VMEM((1, 1, qk_dim, v_dim), d_state_next.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
            grid=(bsz, num_heads),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((bsz, num_heads, 1, qk_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, 1, qk_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, 1, v_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, 1, 1), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, 1, 1), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, qk_dim, v_dim), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )
    return call(query, key, value, beta, decay, recurrent_state, d_out, d_state_next)


def _single_step_gdr_bwd(
    use_qk_l2norm: bool,
    res: tuple,
    g: tuple[Float[Array, "..."], Float[Array, "..."]],
) -> tuple:
    """Custom-VJP backward for the single-step GDR path.

    Calls the Pallas backward kernel and applies the inverse of L2
    normalisation if ``use_qk_l2norm`` is True.

    Args:
        use_qk_l2norm: Non-diff arg — whether Q/K were normalised forward.
        res: Residual tuple ``(query, key, value, beta, decay,
            recurrent_state, q_inv_norm, k_inv_norm, decay_was_none)``
            saved by the forward rule.
        g: Upstream gradients ``(d_output, d_final_state)``.

    Returns:
        Six-element gradient tuple:
        ``(d_query, d_key, d_value, d_beta, d_decay, d_recurrent_state)``
        cast to the input dtype.  ``d_decay`` is ``None`` if ``decay`` was
        ``None`` in the forward call.
    """
    query, key, value, beta, decay, recurrent_state, q_inv_norm, k_inv_norm, decay_was_none = res
    d_out, d_final_state = g
    input_dtype = query.dtype
    scale = 1.0 / math.sqrt(query.shape[-1])
    d_query, d_key, d_value, d_beta, d_decay, d_state = _run_single_step_backward(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        d_out.astype(jnp.float32),
        d_final_state.astype(jnp.float32),
    )
    d_beta = d_beta[..., 0]
    d_decay = d_decay[..., 0]
    d_query = d_query * scale
    if use_qk_l2norm:
        q_norm = query / scale
        k_norm = key
        d_query = _l2norm_bwd(d_query, q_norm, q_inv_norm.astype(jnp.float32))
        d_key = _l2norm_bwd(d_key, k_norm, k_inv_norm.astype(jnp.float32))
    if decay_was_none:
        d_decay = None
    return (
        _cast_grad(d_query, input_dtype),
        _cast_grad(d_key, input_dtype),
        _cast_grad(d_value, input_dtype),
        _cast_grad(d_beta, input_dtype),
        _cast_grad(d_decay, input_dtype),
        _cast_grad(d_state, input_dtype),
    )
