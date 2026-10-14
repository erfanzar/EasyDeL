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

"""Analytical backward pass for chunked Gated Delta Rule (GDR).

This module implements the hand-derived gradient computation for the chunked
GDR forward pass. The backward runs entirely in float32 to maintain numerical
stability, as the Neumann series matrix inversion amplifies precision errors.

The backward consists of two scans:
    1. Forward state scan: Reconstructs per-chunk states from residuals.
    2. Reverse chunk scan: Propagates gradients backward through chunks,
       computing gradients for query, key, value, beta, decay, and state.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ._xla_impl_fwd import _l2norm_bwd

_MATMUL_PRECISION = lax.Precision.HIGHEST


def _chunk_gdr_bwd(
    chunk_size: int,
    use_qk_l2norm: bool,
    res: tuple,
    g: tuple[Float[Array, "..."], Float[Array, "..."]],
) -> tuple:
    """Analytical backward pass for chunked GDR in float32.

    Uses two scans:
    1. Forward scan to reconstruct per-chunk states from saved residuals.
    2. Reverse scan to propagate gradients back through chunks.

    Args:
        chunk_size: Chunk size (non-diff argument from custom_vjp).
        use_qk_l2norm: Whether L2 norm was applied (non-diff argument).
        res: Residuals saved from the forward pass.
        g: Upstream gradients (d_output, d_final_state).

    Returns:
        Tuple of gradients: (d_query, d_key, d_value, d_beta, d_decay, d_initial_state).
    """
    (
        query,
        key,
        value,
        beta,
        attn,
        decay_mask,
        g_cumsum_exp,
        g_end_exp,
        g_diff_state_exp,
        initial_state,
        q_inv_norm,
        k_inv_norm,
        seq_len,
        pad_size,
        decay_was_none,
        initial_state_was_none,
    ) = res
    d_out, d_final_state = g
    input_dtype = query.dtype

    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    attn = attn.astype(jnp.float32)
    decay_mask = decay_mask.astype(jnp.float32)
    g_cumsum_exp = g_cumsum_exp.astype(jnp.float32)
    g_end_exp = g_end_exp.astype(jnp.float32)
    g_diff_state_exp = g_diff_state_exp.astype(jnp.float32)
    initial_state = initial_state.astype(jnp.float32)

    d_out = d_out.astype(jnp.float32)
    d_final_state = d_final_state.astype(jnp.float32)

    B, H, num_chunks, _, K_dim = query.shape
    V_dim = value.shape[-1]
    total_len = seq_len + pad_size
    scale = 1.0 / (K_dim**0.5)

    if pad_size > 0:
        d_out = jnp.pad(d_out, ((0, 0), (0, 0), (0, pad_size), (0, 0)))

    d_out = d_out.reshape(B, H, num_chunks, chunk_size, V_dim)

    value_beta = value * beta[:, :, :, :, None]
    key_beta = key * beta[:, :, :, :, None]
    key_beta_scaled = key_beta * g_cumsum_exp[:, :, :, :, None]

    value_local = jnp.einsum("bhcij,bhcjv->bhciv", attn, value_beta, precision=_MATMUL_PRECISION)
    key_cumdecay = jnp.einsum("bhcij,bhcjk->bhcik", attn, key_beta_scaled, precision=_MATMUL_PRECISION)

    query_tm = query.transpose(2, 0, 1, 3, 4)
    key_tm = key.transpose(2, 0, 1, 3, 4)
    value_local_tm = value_local.transpose(2, 0, 1, 3, 4)
    key_cumdecay_tm = key_cumdecay.transpose(2, 0, 1, 3, 4)
    g_cumsum_exp_tm = g_cumsum_exp.transpose(2, 0, 1, 3)
    g_end_exp_tm = g_end_exp.transpose(2, 0, 1)
    g_diff_state_exp_tm = g_diff_state_exp.transpose(2, 0, 1, 3)
    decay_mask_tm = decay_mask.transpose(2, 0, 1, 3, 4)
    d_out_tm = d_out.transpose(2, 0, 1, 3, 4)

    def fwd_state_scan(state, inputs):
        k_i, v_i, k_cum_i, g_end_i, g_diff_i = inputs
        v_prime = jnp.einsum("bhik,bhkv->bhiv", k_cum_i, state, precision=_MATMUL_PRECISION)
        v_new = v_i - v_prime

        state_decayed = state * g_end_i[:, :, None, None]
        k_scaled = k_i * g_diff_i[:, :, :, None]
        state_update = jnp.einsum("bhik,bhiv->bhkv", k_scaled, v_new, precision=_MATMUL_PRECISION)
        new_state = state_decayed + state_update

        return new_state, state

    _, state_pre_tm = lax.scan(
        fwd_state_scan,
        initial_state,
        (
            key_tm,
            value_local_tm,
            key_cumdecay_tm,
            g_end_exp_tm,
            g_diff_state_exp_tm,
        ),
    )

    def rev_chunk_scan(d_state_next, inputs):
        (
            state_i,
            q_i,
            k_i,
            v_i,
            k_cum_i,
            g_exp_i,
            g_end_i,
            g_diff_i,
            decay_i,
            d_core_i,
        ) = inputs

        attn_qk_base = jnp.einsum("bhik,bhjk->bhij", q_i, k_i, precision=_MATMUL_PRECISION)
        attn_qk = attn_qk_base * decay_i
        q_scaled = q_i * g_exp_i[:, :, :, None]
        v_prime = jnp.einsum("bhik,bhkv->bhiv", k_cum_i, state_i, precision=_MATMUL_PRECISION)
        v_new = v_i - v_prime
        k_scaled = k_i * g_diff_i[:, :, :, None]

        d_state_i = d_state_next * g_end_i[:, :, None, None]
        d_g_end_i = jnp.einsum("bhkv,bhkv->bh", d_state_next, state_i, precision=_MATMUL_PRECISION)

        d_k_scaled = jnp.einsum("bhkv,bhiv->bhik", d_state_next, v_new, precision=_MATMUL_PRECISION)
        d_v_new = jnp.einsum("bhkv,bhik->bhiv", d_state_next, k_scaled, precision=_MATMUL_PRECISION)

        d_attn_qk = jnp.einsum("bhiv,bhjv->bhij", d_core_i, v_new, precision=_MATMUL_PRECISION)
        d_v_new = d_v_new + jnp.einsum("bhij,bhiv->bhjv", attn_qk, d_core_i, precision=_MATMUL_PRECISION)

        d_v_i = d_v_new
        d_v_prime = -d_v_new

        d_k_cum_i = jnp.einsum("bhiv,bhkv->bhik", d_v_prime, state_i, precision=_MATMUL_PRECISION)
        d_state_i = d_state_i + jnp.einsum("bhik,bhiv->bhkv", k_cum_i, d_v_prime, precision=_MATMUL_PRECISION)

        d_q_scaled = jnp.einsum("bhiv,bhkv->bhik", d_core_i, state_i, precision=_MATMUL_PRECISION)
        d_state_i = d_state_i + jnp.einsum("bhik,bhiv->bhkv", q_scaled, d_core_i, precision=_MATMUL_PRECISION)

        d_q_i = d_q_scaled * g_exp_i[:, :, :, None]
        d_g_exp_i = jnp.einsum("bhik,bhik->bhi", d_q_scaled, q_i, precision=_MATMUL_PRECISION)

        d_k_i = d_k_scaled * g_diff_i[:, :, :, None]
        d_g_diff_i = jnp.einsum("bhik,bhik->bhi", d_k_scaled, k_i, precision=_MATMUL_PRECISION)

        d_attn_qk_base = d_attn_qk * decay_i
        d_decay_i = d_attn_qk * attn_qk_base

        d_q_i = d_q_i + jnp.einsum("bhij,bhjk->bhik", d_attn_qk_base, k_i, precision=_MATMUL_PRECISION)
        d_k_i = d_k_i + jnp.einsum("bhji,bhjk->bhik", d_attn_qk_base, q_i, precision=_MATMUL_PRECISION)

        return d_state_i, (d_q_i, d_k_i, d_v_i, d_k_cum_i, d_g_exp_i, d_g_end_i, d_g_diff_i, d_decay_i)

    d_initial_state, grads_tm = lax.scan(
        rev_chunk_scan,
        d_final_state,
        (
            state_pre_tm,
            query_tm,
            key_tm,
            value_local_tm,
            key_cumdecay_tm,
            g_cumsum_exp_tm,
            g_end_exp_tm,
            g_diff_state_exp_tm,
            decay_mask_tm,
            d_out_tm,
        ),
        reverse=True,
    )

    (
        d_query_tm,
        d_key_tm,
        d_value_local_tm,
        d_key_cum_tm,
        d_g_exp_tm,
        d_g_end_tm,
        d_g_diff_tm,
        d_decay_mask_tm,
    ) = grads_tm

    d_query = d_query_tm.transpose(1, 2, 0, 3, 4)
    d_key = d_key_tm.transpose(1, 2, 0, 3, 4)
    d_value_local = d_value_local_tm.transpose(1, 2, 0, 3, 4)
    d_key_cum = d_key_cum_tm.transpose(1, 2, 0, 3, 4)
    d_g_exp = d_g_exp_tm.transpose(1, 2, 0, 3)
    d_g_end = d_g_end_tm.transpose(1, 2, 0)
    d_g_diff = d_g_diff_tm.transpose(1, 2, 0, 3)
    d_decay_mask = d_decay_mask_tm.transpose(1, 2, 0, 3, 4)

    d_attn = jnp.einsum("bhciv,bhcjv->bhcij", d_value_local, value_beta, precision=_MATMUL_PRECISION) + jnp.einsum(
        "bhcik,bhcjk->bhcij", d_key_cum, key_beta_scaled, precision=_MATMUL_PRECISION
    )
    d_value_beta = jnp.einsum("bhcij,bhciv->bhcjv", attn, d_value_local, precision=_MATMUL_PRECISION)
    d_key_beta_scaled = jnp.einsum("bhcij,bhcik->bhcjk", attn, d_key_cum, precision=_MATMUL_PRECISION)

    d_key_beta = d_key_beta_scaled * g_cumsum_exp[:, :, :, :, None]
    d_g_exp = d_g_exp + jnp.sum(d_key_beta_scaled * key_beta, axis=-1)

    tmp = jnp.einsum("bhcji,bhcjk->bhcik", attn, d_attn, precision=_MATMUL_PRECISION)
    d_k_attn = -jnp.einsum("bhcij,bhckj->bhcik", tmp, attn, precision=_MATMUL_PRECISION)

    strict_lower = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
    lower_inclusive = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=0)
    d_k_attn = d_k_attn * strict_lower

    kk = jnp.einsum("bhcik,bhcjk->bhcij", key_beta, key, precision=_MATMUL_PRECISION)
    d_kk = d_k_attn * decay_mask
    d_decay_mask = (d_decay_mask + d_k_attn * kk) * lower_inclusive

    d_key_beta = d_key_beta + jnp.einsum("bhcij,bhcjk->bhcik", d_kk, key, precision=_MATMUL_PRECISION)
    d_key = d_key + jnp.einsum("bhcji,bhcjk->bhcik", d_kk, key_beta, precision=_MATMUL_PRECISION)

    d_value = d_value_beta * beta[:, :, :, :, None]
    d_beta = jnp.sum(d_value_beta * value, axis=-1)

    d_key = d_key + d_key_beta * beta[:, :, :, :, None]
    d_beta = d_beta + jnp.sum(d_key_beta * key, axis=-1)

    d_decay_f = d_decay_mask * decay_mask
    d_g = jnp.sum(d_decay_f, axis=-1) - jnp.sum(d_decay_f, axis=-2)
    d_g = d_g + d_g_exp * g_cumsum_exp

    d_g_diff_term = d_g_diff * g_diff_state_exp
    d_g_end_total = jnp.sum(d_g_diff_term, axis=-1) + d_g_end * g_end_exp
    d_g = d_g - d_g_diff_term
    d_g = d_g.at[:, :, :, -1].add(d_g_end_total)
    d_decay = jnp.flip(jnp.cumsum(jnp.flip(d_g, axis=-1), axis=-1), axis=-1)

    d_query = d_query.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
    d_key = d_key.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
    d_value = d_value.reshape(B, H, total_len, V_dim)[:, :, :seq_len, :]
    d_beta = d_beta.reshape(B, H, total_len)[:, :, :seq_len]
    d_decay = d_decay.reshape(B, H, total_len)[:, :, :seq_len]

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

    def _cast_grad(x):
        if x is None:
            return None
        return x.astype(input_dtype) if x.dtype != input_dtype else x

    return (
        _cast_grad(d_query),
        _cast_grad(d_key),
        _cast_grad(d_value),
        _cast_grad(d_beta),
        _cast_grad(d_decay),
        _cast_grad(d_initial_state),
    )
