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
all forward intermediates (strict-lower inverse, decay masks) from raw inputs
+ saved per-chunk states, then computes all gradients in one fused pass.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float

from ...._xla.gated_delta_rule._xla_impl_fwd import _l2norm_bwd
from ._pallas_impl_fwd import (
    _chunk_blockspec,
    _dot,
    _neumann_inv,
    _PrimalDTypes,
    _segmented_chunk_cumsum,
)


def _bwd_one_chunk(q, k, v, beta, decay, d_out, state_pre, d_state_next, C):
    """Compute gradients for a single GDR chunk via full re-materialisation.

    Recomputes all forward intermediates (strict-lower inverse, decay masks,
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

    kk = _dot(k_beta, k.T)
    attn_neg = -(kk * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn = _neumann_inv(attn_neg, C, strict_lower=strict_lower, lower_mask=lower_mask)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))
    g_end = g_cumsum[C - 1 : C, :]
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0))
    g_diff_state_exp = jnp.exp(jnp.clip(g_end - g_cumsum, -20.0, 20.0))

    def _s(x):
        """Sanitize an intermediate tensor by replacing NaN/Inf with zero.

        Shorthand used throughout the backward chain-rule computation to keep matmul outputs finite.

        Args:
            x: Tensor to sanitize.

        Returns:
            ``x`` with NaN, +Inf and -Inf entries all replaced by ``0.0``.
        """
        return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    k_beta_scaled = k_beta * g_cumsum_exp
    V_dim = v_beta.shape[-1]
    vk_local = _s(_dot(attn, jnp.concatenate([v_beta, k_beta_scaled], axis=-1)))
    value_local = vk_local[:, :V_dim]
    k_cumdecay = vk_local[:, V_dim:]

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
    d_vk_beta = _s(_dot(attn.T, jnp.concatenate([d_value_local, d_k_cumdecay], axis=-1)))
    d_value_beta = d_vk_beta[:, :V_dim]
    d_key_beta_scaled = d_vk_beta[:, V_dim:]

    d_key_beta = d_key_beta_scaled * g_cumsum_exp
    d_g_exp = d_g_exp + jnp.sum(d_key_beta_scaled * k_beta, axis=-1, keepdims=True)

    tmp = _dot(attn.T, d_attn)
    d_k_attn = _s(-_dot(tmp, attn.T))
    d_k_attn = d_k_attn * strict_lower

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


def _bwd_one_chunk_segmented(
    q,
    k,
    v,
    beta,
    decay,
    g_cumsum,
    d_out,
    state_pre,
    d_state_next,
    seg,
    state_seg,
    has_valid,
    seg_last,
    g_end_value,
    C,
):
    """Compute gradients for one packed GDR chunk with explicit segment masks."""
    lower_base = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
    strict_base = lower_base - jnp.eye(C, dtype=jnp.float32)
    future_base = jnp.triu(jnp.ones((C, C), dtype=jnp.float32))

    valid_bool = seg >= 0
    valid = valid_bool.astype(jnp.float32)
    active = has_valid.astype(jnp.float32)
    same_segment = (seg[:, None] == seg[None, :]).astype(jnp.float32) * valid[:, None] * valid[None, :]
    lower_mask = lower_base * same_segment
    strict_lower = strict_base * same_segment
    future_same = future_base * same_segment
    same_as_last = (seg == seg_last).astype(jnp.float32)[:, None] * valid[:, None] * active
    read = (seg == state_seg).astype(jnp.float32)[:, None] * valid[:, None]

    v_beta = v * beta[:, None] * valid[:, None]
    k_beta = k * beta[:, None] * valid[:, None]
    g_cumsum = g_cumsum[:, None]
    g_diff = g_cumsum - g_cumsum.T
    decay_mask = jnp.exp(jnp.clip(g_diff * lower_mask, -20.0, 20.0)) * lower_mask

    kk = _dot(k_beta, k.T)
    attn_neg = -(kk * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn = _neumann_inv(attn_neg, C, strict_lower=strict_lower, lower_mask=lower_mask)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))
    g_end = g_end_value.astype(jnp.float32).reshape(1, 1) * active
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0))
    g_diff_state_exp = jnp.exp(jnp.clip(g_end - g_cumsum, -20.0, 20.0))

    def _s(x):
        return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    k_beta_scaled = k_beta * g_cumsum_exp
    V_dim = v_beta.shape[-1]
    combined_fwd = _s(_dot(attn, jnp.concatenate([v_beta, k_beta_scaled], axis=-1)))
    value_local = combined_fwd[:, :V_dim]
    k_cumdecay = combined_fwd[:, V_dim:]

    attn_qk_base = _dot(q, k.T)
    attn_qk = _s(attn_qk_base * decay_mask)
    q_scaled = q * g_cumsum_exp * valid[:, None]
    read_state = read.astype(state_pre.dtype)
    v_prime = _s(_dot(k_cumdecay, state_pre)) * read_state
    v_new = _s(value_local - v_prime)
    k_scaled = k * g_diff_state_exp * same_as_last

    d_out = d_out * valid[:, None]
    survive = (seg_last == state_seg).astype(jnp.float32) * active
    d_state_update = d_state_next * active
    d_state = _s(d_state_next * (1.0 - active) + d_state_update * g_end_exp * survive)
    d_g_end = jnp.sum(d_state_update * state_pre) * survive

    d_k_scaled = _s(_dot(v_new, d_state_update.T))
    d_v_new = _s(_dot(k_scaled, d_state_update))
    d_attn_qk = _s(_dot(d_out, v_new.T))
    d_v_new = _s(d_v_new + _dot(attn_qk.T, d_out))

    d_value_local = d_v_new
    d_v_prime = -d_v_new

    d_v_prime_state = d_v_prime * read_state
    d_k_cumdecay = _s(_dot(d_v_prime_state, state_pre.T))
    d_state = _s(d_state + _dot(k_cumdecay.T, d_v_prime_state))
    d_attn_inter = d_out * read_state
    d_q_scaled = _s(_dot(d_attn_inter, state_pre.T))
    d_state = _s(d_state + _dot(q_scaled.T, d_attn_inter))

    d_q = d_q_scaled * g_cumsum_exp * valid[:, None]
    d_g_exp = jnp.sum(d_q_scaled * q * valid[:, None], axis=-1, keepdims=True)
    d_k = d_k_scaled * g_diff_state_exp * same_as_last
    d_g_diff = jnp.sum(d_k_scaled * k * same_as_last, axis=-1, keepdims=True)

    d_attn_qk_base = _s(d_attn_qk * decay_mask)
    d_decay_mask_from_qk = d_attn_qk * attn_qk_base
    d_q = d_q + _dot(d_attn_qk_base, k)
    d_k = d_k + _dot(d_attn_qk_base.T, q)

    d_attn = _s(_dot(d_value_local, v_beta.T) + _dot(d_k_cumdecay, k_beta_scaled.T))
    combined_bwd = _s(_dot(attn.T, jnp.concatenate([d_value_local, d_k_cumdecay], axis=-1)))
    d_value_beta = combined_bwd[:, :V_dim]
    d_key_beta_scaled = combined_bwd[:, V_dim:]

    d_key_beta = d_key_beta_scaled * g_cumsum_exp
    d_g_exp = d_g_exp + jnp.sum(d_key_beta_scaled * k_beta, axis=-1, keepdims=True)

    tmp = _dot(attn.T, d_attn)
    d_k_attn = _s(-_dot(tmp, attn.T)) * strict_lower

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

    d_g_diff_term = d_g_diff * g_diff_state_exp * same_as_last
    d_g_end_total = jnp.sum(d_g_diff_term) + d_g_end * g_end_exp.reshape(())
    d_g = d_g - d_g_diff_term
    d_decay_final = _dot(future_same, d_g) + same_as_last * d_g_end_total

    d_state = _s(d_state)
    return d_state, d_q, d_k, d_v, d_beta, d_decay_final


def _run_grouped_dense_bwd_vmem_scan(
    state_pre,
    query_c,
    key_c,
    value_c,
    beta_c,
    decay_c,
    d_out_c,
    d_final_state,
    *,
    expand_ratio: int,
    grad_dtype=jnp.float32,
):
    """Single-launch reverse VMEM-resident Pallas backward for dense (unpacked) GDR.

    Processes chunks in reverse order (``rc = num_chunks - 1 - c``) inside a single
    ``pl.pallas_call`` with ``d_state`` held in VMEM scratch across all chunks. Works
    for both grouped (``expand_ratio > 1``) and non-grouped (``expand_ratio == 1``)
    layouts with zero inter-chunk forward recomputation and zero Q/K repetition.
    """
    bsz, num_value_heads, num_chunks, chunk_size, v_dim = value_c.shape
    qk_dim = query_c.shape[-1]

    beta_in = beta_c.astype(jnp.float32)[:, :, :, None, :]
    decay_in = decay_c.astype(jnp.float32)[:, :, :, None, :]
    d_final_in = d_final_state.astype(jnp.float32)

    def _bh_spec(shape):
        return pl.BlockSpec(shape, lambda b, h, c: (b, h, *([0] * (len(shape) - 2))))

    def _rev_bhc_spec(shape):
        return pl.BlockSpec(shape, lambda b, h, c: (b, h, num_chunks - 1 - c, *([0] * (len(shape) - 3))))

    def _rev_qk_spec(shape):
        return pl.BlockSpec(
            shape,
            lambda b, h, c: (b, h // expand_ratio, num_chunks - 1 - c, *([0] * (len(shape) - 3))),
        )

    def _kernel(
        state_pre_ref,
        q_ref,
        k_ref,
        v_ref,
        beta_ref,
        decay_ref,
        d_out_ref,
        d_final_state_ref,
        d_initial_state_ref,
        d_q_ref,
        d_k_ref,
        d_v_ref,
        d_beta_ref,
        d_decay_ref,
        d_state_scratch,
    ):
        c_idx = pl.program_id(2)

        @pl.when(c_idx == 0)
        def _init_d_state():
            d_state_scratch[...] = jnp.nan_to_num(
                d_final_state_ref[0, 0].astype(jnp.float32), nan=0.0, posinf=0.0, neginf=0.0
            )

        d_state_next = d_state_scratch[...]
        state_p = jnp.nan_to_num(state_pre_ref[0, 0, 0].astype(jnp.float32), nan=0.0, posinf=0.0, neginf=0.0)

        C = q_ref.shape[3]
        q = q_ref[0, 0, 0].astype(jnp.float32)
        k = k_ref[0, 0, 0].astype(jnp.float32)
        v = v_ref[0, 0, 0].astype(jnp.float32)
        beta = beta_ref[0, 0, 0, 0]
        decay = decay_ref[0, 0, 0, 0]
        d_out = d_out_ref[0, 0, 0].astype(jnp.float32)

        d_st, d_q_c, d_k_c, d_v_c, d_beta_c, d_decay_c = _bwd_one_chunk(
            q,
            k,
            v,
            beta,
            decay,
            d_out,
            state_p,
            d_state_next,
            C,
        )
        d_st_clean = jnp.nan_to_num(d_st, nan=0.0, posinf=0.0, neginf=0.0)
        d_state_scratch[...] = d_st_clean

        d_q_ref[0, 0, 0] = jnp.nan_to_num(d_q_c, nan=0.0, posinf=0.0, neginf=0.0).astype(d_q_ref.dtype)
        d_k_ref[0, 0, 0] = jnp.nan_to_num(d_k_c, nan=0.0, posinf=0.0, neginf=0.0).astype(d_k_ref.dtype)
        d_v_ref[0, 0, 0] = jnp.nan_to_num(d_v_c, nan=0.0, posinf=0.0, neginf=0.0).astype(d_v_ref.dtype)
        d_beta_ref[0, 0, 0, 0] = jnp.nan_to_num(d_beta_c[:, 0], nan=0.0, posinf=0.0, neginf=0.0).astype(d_beta_ref.dtype)
        d_decay_ref[0, 0, 0, 0] = jnp.nan_to_num(d_decay_c[:, 0], nan=0.0, posinf=0.0, neginf=0.0).astype(
            d_decay_ref.dtype
        )

        @pl.when(c_idx == num_chunks - 1)
        def _flush_d_init():
            d_initial_state_ref[0, 0] = d_st_clean.astype(d_initial_state_ref.dtype)

    call = pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                _rev_bhc_spec((1, 1, 1, qk_dim, v_dim)),
                _rev_qk_spec((1, 1, 1, chunk_size, qk_dim)),
                _rev_qk_spec((1, 1, 1, chunk_size, qk_dim)),
                _rev_bhc_spec((1, 1, 1, chunk_size, v_dim)),
                _rev_bhc_spec((1, 1, 1, 1, chunk_size)),
                _rev_bhc_spec((1, 1, 1, 1, chunk_size)),
                _rev_bhc_spec((1, 1, 1, chunk_size, v_dim)),
                _bh_spec((1, 1, qk_dim, v_dim)),
            ],
            out_specs=[
                _bh_spec((1, 1, qk_dim, v_dim)),
                _rev_bhc_spec((1, 1, 1, chunk_size, qk_dim)),
                _rev_bhc_spec((1, 1, 1, chunk_size, qk_dim)),
                _rev_bhc_spec((1, 1, 1, chunk_size, v_dim)),
                _rev_bhc_spec((1, 1, 1, 1, chunk_size)),
                _rev_bhc_spec((1, 1, 1, 1, chunk_size)),
            ],
            scratch_shapes=[pltpu.VMEM((qk_dim, v_dim), jnp.float32)],
            grid=(bsz, num_value_heads, num_chunks),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((bsz, num_value_heads, qk_dim, v_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_value_heads, num_chunks, chunk_size, qk_dim), grad_dtype),
            jax.ShapeDtypeStruct((bsz, num_value_heads, num_chunks, chunk_size, qk_dim), grad_dtype),
            jax.ShapeDtypeStruct((bsz, num_value_heads, num_chunks, chunk_size, v_dim), grad_dtype),
            jax.ShapeDtypeStruct((bsz, num_value_heads, num_chunks, 1, chunk_size), grad_dtype),
            jax.ShapeDtypeStruct((bsz, num_value_heads, num_chunks, 1, chunk_size), grad_dtype),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )
    d_init, d_q_rep, d_k_rep, d_v_c, d_beta_out, d_decay_out = call(
        state_pre,
        query_c,
        key_c,
        value_c,
        beta_in,
        decay_in,
        d_out_c,
        d_final_in,
    )
    return d_init, d_q_rep, d_k_rep, d_v_c, d_beta_out[:, :, :, 0, :], d_decay_out[:, :, :, 0, :]


def _run_grouped_segmented_bwd_vmem_scan(
    state_pre,
    state_seg_pre,
    query_c,
    key_c,
    value_c,
    beta_c,
    decay_c,
    g_cumsum_c,
    d_out_c,
    seg_c,
    chunk_has_valid,
    seg_last,
    g_end_c,
    d_final_state,
    *,
    expand_ratio: int,
    grad_dtype=jnp.float32,
):
    """Single-launch reverse VMEM-resident Pallas backward for packed grouped GDR.

    Processes chunks in reverse order (``rc = num_chunks - 1 - c``) inside a single
    ``pl.pallas_call`` with ``d_state`` held in VMEM scratch across all chunks,
    eliminating ``num_chunks`` separate Pallas launches, HBM ``d_state`` round-trips,
    per-chunk ``jnp.repeat`` of Q/K, and 6-D transpose copies.
    """
    bsz, num_value_heads, num_chunks, chunk_size, v_dim = value_c.shape
    qk_dim = query_c.shape[-1]

    state_seg_in = state_seg_pre.astype(jnp.int32)[:, :, :, None, None]
    beta_in = beta_c.astype(jnp.float32)[:, :, :, None, :]
    decay_in = decay_c.astype(jnp.float32)[:, :, :, None, :]
    g_cumsum_in = g_cumsum_c.astype(jnp.float32)[:, :, :, None, :]
    seg_in = seg_c.astype(jnp.int32)[:, :, :, None, :]
    has_valid_in = chunk_has_valid.astype(jnp.float32)[:, :, :, None, None]
    seg_last_in = seg_last.astype(jnp.int32)[:, :, :, None, None]
    g_end_in = g_end_c.astype(jnp.float32)[:, :, :, None, None]
    d_final_in = d_final_state.astype(jnp.float32)

    def _bh_spec(shape):
        return pl.BlockSpec(shape, lambda b, h, c: (b, h, *([0] * (len(shape) - 2))))

    def _rev_bhc_spec(shape):
        return pl.BlockSpec(shape, lambda b, h, c: (b, h, num_chunks - 1 - c, *([0] * (len(shape) - 3))))

    def _rev_qk_spec(shape):
        return pl.BlockSpec(
            shape,
            lambda b, h, c: (b, h // expand_ratio, num_chunks - 1 - c, *([0] * (len(shape) - 3))),
        )

    def _kernel(
        state_pre_ref,
        state_seg_ref,
        q_ref,
        k_ref,
        v_ref,
        beta_ref,
        decay_ref,
        g_cumsum_ref,
        d_out_ref,
        seg_ref,
        has_valid_ref,
        seg_last_ref,
        g_end_ref,
        d_final_state_ref,
        d_initial_state_ref,
        d_q_ref,
        d_k_ref,
        d_v_ref,
        d_beta_ref,
        d_decay_ref,
        d_state_scratch,
    ):
        c_idx = pl.program_id(2)

        @pl.when(c_idx == 0)
        def _init_d_state():
            d_state_scratch[...] = jnp.nan_to_num(
                d_final_state_ref[0, 0].astype(jnp.float32), nan=0.0, posinf=0.0, neginf=0.0
            )

        d_state_next = d_state_scratch[...]
        state_p = jnp.nan_to_num(state_pre_ref[0, 0, 0].astype(jnp.float32), nan=0.0, posinf=0.0, neginf=0.0)
        state_seg_val = state_seg_ref[0, 0, 0, 0, 0]

        C = q_ref.shape[3]
        q = q_ref[0, 0, 0].astype(jnp.float32)
        k = k_ref[0, 0, 0].astype(jnp.float32)
        v = v_ref[0, 0, 0].astype(jnp.float32)
        beta = beta_ref[0, 0, 0, 0]
        decay = decay_ref[0, 0, 0, 0]
        g_cumsum = g_cumsum_ref[0, 0, 0, 0].astype(jnp.float32)
        d_out = d_out_ref[0, 0, 0].astype(jnp.float32)
        seg = seg_ref[0, 0, 0, 0]
        has_valid = has_valid_ref[0, 0, 0, 0, 0].astype(jnp.float32)
        seg_last_val = seg_last_ref[0, 0, 0, 0, 0]
        g_end_value = g_end_ref[0, 0, 0, 0, 0].astype(jnp.float32)

        d_st, d_q_c, d_k_c, d_v_c, d_beta_c, d_decay_c = _bwd_one_chunk_segmented(
            q,
            k,
            v,
            beta,
            decay,
            g_cumsum,
            d_out,
            state_p,
            d_state_next,
            seg,
            state_seg_val,
            has_valid,
            seg_last_val,
            g_end_value,
            C,
        )
        d_st_clean = jnp.nan_to_num(d_st, nan=0.0, posinf=0.0, neginf=0.0)
        d_state_scratch[...] = d_st_clean

        d_q_ref[0, 0, 0] = jnp.nan_to_num(d_q_c, nan=0.0, posinf=0.0, neginf=0.0).astype(d_q_ref.dtype)
        d_k_ref[0, 0, 0] = jnp.nan_to_num(d_k_c, nan=0.0, posinf=0.0, neginf=0.0).astype(d_k_ref.dtype)
        d_v_ref[0, 0, 0] = jnp.nan_to_num(d_v_c, nan=0.0, posinf=0.0, neginf=0.0).astype(d_v_ref.dtype)
        d_beta_ref[0, 0, 0, 0] = jnp.nan_to_num(d_beta_c[:, 0], nan=0.0, posinf=0.0, neginf=0.0).astype(d_beta_ref.dtype)
        d_decay_ref[0, 0, 0, 0] = jnp.nan_to_num(d_decay_c[:, 0], nan=0.0, posinf=0.0, neginf=0.0).astype(
            d_decay_ref.dtype
        )

        @pl.when(c_idx == num_chunks - 1)
        def _flush_d_init():
            d_initial_state_ref[0, 0] = d_st_clean.astype(d_initial_state_ref.dtype)

    call = pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                _rev_bhc_spec((1, 1, 1, qk_dim, v_dim)),
                _rev_bhc_spec((1, 1, 1, 1, 1)),
                _rev_qk_spec((1, 1, 1, chunk_size, qk_dim)),
                _rev_qk_spec((1, 1, 1, chunk_size, qk_dim)),
                _rev_bhc_spec((1, 1, 1, chunk_size, v_dim)),
                _rev_bhc_spec((1, 1, 1, 1, chunk_size)),
                _rev_bhc_spec((1, 1, 1, 1, chunk_size)),
                _rev_bhc_spec((1, 1, 1, 1, chunk_size)),
                _rev_bhc_spec((1, 1, 1, chunk_size, v_dim)),
                _rev_bhc_spec((1, 1, 1, 1, chunk_size)),
                _rev_bhc_spec((1, 1, 1, 1, 1)),
                _rev_bhc_spec((1, 1, 1, 1, 1)),
                _rev_bhc_spec((1, 1, 1, 1, 1)),
                _bh_spec((1, 1, qk_dim, v_dim)),
            ],
            out_specs=[
                _bh_spec((1, 1, qk_dim, v_dim)),
                _rev_bhc_spec((1, 1, 1, chunk_size, qk_dim)),
                _rev_bhc_spec((1, 1, 1, chunk_size, qk_dim)),
                _rev_bhc_spec((1, 1, 1, chunk_size, v_dim)),
                _rev_bhc_spec((1, 1, 1, 1, chunk_size)),
                _rev_bhc_spec((1, 1, 1, 1, chunk_size)),
            ],
            scratch_shapes=[pltpu.VMEM((qk_dim, v_dim), jnp.float32)],
            grid=(bsz, num_value_heads, num_chunks),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((bsz, num_value_heads, qk_dim, v_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_value_heads, num_chunks, chunk_size, qk_dim), grad_dtype),
            jax.ShapeDtypeStruct((bsz, num_value_heads, num_chunks, chunk_size, qk_dim), grad_dtype),
            jax.ShapeDtypeStruct((bsz, num_value_heads, num_chunks, chunk_size, v_dim), grad_dtype),
            jax.ShapeDtypeStruct((bsz, num_value_heads, num_chunks, 1, chunk_size), grad_dtype),
            jax.ShapeDtypeStruct((bsz, num_value_heads, num_chunks, 1, chunk_size), grad_dtype),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )
    d_init, d_q_rep, d_k_rep, d_v_c, d_beta_out, d_decay_out = call(
        state_pre,
        state_seg_in,
        query_c,
        key_c,
        value_c,
        beta_in,
        decay_in,
        g_cumsum_in,
        d_out_c,
        seg_in,
        has_valid_in,
        seg_last_in,
        g_end_in,
        d_final_in,
    )
    return d_init, d_q_rep, d_k_rep, d_v_c, d_beta_out[:, :, :, 0, :], d_decay_out[:, :, :, 0, :]


def _cast_grad(x, dtype):
    """Cast a gradient array to ``dtype``, passing ``None`` through unchanged.

    Args:
        x: Gradient array, or ``None`` for a non-differentiated / absent grad.
        dtype: Target dtype to cast to.

    Returns:
        ``None`` if ``x`` is ``None``; otherwise ``x`` cast to ``dtype`` (returned
        unchanged when it already has that dtype).
    """
    if x is None:
        return None
    return x.astype(dtype) if x.dtype != dtype else x


def _resolve_primal_dtypes(primal_meta, fallback_dtype):
    """Resolve ``(beta_dtype, decay_dtype, initial_state_dtype)`` from backward residual metadata.

    Handles both :class:`_PrimalDTypes` tokens stored by the forward rules and
    legacy residual tuples whose slot holds a raw ``initial_state`` array or
    ``None``.
    """
    if isinstance(primal_meta, _PrimalDTypes):
        return (
            primal_meta.beta or fallback_dtype,
            primal_meta.decay or fallback_dtype,
            primal_meta.initial_state or fallback_dtype,
        )
    state_dtype = getattr(primal_meta, "dtype", None) or fallback_dtype
    return fallback_dtype, fallback_dtype, state_dtype


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
    if len(res) == 19:
        (
            query,
            key,
            value,
            beta,
            decay,
            state_pre_all,
            primal_meta,
            q_inv_norm,
            k_inv_norm,
            seq_len,
            pad_size,
            decay_was_none,
            initial_state_was_none,
            effective_chunk_size,
            seg_c,
            valid_c,
            chunk_has_valid,
            seg_last,
            state_seg_pre_all,
        ) = res
    elif len(res) == 14:
        (
            query,
            key,
            value,
            beta,
            decay,
            state_pre_all,
            primal_meta,
            q_inv_norm,
            k_inv_norm,
            seq_len,
            pad_size,
            decay_was_none,
            initial_state_was_none,
            effective_chunk_size,
        ) = res
        seg_c = valid_c = chunk_has_valid = seg_last = state_seg_pre_all = None
    else:
        raise ValueError(f"Unexpected GDR backward residual length: {len(res)}")
    chunk_size = effective_chunk_size
    d_out, d_final_state = g
    input_dtype = query.dtype
    beta_dtype, decay_dtype, initial_state_dtype = _resolve_primal_dtypes(primal_meta, input_dtype)
    B, H, num_chunks, _C, K_dim = query.shape
    V_dim = value.shape[-1]
    scale = 1.0 / math.sqrt(K_dim)

    if pad_size > 0:
        d_out = jnp.pad(d_out, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    d_out_c = d_out.reshape(B, H, num_chunks, chunk_size, V_dim)

    segmented = seg_c is not None
    if int(state_pre_all.shape[2]) != num_chunks:
        raise ValueError(
            f"Unexpected GDR state residual chunks: got {state_pre_all.shape[2]}, expected {num_chunks}."
        )
    if segmented:
        g_cumsum_c = _segmented_chunk_cumsum(decay[:, :, :, None, :], seg_c, valid_c)
        positions = jnp.arange(chunk_size, dtype=jnp.int32)
        last_valid_idx = jnp.max(jnp.where(valid_c, positions[None, None, None, :], 0), axis=-1)
        g_end_c = jnp.take_along_axis(g_cumsum_c, last_valid_idx[..., None], axis=-1)[..., 0]
        g_end_c = g_end_c * chunk_has_valid.astype(jnp.float32)
        d_initial_state, d_q_c, d_k_c, d_v_c, d_beta_c, d_decay_c = _run_grouped_segmented_bwd_vmem_scan(
            state_pre_all,
            state_seg_pre_all,
            query,
            key,
            value,
            beta,
            decay,
            g_cumsum_c,
            d_out_c,
            seg_c,
            chunk_has_valid,
            seg_last,
            g_end_c,
            d_final_state,
            expand_ratio=1,
            grad_dtype=input_dtype,
        )
    else:
        d_initial_state, d_q_c, d_k_c, d_v_c, d_beta_c, d_decay_c = _run_grouped_dense_bwd_vmem_scan(
            state_pre_all,
            query,
            key,
            value,
            beta,
            decay,
            d_out_c,
            d_final_state,
            expand_ratio=1,
            grad_dtype=input_dtype,
        )

    total_len = seq_len + pad_size
    d_query = d_q_c.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
    d_key = d_k_c.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
    d_value = d_v_c.reshape(B, H, total_len, V_dim)[:, :, :seq_len, :]
    d_beta = d_beta_c.reshape(B, H, total_len)[:, :, :seq_len]
    d_decay = d_decay_c.reshape(B, H, total_len)[:, :, :seq_len]

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
        """Sanitize a gradient and cast it to the input dtype.

        Used to finalize each returned gradient: replaces NaN/Inf with zero, clips to the representable range for
        low-precision targets (bf16/fp16) to avoid overflow on cast, and casts only when the dtype differs.

        Args:
            x: Gradient tensor to finalize, or ``None`` (e.g. an absent ``d_decay``/``d_initial_state``).
            dtype: Target dtype, typically the original input dtype.

        Returns:
            ``None`` if ``x`` is ``None``; otherwise the sanitized, optionally clipped tensor cast to ``dtype``.
        """
        if x is None:
            return None
        x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if dtype == jnp.bfloat16 or dtype == jnp.float16:
            x = jnp.clip(x, -65000.0, 65000.0)
        return x.astype(dtype) if x.dtype != dtype else x

    return (
        _safe_cast(d_query, query.dtype),
        _safe_cast(d_key, key.dtype),
        _safe_cast(d_value, value.dtype),
        _safe_cast(d_beta, beta_dtype),
        _safe_cast(d_decay, decay_dtype),
        _safe_cast(d_initial_state, initial_state_dtype),
    )


def _chunk_gdr_grouped_bwd(
    chunk_size: int,
    use_qk_l2norm: bool,
    res: tuple,
    g: tuple[Float[Array, "..."], Float[Array, "..."]],
) -> tuple:
    """Grouped-head backward without materializing repeated Q/K over the full sequence."""
    if len(res) == 20:
        (
            query,
            key,
            value,
            beta,
            decay,
            state_group_pre,
            primal_meta,
            q_inv_norm,
            k_inv_norm,
            seq_len,
            pad_size,
            decay_was_none,
            initial_state_was_none,
            effective_chunk_size,
            group_size,
            seg_c,
            valid_c,
            chunk_has_valid,
            seg_last,
            state_seg_group_pre,
        ) = res
        segmented = True
    elif len(res) == 15:
        (
            query,
            key,
            value,
            beta,
            decay,
            state_group_pre,
            primal_meta,
            q_inv_norm,
            k_inv_norm,
            seq_len,
            pad_size,
            decay_was_none,
            initial_state_was_none,
            effective_chunk_size,
            group_size,
        ) = res
        seg_c = valid_c = chunk_has_valid = seg_last = state_seg_group_pre = None
        segmented = False
    else:
        raise ValueError(f"Unexpected grouped GDR backward residual length: {len(res)}")
    chunk_size = effective_chunk_size
    d_out, d_final_state = g
    input_dtype = query.dtype
    beta_dtype, decay_dtype, initial_state_dtype = _resolve_primal_dtypes(primal_meta, input_dtype)
    B, Hq, num_chunks, _C, K_dim = query.shape
    Hv = value.shape[1]
    V_dim = value.shape[-1]
    if Hv % Hq != 0:
        raise ValueError(f"grouped GDR requires value heads ({Hv}) to be a multiple of query heads ({Hq})")
    expand_ratio = Hv // Hq
    if int(group_size) != 1 or int(state_group_pre.shape[2]) != num_chunks:
        raise ValueError(
            f"Unexpected grouped GDR state residual chunks: got {state_group_pre.shape[2]} "
            f"(group_size={group_size}), expected {num_chunks} per-chunk states."
        )
    scale = 1.0 / math.sqrt(K_dim)

    if pad_size > 0:
        d_out = jnp.pad(d_out, ((0, 0), (0, 0), (0, pad_size), (0, 0)))

    total_len = seq_len + pad_size
    d_out_c = d_out.reshape(B, Hv, num_chunks, chunk_size, V_dim)
    if segmented:
        g_cumsum_c = _segmented_chunk_cumsum(decay[:, :, :, None, :], seg_c, valid_c)
        positions = jnp.arange(chunk_size, dtype=jnp.int32)
        last_valid_idx = jnp.max(jnp.where(valid_c, positions[None, None, None, :], 0), axis=-1)
        g_end_c = jnp.take_along_axis(g_cumsum_c, last_valid_idx[..., None], axis=-1)[..., 0]
        g_end_c = g_end_c * chunk_has_valid.astype(jnp.float32)

        d_initial_state, d_q_rep, d_k_rep, d_v_c, d_beta_c, d_decay_c = _run_grouped_segmented_bwd_vmem_scan(
            state_group_pre,
            state_seg_group_pre,
            query,
            key,
            value,
            beta,
            decay,
            g_cumsum_c,
            d_out_c,
            seg_c,
            chunk_has_valid,
            seg_last,
            g_end_c,
            d_final_state,
            expand_ratio=expand_ratio,
            grad_dtype=input_dtype,
        )
    else:
        d_initial_state, d_q_rep, d_k_rep, d_v_c, d_beta_c, d_decay_c = _run_grouped_dense_bwd_vmem_scan(
            state_group_pre,
            query,
            key,
            value,
            beta,
            decay,
            d_out_c,
            d_final_state,
            expand_ratio=expand_ratio,
            grad_dtype=input_dtype,
        )
    d_query = (
        d_q_rep.reshape(B, Hq, expand_ratio, total_len, K_dim).sum(axis=2, dtype=jnp.float32)[:, :, :seq_len, :]
    )
    d_key = (
        d_k_rep.reshape(B, Hq, expand_ratio, total_len, K_dim).sum(axis=2, dtype=jnp.float32)[:, :, :seq_len, :]
    )
    d_value = d_v_c.reshape(B, Hv, total_len, V_dim)[:, :, :seq_len, :]
    d_beta = d_beta_c.reshape(B, Hv, total_len)[:, :, :seq_len]
    d_decay = d_decay_c.reshape(B, Hv, total_len)[:, :, :seq_len]

    d_query = d_query * scale
    if use_qk_l2norm:
        q_norm = query.reshape(B, Hq, total_len, K_dim)[:, :, :seq_len, :] / scale
        k_norm = key.reshape(B, Hq, total_len, K_dim)[:, :, :seq_len, :]
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
        _safe_cast(d_query, query.dtype),
        _safe_cast(d_key, key.dtype),
        _safe_cast(d_value, value.dtype),
        _safe_cast(d_beta, beta_dtype),
        _safe_cast(d_decay, decay_dtype),
        _safe_cast(d_initial_state, initial_state_dtype),
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
    ``kv_mem``, ``delta``, and ``state`` (the new state), then applies the chain
    rule. Refs are per-grid-block slices (leading batch/head dims already
    indexed to size 1 by the BlockSpec).

    Grid: ``(batch, num_heads)``; both axes are "parallel".

    Args:
        q_ref: [1, 1, 1, K] — query for this step.
        k_ref: [1, 1, 1, K] — key for this step.
        v_ref: [1, 1, 1, V] — value for this step.
        beta_ref: [1, 1, 1, 1] — scalar gate.
        decay_ref: [1, 1, 1, 1] — scalar log decay.
        state_prev_ref: [1, 1, K, V] — state before the step.
        d_out_ref: [1, 1, 1, V] — output gradient.
        d_state_next_ref: [1, 1, K, V] — state gradient from downstream.
        d_q_ref: [1, 1, 1, K] — output ref for the query gradient.
        d_k_ref: [1, 1, 1, K] — output ref for the key gradient.
        d_v_ref: [1, 1, 1, V] — output ref for the value gradient.
        d_beta_ref: [1, 1, 1, 1] — output ref for the scalar beta gradient.
        d_decay_ref: [1, 1, 1, 1] — output ref for the scalar decay gradient.
        d_state_ref: [1, 1, K, V] — output ref for the gradient w.r.t. ``state_prev``.

    Returns:
        None. All results are written in place into the output refs listed above.
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

    Same math as :func:`_gdr_single_step_bwd_kernel`: it recomputes the forward
    pass inline (``state_decayed``, ``kv_mem``, ``delta``, ``state``) and then
    applies the chain rule. The only difference is that the two ``[K, V]`` state
    matrices are streamed in via asynchronous DMA copies into VMEM scratch so the
    semaphore cost is amortized; the per-token vectors and scalar beta/decay
    inputs stay as direct loads because they are consumed once and are small
    relative to the state.

    Grid: ``(batch, num_heads)``; both axes are "parallel".

    Args:
        q_ref: [1, 1, 1, K] — query for this step (loaded directly, late).
        k_ref: [1, 1, 1, K] — key for this step.
        v_ref: [1, 1, 1, V] — value for this step.
        beta_ref: [1, 1, 1, 1] — scalar gate.
        decay_ref: [1, 1, 1, 1] — scalar log decay.
        state_prev_ref: [1, 1, K, V] — state before the step (DMA source).
        d_out_ref: [1, 1, 1, V] — output gradient (loaded directly, late).
        d_state_next_ref: [1, 1, K, V] — state gradient from downstream (DMA source).
        d_q_ref: [1, 1, 1, K] — output ref for the query gradient.
        d_k_ref: [1, 1, 1, K] — output ref for the key gradient.
        d_v_ref: [1, 1, 1, V] — output ref for the value gradient.
        d_beta_ref: [1, 1, 1, 1] — output ref for the scalar beta gradient.
        d_decay_ref: [1, 1, 1, 1] — output ref for the scalar decay gradient.
        d_state_ref: [1, 1, K, V] — output ref for the gradient w.r.t. ``state_prev``.
        state_tile_ref: [1, 1, K, V] VMEM scratch — DMA destination for ``state_prev``.
        d_state_next_tile_ref: [1, 1, K, V] VMEM scratch — DMA destination for ``d_state_next``.
        dma_sem_ref: DMA semaphore array of length 2, one per in-flight copy.

    Returns:
        None. All results are written in place into the output refs listed above.
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
    if len(res) == 10:
        query, key, value, beta, decay, recurrent_state, q_inv_norm, k_inv_norm, decay_was_none, primal_meta = res
    elif len(res) == 9:
        query, key, value, beta, decay, recurrent_state, q_inv_norm, k_inv_norm, decay_was_none = res
        primal_meta = recurrent_state
    else:
        raise ValueError(f"Unexpected single-step GDR backward residual length: {len(res)}")
    d_out, d_final_state = g
    input_dtype = query.dtype
    beta_dtype, decay_dtype, state_dtype = _resolve_primal_dtypes(primal_meta, input_dtype)
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
        _cast_grad(d_query, query.dtype),
        _cast_grad(d_key, key.dtype),
        _cast_grad(d_value, value.dtype),
        _cast_grad(d_beta, beta_dtype),
        _cast_grad(d_decay, decay_dtype),
        _cast_grad(d_state, state_dtype),
    )
