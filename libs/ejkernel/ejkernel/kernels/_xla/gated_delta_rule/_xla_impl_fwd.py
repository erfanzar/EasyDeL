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

"""Forward pass implementations for Gated Delta Rule (GDR) linear attention.

This module provides three forward pass variants for GDR:

1. Recurrent (_recurrent_gdr_fwd):
   Pure sequential scan with O(L) time complexity. Best for very long
   sequences or memory-constrained inference.

2. Chunked (_chunk_gdr_fwd):
   Hybrid approach with parallel intra-chunk computation (Neumann series)
   and sequential inter-chunk state propagation. Includes a custom VJP
   with an analytical backward pass for numerical stability.

3. Single-step (_single_step_gdr_fwd):
   Optimized path for seq_len=1 during autoregressive inference.

The GDR update rule:
    h_t = exp(decay_t) * h_{t-1} + k_t (x) (beta_t * (v_t - h_{t-1} @ k_t))
    o_t = h_t @ q_t

Where h_t is the [head_dim, d_state] memory matrix that stores key-value
associations and supports efficient retrieval via query projection.

References:
    - Qwen3Next: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int

_MATMUL_PRECISION = lax.Precision.HIGHEST


def _l2norm(x: Float[Array, "..."], axis: int = -1, eps: float = 1e-6) -> Float[Array, "..."]:
    """Apply L2 normalization along specified axis.

    Args:
        x: Input tensor to normalize.
        axis: Axis along which to normalize (default: -1).
        eps: Small constant for numerical stability.

    Returns:
        L2-normalized tensor with same shape as input.
    """
    inv_norm = lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _l2norm_with_inv(
    x: Float[Array, "..."], axis: int = -1, eps: float = 1e-6
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Returns both normalized tensor and inverse norm.

    Args:
        x: Input tensor to normalize.
        axis: Axis along which to normalize (default: -1).
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (normalized tensor, inverse norm).
    """
    inv_norm = lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm, inv_norm


def _l2norm_bwd(grad_y, y, inv_norm):
    """Backward pass for y = l2norm(x).

    Args:
        grad_y: Upstream gradient.
        y: Normalized output from forward pass.
        inv_norm: Inverse norm saved from forward pass.

    Returns:
        Gradient with respect to x.
    """
    proj = jnp.sum(grad_y * y, axis=-1, keepdims=True)
    return inv_norm * (grad_y - y * proj)


def _strict_lower_inverse(matrix_strict_lower: jax.Array) -> jax.Array:
    """Compute the inverse of (I + L) where L is a batch of strict-lower-triangular matrices.

    This is used in the chunked GDR forward pass to solve the intra-chunk
    linear system ``(I + S) v_beta = v_final`` efficiently.  Because ``(I + L)``
    is unit-lower-triangular (ones on the diagonal), the inverse is computed
    via a single triangular solve rather than a general LU factorisation.

    Args:
        matrix_strict_lower: Batch of strict-lower-triangular matrices with
            shape ``[..., n, n]``.  The diagonal and upper triangle are
            assumed to be zero; only the strict lower triangle is used.

    Returns:
        Batch of inverses of ``(I + matrix_strict_lower)`` with the same
        shape as the input.
    """
    n = int(matrix_strict_lower.shape[-1])
    eye = jnp.eye(n, dtype=matrix_strict_lower.dtype)
    flat = matrix_strict_lower.reshape((-1, n, n))

    def _solve_one(lhs):
        return lax.linalg.triangular_solve(
            lhs,
            eye,
            left_side=True,
            lower=True,
            unit_diagonal=True,
        )

    inv_flat = jax.vmap(_solve_one)(flat)
    return inv_flat.reshape(matrix_strict_lower.shape)


def _recurrent_gdr_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
    chunk_size: int = 64,
    seg_ids: Int[Array, "batch seq_len"] | None = None,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Chunked forward pass for Gated Delta Rule using exact triangular solve.

    Processes the sequence in chunks of ``chunk_size``, computing the
    intra-chunk linear system via ``jax.scipy.linalg.solve_triangular``
    (exact ``(I+S)^{-1}`` rather than a Neumann approximation) and
    propagating the recurrent state across chunks with ``lax.scan``.

    This is the *production* multi-token path for GDR: the custom-VJP
    Neumann variant (``_chunk_gdr_fwd``) delegates to this function because
    the Neumann series diverges on padded batches at training time.

    The per-chunk update solves:
        S[i,j] = sum_{j'<i} exp(g_cumsum[i] - g_cumsum[j']) * beta[j'] * (k[j'] ⊗ k[i])
        v_final = (I + S)^{-1} v_beta           (triangular solve)
        h_next  = h * exp(g_end) + K_scaled^T @ v_final

    Note:
        Internal tensor layout is [batch, num_heads, seq_len, dim] (heads-first),
        which is the opposite of the public-API convention.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim].
            Must already be in heads-first layout.
        key: Key tensor [batch, num_heads, seq_len, head_dim].
        value: Value tensor [batch, num_heads, seq_len, d_state].
        beta: Per-token gating [batch, num_heads, seq_len].
        decay: Per-token log-decay [batch, num_heads, seq_len], or None for
            zero decay (full memory retention).
        initial_state: Optional initial recurrent state
            [batch, num_heads, head_dim, d_state].  Defaults to all-zeros.
        use_qk_l2norm: Whether to L2-normalize queries and keys before
            the attention computation.  Strongly recommended for stability.
        chunk_size: Number of tokens per chunk.  The sequence is padded to
            a multiple of this value before processing.

    Returns:
        Tuple of (outputs, final_state) where:
            - outputs: [batch, num_heads, seq_len, d_state], cast to
              ``query.dtype``.
            - final_state: [batch, num_heads, head_dim, d_state], cast to
              ``query.dtype``.
    """
    B, H, L, K_dim = query.shape
    V_dim = value.shape[-1]
    input_dtype = query.dtype
    C = chunk_size

    if use_qk_l2norm:
        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    scale = 1.0 / (K_dim**0.5)
    query = (query * scale).astype(input_dtype)
    key = key.astype(input_dtype)
    value = value.astype(input_dtype)
    beta = beta.astype(input_dtype)

    if initial_state is None:
        initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
    else:
        initial_state = initial_state.astype(jnp.float32)

    if decay is None:
        decay = jnp.zeros((B, H, L), dtype=input_dtype)
    else:
        decay = decay.astype(input_dtype)

    seg_hc = None
    if seg_ids is not None:
        seg_full = jnp.broadcast_to(seg_ids.astype(jnp.int32)[:, None, :], (B, H, L))

    pad_size = (C - L % C) % C
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        decay = jnp.pad(decay, ((0, 0), (0, 0), (0, pad_size)))
        if seg_ids is not None:
            seg_full = jnp.pad(seg_full, ((0, 0), (0, 0), (0, pad_size)), constant_values=-1)

    NC = (L + pad_size) // C

    q_c = query.reshape(B, H, NC, C, K_dim)
    k_c = key.reshape(B, H, NC, C, K_dim)
    v_c = value.reshape(B, H, NC, C, V_dim)
    beta_c = beta.reshape(B, H, NC, C)
    g_c = decay.reshape(B, H, NC, C)

    if seg_ids is not None:
        seg_hc = seg_full.reshape(B, H, NC, C)  # [B, H, NC, C] document id per token
        is_start = jnp.concatenate(
            [jnp.ones((B, H, NC, 1), dtype=jnp.bool_), seg_hc[..., 1:] != seg_hc[..., :-1]],
            axis=-1,
        )
        raw_cumsum = jnp.cumsum(g_c, axis=-1)
        idx = jnp.arange(C)
        start_idx = jax.lax.cummax(jnp.where(is_start, idx[None, None, None, :], 0), axis=3)
        cumsum_at_start = jnp.take_along_axis(
            jnp.concatenate([jnp.zeros((B, H, NC, 1), raw_cumsum.dtype), raw_cumsum[..., :-1]], axis=-1),
            start_idx,
            axis=-1,
        )
        g_cumsum = raw_cumsum - cumsum_at_start
        same_seg_pair = seg_hc[..., :, None] == seg_hc[..., None, :]  # [B, H, NC, C, C]
    else:
        g_cumsum = jnp.cumsum(g_c, axis=-1)
        same_seg_pair = None

    k_beta = k_c * beta_c[..., None]
    v_beta = v_c * beta_c[..., None]

    S = jnp.einsum("bhcik,bhcjk->bhcij", k_beta, k_c, precision=_MATMUL_PRECISION).astype(jnp.float32)

    g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
    strict_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_), k=-1)
    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    if same_seg_pair is not None:
        # Intra-chunk attention only within the same document.
        strict_lower = jnp.logical_and(strict_lower, same_seg_pair)
        lower_mask = jnp.logical_and(lower_mask, same_seg_pair)
    g_diff = jnp.where(strict_lower, g_diff, -1e30)
    S = jnp.where(strict_lower, S * jnp.exp(jnp.clip(g_diff, -20.0, 20.0)), 0.0)

    eye = jnp.eye(C, dtype=jnp.float32)
    lhs = jnp.broadcast_to(eye, S.shape) + S

    lhs_flat = lhs.reshape(-1, C, C)
    jnp.broadcast_to(eye, lhs_flat.shape)

    def _solve_one(m):
        return jax.scipy.linalg.solve_triangular(m, eye, lower=True, unit_diagonal=True)

    A_flat = jax.vmap(_solve_one)(lhs_flat)
    A = A_flat.reshape(B, H, NC, C, C)

    u_chunks = jnp.einsum("bhcij,bhcjv->bhciv", A, v_beta.astype(jnp.float32), precision=_MATMUL_PRECISION).astype(
        input_dtype
    )

    k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))[..., None]
    w_chunks = jnp.einsum("bhcij,bhcjk->bhcik", A, k_beta_g, precision=_MATMUL_PRECISION).astype(input_dtype)

    attn_qk = jnp.einsum("bhcik,bhcjk->bhcij", q_c, k_c, precision=_MATMUL_PRECISION).astype(jnp.float32)
    g_diff_intra = g_cumsum[..., :, None] - g_cumsum[..., None, :]
    g_diff_intra = jnp.where(lower_mask, g_diff_intra, -1e30)
    attn_i = jnp.where(lower_mask, attn_qk * jnp.exp(jnp.clip(g_diff_intra, -20.0, 20.0)), 0.0).astype(input_dtype)

    q_g = (q_c.astype(jnp.float32) * jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))[..., None]).astype(input_dtype)
    g_end_exp = jnp.exp(jnp.clip(g_cumsum[..., -1], -20.0, 20.0))[..., None, None]
    g_diff_state = jnp.exp(jnp.clip(g_cumsum[..., -1, None] - g_cumsum, -20.0, 20.0))[..., None]
    k_g_diff = (k_c.astype(jnp.float32) * g_diff_state).astype(input_dtype)

    if seg_hc is not None:
        # Only the LAST segment of a chunk carries recurrent state OUT of the chunk; mask the
        # state-update so only last-segment keys are folded in. (The inter-chunk READ mask
        # and the old-state survival are segment-dependent on the *incoming* state and are
        # applied inside the scan, where ``state_seg`` is known.)
        seg_last = seg_hc[..., -1]  # [B, H, NC]
        same_as_last = (seg_hc == seg_last[..., None]).astype(k_g_diff.dtype)  # [B, H, NC, C]
        k_g_diff = k_g_diff * same_as_last[..., None]

    xs = (
        u_chunks.transpose(2, 0, 1, 3, 4),
        w_chunks.transpose(2, 0, 1, 3, 4),
        q_g.transpose(2, 0, 1, 3, 4),
        attn_i.transpose(2, 0, 1, 3, 4),
        g_end_exp.transpose(2, 0, 1, 3, 4),
        k_g_diff.transpose(2, 0, 1, 3, 4),
    )

    def scan_body(state, inputs):
        """Process one chunk: apply inter-chunk state contribution and update state.

        Args:
            state: Current recurrent state [batch, heads, head_dim, d_state].
            inputs: Tuple of pre-computed chunk tensors
                (u, w, q_scaled, attn_intra, g_last_exp, k_scaled).

        Returns:
            Tuple of (new_state, core_out) where core_out has shape
            [batch, heads, chunk_size, d_state].
        """
        u, w, q_scaled, attn_intra, g_last_exp, k_scaled = inputs

        v_prime = jnp.einsum("bhck,bhkv->bhcv", w, state, precision=_MATMUL_PRECISION)
        attn_inter = jnp.einsum("bhck,bhkv->bhcv", q_scaled, state, precision=_MATMUL_PRECISION)
        v_new = u.astype(jnp.float32) - v_prime
        core_out = attn_inter + jnp.einsum("bhcr,bhrv->bhcv", attn_intra, v_new, precision=_MATMUL_PRECISION)

        state_update = jnp.einsum("bhkc,bhcv->bhkv", k_scaled.transpose(0, 1, 3, 2), v_new, precision=_MATMUL_PRECISION)
        new_state = jnp.nan_to_num(state * g_last_exp + state_update, nan=0.0, posinf=0.0, neginf=0.0)

        return new_state, core_out

    def scan_body_seg(carry, inputs):
        """Segment-aware chunk step for sequence packing.

        Carries ``(state, state_seg)`` where ``state_seg`` is the document id the recurrent
        ``state`` belongs to. A token reads the incoming state only if its segment matches
        ``state_seg`` (so a new document never inherits the previous document's memory); the
        old state survives to the next chunk only if the chunk's last token continues that
        same document. The outgoing state always represents the chunk's last token's segment.
        """
        state, state_seg = carry
        u, w, q_scaled, attn_intra, g_last_exp, k_scaled, seg_c, seg_last_c = inputs

        read = (seg_c == state_seg[..., None]).astype(state.dtype)[..., None]  # [B, H, C, 1]
        v_prime = jnp.einsum("bhck,bhkv->bhcv", w, state, precision=_MATMUL_PRECISION) * read
        attn_inter = jnp.einsum("bhck,bhkv->bhcv", q_scaled, state, precision=_MATMUL_PRECISION) * read
        v_new = u.astype(jnp.float32) - v_prime
        core_out = attn_inter + jnp.einsum("bhcr,bhrv->bhcv", attn_intra, v_new, precision=_MATMUL_PRECISION)

        state_update = jnp.einsum("bhkc,bhcv->bhkv", k_scaled.transpose(0, 1, 3, 2), v_new, precision=_MATMUL_PRECISION)
        survive = (seg_last_c == state_seg).astype(state.dtype)[..., None, None]
        new_state = jnp.nan_to_num(state * g_last_exp * survive + state_update, nan=0.0, posinf=0.0, neginf=0.0)
        return (new_state, seg_last_c), core_out

    if seg_hc is None:
        final_state, core_out_tm = lax.scan(scan_body, initial_state, xs)
    else:
        xs_seg = (*xs, seg_hc.transpose(2, 0, 1, 3), seg_last.transpose(2, 0, 1))
        state_seg0 = seg_hc[:, :, 0, 0]  # segment the (zero) initial state nominally belongs to
        (final_state, _), core_out_tm = lax.scan(scan_body_seg, (initial_state, state_seg0), xs_seg)

    core_out = core_out_tm.transpose(1, 2, 0, 3, 4)
    outputs = core_out.reshape(B, H, -1, V_dim)[:, :, :L, :].astype(input_dtype)

    return outputs, final_state.astype(input_dtype)


def _chunk_gdr_fwd_impl(
    query,
    key,
    value,
    beta,
    decay,
    chunk_size,
    initial_state,
    use_qk_l2norm,
):
    """Core implementation for the Neumann-series chunked GDR forward.

    Delegates to ``_chunk_gdr_fwd_core`` with ``save_residual=False``.
    This helper is used exclusively by the superseded private
    ``_chunk_gdr_fwd_neumann`` custom-VJP variant. The public
    ``_chunk_gdr_fwd`` uses the exact triangular-solve formulation.
    """
    output, final_state, _ = _chunk_gdr_fwd_core(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        chunk_size=chunk_size,
        initial_state=initial_state,
        use_qk_l2norm=use_qk_l2norm,
        save_residual=False,
    )
    return output, final_state


def _chunk_gdr_fwd_core(
    query,
    key,
    value,
    beta,
    decay,
    chunk_size,
    initial_state,
    use_qk_l2norm,
    save_residual: bool,
):
    """Shared chunked forward path, optionally capturing backward residuals.

    This is the core Neumann-series-based chunked GDR computation. It processes
    the sequence in chunks of ``chunk_size``, computing intra-chunk attention in
    parallel and propagating state across chunks via lax.scan.

    Args:
        query: [batch, num_heads, seq_len, head_dim].
        key: [batch, num_heads, seq_len, head_dim].
        value: [batch, num_heads, seq_len, d_state].
        beta: [batch, num_heads, seq_len].
        decay: [batch, num_heads, seq_len] or None.
        chunk_size: Size of each chunk.
        initial_state: [batch, num_heads, head_dim, d_state] or None.
        use_qk_l2norm: Whether to L2-normalize queries and keys.
        save_residual: If True, save tensors needed for backward pass.

    Returns:
        Tuple of (output, final_state, residuals_or_none).
    """
    B, H, L, K_dim = query.shape
    V_dim = value.shape[-1]
    input_dtype = query.dtype
    decay_was_none = decay is None
    initial_state_was_none = initial_state is None

    q_inv_norm = None
    k_inv_norm = None
    if use_qk_l2norm:
        query, q_inv_norm = _l2norm_with_inv(query, axis=-1, eps=1e-6)
        key, k_inv_norm = _l2norm_with_inv(key, axis=-1, eps=1e-6)

    if decay is None:
        decay = jnp.zeros((B, H, L), dtype=jnp.float32)
    else:
        decay = decay.astype(jnp.float32)

    pad_size = (chunk_size - L % chunk_size) % chunk_size
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        decay = jnp.pad(decay, ((0, 0), (0, 0), (0, pad_size)))

    total_len = L + pad_size
    num_chunks = total_len // chunk_size

    scale = 1.0 / (K_dim**0.5)
    query = query * scale

    v_beta = value * beta[:, :, :, None]
    k_beta = key * beta[:, :, :, None]

    query = query.reshape(B, H, num_chunks, chunk_size, K_dim)
    key = key.reshape(B, H, num_chunks, chunk_size, K_dim)
    value = value.reshape(B, H, num_chunks, chunk_size, V_dim)
    beta = beta.reshape(B, H, num_chunks, chunk_size)
    k_beta = k_beta.reshape(B, H, num_chunks, chunk_size, K_dim)
    v_beta = v_beta.reshape(B, H, num_chunks, chunk_size, V_dim)
    g = decay.reshape(B, H, num_chunks, chunk_size)

    mask_triu = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)

    g_cumsum = jnp.cumsum(g, axis=-1)

    g_diff = g_cumsum[:, :, :, :, None] - g_cumsum[:, :, :, None, :]
    g_diff = jnp.tril(g_diff)
    decay_mask = jnp.exp(jnp.clip(g_diff, -20.0, 20.0))
    decay_mask = jnp.tril(decay_mask)

    attn = jnp.einsum("bhcik,bhcjk->bhcij", k_beta, key, precision=_MATMUL_PRECISION)
    attn = -(attn * decay_mask).astype(jnp.float32)
    attn = jnp.where(mask_triu, 0.0, attn)

    attn = jnp.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    inv = _strict_lower_inverse(-attn)

    attn = jnp.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0).astype(input_dtype)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0)).astype(input_dtype)
    g_end = g_cumsum[:, :, :, -1]
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0)).astype(input_dtype)
    g_diff_state_exp = jnp.exp(jnp.clip(g_end[:, :, :, None] - g_cumsum, -20.0, 20.0)).astype(input_dtype)

    value_local = jnp.einsum("bhcij,bhcjv->bhciv", attn, v_beta, precision=_MATMUL_PRECISION)
    k_beta_scaled = k_beta * g_cumsum_exp[:, :, :, :, None]
    k_cumdecay = jnp.einsum("bhcij,bhcjk->bhcik", attn, k_beta_scaled, precision=_MATMUL_PRECISION)

    if initial_state is None:
        initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=input_dtype)
    else:
        initial_state = initial_state.astype(input_dtype)

    mask_triu_inner = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

    xs = (
        query.transpose(2, 0, 1, 3, 4),
        key.transpose(2, 0, 1, 3, 4),
        value_local.transpose(2, 0, 1, 3, 4),
        k_cumdecay.transpose(2, 0, 1, 3, 4),
        g_cumsum_exp.transpose(2, 0, 1, 3),
        g_end_exp.transpose(2, 0, 1),
        g_diff_state_exp.transpose(2, 0, 1, 3),
        decay_mask.astype(input_dtype).transpose(2, 0, 1, 3, 4),
    )

    def chunk_step(state, inputs):
        q_i, k_i, v_i, k_cumdecay_i, g_exp_i, g_end_exp_i, g_diff_exp_i, decay_mask_i = inputs

        attn_qk = jnp.einsum("bhik,bhjk->bhij", q_i, k_i, precision=_MATMUL_PRECISION)
        attn_qk = attn_qk * decay_mask_i
        attn_qk = jnp.where(mask_triu_inner, 0.0, attn_qk)

        q_scaled = q_i * g_exp_i[:, :, :, None]
        qk_fused = jnp.stack([k_cumdecay_i, q_scaled], axis=0)
        both = jnp.einsum("nbhik,bhkv->nbhiv", qk_fused, state, precision=_MATMUL_PRECISION)
        v_prime = jnp.nan_to_num(both[0], nan=0.0, posinf=0.0, neginf=0.0)
        attn_inter = jnp.nan_to_num(both[1], nan=0.0, posinf=0.0, neginf=0.0)

        v_new = jnp.nan_to_num(v_i - v_prime, nan=0.0, posinf=0.0, neginf=0.0)
        core_out = jnp.nan_to_num(
            attn_inter + jnp.einsum("bhij,bhjv->bhiv", attn_qk, v_new, precision=_MATMUL_PRECISION),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        state_decayed = state * g_end_exp_i[:, :, None, None]
        k_scaled = k_i * g_diff_exp_i[:, :, :, None]
        state_update = jnp.einsum("bhik,bhiv->bhkv", k_scaled, v_new, precision=_MATMUL_PRECISION)
        new_state = jnp.nan_to_num(state_decayed + state_update, nan=0.0, posinf=0.0, neginf=0.0).astype(state.dtype)

        return new_state, core_out.astype(input_dtype)

    final_state, core_attn_out = lax.scan(chunk_step, initial_state, xs)

    core_attn_out = core_attn_out.transpose(1, 2, 0, 3, 4)
    core_attn_out = core_attn_out.reshape(B, H, -1, V_dim)
    core_attn_out = core_attn_out[:, :, :L, :]

    if not save_residual:
        return core_attn_out, final_state, None

    residual = (
        query,
        key,
        value,
        beta,
        attn,
        decay_mask.astype(input_dtype),
        g_cumsum_exp,
        g_end_exp,
        g_diff_state_exp,
        initial_state,
        q_inv_norm,
        k_inv_norm,
        L,
        pad_size,
        decay_was_none,
        initial_state_was_none,
    )
    return core_attn_out, final_state, residual


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 7))
def _chunk_gdr_fwd_neumann(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Chunked forward pass for gated delta rule with custom backward.

    Forward runs in ``input_dtype`` (bf16) for memory efficiency.
    Backward uses a hand-derived analytical reverse pass in float32,
    avoiding mixed-type issues inside ``shard_map`` backward.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim].
        key: Key tensor [batch, num_heads, seq_len, head_dim].
        value: Value tensor [batch, num_heads, seq_len, d_state].
        beta: Gating tensor [batch, num_heads, seq_len].
        decay: Per-token decay [batch, num_heads, seq_len].
        chunk_size: Size of chunks for parallel processing (non-diff).
        initial_state: Optional initial recurrent state.
        use_qk_l2norm: Whether to apply L2 normalization (non-diff).

    Returns:
        Tuple of (outputs, final_state).

    Note:
        This Neumann-series variant uses a hand-written, SEGMENT-BLIND backward
        (``_xla_impl_bwd._chunk_gdr_bwd``). It therefore does NOT support sequence
        packing — there is intentionally no ``seg_ids`` parameter, and packed
        training must route through ``_chunk_gdr_fwd`` -> ``_recurrent_gdr_fwd``
        (plain autodiff) so segment boundaries are honored in the backward pass.
    """
    return _chunk_gdr_fwd_impl(
        query,
        key,
        value,
        beta,
        decay,
        chunk_size,
        initial_state,
        use_qk_l2norm,
    )


def _chunk_gdr_fwd_rule(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm):
    """Forward rule for the custom_vjp of the Neumann-series chunked GDR.

    Runs the chunked forward pass with ``save_residual=True`` so that the
    tensors required by the hand-derived backward are stored in the VJP
    residual tuple.  This is the ``fwd`` half of the ``custom_vjp`` pair
    registered on the private ``_chunk_gdr_fwd_neumann`` helper. The public
    ``_chunk_gdr_fwd`` below uses the exact triangular-solve formulation.

    Args:
        query: [batch, num_heads, seq_len, head_dim].
        key: [batch, num_heads, seq_len, head_dim].
        value: [batch, num_heads, seq_len, d_state].
        beta: [batch, num_heads, seq_len].
        decay: [batch, num_heads, seq_len] or None.
        chunk_size: Chunk size (non-diff, captured via nondiff_argnums).
        initial_state: [batch, num_heads, head_dim, d_state] or None.
        use_qk_l2norm: Whether to L2-normalize (non-diff).

    Returns:
        Tuple of ((output, final_state), residual_tuple).
    """
    output, final_state, residual = _chunk_gdr_fwd_core(
        query,
        key,
        value,
        beta,
        decay,
        chunk_size,
        initial_state,
        use_qk_l2norm,
        save_residual=True,
    )
    return (output, final_state), residual


def _chunk_gdr_bwd_rule(chunk_size, use_qk_l2norm, res, g):
    """Backward rule for the custom_vjp of the Neumann-series chunked GDR.

    Delegates gradient computation to the hand-derived backward in
    ``_xla_impl_bwd._chunk_gdr_bwd``.  Like ``_chunk_gdr_fwd_rule``, this
    is registered on the private Neumann helper.

    Args:
        chunk_size: Chunk size (non-diff, bound via nondiff_argnums).
        use_qk_l2norm: Whether L2-norm was applied (non-diff).
        res: Residual tuple from ``_chunk_gdr_fwd_rule``.
        g: Upstream gradient tuple (d_output, d_final_state).

    Returns:
        Tuple of gradients (d_query, d_key, d_value, d_beta, d_decay,
        d_initial_state).
    """
    from ._xla_impl_bwd import _chunk_gdr_bwd

    return _chunk_gdr_bwd(chunk_size, use_qk_l2norm, res, g)


_chunk_gdr_fwd_neumann.defvjp(_chunk_gdr_fwd_rule, _chunk_gdr_bwd_rule)


def _chunk_gdr_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
    seg_ids: Int[Array, "batch seq_len"] | None = None,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Exact multi-token chunked GDR forward path (triangular-solve formulation).

    This is the active public entry point for multi-token GDR inference and
    training. It uses the exact formulation because the older Neumann-series
    approximation diverges catastrophically on padded SFT batches.

    Delegates entirely to ``_recurrent_gdr_fwd``, which uses
    ``jax.scipy.linalg.solve_triangular`` for an exact intra-chunk solve and
    relies on standard JAX autodiff for gradient computation.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim].
        key: Key tensor [batch, num_heads, seq_len, head_dim].
        value: Value tensor [batch, num_heads, seq_len, d_state].
        beta: Per-token gating [batch, num_heads, seq_len].
        decay: Per-token log-decay [batch, num_heads, seq_len], or None.
        chunk_size: Number of tokens per chunk (passed to ``_recurrent_gdr_fwd``).
        initial_state: Optional initial recurrent state
            [batch, num_heads, head_dim, d_state].
        use_qk_l2norm: Whether to L2-normalize queries and keys.

    Returns:
        Tuple of (outputs, final_state) — see ``_recurrent_gdr_fwd`` for details.
    """
    # IMPORTANT (sequence packing): this delegates to ``_recurrent_gdr_fwd``, which has NO
    # custom_vjp — JAX autodiff differentiates through its segmented cumsum + same-segment
    # masks, so ``seg_ids`` is honored in the BACKWARD pass for free. Do NOT reroute this to
    # the ``_chunk_gdr_fwd_neumann`` custom_vjp variant when ``seg_ids`` is set: that path's
    # hand-written backward (``_xla_impl_bwd._chunk_gdr_bwd``) is segment-blind and would
    # silently produce wrong gradients across document boundaries.
    return _recurrent_gdr_fwd(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        initial_state=initial_state,
        use_qk_l2norm=use_qk_l2norm,
        chunk_size=chunk_size,
        seg_ids=seg_ids,
    )


def _single_step_gdr_fwd(
    query: Float[Array, "batch num_heads 1 head_dim"],
    key: Float[Array, "batch num_heads 1 head_dim"],
    value: Float[Array, "batch num_heads 1 d_state"],
    beta: Float[Array, "batch num_heads 1"],
    decay: Float[Array, "batch num_heads 1"] | None,
    recurrent_state: Float[Array, "batch num_heads head_dim d_state"],
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads 1 d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Single-step GDR update optimized for autoregressive inference.

    When seq_len=1 and we have an existing state, this function provides
    an optimized path that avoids the overhead of scan/chunk machinery.

    Args:
        query: Single query token [batch, num_heads, 1, head_dim].
        key: Single key token [batch, num_heads, 1, head_dim].
        value: Single value token [batch, num_heads, 1, d_state].
        beta: Gating for this token [batch, num_heads, 1].
        decay: Decay for this token [batch, num_heads, 1] or None.
        recurrent_state: Current memory state [batch, num_heads, head_dim, d_state].
        use_qk_l2norm: Whether to L2-normalize query and key.

    Returns:
        Tuple of (output, new_state).
    """
    if use_qk_l2norm:
        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    query = query.squeeze(2)
    key = key.squeeze(2)
    value = value.squeeze(2)
    beta = beta.squeeze(2)

    head_dim = query.shape[-1]
    scale = 1.0 / (head_dim**0.5)
    query = query * scale

    if decay is not None:
        decay = decay.squeeze(2)
        g_exp = jnp.exp(decay.astype(jnp.float32)).astype(recurrent_state.dtype)
        recurrent_state = recurrent_state * g_exp[:, :, None, None]

    kv_mem = jnp.sum(recurrent_state * key[:, :, :, None], axis=-2)

    beta_scaled = beta[:, :, None]
    delta = (value - kv_mem) * beta_scaled

    new_state = recurrent_state + key[:, :, :, None] * delta[:, :, None, :]

    output = jnp.sum(new_state * query[:, :, :, None], axis=-2)
    output = output[:, :, None, :]
    return output, new_state
