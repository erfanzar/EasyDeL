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

"""Ragged Gated Delta Rule XLA forward implementations.

Provides three core functions for processing variable-length sequences
packed into a flat token stream:

1. ``_ragged_gdr_decode_only``: Sequential scan for decode (all seq_len=1).
2. ``_ragged_gdr_chunked_prefill``: Chunked parallel prefill with
   triangular-solve inversion and inter-chunk state propagation.
3. ``ragged_gated_delta_rule_dispatch``: Main entry point that routes
   to decode or prefill based on sequence lengths.

The chunked prefill algorithm:
    1. Pack ragged sequences into a chunk-aligned stream with padding.
    2. Compute intra-chunk attention via ``solve_triangular((I+S), I)``
       for the exact ``(I + k_beta @ k^T * decay_mask)^{-1}`` inverse.
    3. Pre-compute per-chunk quantities: ``u = A @ v_beta``,
       ``w = A @ (k_beta * exp(g_cumsum))``, ``attn_i``, etc.
    4. Run a lightweight ``lax.scan`` across chunks with 4 matmuls
       per step, resetting state at sequence boundaries via
       ``reset_mask``.
    5. Unpack outputs and scatter final states back to the pool.

Reference:
    vllm tpu-inference ``ragged_gated_delta_rule_chunked.py``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ..gated_delta_rule._xla_impl_fwd import _l2norm

_P = lax.Precision.HIGHEST
_F32 = jnp.float32


def _ragged_gdr_step(query, key, value, g, beta, state):
    """Execute one step of the GDR recurrence for a single token.

    Implements the gated delta rule update:
        k_state = k @ state
        v_diff = v - exp(g) * k_state
        v_new = beta * v_diff
        out = exp(g) * (q @ state) + (q . k) * v_new
        state_new = state * exp(g) + k outer v_new

    All computation is done in the input dtype (no internal upcast).

    Args:
        query: Query vector of shape ``(H, d_k)``.
        key: Key vector of shape ``(H, d_k)``.
        value: Value vector of shape ``(H, d_v)``.
        g: Log-space decay of shape ``(H,)``.
        beta: Gating coefficient of shape ``(H,)``.
        state: Recurrent state of shape ``(H, d_k, d_v)``.

    Returns:
        A tuple ``(output, new_state)`` with shapes
        ``(H, d_v)`` and ``(H, d_k, d_v)`` respectively.
    """
    d_k = query.shape[-1]
    scale = d_k**-0.5
    q = query * scale

    k_state = jnp.einsum("hd,hdm->hm", key, state)
    g_exp = jnp.exp(g)
    v_diff = value - g_exp[:, None] * k_state
    v_new = beta[:, None] * v_diff

    q_state = jnp.einsum("hd,hdm->hm", q, state)
    q_k = jnp.sum(q * key, axis=-1, keepdims=True)
    out = g_exp[:, None] * q_state + q_k * v_new

    k_v_new = jnp.einsum("hd,hm->hdm", key, v_new)
    new_state = state * g_exp[:, None, None] + k_v_new

    return out, new_state


def _ragged_gdr_decode_only(
    query,
    key,
    value,
    beta,
    decay,
    recurrent_state,
    query_start_loc,
    state_indices,
    use_qk_l2norm,
):
    """Decode-only forward path for ragged GDR (all sequences have length 1).

    Uses ``lax.scan`` to iterate over tokens sequentially. Each token's
    request index is determined from ``query_start_loc``, and the
    corresponding state is gathered from ``recurrent_state`` via
    ``state_indices``. After the single-step GDR update, the new state
    is scattered back into the pool.

    Args:
        query: Flat queries of shape ``(num_tokens, H, d_k)``.
        key: Flat keys of shape ``(num_tokens, H, d_k)``.
        value: Flat values of shape ``(num_tokens, H, d_v)``.
        beta: Gating of shape ``(num_tokens, H)``.
        decay: Log-space decay of shape ``(num_tokens, H)``.
        recurrent_state: State pool of shape ``(num_slots, H, d_k, d_v)``.
        query_start_loc: Cumulative offsets of shape ``(num_requests + 1,)``.
        state_indices: Request-to-slot mapping of shape ``(num_requests,)``.
        use_qk_l2norm: Whether to L2-normalize queries and keys.

    Returns:
        A tuple ``(updated_recurrent_state, output)`` with shapes
        ``(num_slots, H, d_k, d_v)`` and ``(num_tokens, H, d_v)``.
    """
    num_tokens = query.shape[0]
    max_reqs = recurrent_state.shape[0]

    if use_qk_l2norm:
        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    token_idx = jnp.arange(num_tokens)
    req_indices = jnp.sum(token_idx[:, None] >= query_start_loc[None, :], axis=1) - 1
    req_indices = jnp.clip(req_indices, 0, max_reqs - 1)
    valid_mask = token_idx < query_start_loc[-1]

    def scan_fn(carry, xs):
        recurrent_state_all = carry
        curr_q, curr_k, curr_v, curr_beta, curr_g, req_idx, is_valid = xs

        state_idx = state_indices[req_idx]
        state = recurrent_state_all[state_idx]

        output, new_state = _ragged_gdr_step(
            curr_q.astype(_F32),
            curr_k.astype(_F32),
            curr_v.astype(_F32),
            curr_g.astype(_F32),
            curr_beta.astype(_F32),
            state.astype(_F32),
        )

        recurrent_state_all = jnp.where(
            is_valid,
            recurrent_state_all.at[state_idx].set(new_state.astype(recurrent_state_all.dtype)),
            recurrent_state_all,
        )

        return recurrent_state_all, output.astype(query.dtype)

    new_recurrent_state, output = lax.scan(
        scan_fn,
        recurrent_state,
        (query, key, value, beta, decay, req_indices, valid_mask),
    )

    return new_recurrent_state, output


def pack_inputs_single_stream(query, key, value, g, beta, query_start_loc, chunk_size):
    """Pack ragged sequences into a chunk-aligned contiguous stream.

    Each sequence is padded to the next multiple of ``chunk_size``, then
    all padded sequences are concatenated into a single flat stream. A
    ``reset_mask`` is produced to indicate which chunks start a new
    sequence (used to reset recurrent state during the inter-chunk scan).

    Example with ``chunk_size=4``::

        Original:  [A, A, A] [B, B, B, B, B] [C, C]
        Packed:    [A, A, A, 0] [B, B, B, B] [B, 0, 0, 0] [C, C, 0, 0]
        Chunks:     Chunk 0      Chunk 1      Chunk 2       Chunk 3
        reset:     [True,        True,         False,        True]

    Args:
        query: Flat queries of shape ``(num_tokens, H, d_k)``.
        key: Flat keys of shape ``(num_tokens, H, d_k)``.
        value: Flat values of shape ``(num_tokens, H, d_v)``.
        g: Flat decay values of shape ``(num_tokens, H)``.
        beta: Flat gating values of shape ``(num_tokens, H)``.
        query_start_loc: Cumulative token offsets of shape
            ``(num_seqs + 1,)``.
        chunk_size: Target chunk size for padding alignment.

    Returns:
        An 8-tuple of:

        - ``packed_q``: Packed queries ``(max_packed_tokens, H, d_k)``.
        - ``packed_k``: Packed keys ``(max_packed_tokens, H, d_k)``.
        - ``packed_v``: Packed values ``(max_packed_tokens, H, d_v)``.
        - ``packed_g``: Packed decay ``(max_packed_tokens, H)``.
        - ``packed_beta``: Packed gating ``(max_packed_tokens, H)``.
        - ``reset_mask``: Boolean mask ``(num_packed_chunks,)`` —
          ``True`` at the first chunk of each sequence.
        - ``new_query_start_loc``: Offsets in the packed stream
          ``(num_seqs + 1,)``.
        - ``padded_indices_valid``: Mapping from original token
          positions to packed positions ``(num_tokens,)``.
    """
    num_tokens = query.shape[0]
    num_seqs = query_start_loc.shape[0] - 1

    seq_lengths = query_start_loc[1:] - query_start_loc[:-1]
    num_chunks = (seq_lengths + chunk_size - 1) // chunk_size
    padded_lengths = num_chunks * chunk_size

    new_query_start_loc = jnp.cumsum(jnp.concatenate([jnp.array([0]), padded_lengths]))

    seq_id = jnp.searchsorted(query_start_loc, jnp.arange(num_tokens), side="right") - 1
    original_start = query_start_loc[seq_id]
    new_start = new_query_start_loc[seq_id]
    padded_indices_valid = new_start + (jnp.arange(num_tokens) - original_start)

    max_packed_tokens = num_tokens + num_seqs * chunk_size
    max_packed_tokens = ((max_packed_tokens + chunk_size - 1) // chunk_size) * chunk_size

    def _pack(data, fill=0.0):
        out_shape = (max_packed_tokens, *data.shape[1:])
        packed = jnp.full(out_shape, fill, dtype=data.dtype)
        packed = packed.at[padded_indices_valid].set(data)
        return packed

    packed_q = _pack(query)
    packed_k = _pack(key)
    packed_v = _pack(value)
    packed_g = _pack(g)
    packed_beta = _pack(beta)

    num_chunks_total = max_packed_tokens // chunk_size
    reset_mask = jnp.zeros((num_chunks_total,), dtype=jnp.bool_)
    start_chunk_indices = new_query_start_loc[:-1] // chunk_size
    reset_mask = reset_mask.at[start_chunk_indices].set(True)

    return (
        packed_q,
        packed_k,
        packed_v,
        packed_g,
        packed_beta,
        reset_mask,
        new_query_start_loc,
        padded_indices_valid,
    )


def _ragged_gdr_chunked_prefill(
    query,
    key,
    value,
    beta,
    decay,
    recurrent_state,
    query_start_loc,
    state_indices,
    chunk_size,
    use_qk_l2norm,
    compute_dtype=jnp.bfloat16,
):
    """Chunked prefill forward path for ragged GDR with mixed-length sequences.

    Processes all sequences in a packed flat stream using three stages:

    **Stage 1 — Packing**: Calls ``pack_inputs_single_stream`` to pad
    each sequence to a multiple of ``chunk_size`` and concatenate into a
    single stream with ``reset_mask`` for state resets.

    **Stage 2 — Intra-chunk precomputation**: For each chunk, computes
    the delta-rule inverse ``A = (I + S)^{-1}`` via
    ``jax.scipy.linalg.solve_triangular`` (exact, no approximation),
    then derives per-chunk quantities:

    - ``u_chunks = A @ v_beta`` — local value contribution.
    - ``w_chunks = A @ (k_beta * exp(g_cumsum))`` — state read term.
    - ``attn_i`` — intra-chunk attention scores with decay.
    - ``q_g``, ``k_g_diff``, ``g_end_exp`` — scaled q/k and decay.

    **Stage 3 — Inter-chunk scan**: A lightweight ``lax.scan`` with
    4 matmuls per step propagates state across chunks. At sequence
    boundaries (``reset_mask[i] == True``), the state is reset to the
    initial state from the global pool.

    Args:
        query: Flat queries of shape ``(num_tokens, H, d_k)``.
        key: Flat keys of shape ``(num_tokens, H, d_k)``.
        value: Flat values of shape ``(num_tokens, H, d_v)``.
        beta: Gating of shape ``(num_tokens, H)``.
        decay: Log-space decay of shape ``(num_tokens, H)``.
        recurrent_state: State pool of shape ``(num_slots, H, d_k, d_v)``.
        query_start_loc: Cumulative offsets ``(num_seqs + 1,)``.
        state_indices: Request-to-slot mapping ``(num_seqs,)``.
        chunk_size: Tokens per chunk for the blocked computation.
        use_qk_l2norm: Whether to L2-normalize queries and keys.
        compute_dtype: Dtype for matmul operands (default: bfloat16).

    Returns:
        A tuple ``(updated_recurrent_state, output)`` with shapes
        ``(num_slots, H, d_k, d_v)`` and ``(num_tokens, H, d_v)``.
    """
    initial_dtype = query.dtype

    if use_qk_l2norm:
        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    (
        packed_q,
        packed_k,
        packed_v,
        packed_g,
        packed_beta,
        reset_mask,
        new_query_start_loc,
        padded_indices_valid,
    ) = pack_inputs_single_stream(query, key, value, decay, beta, query_start_loc, chunk_size)

    packed_g = packed_g.astype(_F32)
    scale = lax.rsqrt(jnp.array(packed_q.shape[-1], dtype=_F32)).astype(compute_dtype)
    packed_q = packed_q.astype(compute_dtype) * scale
    packed_k = packed_k.astype(compute_dtype)
    packed_v = packed_v.astype(compute_dtype)
    packed_beta = packed_beta.astype(compute_dtype)

    total_tokens = packed_q.shape[0]
    num_chunks = total_tokens // chunk_size
    H = packed_q.shape[1]
    K_dim = packed_q.shape[2]
    V_dim = packed_v.shape[2]
    C = chunk_size

    def to_chunk(x):
        return x.reshape(num_chunks, C, H, -1).transpose(0, 2, 1, 3)

    def to_chunk_scalar(x):
        return x.reshape(num_chunks, C, H).transpose(0, 2, 1)

    q_c = to_chunk(packed_q)
    k_c = to_chunk(packed_k)
    v_c = to_chunk(packed_v)
    g_c = to_chunk_scalar(packed_g)
    beta_c = to_chunk_scalar(packed_beta)

    g_cumsum = jnp.cumsum(g_c, axis=-1)
    k_beta = k_c * beta_c[..., None]

    S = jnp.matmul(k_beta, k_c.swapaxes(-1, -2), precision=_P, preferred_element_type=_F32).astype(_F32)

    g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
    strict_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_), k=-1)
    g_diff = jnp.where(strict_lower, g_diff, -1e30)
    S = jnp.where(strict_lower, S * jnp.exp(g_diff), 0.0)

    eye = jnp.eye(C, dtype=_F32)
    eye_bc = jnp.broadcast_to(eye, S.shape)
    A = jax.scipy.linalg.solve_triangular(eye + S, eye_bc, lower=True, unit_diagonal=True)

    v_beta = v_c * beta_c[..., None]
    u_chunks = jnp.matmul(A, v_beta.astype(_F32), precision=_P, preferred_element_type=_F32).astype(compute_dtype)

    k_beta_g = k_beta.astype(_F32) * jnp.exp(g_cumsum)[..., None]
    w_chunks = jnp.matmul(A, k_beta_g, precision=_P, preferred_element_type=_F32).astype(compute_dtype)

    attn_qk = jnp.matmul(q_c, k_c.swapaxes(-1, -2), precision=_P, preferred_element_type=_F32).astype(_F32)
    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    g_diff_intra = g_cumsum[..., :, None] - g_cumsum[..., None, :]
    g_diff_intra = jnp.where(lower_mask, g_diff_intra, -1e30)
    attn_i = jnp.where(lower_mask, attn_qk * jnp.exp(g_diff_intra), 0.0).astype(compute_dtype)

    q_g = (q_c.astype(_F32) * jnp.exp(g_cumsum)[..., None]).astype(compute_dtype)
    g_end_exp = jnp.exp(g_cumsum[..., -1, None, None])
    g_diff_state = jnp.exp(g_cumsum[..., -1, None] - g_cumsum)[..., None]
    k_g_diff = (k_c.astype(_F32) * g_diff_state).astype(compute_dtype)

    init_h_per_chunk = jnp.zeros((num_chunks, H, K_dim, V_dim), dtype=recurrent_state.dtype)
    start_chunk_indices = new_query_start_loc[:-1] // C
    init_states_for_seqs = recurrent_state[state_indices]
    init_h_per_chunk = init_h_per_chunk.at[start_chunk_indices].set(init_states_for_seqs)

    h_init = jnp.zeros((H, K_dim, V_dim), dtype=_F32)

    xs = (w_chunks, u_chunks, q_g, attn_i, g_end_exp, k_g_diff, reset_mask, init_h_per_chunk)

    def scan_body(h, args):
        w, u, q_scaled, attn_intra, g_last_exp, k_scaled, reset, init_h = args

        h = jnp.where(reset, init_h.astype(_F32), h)

        attn_inter = jnp.matmul(q_scaled, h, precision=_P, preferred_element_type=_F32)
        v_prime = jnp.matmul(w.astype(_F32), h, precision=_P, preferred_element_type=_F32)
        v_new = u.astype(_F32) - v_prime
        term2 = jnp.matmul(attn_intra, v_new, precision=_P, preferred_element_type=_F32)
        o_c = attn_inter + term2

        h_new = h * g_last_exp
        update = jnp.matmul(k_scaled.swapaxes(-1, -2), v_new, precision=_P, preferred_element_type=_F32)
        h_new = h_new + update

        return h_new, (o_c, h_new)

    _, (o_chunks, h_chunks) = lax.scan(scan_body, h_init, xs)

    output = o_chunks.transpose(0, 2, 1, 3).reshape(-1, H, V_dim)
    output = output.astype(initial_dtype)

    packed_output = output[padded_indices_valid]

    last_chunk_indices = (new_query_start_loc[1:] // C) - 1
    final_states = h_chunks[last_chunk_indices]
    updated_recurrent_state = recurrent_state.at[state_indices].set(final_states.astype(recurrent_state.dtype))

    return updated_recurrent_state, packed_output


def ragged_gated_delta_rule_dispatch(
    query,
    key,
    value,
    beta,
    decay,
    recurrent_state,
    query_start_loc,
    state_indices,
    *,
    chunk_size=64,
    use_qk_l2norm=True,
):
    """Dispatch ragged GDR to decode-only or chunked prefill path.

    Determines the execution path based on sequence lengths derived
    from ``query_start_loc``. If all sequences have length <= 1
    (pure decode), routes to the sequential scan path. Otherwise,
    routes to the chunked prefill path.

    The dispatch uses ``lax.cond`` so both branches are traced but
    only one executes at runtime.

    Args:
        query: Flat queries of shape ``(num_tokens, H, d_k)``.
        key: Flat keys of shape ``(num_tokens, H, d_k)``.
        value: Flat values of shape ``(num_tokens, H, d_v)``.
        beta: Gating of shape ``(num_tokens, H)``.
        decay: Log-space decay of shape ``(num_tokens, H)`` or ``None``.
        recurrent_state: State pool of shape ``(num_slots, H, d_k, d_v)``.
        query_start_loc: Cumulative offsets ``(num_requests + 1,)``.
        state_indices: Request-to-slot mapping ``(num_requests,)``.
        chunk_size: Chunk size for prefill path.
        use_qk_l2norm: Whether to L2-normalize queries and keys.

    Returns:
        A tuple ``(output, updated_recurrent_state)``.
    """
    if decay is None:
        decay = jnp.zeros_like(beta)

    seq_lengths = query_start_loc[1:] - query_start_loc[:-1]
    is_all_decode = jnp.all(seq_lengths <= 1)

    def decode_fn(_):
        new_state, out = _ragged_gdr_decode_only(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            query_start_loc,
            state_indices,
            use_qk_l2norm,
        )
        return out, new_state

    def prefill_fn(_):
        new_state, out = _ragged_gdr_chunked_prefill(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            query_start_loc,
            state_indices,
            chunk_size,
            use_qk_l2norm,
        )
        return out, new_state

    output, updated_state = lax.cond(is_all_decode, decode_fn, prefill_fn, operand=None)

    return output, updated_state
