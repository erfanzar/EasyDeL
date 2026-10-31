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

"""Forward pass implementations for Kernel Delta Attention (KDA).

This module provides three forward pass variants for KDA:

1. Recurrent (_recurrent_kda_fwd):
   Pure sequential scan with O(L) time complexity. Best for very long
   sequences or memory-constrained inference.

2. Chunked (_chunk_kda_fwd):
   Hybrid approach with parallel intra-chunk computation and sequential
   inter-chunk state propagation. Best for training with moderate sequences.

3. Single-step (_single_step_kda_fwd):
   Optimized path for seq_len=1 during autoregressive inference.

The KDA update rule:
    h_t = exp(decay_t) * h_{t-1} + k_t ⊗ (beta_t * (v_t - h_{t-1} @ k_t))
    o_t = h_t @ q_t

Where h_t is the [head_dim, value_dim] memory matrix that stores key-value
associations and supports efficient retrieval via query projection.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

_MATMUL_PRECISION = lax.Precision.HIGHEST


def _l2norm(x: Float[Array, "..."], axis: int = -1, eps: float = 1e-6) -> Float[Array, "..."]:
    """Apply L2 normalization along specified axis.

    Args:
        x: Input tensor to normalize
        axis: Axis along which to normalize (default: -1)
        eps: Small constant for numerical stability

    Returns:
        L2-normalized tensor with same shape as input
    """
    inv_norm = lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _recurrent_kda_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len value_dim"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    softmax_scale: float,
    initial_state: Float[Array, "batch num_heads head_dim value_dim"] | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads seq_len value_dim"],
    Float[Array, "batch num_heads head_dim value_dim"],
]:
    """Recurrent (scan) forward pass for Kernel Delta Attention.

    Uses JAX's lax.scan for sequential processing with O(L) time complexity.
    This is the most memory-efficient variant but may be slower than chunked
    for moderate sequence lengths due to lack of parallelism.

    The per-step update:
        state = state * exp(decay_t)           # Apply decay
        kv_mem = state @ k_t                   # Retrieve current memory
        delta = beta_t * (v_t - kv_mem)        # Compute what's new
        state = state + outer(k_t, delta)      # Update memory
        output_t = state @ q_t                 # Query the memory

    Args:
        query: Query tensor (transposed to [batch, heads, seq, dim])
            Shape: [batch, num_heads, seq_len, head_dim]
        key: Key tensor for memory addressing
            Shape: [batch, num_heads, seq_len, head_dim]
        value: Value tensor to store
            Shape: [batch, num_heads, seq_len, value_dim]
        beta: Learning rate for delta updates
            Shape: [batch, num_heads, seq_len]
        decay: Memory decay factor (should be <= 0, or None for no decay)
            Shape: [batch, num_heads, seq_len]
        softmax_scale: Scaling factor applied to queries
        initial_state: Optional starting memory state
            Shape: [batch, num_heads, head_dim, value_dim]
        use_qk_l2norm: Whether to L2-normalize queries and keys

    Returns:
        Tuple of:
            - outputs: Attention outputs [batch, num_heads, seq_len, value_dim]
            - final_state: Final memory state [batch, num_heads, head_dim, value_dim]
    """
    batch, num_heads, seq_len, head_dim = query.shape
    value_dim = value.shape[-1]

    if use_qk_l2norm:
        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    query = (query * softmax_scale).astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    decay = jnp.zeros((batch, num_heads, seq_len), dtype=jnp.float32) if decay is None else decay.astype(jnp.float32)

    if initial_state is None:
        initial_state = jnp.zeros((batch, num_heads, head_dim, value_dim), dtype=jnp.float32)
    else:
        initial_state = initial_state.astype(jnp.float32)

    q_seq = query.transpose(2, 0, 1, 3)
    k_seq = key.transpose(2, 0, 1, 3)
    v_seq = value.transpose(2, 0, 1, 3)
    g_seq = decay.transpose(2, 0, 1)
    b_seq = beta.transpose(2, 0, 1)

    def step_fn(state, inputs):
        """Execute a single recurrent KDA step.

        Applies decay, computes delta update, modifies state, and
        produces output for one timestep.

        Args:
            state: Current memory state [batch, num_heads, head_dim, value_dim].
            inputs: Tuple of (q_t, k_t, v_t, g_t, beta_t) for this timestep.

        Returns:
            Tuple of (new_state, output) for this timestep.
        """
        q_t, k_t, v_t, g_t, beta_t = inputs
        g_exp = jnp.exp(g_t)[:, :, None, None]
        beta_scaled = beta_t[:, :, None]

        state = state * g_exp
        kv_mem = jnp.sum(state * k_t[:, :, :, None], axis=-2)
        delta = (v_t - kv_mem) * beta_scaled
        state = state + k_t[:, :, :, None] * delta[:, :, None, :]
        output = jnp.sum(state * q_t[:, :, :, None], axis=-2)
        return state, output

    final_state, outputs = lax.scan(step_fn, initial_state, (q_seq, k_seq, v_seq, g_seq, b_seq))
    outputs = outputs.transpose(1, 2, 0, 3)
    return outputs, final_state


def _chunk_kda_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len value_dim"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    softmax_scale: float,
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads head_dim value_dim"] | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads seq_len value_dim"],
    Float[Array, "batch num_heads head_dim value_dim"],
]:
    """Chunked forward pass for KDA with parallel intra-chunk computation.

    This hybrid approach divides the sequence into chunks and:
    1. Computes intra-chunk attention in parallel (similar to standard attention)
    2. Propagates state across chunks sequentially via lax.scan

    This offers better throughput than pure recurrent for training workloads
    while maintaining O(L) memory complexity for the recurrent state.

    Algorithm Overview:
        1. Pad sequence to multiple of chunk_size
        2. Reshape into [batch, heads, num_chunks, chunk_size, dim]
        3. For each chunk:
           - Compute local attention within chunk (parallel)
           - Update chunk output with inter-chunk state contribution
           - Update state for next chunk
        4. Concatenate chunk outputs and trim padding

    The intra-chunk computation uses a lower-triangular attention pattern
    with cumulative decay weighting, resolved via iterative refinement.

    Args:
        query: Query tensor
            Shape: [batch, num_heads, seq_len, head_dim]
        key: Key tensor
            Shape: [batch, num_heads, seq_len, head_dim]
        value: Value tensor
            Shape: [batch, num_heads, seq_len, value_dim]
        beta: Learning rate for delta updates
            Shape: [batch, num_heads, seq_len]
        decay: Memory decay factor (should be <= 0, or None for no decay)
            Shape: [batch, num_heads, seq_len]
        softmax_scale: Scaling factor applied to queries
        chunk_size: Size of each chunk for parallel computation (default: 64)
        initial_state: Optional starting memory state
            Shape: [batch, num_heads, head_dim, value_dim]
        use_qk_l2norm: Whether to L2-normalize queries and keys

    Returns:
        Tuple of:
            - outputs: Attention outputs [batch, num_heads, seq_len, value_dim]
            - final_state: Final memory state [batch, num_heads, head_dim, value_dim]

    Note:
        Chunk size should be chosen based on hardware. Larger chunks increase
        parallelism but also increase memory for intra-chunk attention matrices.
        64-128 is typically a good default for modern accelerators.
    """
    batch, num_heads, seq_len, head_dim = query.shape
    value_dim = value.shape[-1]

    if use_qk_l2norm:
        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    if decay is None:
        decay = jnp.zeros((batch, num_heads, seq_len), dtype=jnp.float32)
    else:
        decay = decay.astype(jnp.float32)

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        decay = jnp.pad(decay, ((0, 0), (0, 0), (0, pad_size)))

    total_len = seq_len + pad_size
    num_chunks = total_len // chunk_size

    query = query * softmax_scale

    v_beta = value * beta[:, :, :, None]
    k_beta = key * beta[:, :, :, None]

    query = query.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    key = key.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    value = value.reshape(batch, num_heads, num_chunks, chunk_size, value_dim)
    k_beta = k_beta.reshape(batch, num_heads, num_chunks, chunk_size, head_dim)
    v_beta = v_beta.reshape(batch, num_heads, num_chunks, chunk_size, value_dim)
    g = decay.reshape(batch, num_heads, num_chunks, chunk_size)

    mask_triu = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)

    g_cumsum = jnp.cumsum(g, axis=-1)

    g_diff = g_cumsum[:, :, :, :, None] - g_cumsum[:, :, :, None, :]
    decay_mask = jnp.tril(jnp.exp(jnp.tril(g_diff)))

    attn = jnp.einsum("bhcik,bhcjk->bhcij", k_beta, key, precision=_MATMUL_PRECISION)
    attn = -(attn * decay_mask)
    attn = jnp.where(mask_triu, 0.0, attn)

    def resolve_intra_chunk_row(attn_chunk, i):
        """Resolve delta dependencies for a single row of the intra-chunk attention matrix.

        Uses iterative refinement to account for cascading delta updates
        within the chunk. Row i's dependencies on rows 0..i-1 are resolved
        by accumulating their contributions.

        Args:
            attn_chunk: Current attention matrix [chunk_size, chunk_size].
            i: Row index being resolved.

        Returns:
            Tuple of (updated_attn_chunk, None) with row i resolved.
        """
        row = attn_chunk[i, :]
        idx = jnp.arange(chunk_size)
        mask_lt_i = idx < i

        contribution = jnp.sum(
            row[:, None] * attn_chunk * mask_lt_i[:, None] * mask_lt_i[None, :],
            axis=0,
        )
        new_row = jnp.where(mask_lt_i, row + contribution, row)
        return attn_chunk.at[i].set(new_row), None

    def resolve_single_chunk(attn_single):
        """Resolve all intra-chunk delta dependencies for one attention matrix.

        Iterates through all rows sequentially, resolving cascading
        dependencies to produce the final effective attention coefficients.

        Args:
            attn_single: Intra-chunk attention matrix [chunk_size, chunk_size].

        Returns:
            Resolved attention matrix [chunk_size, chunk_size] with all
            delta dependencies accounted for.
        """
        resolved, _ = lax.scan(resolve_intra_chunk_row, attn_single, jnp.arange(chunk_size))
        return resolved

    attn_flat = attn.reshape(-1, chunk_size, chunk_size)
    attn_resolved = jax.vmap(resolve_single_chunk)(attn_flat)
    attn = attn_resolved.reshape(batch, num_heads, num_chunks, chunk_size, chunk_size)

    attn = attn + jnp.eye(chunk_size, dtype=attn.dtype)

    value_local = jnp.einsum("bhcij,bhcjv->bhciv", attn, v_beta, precision=_MATMUL_PRECISION)
    k_beta_scaled = k_beta * jnp.exp(g_cumsum)[:, :, :, :, None]
    k_cumdecay = jnp.einsum("bhcij,bhcjk->bhcik", attn, k_beta_scaled, precision=_MATMUL_PRECISION)

    if initial_state is None:
        initial_state = jnp.zeros((batch, num_heads, head_dim, value_dim), dtype=jnp.float32)
    else:
        initial_state = initial_state.astype(jnp.float32)

    mask_triu_inner = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

    xs = {
        "query": query.transpose(2, 0, 1, 3, 4),
        "key": key.transpose(2, 0, 1, 3, 4),
        "value": value_local.transpose(2, 0, 1, 3, 4),
        "k_cumdecay": k_cumdecay.transpose(2, 0, 1, 3, 4),
        "g_cumsum": g_cumsum.transpose(2, 0, 1, 3),
        "decay_mask": decay_mask.transpose(2, 0, 1, 3, 4),
    }

    def chunk_step(state, inputs):
        """Process a single chunk in the chunked KDA algorithm.

        Combines intra-chunk parallel computation with inter-chunk
        state propagation. For each chunk:
        1. Computes the intra-chunk QK attention with decay weighting
        2. Subtracts the inter-chunk state contribution from values
        3. Combines local and cross-chunk attention outputs
        4. Updates the recurrent state for the next chunk

        Args:
            state: Current memory state [batch, num_heads, head_dim, value_dim].
            inputs: Dictionary with keys 'query', 'key', 'value',
                'k_cumdecay', 'g_cumsum', 'decay_mask' containing the
                chunk's data tensors.

        Returns:
            Tuple of (new_state, chunk_output) where chunk_output has
            shape [batch, num_heads, chunk_size, value_dim].
        """
        q_i = inputs["query"]
        k_i = inputs["key"]
        v_i = inputs["value"]
        k_cumdecay_i = inputs["k_cumdecay"]
        g_cumsum_i = inputs["g_cumsum"]
        decay_mask_i = inputs["decay_mask"]

        attn_qk = jnp.einsum("bhik,bhjk->bhij", q_i, k_i, precision=_MATMUL_PRECISION)
        attn_qk = jnp.where(mask_triu_inner, 0.0, attn_qk * decay_mask_i)

        v_prime = jnp.einsum("bhik,bhkv->bhiv", k_cumdecay_i, state, precision=_MATMUL_PRECISION)
        v_new = v_i - v_prime

        q_scaled = q_i * jnp.exp(g_cumsum_i)[:, :, :, None]
        attn_inter = jnp.einsum("bhik,bhkv->bhiv", q_scaled, state, precision=_MATMUL_PRECISION)

        core_out = attn_inter + jnp.einsum("bhij,bhjv->bhiv", attn_qk, v_new, precision=_MATMUL_PRECISION)

        g_end = g_cumsum_i[:, :, -1]
        state_decayed = state * jnp.exp(g_end)[:, :, None, None]

        g_diff_state = g_end[:, :, None] - g_cumsum_i
        k_scaled = k_i * jnp.exp(g_diff_state)[:, :, :, None]
        state_update = jnp.einsum("bhik,bhiv->bhkv", k_scaled, v_new, precision=_MATMUL_PRECISION)
        new_state = state_decayed + state_update

        return new_state, core_out

    final_state, core_attn_out = lax.scan(chunk_step, initial_state, xs)
    core_attn_out = core_attn_out.transpose(1, 2, 0, 3, 4)
    core_attn_out = core_attn_out.reshape(batch, num_heads, -1, value_dim)[:, :, :seq_len, :]
    return core_attn_out, final_state


def _single_step_kda_fwd(
    query: Float[Array, "batch num_heads 1 head_dim"],
    key: Float[Array, "batch num_heads 1 head_dim"],
    value: Float[Array, "batch num_heads 1 value_dim"],
    beta: Float[Array, "batch num_heads 1"],
    decay: Float[Array, "batch num_heads 1"] | None,
    softmax_scale: float,
    recurrent_state: Float[Array, "batch num_heads head_dim value_dim"],
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads 1 value_dim"],
    Float[Array, "batch num_heads head_dim value_dim"],
]:
    """Single-step KDA update optimized for autoregressive inference.

    When seq_len=1 and we have an existing state, this function provides
    an optimized path that avoids the overhead of scan/chunk machinery.
    This is the typical case during token-by-token generation.

    The update performs:
        1. Apply decay to existing state: state *= exp(decay)
        2. Retrieve current memory: kv_mem = state @ k
        3. Compute delta: delta = beta * (v - kv_mem)
        4. Update state: state += outer(k, delta)
        5. Query output: output = state @ q

    Args:
        query: Single query token
            Shape: [batch, num_heads, 1, head_dim]
        key: Single key token
            Shape: [batch, num_heads, 1, head_dim]
        value: Single value token
            Shape: [batch, num_heads, 1, value_dim]
        beta: Learning rate for this token
            Shape: [batch, num_heads, 1]
        decay: Decay factor for this token (should be <= 0, or None)
            Shape: [batch, num_heads, 1]
        softmax_scale: Scaling factor applied to query
        recurrent_state: Current memory state from previous tokens
            Shape: [batch, num_heads, head_dim, value_dim]
        use_qk_l2norm: Whether to L2-normalize query and key

    Returns:
        Tuple of:
            - output: Attention output for this token
                Shape: [batch, num_heads, 1, value_dim]
            - new_state: Updated memory state for next token
                Shape: [batch, num_heads, head_dim, value_dim]
    """
    if use_qk_l2norm:
        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    query = query.squeeze(2).astype(jnp.float32)
    key = key.squeeze(2).astype(jnp.float32)
    value = value.squeeze(2).astype(jnp.float32)
    beta = beta.squeeze(2).astype(jnp.float32)
    recurrent_state = recurrent_state.astype(jnp.float32)

    query = query * softmax_scale

    if decay is not None:
        g_exp = jnp.exp(decay.squeeze(2).astype(jnp.float32))[:, :, None, None]
        recurrent_state = recurrent_state * g_exp

    kv_mem = jnp.sum(recurrent_state * key[:, :, :, None], axis=-2)
    delta = (value - kv_mem) * beta[:, :, None]
    new_state = recurrent_state + key[:, :, :, None] * delta[:, :, None, :]

    output = jnp.sum(new_state * query[:, :, :, None], axis=-2)[:, :, None, :]
    return output, new_state
