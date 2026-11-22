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

"""Decode Attention forward pass implementation using XLA/JAX.

This module provides the forward pass for decode-phase attention, optimized
for single-token generation with cached key-value tensors.

Key Components:
    - _decode_attention_fwd: Main forward function for decode phase

Algorithm:
    Decode attention for autoregressive generation:
    1. Query is a single token per sequence (batch dim represents sequences)
    2. KV buffer contains cached keys/values from previous tokens
    3. req_to_tokens maps each request to its token indices in the buffer
    4. Compute attention over the cached sequence up to seq_lens

Features:
    - Single-token query optimization (seq_len_q = 1)
    - Page-based KV buffer with req_to_tokens indirection
    - Variable sequence lengths per request
    - Optional logit soft capping for numerical stability
    - GQA/MQA support with head broadcasting

Memory Layout:
    - query: [batch, num_q_heads, head_dim] (single token per sequence)
    - key_buffer/value_buffer: [total_tokens, num_kv_heads, head_dim]
      (internally reshaped to [total_tokens // page_size, page_size, ...])
    - req_to_tokens: [batch, max_pages] indirect page addressing

Note:
    This kernel is optimized for the decode phase where each sequence
    generates one token at a time. For prefill (multiple tokens), use
    the chunked_prefill_paged_decode kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.callib import ejit


@ejit(static_argnames=("softmax_scale", "page_size", "logits_soft_cap"))
def _decode_attention_fwd(
    *,
    query: jax.Array,
    key_buffer: jax.Array,
    value_buffer: jax.Array,
    req_to_tokens: jax.Array,
    seq_lens: jax.Array,
    softmax_scale: float,
    page_size: int,
    logits_soft_cap: float | None,
) -> tuple[jax.Array, jax.Array]:
    """XLA forward pass for paged decode attention.

    Uses JAX gather operations to collect KV pages per sequence, then computes
    attention using einsum and standard softmax operations. All operations are
    hardware-agnostic and compatible with any XLA backend.

    Args:
        query: Query vectors [batch, num_q_heads, head_dim]
        key_buffer: Paged key buffer [total_tokens, num_kv_heads, head_dim]
        value_buffer: Paged value buffer [total_tokens, num_kv_heads, head_dim]
        req_to_tokens: Page table mapping [batch, max_pages]
        seq_lens: Context length per sequence [batch]
        softmax_scale: Attention scaling factor
        page_size: Memory page size in tokens
        logits_soft_cap: Optional logit capping value

    Returns:
        Tuple of (attention_output, lse):
            - attention_output: [batch, num_q_heads, head_dim]
            - lse: Log-sum-exp values [batch, num_q_heads] (float32)

    Raises:
        ValueError: If input shapes are inconsistent or invalid
    """
    if query.ndim != 3:
        raise ValueError("query must be [batch, num_q_heads, head_dim]")
    if key_buffer.ndim != 3 or value_buffer.ndim != 3:
        raise ValueError("key_buffer/value_buffer must be [total_tokens, num_kv_heads, head_dim]")
    if key_buffer.shape != value_buffer.shape:
        raise ValueError("key_buffer/value_buffer shape mismatch")
    if req_to_tokens.ndim != 2:
        raise ValueError("req_to_tokens must be [batch, max_pages]")
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must be [batch]")
    if req_to_tokens.dtype != jnp.int32 or seq_lens.dtype != jnp.int32:
        raise ValueError("req_to_tokens and seq_lens must be int32")

    batch, num_q_heads, head_dim = map(int, query.shape)
    total_tokens, num_kv_heads, head_dim_kv = map(int, key_buffer.shape)
    if head_dim_kv != head_dim:
        raise ValueError("head_dim mismatch between query and key/value buffers")
    if batch != int(req_to_tokens.shape[0]) or batch != int(seq_lens.shape[0]):
        raise ValueError("batch mismatch among query/req_to_tokens/seq_lens")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads (GQA)")
    if page_size <= 0:
        raise ValueError("page_size must be > 0")
    if total_tokens % page_size != 0:
        raise ValueError("key/value buffer first dim must be a multiple of page_size")

    max_pages = int(req_to_tokens.shape[1])
    k_max = max_pages * int(page_size)

    num_pages_total = total_tokens // page_size
    k_pages = key_buffer.reshape(num_pages_total, page_size, num_kv_heads, head_dim)
    v_pages = value_buffer.reshape(num_pages_total, page_size, num_kv_heads, head_dim)

    k_g = k_pages[req_to_tokens]
    v_g = v_pages[req_to_tokens]

    k_tokens = k_g.reshape(batch, k_max, num_kv_heads, head_dim)
    v_tokens = v_g.reshape(batch, k_max, num_kv_heads, head_dim)

    q_group = num_q_heads // num_kv_heads
    q_grouped = query.reshape(batch, num_kv_heads, q_group, head_dim).astype(jnp.float32)
    k_tokens_f32 = k_tokens.astype(jnp.float32)
    v_tokens_f32 = v_tokens.astype(jnp.float32)

    logits = jnp.einsum("bhgd,bkhd->bhgk", q_grouped, k_tokens_f32, preferred_element_type=jnp.float32) * float(
        softmax_scale
    )

    if logits_soft_cap is None:
        cap = 0.0
    else:
        cap = float(logits_soft_cap)
    if cap > 0.0:
        logits = cap * jnp.tanh(logits / cap)

    positions = jnp.arange(k_max, dtype=jnp.int32)[None, :]
    token_mask = positions < seq_lens[:, None]
    token_mask = token_mask[:, None, None, :]

    logits_masked = jnp.where(token_mask, logits, -jnp.inf)
    m = jnp.max(logits_masked, axis=-1, keepdims=True)
    has_any = m > -jnp.inf
    m_safe = jnp.where(has_any, m, 0.0)

    exp_logits = jnp.where(token_mask, jnp.exp(logits - m_safe), 0.0)
    sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)
    sum_exp_safe = jnp.where(sum_exp > 0.0, sum_exp, 1.0)

    weights = exp_logits / sum_exp_safe
    out_grouped = jnp.einsum("bhgk,bkhd->bhgd", weights, v_tokens_f32, preferred_element_type=jnp.float32)
    out = out_grouped.reshape(batch, num_q_heads, head_dim).astype(query.dtype)

    lse = jnp.where(sum_exp > 0.0, m_safe + jnp.log(sum_exp_safe), -jnp.inf)
    lse = jnp.squeeze(lse, axis=-1).reshape(batch, num_q_heads).astype(jnp.float32)
    return out, lse
