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


"""XLA backend for Prefill Paged Attention.

This submodule provides an XLA reference implementation of paged attention
specifically designed for the prefill phase of sequence processing. During
prefill, the model processes the entire input prompt and populates the KV
cache with all tokens.

Prefill paged attention differs from decode attention in that:
1. Multiple query tokens are processed at once (the entire prompt)
2. The KV cache is being filled rather than read-only
3. Causal masking is applied (each position attends to previous)
4. Higher arithmetic intensity allows for better GPU/TPU utilization

This implementation processes prefill in chunks to manage memory usage
while maintaining the paged KV cache structure for efficient subsequent
decode operations.

Key features:
1. Chunked prefill: Processes long prompts in fixed-size chunks
2. Paged KV cache: Non-contiguous memory for efficient allocation
3. GQA/MQA support: Flexible query/KV head configurations
4. Causal masking: Proper autoregressive attention pattern
5. Sliding window: Optional local attention for long contexts
6. Logits soft capping: Numerical stability via tanh capping

Memory layout:
- Query: [chunk_size, num_q_heads, head_dim] - Current chunk being processed
- KV cache: [num_kv_heads, total_pages, page_size, head_dim] - All cached KV
- Page indices: [num_pages] - Maps logical to physical pages for this sequence

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._xla.prefill_page_attention import prefill_page_attention
    >>>
    >>> # Prefill with 128-token chunk
    >>> chunk_size, num_heads, head_dim = 128, 8, 64
    >>> num_kv_heads = 8
    >>> query = jnp.ones((chunk_size, num_heads, head_dim))
    >>>
    >>> # Paged KV cache
    >>> key_cache = jnp.ones((num_kv_heads, 100, 16, head_dim))
    >>> value_cache = jnp.ones((num_kv_heads, 100, 16, head_dim))
    >>>
    >>> # Context info
    >>> context_len = jnp.array([512])  # Total context so far
    >>> page_indices = jnp.arange(32)   # Pages for this sequence
    >>>
    >>> output = prefill_page_attention(
    ...     query, key_cache, value_cache,
    ...     context_len, page_indices
    ... )
    >>> output.shape
    (128, 8, 64)

Note:
    This is an XLA reference implementation. For production TPU workloads,
    prefer the Pallas implementation for better performance.
"""

from ._interface import prefill_page_attention

__all__ = ("prefill_page_attention",)
