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

"""Paged decode attention implemented in Triton (GPU).

This is a JAX/Triton port of vLLM's decode attention, operating on a paged KV cache
addressed via a per-sequence page table (`req_to_tokens`).

Core inputs:
- `query`: [batch, num_q_heads, head_dim]
- `key_buffer` / `value_buffer`: [num_pages*page_size, num_kv_heads, head_dim]
- `req_to_tokens`: [batch, max_pages] mapping logical page -> physical page id
- `seq_lens`: [batch] context lengths in tokens

Returns:
- `out`: [batch, num_q_heads, head_dim]
- `lse`: [batch, num_q_heads] (float32), the log-sum-exp of attention logits.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import decode_attention_triton


@kernel_registry.register("decode_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def decode_attention(
    query: Float[Array, "batch num_q_heads head_dim"],
    key_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
    value_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
    req_to_tokens: Int32[Array, "batch max_pages"],
    seq_lens: Int32[Array, "batch"],
    *,
    softmax_scale: float | None = None,
    num_kv_splits: int = 16,
    page_size: int = 1,
    logits_soft_cap: float | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[
    Float[Array, "batch num_q_heads head_dim"],
    Float[Array, "batch num_q_heads"],
]:
    """Triton GPU implementation of vLLM-style paged decode attention.

    Performs decode-phase attention where each query attends to its full context
    stored in a paged KV cache. Uses a two-stage approach: first splits KV context
    across multiple blocks for parallel processing, then combines results.

    Args:
        query: Query vectors [batch, num_q_heads, head_dim]
        key_buffer: Paged key buffer [total_tokens, num_kv_heads, head_dim]
        value_buffer: Paged value buffer [total_tokens, num_kv_heads, head_dim]
        req_to_tokens: Page table mapping [batch, max_pages]
        seq_lens: Context length per sequence [batch]
        softmax_scale: Attention scaling factor (default: 1/√head_dim)
        num_kv_splits: Number of KV splits for parallelization (default: 16)
        page_size: Memory page size in tokens (default: 1)
        logits_soft_cap: Optional logit capping value
        num_warps: Number of Triton warps
        num_stages: Number of pipeline stages

    Returns:
        Tuple of (attention_output, lse):
            - attention_output: [batch, num_q_heads, head_dim]
            - lse: Log-sum-exp values [batch, num_q_heads] (float32)
    """
    return decode_attention_triton(
        query=query,
        key_buffer=key_buffer,
        value_buffer=value_buffer,
        req_to_tokens=req_to_tokens,
        seq_lens=seq_lens,
        softmax_scale=softmax_scale,
        num_kv_splits=num_kv_splits,
        page_size=page_size,
        logits_soft_cap=logits_soft_cap,
        num_warps=num_warps,
        num_stages=num_stages,
    )
