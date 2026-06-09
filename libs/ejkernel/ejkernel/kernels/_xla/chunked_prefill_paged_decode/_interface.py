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

"""Chunked prefill + paged decode over block-tabled KV cache (XLA).

This kernel updates a block-tabled KV cache with the provided `keys` / `values`
for the packed `queries`, then computes causal attention outputs for the packed
queries against the updated cache.

It is a JAX-friendly equivalent to vLLM's `chunked_prefill_paged_decode.py`.
"""

from __future__ import annotations

import math

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import _chunked_prefill_paged_decode_fwd


@kernel_registry.register("chunked_prefill_paged_decode", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def chunked_prefill_paged_decode(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
]:
    """XLA/pure JAX implementation of chunked prefill + paged decode attention.

    Hardware-agnostic implementation that updates a block-tabled KV cache and computes
    causal attention outputs. Works across all XLA-supported backends (CPU, GPU, TPU).

    Args:
        queries: Packed query vectors [total_tokens, num_q_heads, head_dim]
        keys: Packed keys to insert [total_tokens, num_kv_heads, head_dim]
        values: Packed values to insert [total_tokens, num_kv_heads, head_dim]
        key_cache: Block-tabled key cache [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Block-tabled value cache [num_blocks, block_size, num_kv_heads, head_dim]
        kv_lens: KV lengths per sequence [num_seqs]
        block_tables: Logical-to-physical block mapping [num_seqs, max_blocks_per_seq]
        query_start_loc: Cumulative query positions [num_seqs + 1]
        alibi_slopes: Optional ALiBi slopes [num_q_heads]
        softmax_aux: Optional auxiliary softmax parameters [num_q_heads]
        softmax_scale: Attention scaling factor (default: 1/√head_dim)
        causal: Apply causal masking (default: True)
        sliding_window: Optional local attention window size
        logits_soft_cap: Optional logit capping value
        seq_threshold_3d: Ignored (Triton-specific)
        num_par_softmax_segments: Ignored (Triton-specific)
        num_warps: Ignored (Triton-specific)
        num_stages: Ignored (Triton-specific)

    Returns:
        Tuple of (attention_output, updated_key_cache, updated_value_cache)

    Raises:
        NotImplementedError: If causal=False
        ValueError: If input shapes are invalid or inconsistent
    """
    del seq_threshold_3d, num_par_softmax_segments, num_warps, num_stages
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(queries.shape[-1])
    return _chunked_prefill_paged_decode_fwd(
        queries=queries,
        keys=keys,
        values=values,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        alibi_slopes=alibi_slopes,
        softmax_aux=softmax_aux,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
    )
