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

"""Public interface for the CUDA ragged paged attention v3 kernel.

This module defines the top-level :func:`ragged_page_attention_v3` function,
which is registered with the ejKernel kernel registry for the CUDA platform
and GPU backend. The function is decorated with :mod:`jaxtyping` and
:mod:`beartype` for runtime shape and type checking and delegates
execution to the low-level CUDA implementation through
:func:`ejit <ejkernel.callib._ejit.ejit>`.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ejkernel.callib._ejit import ejit

from ..._registry import Backend, Platform, kernel_registry
from ._cuda_impl import ragged_page_attention_v3_cuda


@kernel_registry.register("ragged_page_attention_v3", Platform.CUDA, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_page_attention_v3(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
]:
    """Compute ragged paged attention v3 on GPU via the CUDA kernel.

    This is the registered public entry point for the CUDA ragged paged
    attention v3 kernel. It accepts type-annotated JAX arrays (enforced
    at runtime by *jaxtyping* + *beartype*), wraps the low-level CUDA
    implementation with :func:`ejit`, and forwards all relevant
    arguments.

    Note:
        The keyword arguments ``chunk_prefill_size``,
        ``num_kv_pages_per_block``, ``num_queries_per_block``, and
        ``vmem_limit_bytes`` are accepted for API compatibility with
        other backends (e.g., XLA/TPU) but are **not used** by the CUDA
        implementation and are silently discarded.

    Args:
        queries: Query tensor of shape
            ``(total_tokens, num_q_heads, head_dim)``.
        keys: Key tensor of shape
            ``(total_tokens, num_kv_heads, head_dim)``.
        values: Value tensor of shape
            ``(total_tokens, num_kv_heads, head_dim)``.
        kv_cache: Paged key-value cache of shape
            ``(num_pages, page_size, num_kv_heads_x2_per_kv_packing,
            kv_packing, head_dim_padded)``.
        kv_lens: Number of valid KV tokens for each sequence, shape
            ``(max_num_seqs,)``, dtype ``int32``.
        block_tables: Flat block-table mapping sequences to cache pages,
            shape ``(max_num_seqs * pages_per_seq,)``, dtype ``int32``.
        query_start_loc: Cumulative query-token offsets per sequence,
            shape ``(max_num_seqs + 1,)``, dtype ``int32``.
        distribution: Integer tensor of shape ``(3,)`` encoding
            workload distribution metadata for the kernel.
        softmax_aux: Optional auxiliary tensor of shape
            ``(num_q_heads,)`` for sink-token softmax correction.
            Defaults to ``None`` (sink logic disabled).
        softmax_scale: Scaling factor applied to QK^T dot products
            before softmax. Defaults to ``1.0``.
        sliding_window: When set, restricts attention to the most recent
            *sliding_window* KV tokens. ``None`` uses the full context.
        logits_soft_cap: Soft-capping value (``tanh``-based) applied to
            attention logits. ``None`` disables capping.
        q_scale: Optional FP8 dequantization scale for queries.
        k_scale: Optional FP8 dequantization scale for keys.
        v_scale: Optional FP8 dequantization scale for values.
        chunk_prefill_size: *Unused by this backend.* Accepted for API
            compatibility.
        num_kv_pages_per_block: *Unused by this backend.* Accepted for
            API compatibility.
        num_queries_per_block: *Unused by this backend.* Accepted for
            API compatibility.
        vmem_limit_bytes: *Unused by this backend.* Accepted for API
            compatibility.

    Returns:
        A tuple of two JAX arrays:
            - **output**: Attention result with the same shape and dtype
              as *queries* (``total_tokens, num_q_heads, head_dim``).
            - **kv_cache**: The (potentially updated) paged KV cache,
              retaining the same shape and dtype as the input *kv_cache*.
    """
    del chunk_prefill_size, num_kv_pages_per_block, num_queries_per_block, vmem_limit_bytes

    return ejit(
        func=ragged_page_attention_v3_cuda,
        static_argnames=[
            "softmax_scale",
            "sliding_window",
            "logits_soft_cap",
            "q_scale",
            "k_scale",
            "v_scale",
        ],
    )(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
