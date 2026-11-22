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


"""Page Attention implementation using Triton kernels.

This module implements paged attention, a memory-efficient attention mechanism
designed for inference workloads where key-value caches are stored in paged
memory blocks. This is particularly useful for serving large language models
where managing KV cache memory efficiently is critical.

Paged attention addresses the challenge of dynamic memory allocation during
autoregressive generation. Instead of allocating a large contiguous buffer
for each sequence's KV cache, memory is organized into fixed-size pages that
can be allocated on-demand and potentially shared across sequences.

Key concepts:
- **KV Cache Pages**: Fixed-size blocks storing key and value vectors
- **Block Tables**: Mapping from logical positions to physical page indices
- **Variable Context Lengths**: Each sequence can have different lengths
- **Memory Efficiency**: Pages can be allocated/deallocated dynamically

Architecture benefits:
1. Reduced memory fragmentation
2. Support for extremely long contexts via pagination
3. Efficient batching of variable-length sequences
4. Memory sharing for prefix caching scenarios

The implementation supports two modes:
1. Single-partition mode (num_splits=1): Direct attention computation
2. Multi-partition mode (num_splits>1): Splits long contexts for parallelization

Features:
- Grouped-query attention (GQA) and multi-query attention (MQA)
- Automatic splitting for long contexts
- Optimized memory access patterns
- GPU-accelerated via Triton kernels

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.page_attention import page_attention
    >>>
    >>> num_seqs, num_heads, head_dim = 4, 8, 64
    >>> num_blocks, num_kv_heads, block_size = 100, 8, 16
    >>>
    >>>
    >>> query = jnp.ones((num_seqs, num_heads, head_dim))
    >>>
    >>>
    >>> key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
    >>> value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
    >>>
    >>>
    >>> context_lens = jnp.array([50, 100, 75, 120])
    >>> block_tables = jnp.zeros((num_seqs, 10), dtype=jnp.int32)
    >>>
    >>>
    >>> output = page_attention(query, key_cache, value_cache, context_lens, block_tables)

Reference:
    Efficient Memory Management for Large Language Model Serving with PagedAttention
    https://arxiv.org/abs/2309.06180
"""

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.callib import cdiv, strides_from_shape, triton_call

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import _paged_attn_kernel, _paged_attn_v2_reduce_kernel

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


@kernel_registry.register("page_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def page_attention(
    query: Float[Array, "num_seqs num_heads head_dim"],
    key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs max_blocks"],
    attn_scale: float | None = None,
    max_context_len: int | None = None,
    num_splits: int = 0,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int | None = None,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
    sliding_window: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Float[Array, "num_seqs num_heads head_dim"]:
    """Compute paged attention with key-value caches stored in memory pages.

    This function performs attention where the key-value cache is organized into
    fixed-size pages, enabling efficient memory management for LLM serving workloads.
    The implementation automatically decides whether to use single-partition or
    multi-partition computation based on context lengths and available resources.

    Args:
        query: Query tensor of shape (num_seqs, num_heads, head_dim). Each row
            represents the query for one sequence in the batch (typically during
            autoregressive decoding).
        key_cache: Paged key cache of shape (num_blocks, num_kv_heads, block_size, head_dim).
            Keys are stored in fixed-size blocks that can be non-contiguous.
        value_cache: Paged value cache of shape (num_blocks, num_kv_heads, block_size, head_dim).
            Values are stored in fixed-size blocks matching key_cache organization.
        context_lens: Length of context for each sequence, shape (num_seqs,).
            Specifies how many tokens in the KV cache are valid for each sequence.
        block_tables: Mapping from logical blocks to physical blocks, shape
            (num_seqs, max_blocks). For each sequence, maps logical block indices
            to physical block indices in key_cache/value_cache.
        attn_scale: Attention scaling factor. If None, defaults to 1/sqrt(head_dim).
        max_context_len: Maximum context length across all sequences. If None,
            computed as the maximum of context_lens.
        num_splits: Number of partitions for splitting long contexts. If 0, the
            implementation automatically determines the optimal number of splits.
            Set to 1 to force single-partition mode.
        mask_value: Value to use for masked positions (default: -2.38e38).
        attn_logits_soft_cap: Not supported in Triton implementation (raises error).
        pages_per_compute_block: Not supported in Triton implementation (raises error).
        megacore_mode: Not supported in Triton implementation (raises error).
        inline_seq_dim: Must be True for Triton implementation (raises error if False).
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        Attention output of shape (num_seqs, num_heads, head_dim).

    Raises:
        NotImplementedError: If unsupported parameters are provided (attn_logits_soft_cap,
            pages_per_compute_block, megacore_mode, or inline_seq_dim=False).
        AssertionError: If head_size is not in {16, 32, 64, 128, 256, 512} or if
            block_size constraints are violated.

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._triton.page_attention import page_attention
        >>>
        >>>
        >>> num_seqs, num_heads, head_dim = 4, 8, 64
        >>> query = jnp.ones((num_seqs, num_heads, head_dim))
        >>>
        >>>
        >>> num_blocks, num_kv_heads, block_size = 100, 8, 16
        >>> key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        >>> value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        >>>
        >>>
        >>> context_lens = jnp.array([32, 64, 48, 80])
        >>>
        >>>
        >>>
        >>> block_tables = jnp.array([
        ...     [0, 1, -1, -1, -1],
        ...     [2, 3, 4, 5, -1],
        ...     [6, 7, 8, -1, -1],
        ...     [9, 10, 11, 12, 13]
        ... ])
        >>>
        >>> output = page_attention(query, key_cache, value_cache, context_lens, block_tables)
        >>> print(output.shape)
    """
    if pages_per_compute_block is not None:
        raise NotImplementedError("pages_per_compute_block is not supported in Triton implementation")
    if megacore_mode is not None:
        raise NotImplementedError("megacore_mode is not supported in Triton implementation")
    if not inline_seq_dim:
        raise NotImplementedError("inline_seq_dim=False is not supported in Triton implementation")
    if attn_logits_soft_cap is not None:
        raise NotImplementedError("attn_logits_soft_cap is not supported in Triton implementation")

    num_seqs = query.shape[0]
    num_kv_heads = key_cache.shape[1]
    kv_blocksize = key_cache.shape[2]
    head_size = key_cache.shape[3]
    query_group_size = query.shape[1] // num_kv_heads

    if attn_scale is None:
        attn_scale = 1.0 / (head_size**0.5)

    if max_context_len is None:
        max_context_len = int(jax.device_get(context_lens).max())

    if query_group_size == 1:
        padded_group_size = 1
    elif query_group_size < 16:
        padded_group_size = 16
    else:
        padded_group_size = 1 << (query_group_size - 1).bit_length()

    assert head_size in (16, 32, 64, 128, 256, 512), f"head_size={head_size}"
    assert padded_group_size == 1 or kv_blocksize >= 16, f"kv_blocksize={kv_blocksize}"

    num_sms = 108
    if num_splits == 0:
        if num_seqs * num_kv_heads > 2 * num_sms:
            num_splits = 1
            if max_context_len >= 4096:
                partition_size = max(256, kv_blocksize)
                num_splits = cdiv(max_context_len, partition_size)
        else:
            partition_size = max(256, kv_blocksize)
            num_splits = cdiv(max_context_len, partition_size)
            if max_context_len <= 1024 or kv_blocksize >= 256:
                num_splits = 1
    elif num_splits > 1:
        partition_size = cdiv(max_context_len, num_splits)
        partition_size = 1 << (partition_size - 1).bit_length()

    stride_bt0, stride_bt1 = strides_from_shape(block_tables.shape)
    stride_q0, stride_q1, stride_q2 = strides_from_shape(query.shape)
    stride_kv0, stride_kv1, stride_kv2, stride_kv3 = strides_from_shape(key_cache.shape)

    if num_splits == 1:
        out_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)

        def grid(meta):
            return (num_seqs, num_kv_heads, 1)

        stride_o0 = stride_q0
        stride_o1 = stride_q1
        stride_o2 = stride_q2
        stride_o3 = stride_q1
        stride_o4 = stride_q2

        metaparams = dict(
            grid=grid,
            attn_scale=attn_scale,
            stride_bt0=stride_bt0,
            stride_bt1=stride_bt1,
            stride_q0=stride_q0,
            stride_q1=stride_q1,
            stride_q2=stride_q2,
            stride_kv0=stride_kv0,
            stride_kv1=stride_kv1,
            stride_kv2=stride_kv2,
            stride_kv3=stride_kv3,
            stride_o0=stride_o0,
            stride_o1=stride_o1,
            stride_o2=stride_o2,
            stride_o3=stride_o3,
            stride_o4=stride_o4,
            HEAD_SIZE=head_size,
            QUERY_GROUP_SIZE=query_group_size,
            PADDED_QUERY_GROUP_SIZE=padded_group_size,
            NUM_KV_HEADS=num_kv_heads,
            KV_BLOCK_SIZE=kv_blocksize,
            PARTITION_SIZE=0,
        )

        dummy_m_shape = jax.ShapeDtypeStruct((num_seqs, num_kv_heads, 1, query_group_size), jnp.float32)
        dummy_l_shape = jax.ShapeDtypeStruct((num_seqs, num_kv_heads, 1, query_group_size), jnp.float32)

        *_, out = triton_call(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            kernel=_paged_attn_kernel,
            out_shape=(dummy_m_shape, dummy_l_shape, out_shape),
            name="ejkernel::triton::page_attn_fwd",
            num_warps=num_warps,
            num_stages=num_stages,
            **metaparams,
        )

    else:
        tmp_out_shape = jax.ShapeDtypeStruct(
            (num_seqs, num_kv_heads, num_splits, query_group_size, head_size),
            query.dtype,
        )
        m_i_shape = jax.ShapeDtypeStruct(
            (num_seqs, num_kv_heads, num_splits, query_group_size),
            jnp.float32,
        )
        l_i_shape = m_i_shape

        def grid(meta):
            return (num_seqs, num_kv_heads, num_splits)

        assert (partition_size >= kv_blocksize) and (partition_size % kv_blocksize == 0), (
            f"partition_size={partition_size}, kv_blocksize={kv_blocksize}"
        )

        metaparams = dict(
            grid=grid,
            HEAD_SIZE=head_size,
            QUERY_GROUP_SIZE=query_group_size,
            PADDED_QUERY_GROUP_SIZE=padded_group_size,
            NUM_KV_HEADS=num_kv_heads,
            KV_BLOCK_SIZE=kv_blocksize,
            PARTITION_SIZE=partition_size,
        )

        stride_tmp0, stride_tmp1, stride_tmp2, stride_tmp3, stride_tmp4 = strides_from_shape(
            (num_seqs, num_kv_heads, num_splits, query_group_size, head_size)
        )

        metaparams["attn_scale"] = attn_scale
        metaparams["stride_bt0"] = stride_bt0
        metaparams["stride_bt1"] = stride_bt1
        metaparams["stride_q0"] = stride_q0
        metaparams["stride_q1"] = stride_q1
        metaparams["stride_q2"] = stride_q2
        metaparams["stride_kv0"] = stride_kv0
        metaparams["stride_kv1"] = stride_kv1
        metaparams["stride_kv2"] = stride_kv2
        metaparams["stride_kv3"] = stride_kv3
        metaparams["stride_o0"] = stride_tmp0
        metaparams["stride_o1"] = stride_tmp1
        metaparams["stride_o2"] = stride_tmp2
        metaparams["stride_o3"] = stride_tmp3
        metaparams["stride_o4"] = stride_tmp4

        m_i, l_i, tmp_out = triton_call(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            kernel=_paged_attn_kernel,
            out_shape=(m_i_shape, l_i_shape, tmp_out_shape),
            name="ejkernel::triton::page_attn_fwd_split",
            num_warps=num_warps,
            num_stages=num_stages,
            **metaparams,
        )

        out_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)

        def reduce_grid(meta):
            return (num_seqs, num_kv_heads)

        next_num_splits = 1 << (num_splits - 1).bit_length()

        reduce_metaparams = dict(
            grid=reduce_grid,
            max_num_partitions=num_splits,
            stride_o0=stride_q0,
            stride_o1=stride_q1,
            stride_o2=stride_q2,
            HEAD_SIZE=head_size,
            QUERY_GROUP_SIZE=query_group_size,
            NUM_KV_HEADS=num_kv_heads,
            PARTITION_SIZE=partition_size,
            NUM_PARTITIONS=next_num_splits,
        )

        out = triton_call(
            m_i,
            l_i,
            tmp_out,
            context_lens,
            kernel=_paged_attn_v2_reduce_kernel,
            out_shape=out_shape,
            name="ejkernel::triton::page_attn_reduce",
            **reduce_metaparams,
        )

    return out
