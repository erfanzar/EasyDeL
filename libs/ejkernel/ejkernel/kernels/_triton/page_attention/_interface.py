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
memory blocks.
"""

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.callib import cdiv, next_power_of_2, strides_from_shape, triton_call

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import _paged_attn_kernel, _paged_attn_v2_reduce_kernel

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


@kernel_registry.register("page_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def page_attention(
    query: Float[Array, "num_seqs num_heads head_dim"],
    # Note: axis names match the cross-backend kernel_registry signature contract;
    # the Triton layout is blocks-first: [num_blocks, num_kv_heads, block_size, head_dim].
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

    Args:
        query: Query tensor of shape (num_seqs, num_heads, head_dim).
        key_cache: Paged key cache of shape (num_blocks, num_kv_heads, block_size, head_dim).
        value_cache: Paged value cache of shape (num_blocks, num_kv_heads, block_size, head_dim).
        context_lens: Length of context for each sequence, shape (num_seqs,).
        block_tables: Mapping from logical blocks to physical blocks, shape
            (num_seqs, max_blocks).
        attn_scale: Attention scaling factor. If None, defaults to 1/sqrt(head_dim).
        max_context_len: Maximum context length across all sequences. If None,
            uses the static upper bound ``block_tables.shape[1] * block_size``
            so execution remains ``jax.jit``-compatible without host sync.
        num_splits: Number of partitions for splitting long contexts (0 = auto,
            1 = single-partition).
        mask_value: Not supported in Triton (raises EjkernelRuntimeError if non-default).
        attn_logits_soft_cap: Optional soft capping value for attention logits.
        pages_per_compute_block: Not supported in Triton implementation.
        megacore_mode: Not supported in Triton implementation.
        inline_seq_dim: Must be True for Triton implementation.
        sliding_window: Not supported in Triton (raises EjkernelRuntimeError if provided).
        num_warps: Number of Triton warps selected by the operation config.
        num_stages: Number of Triton pipeline stages selected by the operation config.

    Returns:
        Attention output of shape (num_seqs, num_heads, head_dim).
    """
    del sliding_window, mask_value

    if pages_per_compute_block is not None:
        raise NotImplementedError("pages_per_compute_block is not supported in Triton implementation")
    if megacore_mode is not None:
        raise NotImplementedError("megacore_mode is not supported in Triton implementation")
    if not inline_seq_dim:
        raise NotImplementedError("inline_seq_dim=False is not supported in Triton implementation")

    num_seqs = int(query.shape[0])
    num_kv_heads = int(key_cache.shape[1])
    kv_blocksize = int(key_cache.shape[2])
    head_size = int(key_cache.shape[3])
    query_group_size = int(query.shape[1]) // num_kv_heads

    if attn_scale is None:
        attn_scale = 1.0 / (head_size**0.5)

    # Static upper bound avoids host-sync jax.device_get() and works under jax.jit.
    if max_context_len is None:
        max_context_len = int(block_tables.shape[1]) * kv_blocksize
    else:
        max_context_len = int(max_context_len)

    if query_group_size == 1:
        padded_group_size = 1
    elif query_group_size < 16:
        padded_group_size = 16
    else:
        padded_group_size = int(next_power_of_2(query_group_size))

    padded_group_size_reduce = max(1, int(next_power_of_2(query_group_size)))

    assert head_size in (16, 32, 64, 128, 256, 512), f"head_size={head_size}"
    assert padded_group_size == 1 or kv_blocksize >= 16, f"kv_blocksize={kv_blocksize}"

    has_softcap = attn_logits_soft_cap is not None and float(attn_logits_soft_cap) > 0.0
    softcap_val = float(attn_logits_soft_cap) if has_softcap else 0.0
    use_fp32_ieee = query.dtype == jnp.float32

    num_sms = 108
    partition_size = 0
    if num_splits == 0:
        if num_seqs * num_kv_heads > 2 * num_sms:
            num_splits = 1
            if max_context_len >= 4096:
                partition_size = max(256, kv_blocksize)
                num_splits = max(1, cdiv(max_context_len, partition_size))
        else:
            partition_size = max(256, kv_blocksize)
            num_splits = max(1, cdiv(max_context_len, partition_size))
            if max_context_len <= 1024 or kv_blocksize >= 256:
                num_splits = 1
    elif num_splits > 1:
        partition_size = max(kv_blocksize, cdiv(max(1, max_context_len), num_splits))
        partition_size = int(next_power_of_2(partition_size))
        if partition_size % kv_blocksize != 0:
            partition_size = cdiv(partition_size, kv_blocksize) * kv_blocksize
        num_splits = max(1, cdiv(max(1, max_context_len), partition_size))

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
            attn_scale=float(attn_scale),
            softcap=float(softcap_val),
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
            HAS_SOFTCAP=bool(has_softcap),
            USE_FP32_IEEE=bool(use_fp32_ieee),
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
            HAS_SOFTCAP=bool(has_softcap),
            USE_FP32_IEEE=bool(use_fp32_ieee),
        )

        stride_tmp0, stride_tmp1, stride_tmp2, stride_tmp3, stride_tmp4 = strides_from_shape(
            (num_seqs, num_kv_heads, num_splits, query_group_size, head_size)
        )

        metaparams["attn_scale"] = float(attn_scale)
        metaparams["softcap"] = float(softcap_val)
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

        next_num_splits = int(next_power_of_2(num_splits))

        reduce_metaparams = dict(
            grid=reduce_grid,
            max_num_partitions=num_splits,
            stride_o0=stride_q0,
            stride_o1=stride_q1,
            stride_o2=stride_q2,
            HEAD_SIZE=head_size,
            QUERY_GROUP_SIZE=query_group_size,
            PADDED_QUERY_GROUP_SIZE=padded_group_size_reduce,
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
            num_warps=num_warps,
            num_stages=num_stages,
            **reduce_metaparams,
        )

    return out
