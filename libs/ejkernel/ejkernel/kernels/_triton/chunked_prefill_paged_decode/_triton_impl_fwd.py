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

"""Chunked prefill with paged decode attention implementation for Triton.

Updates the block-tabled KV cache with a single fused Triton scatter kernel
(1 program per ``(token, kv_head)`` using ``input_output_aliases`` to preserve
existing prefix cache slots), then computes causal attention using
``unified_attention``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from ejkernel.callib import next_power_of_2, strides_from_shape, triton_call
from ejkernel.kernels._triton.unified_attention import unified_attention as triton_unified_attention


@triton.jit
def _scatter_kv_cache_kernel(
    keys_ptr,
    values_ptr,
    in_k_cache_ptr,
    in_v_cache_ptr,
    kv_lens_ptr,
    block_tables_ptr,
    query_start_loc_ptr,
    out_k_cache_ptr,
    out_v_cache_ptr,
    stride_k0: tl.constexpr,
    stride_k1: tl.constexpr,
    stride_k2: tl.constexpr,
    stride_v0: tl.constexpr,
    stride_v1: tl.constexpr,
    stride_v2: tl.constexpr,
    stride_kc0: tl.constexpr,
    stride_kc1: tl.constexpr,
    stride_kc2: tl.constexpr,
    stride_kc3: tl.constexpr,
    stride_vc0: tl.constexpr,
    stride_vc1: tl.constexpr,
    stride_vc2: tl.constexpr,
    stride_vc3: tl.constexpr,
    stride_bt0: tl.constexpr,
    stride_bt1: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS_PER_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused KV-cache scatter kernel: 1 program per ``(tok, kv_head)``.

    ``in_k_cache_ptr`` / ``in_v_cache_ptr`` are aliased to ``out_k_cache_ptr`` /
    ``out_v_cache_ptr`` via ``input_output_aliases`` so untouched prefix slots
    remain intact while new tokens are written in parallel across GPU SMs.
    """
    tok = tl.program_id(0)
    kv_head = tl.program_id(1)

    if tok >= NUM_TOKENS:
        return

    total_active = tl.load(query_start_loc_ptr + NUM_SEQS).to(tl.int32)
    if tok >= total_active:
        return

    left = 0
    right = NUM_SEQS
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_loc_ptr + mid).to(tl.int32)
        if val <= tok:
            left = mid + 1
        else:
            right = mid
    seq_idx = left - 1
    if (seq_idx < 0) | (seq_idx >= NUM_SEQS):
        return

    q_start = tl.load(query_start_loc_ptr + seq_idx).to(tl.int32)
    q_end = tl.load(query_start_loc_ptr + seq_idx + 1).to(tl.int32)
    q_len = q_end - q_start
    kv_len = tl.load(kv_lens_ptr + seq_idx).to(tl.int32)
    ctx_len = kv_len - q_len

    pos = ctx_len + (tok - q_start)
    if (pos < 0) | (pos >= kv_len):
        return

    block_idx = pos // BLOCK_SIZE
    if block_idx >= MAX_BLOCKS_PER_SEQ:
        return

    within = pos - block_idx * BLOCK_SIZE
    phys_block = tl.load(
        block_tables_ptr + seq_idx.to(tl.int64) * stride_bt0 + block_idx.to(tl.int64) * stride_bt1
    ).to(tl.int32)
    if (phys_block < 0) | (phys_block >= NUM_BLOCKS):
        return

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM
    offs_d64 = offs_d.to(tl.int64)

    tok64 = tok.to(tl.int64)
    kv_head64 = kv_head.to(tl.int64)
    phys64 = phys_block.to(tl.int64)
    within64 = within.to(tl.int64)

    src_k_off = tok64 * stride_k0 + kv_head64 * stride_k1 + offs_d64 * stride_k2
    src_v_off = tok64 * stride_v0 + kv_head64 * stride_v1 + offs_d64 * stride_v2
    dst_k_off = phys64 * stride_kc0 + within64 * stride_kc1 + kv_head64 * stride_kc2 + offs_d64 * stride_kc3
    dst_v_off = phys64 * stride_vc0 + within64 * stride_vc1 + kv_head64 * stride_vc2 + offs_d64 * stride_vc3

    k_vec = tl.load(keys_ptr + src_k_off, mask=d_mask, other=0.0)
    v_vec = tl.load(values_ptr + src_v_off, mask=d_mask, other=0.0)

    tl.store(out_k_cache_ptr + dst_k_off, k_vec.to(out_k_cache_ptr.dtype.element_ty), mask=d_mask)
    tl.store(out_v_cache_ptr + dst_v_off, v_vec.to(out_v_cache_ptr.dtype.element_ty), mask=d_mask)


def _update_block_tabled_kv_cache(
    *,
    keys: jax.Array,
    values: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Update block-tabled KV cache with a fused Triton scatter kernel."""
    total_tokens = int(keys.shape[0])
    if total_tokens == 0:
        return key_cache, value_cache

    num_blocks, block_size, num_kv_heads, head_dim = map(int, key_cache.shape)
    num_seqs, max_blocks_per_seq = map(int, block_tables.shape)
    block_d = max(16, int(next_power_of_2(head_dim)))

    stride_k0, stride_k1, stride_k2 = strides_from_shape(keys.shape)
    stride_v0, stride_v1, stride_v2 = strides_from_shape(values.shape)
    stride_kc0, stride_kc1, stride_kc2, stride_kc3 = strides_from_shape(key_cache.shape)
    stride_vc0, stride_vc1, stride_vc2, stride_vc3 = strides_from_shape(value_cache.shape)
    stride_bt0, stride_bt1 = strides_from_shape(block_tables.shape)

    scatter_warps = 2 if head_dim <= 64 else 4
    if num_warps is not None:
        scatter_warps = min(int(num_warps), 4)

    new_key_cache, new_value_cache = triton_call(
        keys,
        values,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        kernel=_scatter_kv_cache_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(key_cache.shape, key_cache.dtype),
            jax.ShapeDtypeStruct(value_cache.shape, value_cache.dtype),
        ),
        input_output_aliases={2: 0, 3: 1},
        grid=lambda META: (total_tokens, num_kv_heads),
        name="ejkernel::triton::chunked_prefill_kv_scatter",
        num_warps=scatter_warps,
        num_stages=1 if num_stages is None else min(int(num_stages), 2),
        stride_k0=stride_k0,
        stride_k1=stride_k1,
        stride_k2=stride_k2,
        stride_v0=stride_v0,
        stride_v1=stride_v1,
        stride_v2=stride_v2,
        stride_kc0=stride_kc0,
        stride_kc1=stride_kc1,
        stride_kc2=stride_kc2,
        stride_kc3=stride_kc3,
        stride_vc0=stride_vc0,
        stride_vc1=stride_vc1,
        stride_vc2=stride_vc2,
        stride_vc3=stride_vc3,
        stride_bt0=stride_bt0,
        stride_bt1=stride_bt1,
        NUM_TOKENS=total_tokens,
        NUM_SEQS=num_seqs,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=block_size,
        MAX_BLOCKS_PER_SEQ=max_blocks_per_seq,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
    )
    return new_key_cache, new_value_cache


def chunked_prefill_paged_decode_triton(
    *,
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    alibi_slopes: jax.Array | None,
    softmax_aux: jax.Array | None,
    softmax_scale: float,
    causal: bool,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    seq_threshold_3d: int | None,
    num_par_softmax_segments: int | None,
    num_warps: int | None,
    num_stages: int | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Triton implementation of chunked prefill + paged decode attention."""
    if not causal:
        raise NotImplementedError("chunked_prefill_paged_decode only supports causal attention.")
    if queries.ndim != 3:
        raise ValueError("queries must be [total_tokens, num_q_heads, head_dim]")
    if keys.ndim != 3 or values.ndim != 3:
        raise ValueError("keys/values must be [total_tokens, num_kv_heads, head_dim]")
    if keys.shape != values.shape:
        raise ValueError("keys/values shape mismatch")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache/value_cache shape mismatch")
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise ValueError("kv_lens/block_tables/query_start_loc must be int32")

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    if keys.shape[0] != total_tokens:
        raise ValueError("keys/values first dim must match total_tokens")
    num_seqs = int(kv_lens.shape[0])
    if query_start_loc.shape != (num_seqs + 1,):
        raise ValueError("query_start_loc must have shape (num_seqs + 1,)")

    new_key_cache, new_value_cache = _update_block_tabled_kv_cache(
        keys=keys,
        values=values,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out = triton_unified_attention(
        queries=queries,
        key_cache=new_key_cache,
        value_cache=new_value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        alibi_slopes=alibi_slopes,
        qq_bias=None,
        softmax_aux=softmax_aux,
        softmax_scale=softmax_scale,
        causal=True,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        seq_threshold_3d=seq_threshold_3d,
        num_par_softmax_segments=num_par_softmax_segments,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    assert out.shape == (total_tokens, num_q_heads, head_dim)
    return out, new_key_cache, new_value_cache
