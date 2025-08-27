# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import jax
import jax.numpy as jnp
from eformer.callib import cdiv, triton_call

from ._forward_triton import _ragged_paged_attn_prefetch_kernel_combined


def ragged_paged_attention(
    queries: jnp.ndarray,  # [T, QH, D]
    kv_pages: jnp.ndarray,  # [P, PS, 2*KVH, D]
    context_lens: jnp.ndarray,  # [S], int32
    block_tables: jnp.ndarray,  # [S, pages_per_seq_max], int32
    query_start_loc: jnp.ndarray,  # [S+1], int32
    num_seqs: jnp.ndarray | int,  # active sequence count
    softmax_scale: float,
    kv_pages_per_block: int = 8,
) -> jnp.ndarray:
    T, QH, D = queries.shape
    P, PS, C, Dk = kv_pages.shape
    assert D == Dk, "head_size mismatch"
    assert C % 2 == 0, "combined kv heads must be even"
    KVH = C // 2
    assert QH % KVH == 0
    QHG = QH // KVH
    pages_per_seq_max = int(block_tables.shape[1])

    # Q reshape/scale + pad
    q4 = (queries * softmax_scale).reshape(T, KVH, QHG, D)
    T_padded = max(T, 1)
    if T_padded > T:
        pad = jnp.zeros((T_padded - T, KVH, QHG, D), dtype=q4.dtype)
        q4 = jnp.concatenate([q4, pad], axis=0)

    # per-row metadata
    starts = query_start_loc[:-1]
    ends = query_start_loc[1:]
    q_lens = (ends - starts).astype(jnp.int32)

    t_idx = jnp.arange(T_padded, dtype=jnp.int32)
    t_clamped = jnp.minimum(t_idx, jnp.int32(max(T - 1, 0)))
    row_seq = jnp.searchsorted(ends, t_clamped, side="right").astype(jnp.int32)

    row_start = starts[row_seq]
    row_qlen = q_lens[row_seq]
    row_kvlen = context_lens[row_seq]
    row_qoff = t_idx - row_start
    row_firstk = (row_kvlen - row_qlen + row_qoff).astype(jnp.int32)

    # active rows gating
    ns_dev = jnp.asarray(num_seqs, dtype=jnp.int32)  # 0-d device scalar
    row_valid = (t_idx < T) & (row_seq < ns_dev)

    KV_PAGES_PER_BLOCK = int(kv_pages_per_block)
    MAX_KV_SUPERBLOCKS = cdiv(pages_per_seq_max, KV_PAGES_PER_BLOCK)

    out_shape = jax.ShapeDtypeStruct((T_padded, KVH, QHG, D), jnp.float32)

    def grid(meta):
        return (T_padded, KVH, QHG)

    out4_padded = triton_call(
        q4,
        kv_pages,
        block_tables.astype(jnp.int32),
        row_seq.astype(jnp.int32),
        row_firstk.astype(jnp.int32),
        row_kvlen.astype(jnp.int32),
        row_valid.astype(jnp.bool_),
        kernel=_ragged_paged_attn_prefetch_kernel_combined,
        out_shape=out_shape,
        grid=grid,
        T=T_padded,
        KVH=KVH,
        QHG=QHG,
        D=D,
        PS=PS,
        PAGES_PER_SEQ_MAX=pages_per_seq_max,
        KV_PAGES_PER_BLOCK=KV_PAGES_PER_BLOCK,
        MAX_KV_SUPERBLOCKS=int(MAX_KV_SUPERBLOCKS),
        num_warps=8,
        num_stages=3,
    )

    return out4_padded[:T].reshape(T, QH, D).astype(queries.dtype)
