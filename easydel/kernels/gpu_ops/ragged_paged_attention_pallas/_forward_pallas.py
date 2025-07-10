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
from jax.experimental import pallas as pl


def _ragged_paged_attention_pallas_kernel(
    # Input/Output References
    q_ref,
    k_pages_ref,
    v_pages_ref,
    sequence_page_indices_ref,
    sequence_lengths_ref,
    q_sequence_indices_ref,
    q_token_positions_ref,
    o_ref,
    # Static parameters
    softmax_scale: float,
    soft_cap: float | None,
    q_heads_per_group: int,
    KV_PAGE_BLOCK_SIZE: int,
):
    q_idx = pl.program_id(0)
    q_head_idx = pl.program_id(1)
    head_size = q_ref.shape[-1]
    q_vec_1d = pl.load(q_ref, (q_idx, q_head_idx, slice(None))).astype(jnp.float32)
    q_vec = q_vec_1d.reshape(1, -1) * softmax_scale
    seq_idx = pl.load(q_sequence_indices_ref, (q_idx,)).astype(jnp.int32)
    query_token_pos = pl.load(q_token_positions_ref, (q_idx,)).astype(jnp.int32)
    kv_cache_len = pl.load(sequence_lengths_ref, (seq_idx,)).astype(jnp.int32)
    kv_head_idx = q_head_idx // q_heads_per_group

    m_i = jnp.full((), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((), dtype=jnp.float32)
    acc = jnp.zeros((1, head_size), dtype=jnp.float32)

    page_size = k_pages_ref.shape[1]
    num_kv_blocks = pl.cdiv(kv_cache_len, page_size * KV_PAGE_BLOCK_SIZE)

    def _process_kv_block(kv_block_idx, carry):
        acc, m_i, l_i = carry

        page_map_start_idx = kv_block_idx * KV_PAGE_BLOCK_SIZE
        page_indices_for_block = pl.load(
            sequence_page_indices_ref,
            (seq_idx, pl.ds(page_map_start_idx, KV_PAGE_BLOCK_SIZE)),
        )
        key_block = k_pages_ref[page_indices_for_block, :, kv_head_idx, :].reshape(-1, head_size)
        value_block = v_pages_ref[page_indices_for_block, :, kv_head_idx, :].reshape(-1, head_size)
        scores = pl.dot(q_vec, key_block.T)
        if soft_cap is not None:
            scores = jnp.tanh(scores / soft_cap) * soft_cap
        kv_block_start_token_idx = kv_block_idx * KV_PAGE_BLOCK_SIZE * page_size
        kv_token_indices = jnp.arange(KV_PAGE_BLOCK_SIZE * page_size) + kv_block_start_token_idx
        causal_mask = kv_token_indices <= query_token_pos
        padding_mask = kv_token_indices < kv_cache_len
        mask = causal_mask & padding_mask
        scores = jnp.where(mask, scores, -jnp.inf)
        m_j = jnp.max(scores)

        def do_update(acc, m_i, l_i):
            m_new = jnp.maximum(m_i, m_j)
            alpha = jnp.exp(m_i - m_new)
            p_j = jnp.exp(scores - m_new)
            p_j = jnp.where(mask, p_j, 0.0)
            l_new = alpha * l_i + jnp.sum(p_j)
            new_acc = acc * alpha
            new_acc += pl.dot(p_j.astype(value_block.dtype), value_block)
            return new_acc, m_new, l_new

        def no_op(acc, m_i, l_i):
            return acc, m_i, l_i

        return jax.lax.cond(m_j > -jnp.inf, do_update, no_op, acc, m_i, l_i)

    acc, m_i, l_i = jax.lax.fori_loop(0, num_kv_blocks, _process_kv_block, (acc, m_i, l_i))

    def calculate_output():
        return (acc / l_i).squeeze(axis=0)

    def return_zeros():
        return jnp.zeros((head_size,), dtype=acc.dtype)

    final_output = jax.lax.cond(l_i > 0, calculate_output, return_zeros)
    pl.store(o_ref, (q_idx, q_head_idx, slice(None)), final_output.astype(o_ref.dtype))
