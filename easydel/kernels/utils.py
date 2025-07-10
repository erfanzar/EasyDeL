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

from easydel.layers.caching.page import PagesMetadata

MAX_GPU_FUSED_SIZE = 65536


def get_stride(array):
    strides = [1]
    for i in range(len(array.shape) - 1, 0, -1):  # Iterate in reverse shape order
        strides.insert(0, strides[0] * array.shape[i])
    return strides


def cdiv(a, b):
    return (a + b - 1) // b


def _pad_to_shape(arr: jax.Array, target_shape: tuple[int, ...], pad_value: float | int = 0) -> jax.Array:
    """Pads a JAX array to a target shape."""
    if arr.shape == target_shape:
        return arr
    padding = tuple((0, max_len - cur_len) for cur_len, max_len in zip(arr.shape, target_shape, strict=False))
    return jnp.pad(arr, padding, constant_values=pad_value)


def generate_ragged_paged_attention_data(
    seq_lens: list[tuple[int, int]],  # List of (q_len, kv_len) for each sequence
    num_heads: tuple[int, int],  # (num_q_heads, num_kv_heads)
    head_dim: int,
    page_size: int,
    max_num_seq: int,
    max_num_batched_tokens: int,
    q_dtype: jnp.dtype = jnp.float32,
    kv_dtype: jnp.dtype = jnp.float32,
    seed: int = 42,
) -> tuple[jax.Array, PagesMetadata, int]:
    """
    Generates realistic test data for a ragged paged attention kernel.

    This function creates a batch of sequences with varying query and key/value lengths.
    The KV cache is organized into a 'paged' memory model where a global pool of pages
    is indexed by each sequence. All generated tensors are padded to maximum
    dimensions to be compatible with JAX's static shape requirements.

    Args:
        seq_lens: A list of (query_length, key_value_length) tuples.
        num_heads: A tuple of (number_of_query_heads, number_of_kv_heads).
        head_dim: The dimension of each attention head.
        page_size: The number of tokens stored per page in the KV cache.
        q_dtype: The data type for the query tensor.
        kv_dtype: The data type for the key/value tensors.
        max_num_pages: The total number of pages available in the global KV cache pool.
        max_num_seq: The maximum number of sequences the batch can hold (for padding).
        max_num_batched_tokens: The maximum total number of query tokens (for padding).
        seed: The random seed for data generation.

    """

    num_q_heads, num_kv_heads = num_heads
    batch_size = len(seq_lens)
    key = jax.random.key(seed)
    q_key, kv_key = jax.random.split(key)
    cu_q_lens_list = [0]
    kv_lens_list = []
    for q_len, kv_len in seq_lens:
        if q_len > kv_len:
            raise ValueError(f"Query length ({q_len}) cannot be greater than KV length ({kv_len}).")
        cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)
        kv_lens_list.append(kv_len)

    total_q_tokens = cu_q_lens_list[-1]
    max_kv_len_in_batch = max(kv_lens_list) if kv_lens_list else 0
    max_pages_per_seq = cdiv(max_kv_len_in_batch, page_size)
    q = jax.random.normal(q_key, (total_q_tokens, num_q_heads, head_dim), dtype=q_dtype)
    all_kv_pages = []
    all_page_indices = []
    page_offset = 0

    for kv_len in kv_lens_list:
        num_tokens_to_generate = cdiv(kv_len, page_size) * page_size
        kv_shape = (num_tokens_to_generate, num_kv_heads * 2, head_dim)
        kv_seq = jax.random.normal(kv_key, kv_shape, dtype=kv_dtype)
        kv_pages_seq = kv_seq.reshape(-1, page_size, num_kv_heads * 2, head_dim)
        all_kv_pages.append(kv_pages_seq)
        num_pages_for_seq = kv_pages_seq.shape[0]
        indices = page_offset + jnp.arange(num_pages_for_seq, dtype=jnp.int32)
        all_page_indices.append(indices)
        page_offset += num_pages_for_seq
    invalid_index_pad_value = -1
    padded_page_indices = [
        _pad_to_shape(pi, (max_pages_per_seq,), pad_value=invalid_index_pad_value) for pi in all_page_indices
    ]
    if padded_page_indices:
        block_tables = jnp.stack(padded_page_indices, axis=0)
    else:
        block_tables = jnp.empty((0, max_pages_per_seq), dtype=jnp.int32)
    q_padded = _pad_to_shape(q, (max_num_batched_tokens, num_q_heads, head_dim), pad_value=0.0)
    page_indices_padded = _pad_to_shape(
        block_tables,
        (max_num_seq, max_pages_per_seq),
        pad_value=invalid_index_pad_value,
    )
    cu_q_lens_padded = _pad_to_shape(jnp.array(cu_q_lens_list, dtype=jnp.int32), (max_num_seq + 1,), pad_value=0)
    kv_lens_padded = _pad_to_shape(jnp.array(kv_lens_list, dtype=jnp.int32), (max_num_seq,), pad_value=0)

    metadata = PagesMetadata(
        slot_mapping=jnp.zeros([max_num_batched_tokens], "i4"),
        block_tables=page_indices_padded,
        context_lens=kv_lens_padded,
        query_start_loc=cu_q_lens_padded,
        num_seqs=jnp.array([batch_size], dtype=jnp.int32),
        page_size=page_size,
    )

    return q_padded, metadata, max_pages_per_seq
