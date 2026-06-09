import jax
import jax.numpy as jnp
import numpy as np

from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd import ref_ragged_paged_attention
from ejkernel.kernels._xla.ragged_page_attention_v3 import ragged_page_attention_v3


def test_ragged_page_attention_v3_correctness():
    jax.config.update("jax_enable_x64", False)

    # Parameters
    num_seqs = 4
    max_num_seqs = 4
    pages_per_seq = 8
    page_size = 16
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64
    q_dtype = jnp.float32
    kv_dtype = jnp.float32

    key = jax.random.PRNGKey(0)

    # Generate lengths
    # kv_lens: total length of sequence (cache + new)
    # q_lens: length of new tokens
    key, k1, k2 = jax.random.split(key, 3)
    kv_lens = jax.random.randint(k1, (num_seqs,), minval=10, maxval=pages_per_seq * page_size, dtype=jnp.int32)
    q_lens = jax.random.randint(k2, (num_seqs,), minval=1, maxval=10, dtype=jnp.int32)
    # Ensure q_len <= kv_len
    q_lens = jnp.minimum(q_lens, kv_lens)

    # Layouts
    total_q = jnp.sum(q_lens)
    query_start_loc = jnp.pad(jnp.cumsum(q_lens), (1, 0))

    # Data
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    queries = jax.random.normal(k1, (total_q, num_q_heads, head_dim), dtype=q_dtype)
    keys = jax.random.normal(k2, (total_q, num_kv_heads, head_dim), dtype=kv_dtype)
    values = jax.random.normal(k3, (total_q, num_kv_heads, head_dim), dtype=kv_dtype)

    # Cache
    # We need to initialize cache with some data (the "past" tokens)
    # But for simplicity, we can init with zeros or random,
    # and the kernel will overwrite the "new" part (tail).
    # The reference implementation also does the merge.
    # So we just provide a cache container.

    total_pages = max_num_seqs * pages_per_seq
    # Cache shape: [num_pages, page_size, num_kv_heads_x2 // pack, pack, head_dim]
    # Assuming float32, pack=1
    pack = 1
    num_kv_heads_x2 = num_kv_heads * 2
    # Pad head_dim to 128
    head_dim_padded = 128
    kv_cache = jax.random.normal(k4, (total_pages, page_size, num_kv_heads_x2, pack, head_dim_padded), dtype=kv_dtype)

    # Block tables
    block_tables = jnp.arange(total_pages, dtype=jnp.int32)
    # Reshape to [max_num_seqs * pages_per_seq]?
    # _interface says: [max_num_seqs_times_pages_per_seq]
    # _kernel says: block_tables.shape[0] // kv_lens.shape[0] is pages_per_seq
    # So it expects a flat array or [num_seqs, pages_per_seq]?
    # _kernel: `pages_per_seq = block_tables.shape[0] // kv_lens.shape[0]`
    # So it expects block_tables to be flattened [num_seqs * pages_per_seq].

    # Distribution
    # [decode_end, prefill_end, mixed_end]
    # We treat all as mixed for simplicity or just set mixed_end = num_seqs
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    # Run Reference
    # ref_ragged_paged_attention expects block_tables as [num_page_indices]
    # It also expects kv_cache with same shape.

    # Note: ref implementation might modify kv_cache inplace or return new one.
    # JAX arrays are immutable, so it returns new one.

    ref_out, ref_cache = ref_ragged_paged_attention(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_scale=1.0,
    )

    # Run Kernel
    out, cache = ragged_page_attention_v3(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_scale=1.0,
    )

    # Compare
    print("Output Shape:", out.shape)
    print("Ref Output Shape:", ref_out.shape)

    np.testing.assert_allclose(out, ref_out, rtol=1e-2, atol=1e-2)
    print("Output matches reference!")

    # Verify Cache Update
    # The cache should be updated at the tail positions.
    # We can check if cache matches ref_cache.
    np.testing.assert_allclose(cache, ref_cache, rtol=1e-2, atol=1e-2)
    print("Cache matches reference!")


if __name__ == "__main__":
    test_ragged_page_attention_v3_correctness()

    print("\nRunning Attention Sink Test...")
    # Reuse parameters but add sink
    jax.config.update("jax_enable_x64", False)

    num_seqs = 4
    max_num_seqs = 4
    pages_per_seq = 8
    page_size = 16
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64
    q_dtype = jnp.float32
    kv_dtype = jnp.float32

    key = jax.random.PRNGKey(1)

    # Generate lengths
    key, k1, k2 = jax.random.split(key, 3)
    kv_lens = jax.random.randint(k1, (num_seqs,), minval=10, maxval=pages_per_seq * page_size, dtype=jnp.int32)
    q_lens = jax.random.randint(k2, (num_seqs,), minval=1, maxval=10, dtype=jnp.int32)
    q_lens = jnp.minimum(q_lens, kv_lens)

    total_q = jnp.sum(q_lens)
    query_start_loc = jnp.pad(jnp.cumsum(q_lens), (1, 0))

    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    queries = jax.random.normal(k1, (total_q, num_q_heads, head_dim), dtype=q_dtype)
    keys = jax.random.normal(k2, (total_q, num_kv_heads, head_dim), dtype=kv_dtype)
    values = jax.random.normal(k3, (total_q, num_kv_heads, head_dim), dtype=kv_dtype)

    total_pages = max_num_seqs * pages_per_seq
    pack = 1
    num_kv_heads_x2 = num_kv_heads * 2
    head_dim_padded = 128
    kv_cache = jax.random.normal(k4, (total_pages, page_size, num_kv_heads_x2, pack, head_dim_padded), dtype=kv_dtype)

    block_tables = jnp.arange(total_pages, dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    # Attention Sink
    softmax_aux = jax.random.normal(k5, (num_q_heads,), dtype=q_dtype)

    ref_out, ref_cache = ref_ragged_paged_attention(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux=softmax_aux,
        softmax_scale=1.0,
    )

    out, cache = ragged_page_attention_v3(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux=softmax_aux,
        softmax_scale=1.0,
    )

    np.testing.assert_allclose(out, ref_out, rtol=1e-2, atol=1e-2)
    print("Output matches reference with softmax_aux!")
    np.testing.assert_allclose(cache, ref_cache, rtol=1e-2, atol=1e-2)
    print("Cache matches reference with softmax_aux!")
