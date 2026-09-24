from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.ragged_page_attention_v3 import ragged_page_attention_v3


@pytest.mark.parametrize(
    "head_dim,num_kv_heads,dtype",
    [
        (64, 2, jnp.float32),
        (32, 2, jnp.float32),  # Only exactly 64 uses K|V on the last axis.
        (128, 2, jnp.float32),
        (192, 2, jnp.float32),  # Generic layout with head-dim padding.
        (64, 1, jnp.bfloat16),  # Padding a true head count smaller than packing.
        (64, 3, jnp.bfloat16),  # Odd true head count, four storage head slots.
        (128, 3, jnp.bfloat16),
        (32, 1, jnp.bfloat16),
    ],
)
@pytest.mark.parametrize("with_sink", [False, True])
def test_ragged_page_attention_v3_correctness(head_dim, num_kv_heads, dtype, with_sink):
    """Compare public outputs and the entire packed cache to a NumPy reference."""
    rng = np.random.default_rng(0)
    num_q_heads = num_kv_heads * 4
    page_size = 4
    pages_per_seq = 4
    # Decode, full prefill, and chunked prefill; cross page/query-block boundaries.
    q_lens = np.array([1, 9, 5], dtype=np.int32)
    kv_lens = np.array([7, 9, 14], dtype=np.int32)
    query_start_loc = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(q_lens, dtype=np.int32)])
    total_q = int(query_start_loc[-1])
    # Non-contiguous physical pages plus an unreferenced page whose contents must survive.
    total_pages = len(q_lens) * pages_per_seq + 1
    block_tables = rng.permutation(total_pages - 1).astype(np.int32).reshape(len(q_lens), pages_per_seq)

    def random_array(shape):
        return jnp.asarray(rng.normal(size=shape).astype(np.float32), dtype=dtype)

    queries = random_array((total_q, num_q_heads, head_dim))
    keys = random_array((total_q, num_kv_heads, head_dim))
    values = random_array((total_q, num_kv_heads, head_dim))
    packing = 4 // np.dtype(dtype).itemsize
    combined_heads = num_kv_heads if head_dim == 64 else 2 * num_kv_heads
    packed_heads = (combined_heads + packing - 1) // packing
    padded_dim = (head_dim + 127) // 128 * 128
    cache_shape = (total_pages, page_size, packed_heads, packing, padded_dim)
    kv_cache = random_array(cache_shape)
    softmax_aux = random_array((num_q_heads,)) if with_sink else None
    softmax_scale = 1.0

    # Independent scalar-slot writes, not the implementation's merge_kv or a
    # backend reference with a different h64 layout. Copy before cache donation.
    expected_cache = np.array(kv_cache, dtype=np.float32, copy=True)
    q_np = np.asarray(queries, dtype=np.float64)
    k_np = np.asarray(keys, dtype=np.float32)
    v_np = np.asarray(values, dtype=np.float32)
    sinks_np = np.asarray(softmax_aux, dtype=np.float64) if with_sink else np.full(num_q_heads, -np.inf)
    expected_out = np.empty(queries.shape, dtype=np.float64)
    for seq, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        q_start = query_start_loc[seq]
        write_start = kv_len - q_len
        for token in range(q_len):
            pos = write_start + token
            page = block_tables[seq, pos // page_size]
            row = expected_cache[page, pos % page_size].reshape(packed_heads * packing, padded_dim)
            row[:] = 0  # New-token padding is zero; untouched cache remains unchanged.
            for head in range(num_kv_heads):
                if head_dim == 64:
                    row[head, :64] = k_np[q_start + token, head]
                    row[head, 64:] = v_np[q_start + token, head]
                else:
                    row[2 * head, :head_dim] = k_np[q_start + token, head]
                    row[2 * head + 1, :head_dim] = v_np[q_start + token, head]

        # Dense per-sequence causal attention using only true (unpadded) heads.
        dense_k = np.empty((kv_len, num_kv_heads, head_dim), dtype=np.float64)
        dense_v = np.empty_like(dense_k)
        for pos in range(kv_len):
            page = block_tables[seq, pos // page_size]
            row = expected_cache[page, pos % page_size].reshape(packed_heads * packing, padded_dim)
            for head in range(num_kv_heads):
                if head_dim == 64:
                    dense_k[pos, head] = row[head, :64]
                    dense_v[pos, head] = row[head, 64:]
                else:
                    dense_k[pos, head] = row[2 * head, :head_dim]
                    dense_v[pos, head] = row[2 * head + 1, :head_dim]
        for token in range(q_len):
            visible = write_start + token + 1
            for head in range(num_q_heads):
                kv_head = head // (num_q_heads // num_kv_heads)
                scores = dense_k[:visible, kv_head] @ q_np[q_start + token, head] * softmax_scale
                sink = sinks_np[head]
                maximum = max(float(scores.max()), sink)
                weights = np.exp(scores - maximum)
                denominator = weights.sum() + np.exp(sink - maximum)
                expected_out[q_start + token, head] = weights @ dense_v[:visible, kv_head] / denominator

    # Float32 NumPy parity requires full-fp32 products, not TPU DEFAULT's
    # reduced-precision multipliers. Scope this to the kernel invocation;
    # the bf16 cases continue to exercise the caller's ambient precision.
    precision_context = jax.default_matmul_precision("highest") if dtype == jnp.float32 else nullcontext()
    with precision_context:
        out, cache = ragged_page_attention_v3(
            queries,
            keys,
            values,
            kv_cache,
            jnp.asarray(kv_lens),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray(query_start_loc),
            jnp.array([1, 2, 3], dtype=jnp.int32),
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            num_kv_pages_per_block=2,
        )

    assert out.shape == queries.shape
    assert out.dtype == queries.dtype
    assert cache.shape == cache_shape
    assert cache.dtype == keys.dtype
    assert np.isfinite(np.asarray(out, dtype=np.float32)).all()
    np.testing.assert_allclose(np.asarray(out, dtype=np.float32), expected_out, rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(np.asarray(cache, dtype=np.float32), expected_cache)


@pytest.mark.parametrize("head_dim,stored_heads", [(64, 4), (128, 2)])
def test_ragged_page_attention_v3_rejects_wrong_cache_layout(head_dim, stored_heads):
    """Reject double-counted h64 pairs and missing generic pairs at the API boundary."""
    queries = jnp.zeros((1, 8, head_dim), dtype=jnp.float32)
    keys = jnp.zeros((1, 2, head_dim), dtype=jnp.float32)
    with pytest.raises(ValueError, match="cache packed head axis must be"):
        ragged_page_attention_v3(
            queries,
            keys,
            keys,
            jnp.zeros((1, 4, stored_heads, 1, 128), dtype=jnp.float32),
            jnp.array([1], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.array([1, 1, 1], dtype=jnp.int32),
        )
