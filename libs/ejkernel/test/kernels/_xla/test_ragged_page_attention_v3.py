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


def _dense_sliding_window_reference(q, k, v, q_pos, sliding_window, sinks, softmax_scale):
    """Dense masked softmax with HF sliding-window semantics (query plus ``W - 1`` previous keys).

    Args:
        q: ``[q_len, num_q_heads, head_dim]`` queries.
        k: ``[kv_len, num_kv_heads, head_dim]`` keys for every cached position.
        v: ``[kv_len, num_kv_heads, head_dim]`` values for every cached position.
        q_pos: ``[q_len]`` absolute query positions.
        sliding_window: Window size ``W`` or ``None`` for full causal attention.
        sinks: ``[num_q_heads]`` sink logits (``-inf`` disables).
        softmax_scale: Logit scale.
    """
    kv_pos = np.arange(k.shape[0])
    visible = kv_pos[None, :] <= q_pos[:, None]
    if sliding_window is not None:
        visible &= kv_pos[None, :] > q_pos[:, None] - sliding_window  # transformers sliding_window_overlay
    group = q.shape[1] // k.shape[1]
    out = np.empty(q.shape, dtype=np.float64)
    for head in range(q.shape[1]):
        scores = q[:, head] @ k[:, head // group].T * softmax_scale
        scores = np.where(visible, scores, -np.inf)
        maximum = np.maximum(scores.max(axis=-1), sinks[head])
        weights = np.exp(scores - maximum[:, None])
        denominator = weights.sum(axis=-1) + np.exp(sinks[head] - maximum)
        out[:, head] = weights @ v[:, head // group] / denominator[:, None]
    return out


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("sliding_window", [1, 4, 8, 16, 33])
@pytest.mark.parametrize("with_sink", [False, True])
def test_ragged_page_attention_v3_sliding_window_matches_dense(head_dim, sliding_window, with_sink):
    """Every query of a prefill chunk keeps its own ``W``-key window.

    Covers decode over a long context, a full prefill (``q_len == kv_len``), a chunked prefill
    (``kv_len > q_len``) and a long chunk whose early KV blocks lie entirely outside the window
    (8-token KV blocks), for windows smaller than, equal to and larger than a KV block.
    """
    rng = np.random.default_rng(sliding_window + 100 * head_dim + int(with_sink))
    num_kv_heads, num_q_heads = 2, 4
    page_size = 4
    q_lens = np.array([1, 24, 13, 37], dtype=np.int32)
    kv_lens = np.array([70, 24, 45, 97], dtype=np.int32)
    pages_per_seq = int(-(-kv_lens.max() // page_size))
    query_start_loc = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(q_lens, dtype=np.int32)])
    total_q = int(query_start_loc[-1])
    total_pages = len(q_lens) * pages_per_seq
    block_tables = rng.permutation(total_pages).astype(np.int32).reshape(len(q_lens), pages_per_seq)

    queries = rng.normal(size=(total_q, num_q_heads, head_dim)).astype(np.float32)
    keys = rng.normal(size=(total_q, num_kv_heads, head_dim)).astype(np.float32)
    values = rng.normal(size=(total_q, num_kv_heads, head_dim)).astype(np.float32)
    combined_heads = num_kv_heads if head_dim == 64 else 2 * num_kv_heads
    padded_dim = (head_dim + 127) // 128 * 128
    kv_cache = rng.normal(size=(total_pages, page_size, combined_heads, 1, padded_dim)).astype(np.float32)
    sinks = rng.normal(size=(num_q_heads,)) if with_sink else np.full(num_q_heads, -np.inf)
    softmax_scale = head_dim**-0.5

    expected = np.empty(queries.shape, dtype=np.float64)
    for seq, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        q_start, write_start = query_start_loc[seq], kv_len - q_len
        dense_k = np.empty((kv_len, num_kv_heads, head_dim), dtype=np.float64)
        dense_v = np.empty_like(dense_k)
        for pos in range(write_start):  # cached prefix, read back from the (unchanged) pages
            row = kv_cache[block_tables[seq, pos // page_size], pos % page_size, :, 0]
            for head in range(num_kv_heads):
                if head_dim == 64:
                    dense_k[pos, head], dense_v[pos, head] = row[head, :64], row[head, 64:]
                else:
                    dense_k[pos, head] = row[2 * head, :head_dim]
                    dense_v[pos, head] = row[2 * head + 1, :head_dim]
        dense_k[write_start:] = keys[q_start : q_start + q_len]
        dense_v[write_start:] = values[q_start : q_start + q_len]
        expected[q_start : q_start + q_len] = _dense_sliding_window_reference(
            queries[q_start : q_start + q_len].astype(np.float64),
            dense_k,
            dense_v,
            write_start + np.arange(q_len),
            sliding_window,
            sinks,
            softmax_scale,
        )

    with jax.default_matmul_precision("highest"):
        out, _ = ragged_page_attention_v3(
            jnp.asarray(queries),
            jnp.asarray(keys),
            jnp.asarray(values),
            jnp.asarray(kv_cache),
            jnp.asarray(kv_lens),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray(query_start_loc),
            jnp.array([0, 0, len(q_lens)], dtype=jnp.int32),
            softmax_aux=jnp.asarray(sinks, dtype=jnp.float32) if with_sink else None,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            num_kv_pages_per_block=2,
        )

    out = np.asarray(out, dtype=np.float32)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)
