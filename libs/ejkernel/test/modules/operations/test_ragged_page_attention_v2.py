from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import ragged_page_attention_v2

from ._utils import assert_allclose, dense_attention_reference, device_platform


def _make_kv_pages(*, num_pages: int, page_size: int, kv_heads: int, head_dim: int, seed: int) -> jax.Array:
    k = jax.random.normal(
        jax.random.PRNGKey(seed), (num_pages, page_size, kv_heads, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    v = jax.random.normal(
        jax.random.PRNGKey(seed ^ 0xA5A5), (num_pages, page_size, kv_heads, head_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    kv = jnp.stack([k, v], axis=3).reshape(num_pages, page_size, kv_heads * 2, head_dim)
    return kv


def _gather_seq_kv(
    kv_pages: jax.Array, block_tables_row: jax.Array, *, context_len: int
) -> tuple[jax.Array, jax.Array]:  # ([kv_len, kv_heads, D], [kv_len, kv_heads, D])
    pages = kv_pages[block_tables_row]  # [pages_per_seq, page_size, kv_heads*2, D]
    pages.shape[1]
    kv_heads = pages.shape[2] // 2
    head_dim = pages.shape[3]
    flat = pages.reshape(-1, kv_heads * 2, head_dim)[:context_len]
    k = flat[:, 0::2, :]
    v = flat[:, 1::2, :]
    return k, v


def test_ragged_page_attention_v2_matches_dense_reference_and_accepts_num_seqs_int_or_array():
    num_seqs = 3
    pages_per_seq = 3
    page_size = 8
    kv_heads, q_heads, head_dim = 2, 4, 32
    num_pages = num_seqs * pages_per_seq

    context_lens = jnp.array([16, 8, 20], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 4, 5, 10], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    queries = jax.random.normal(jax.random.PRNGKey(0), (total_tokens, q_heads, head_dim), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    kv_pages = _make_kv_pages(num_pages=num_pages, page_size=page_size, kv_heads=kv_heads, head_dim=head_dim, seed=1)
    block_tables = jnp.arange(num_pages, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)

    sliding_window = 12
    logits_soft_cap = 9.0
    softmax_aux = jax.random.normal(jax.random.PRNGKey(2), (q_heads,), dtype=jnp.float32).astype(jnp.bfloat16)
    scale = head_dim**-0.5

    out_int = ragged_page_attention_v2(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=scale,
        platform="xla",
    )
    out_arr = ragged_page_attention_v2(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        jnp.array([num_seqs], dtype=jnp.int32),
        softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=scale,
        platform="xla",
    )
    assert out_int.shape == (total_tokens, q_heads, head_dim)
    assert_allclose(out_arr, out_int, atol=0.0)

    refs = []
    for i in range(num_seqs):
        q0 = int(query_start_loc[i])
        q1 = int(query_start_loc[i + 1])
        q_i = queries[q0:q1][None, ...]  # [1, q_len, q_heads, D]
        kv_len = int(context_lens[i])
        k_i, v_i = _gather_seq_kv(kv_pages, block_tables[i], context_len=kv_len)
        k_i = k_i[None, ...]
        v_i = v_i[None, ...]

        q_abs = (kv_len - (q1 - q0)) + jnp.arange(q1 - q0, dtype=jnp.int32)
        kv_pos = jnp.arange(kv_len, dtype=jnp.int32)
        mask = kv_pos[None, :] <= q_abs[:, None]
        mask = mask & (kv_pos[None, :] >= (q_abs[:, None] - sliding_window + 1))
        mask = mask[None, None, :, :]

        ref_i, _ = dense_attention_reference(
            q_i,
            k_i,
            v_i,
            attention_mask=mask,
            softmax_scale=scale,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
        )
        refs.append(ref_i[0])
    ref = jnp.concatenate(refs, axis=0)

    assert_allclose(out_int, ref, atol=0.3)


def test_ragged_page_attention_v2_optimized_flag_runs():
    num_seqs = 2
    pages_per_seq = 2
    page_size = 8
    kv_heads, q_heads, head_dim = 2, 4, 32
    num_pages = num_seqs * pages_per_seq

    context_lens = jnp.array([8, 16], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 2, 4], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    queries = jax.random.normal(jax.random.PRNGKey(3), (total_tokens, q_heads, head_dim), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    kv_pages = _make_kv_pages(num_pages=num_pages, page_size=page_size, kv_heads=kv_heads, head_dim=head_dim, seed=4)
    block_tables = jnp.arange(num_pages, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)

    out = ragged_page_attention_v2(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        None,
        optimized=True,
        platform="xla",
    )
    assert out.shape == (total_tokens, q_heads, head_dim)


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only cross-backend comparison (pallas vs xla)")
def test_ragged_page_attention_v2_pallas_matches_xla_on_tpu():
    num_seqs = 2
    pages_per_seq = 3
    page_size = 32
    kv_heads, q_heads, head_dim = 2, 8, 128
    num_pages = num_seqs * pages_per_seq

    context_lens = jnp.array([64, 96], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 4, 8], dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    queries = jax.random.normal(jax.random.PRNGKey(5), (total_tokens, q_heads, head_dim), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    kv_pages = _make_kv_pages(num_pages=num_pages, page_size=page_size, kv_heads=kv_heads, head_dim=head_dim, seed=6)
    block_tables = jnp.arange(num_pages, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)

    out_xla = ragged_page_attention_v2(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        None,
        sliding_window=12,
        logits_soft_cap=8.0,
        platform="xla",
    )
    out_pallas = ragged_page_attention_v2(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        None,
        sliding_window=12,
        logits_soft_cap=8.0,
        platform="pallas",
    )
    assert out_pallas.shape == out_xla.shape
    assert_allclose(out_pallas, out_xla, atol=0.35)
