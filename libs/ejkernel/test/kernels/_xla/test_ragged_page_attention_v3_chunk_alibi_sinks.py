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

"""Ragged paged attention v3: chunked local attention, ALiBi and multi-sink layouts.

Every case is compared with an independent float64 dense reference (a plain masked softmax over the
fully materialised per-request KV history, sinks modelled as extra zero-valued keys) and carries a
negative control proving the feature changes the output. The XLA reference and the Pallas TPU kernel
run the same cases; Pallas is skipped off TPU. Computation tests must run on an accelerator.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3 import ragged_page_attention_v3 as pallas_v3
from ejkernel.kernels._xla.ragged_page_attention_v3 import ragged_page_attention_v3 as xla_v3

PAGE_SIZE = 16
NUM_Q_HEADS, NUM_KV_HEADS = 8, 2
Q_LENS = np.array([1, 24, 13, 37], dtype=np.int32)
KV_LENS = np.array([70, 24, 45, 97], dtype=np.int32)


def _impl(name):
    if name == "pallas":
        if jax.default_backend() != "tpu":
            pytest.skip("Pallas TPU kernel needs a TPU backend")
        return pallas_v3
    return xla_v3


def _build(head_dim, seed):
    """Random ragged batch: inputs, cache, page table, plus dense per-request K/V histories."""
    rng = np.random.default_rng(seed)
    pages_per_seq = int(-(-KV_LENS.max() // PAGE_SIZE))
    qsl = np.concatenate([np.zeros(1, np.int32), np.cumsum(Q_LENS, dtype=np.int32)])
    total_q = int(qsl[-1])
    total_pages = len(Q_LENS) * pages_per_seq
    tables = rng.permutation(total_pages).astype(np.int32).reshape(len(Q_LENS), pages_per_seq)
    queries = rng.normal(size=(total_q, NUM_Q_HEADS, head_dim)).astype(np.float32)
    keys = rng.normal(size=(total_q, NUM_KV_HEADS, head_dim)).astype(np.float32)
    values = rng.normal(size=(total_q, NUM_KV_HEADS, head_dim)).astype(np.float32)
    combined = NUM_KV_HEADS if head_dim == 64 else 2 * NUM_KV_HEADS
    padded = (head_dim + 127) // 128 * 128
    cache = rng.normal(size=(total_pages, PAGE_SIZE, combined, 1, padded)).astype(np.float32)
    histories = []
    for seq, (q_len, kv_len) in enumerate(zip(Q_LENS, KV_LENS, strict=True)):
        write_start = kv_len - q_len
        dense_k = np.empty((kv_len, NUM_KV_HEADS, head_dim), np.float64)
        dense_v = np.empty_like(dense_k)
        for pos in range(write_start):  # cached prefix, read back from the untouched pages
            row = cache[tables[seq, pos // PAGE_SIZE], pos % PAGE_SIZE, :, 0]
            for head in range(NUM_KV_HEADS):
                if head_dim == 64:
                    dense_k[pos, head], dense_v[pos, head] = row[head, :64], row[head, 64:]
                else:
                    dense_k[pos, head] = row[2 * head, :head_dim]
                    dense_v[pos, head] = row[2 * head + 1, :head_dim]
        dense_k[write_start:] = keys[qsl[seq] : qsl[seq] + q_len]
        dense_v[write_start:] = values[qsl[seq] : qsl[seq] + q_len]
        histories.append((dense_k, dense_v))
    return queries, keys, values, cache, tables, qsl, histories


def _dense_reference(q, k, v, q_pos, *, scale, chunk=None, slopes=None, sinks=None):
    """Masked softmax over the full history; ``sinks`` is ``[Hq, S]`` extra zero-valued keys."""
    kv_pos = np.arange(k.shape[0])
    visible = kv_pos[None, :] <= q_pos[:, None]
    if chunk is not None:
        visible &= (kv_pos[None, :] // chunk) == (q_pos[:, None] // chunk)
    group = q.shape[1] // k.shape[1]
    out = np.empty(q.shape, np.float64)
    for head in range(q.shape[1]):
        scores = q[:, head] @ k[:, head // group].T * scale
        if slopes is not None:
            scores = scores + slopes[head] * kv_pos[None, :]
        scores = np.where(visible, scores, -np.inf)
        num_sinks = 0
        if sinks is not None:
            num_sinks = sinks.shape[1]
            scores = np.concatenate([np.tile(sinks[head][None, :], (scores.shape[0], 1)), scores], axis=1)
        scores = scores - scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=1, keepdims=True)
        out[:, head] = weights[:, num_sinks:] @ v[:, head // group]
    return out


def _expected(queries, qsl, histories, **kwargs):
    out = np.empty(queries.shape, np.float64)
    for seq, (q_len, kv_len) in enumerate(zip(Q_LENS, KV_LENS, strict=True)):
        dense_k, dense_v = histories[seq]
        sl = slice(qsl[seq], qsl[seq] + q_len)
        out[sl] = _dense_reference(
            queries[sl].astype(np.float64), dense_k, dense_v, (kv_len - q_len) + np.arange(q_len), **kwargs
        )
    return out


def _run(impl, head_dim, batch, **kwargs):
    queries, keys, values, cache, tables, qsl, _ = batch
    with jax.default_matmul_precision("highest"):
        out, _ = impl(
            jnp.asarray(queries),
            jnp.asarray(keys),
            jnp.asarray(values),
            jnp.asarray(cache),
            jnp.asarray(KV_LENS),
            jnp.asarray(tables.reshape(-1)),
            jnp.asarray(qsl),
            jnp.array([0, 0, len(Q_LENS)], dtype=jnp.int32),
            softmax_scale=head_dim**-0.5,
            num_kv_pages_per_block=1,
            **kwargs,
        )
    return np.asarray(out, np.float32)


@pytest.mark.parametrize("impl_name", ["xla", "pallas"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("chunk_size", [8, 16, 40])
def test_chunked_local_attention_matches_dense(impl_name, head_dim, chunk_size):
    """``k <= q`` and ``k // C == q // C``; chunk starts fall inside and across 16-token KV blocks."""
    impl = _impl(impl_name)
    batch = _build(head_dim, seed=chunk_size + head_dim)
    scale = head_dim**-0.5
    expected = _expected(batch[0], batch[5], batch[6], scale=scale, chunk=chunk_size)
    out = _run(impl, head_dim, batch, chunk_size=chunk_size)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)
    # Negative control: plain causal attention is a different function once the context exceeds one chunk.
    causal = _expected(batch[0], batch[5], batch[6], scale=scale)
    assert np.abs(causal - expected).max() > 5e-2
    assert np.abs(out - causal).max() > 5e-2


@pytest.mark.parametrize("impl_name", ["xla", "pallas"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_chunk_larger_than_context_is_plain_causal(impl_name, head_dim):
    """A chunk covering every position must not change the result."""
    impl = _impl(impl_name)
    batch = _build(head_dim, seed=7)
    expected = _expected(batch[0], batch[5], batch[6], scale=head_dim**-0.5)
    out = _run(impl, head_dim, batch, chunk_size=4096)
    np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("impl_name", ["xla", "pallas"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_alibi_matches_dense(impl_name, head_dim):
    """Kernel adds ``slope * (k - q)``; the reference adds ``slope * k`` (softmax-equivalent)."""
    impl = _impl(impl_name)
    batch = _build(head_dim, seed=11 + head_dim)
    slopes = np.array([2.0 ** (-(i + 1)) for i in range(NUM_Q_HEADS)], np.float32)
    scale = head_dim**-0.5
    expected = _expected(batch[0], batch[5], batch[6], scale=scale, slopes=slopes.astype(np.float64))
    out = _run(impl, head_dim, batch, alibi_slopes=jnp.asarray(slopes))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)
    # Negative control: without slopes the output differs materially.
    plain = _expected(batch[0], batch[5], batch[6], scale=scale)
    assert np.abs(plain - expected).max() > 5e-2
    assert np.abs(_run(impl, head_dim, batch) - out).max() > 5e-2


@pytest.mark.parametrize("impl_name", ["xla", "pallas"])
def test_chunk_and_alibi_combine(impl_name):
    impl = _impl(impl_name)
    head_dim = 128
    batch = _build(head_dim, seed=23)
    slopes = np.array([2.0 ** (-(i + 1)) for i in range(NUM_Q_HEADS)], np.float32)
    expected = _expected(batch[0], batch[5], batch[6], scale=head_dim**-0.5, chunk=16, slopes=slopes.astype(np.float64))
    out = _run(impl, head_dim, batch, chunk_size=16, alibi_slopes=jnp.asarray(slopes))
    np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("chunk_size", [8, 16])
def test_chunked_prefill_writes_every_new_token_to_cache(head_dim, chunk_size):
    """With several query blocks, the cache-writing (last) block must still store tokens outside its chunk.

    Regression for r10 S02-3: the chunk-local page skip made the last query block start past earlier
    new-token blocks, so those tokens were never written to the returned cache.
    """
    pallas = _impl("pallas")
    batch = _build(head_dim, seed=41 + chunk_size)
    queries, keys, values, cache, tables, qsl, histories = batch

    def run(impl, **kwargs):
        with jax.default_matmul_precision("highest"):
            out, new_cache = impl(
                jnp.asarray(queries),
                jnp.asarray(keys),
                jnp.asarray(values),
                jnp.asarray(cache),
                jnp.asarray(KV_LENS),
                jnp.asarray(tables.reshape(-1)),
                jnp.asarray(qsl),
                jnp.array([0, 0, len(Q_LENS)], dtype=jnp.int32),
                softmax_scale=head_dim**-0.5,
                num_kv_pages_per_block=1,
                num_queries_per_block=8,
                **kwargs,
            )
        return np.asarray(out, np.float32), np.asarray(new_cache, np.float32)

    out, pallas_cache = run(pallas, chunk_size=chunk_size)
    _, xla_cache = run(xla_v3, chunk_size=chunk_size)
    _, unchunked_cache = run(pallas)
    # The cache write is independent of the attention mask: all three must store every new token.
    np.testing.assert_array_equal(pallas_cache, xla_cache)
    np.testing.assert_array_equal(pallas_cache, unchunked_cache)
    expected = _expected(queries, qsl, histories, scale=head_dim**-0.5, chunk=chunk_size)
    np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("impl_name", ["pallas", "xla"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_queries_per_block", [8, 32])
@pytest.mark.parametrize("chunk_size", [8, 16])
def test_chunked_prefill_with_infinite_mask_value_is_finite(impl_name, head_dim, num_queries_per_block, chunk_size):
    """``mask_value=-inf`` with no sinks must not turn fully masked KV blocks into NaN.

    hd128 (r11 S02-1): the cache-writing last query block traverses KV blocks outside its chunk.
    h64 (r12 S02-1): a query tile spanning several chunks starts at its earliest query's block, so
    later rows see fully masked leading blocks. With an infinite mask and no sink the running max is
    ``-inf`` there, and ``exp(-inf - -inf)`` poisoned the accumulator.
    XLA (r14 S04-1): the same chunk mask fully masks leading tiles of the XLA online softmax.
    """
    if impl_name == "xla":
        from ejkernel.kernels._xla.ragged_page_attention_v3._xla_impl_fwd import ragged_paged_attention
    elif (_impl("pallas") is not None) and head_dim == 64:  # _impl skips off TPU
        from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd_h64 import (
            ragged_paged_attention,
        )
    else:
        from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd import ragged_paged_attention

    batch = _build(head_dim, seed=53 + chunk_size + head_dim)
    queries, keys, values, cache, tables, qsl, histories = batch

    def run(mask_value):
        kwargs = {} if mask_value is None else {"mask_value": mask_value}
        with jax.default_matmul_precision("highest"):
            out, _ = ragged_paged_attention(
                jnp.asarray(queries),
                jnp.asarray(keys),
                jnp.asarray(values),
                jnp.asarray(cache),
                jnp.asarray(KV_LENS),
                jnp.asarray(tables.reshape(-1)),
                jnp.asarray(qsl),
                jnp.array([0, 0, len(Q_LENS)], dtype=jnp.int32),
                None,
                softmax_scale=head_dim**-0.5,
                num_kv_pages_per_block=1,
                num_queries_per_block=num_queries_per_block,
                chunk_size=chunk_size,
                **kwargs,
            )
        return np.asarray(out, np.float32)

    out = run(float("-inf"))
    assert np.isfinite(out).all(), "infinite mask value produced non-finite attention output"
    expected = _expected(queries, qsl, histories, scale=head_dim**-0.5, chunk=chunk_size)
    np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)
    # The default finite mask must give the same answer.
    np.testing.assert_allclose(out, run(None), rtol=1e-5, atol=1e-5)


def _sink_layouts(rng, num_sinks=3):
    """Kernel-facing sink layout -> per-query-head ``[Hq, S]`` array used by the reference."""
    group = NUM_Q_HEADS // NUM_KV_HEADS
    per_head = rng.normal(size=(NUM_Q_HEADS, num_sinks))
    per_kv = rng.normal(size=(NUM_KV_HEADS, num_sinks))
    shared = rng.normal(size=(1, num_sinks))
    flat_per_head = rng.normal(size=(NUM_Q_HEADS,))
    return {
        "hq": (flat_per_head, flat_per_head[:, None]),
        "hq_by_s": (per_head, per_head),
        "hkv_by_s": (per_kv, np.repeat(per_kv, group, axis=0)),
        "one_by_s": (shared, np.repeat(shared, NUM_Q_HEADS, axis=0)),
        "shared_1d": (shared[0], np.repeat(shared, NUM_Q_HEADS, axis=0)),
    }


@pytest.mark.parametrize("impl_name", ["xla", "pallas"])
@pytest.mark.parametrize("layout", ["hq", "hq_by_s", "hkv_by_s", "one_by_s", "shared_1d"])
def test_multi_sink_layouts_match_dense(impl_name, layout):
    """S sinks per head equal one logsumexp sink; the reference treats them as S extra zero-value keys."""
    impl = _impl(impl_name)
    head_dim = 128
    batch = _build(head_dim, seed=31)
    kernel_sinks, ref_sinks = _sink_layouts(np.random.default_rng(5))[layout]
    scale = head_dim**-0.5
    expected = _expected(batch[0], batch[5], batch[6], scale=scale, sinks=ref_sinks)
    out = _run(impl, head_dim, batch, softmax_aux=jnp.asarray(kernel_sinks, jnp.float32))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)
    # Negative control: sinks change the output relative to no sinks.
    no_sink = _expected(batch[0], batch[5], batch[6], scale=scale)
    assert np.abs(no_sink - expected).max() > 1e-2
