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

"""Sparse MLA ragged paged attention (XLA) vs an independent float64 numpy reference.

The reference never touches pages: it keeps each sequence's full latent history as a
plain array and applies the selection/causal mask directly. The kernel under test must
resolve the same tokens through a randomly permuted page table.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.multi_latent_ragged_page_attention_v2 import multi_latent_ragged_page_attention_v2
from ejkernel.kernels._xla.sparse_multi_latent_ragged_page_attention import (
    sparse_multi_latent_ragged_page_attention,
)

NOPE, PE, HEADS = 32, 16, 4
PAGE, PAGES_PER_SEQ = 16, 4
CACHE_DIM = 256  # align(NOPE, 128) + align(PE, 128)
PAD_TOKENS = 3


def _build(seed, kv_lens, q_lens, *, chunk_size, width, dtype=jnp.float32):
    rng = np.random.default_rng(seed)
    num_live = len(kv_lens)
    kv_lens_all = [*kv_lens, 0]  # one extra empty slot: max_num_seqs > live seqs
    q_lens_all = [*q_lens, 0]
    max_seqs = len(kv_lens_all)
    pack = 32 // (jnp.dtype(dtype).itemsize * 8)
    num_pages = max_seqs * PAGES_PER_SEQ + 2
    block_tables = rng.permutation(num_pages)[: max_seqs * PAGES_PER_SEQ].astype(np.int32)

    def rnd(*shape):
        return np.asarray(jnp.asarray(rng.standard_normal(shape), dtype).astype(jnp.float32), np.float64)

    cache = rnd(num_pages, PAGE // pack, pack, CACHE_DIM)
    flat = cache.reshape(num_pages * PAGE, CACHE_DIM)
    hist_nope = [rnd(kl, NOPE) for kl in kv_lens]
    hist_pe = [rnd(kl, PE) for kl in kv_lens]
    for s, (kl, ql) in enumerate(zip(kv_lens, q_lens, strict=True)):
        for j in range(kl - ql):  # already-cached history
            phys = block_tables[s * PAGES_PER_SEQ + j // PAGE]
            flat[phys * PAGE + j % PAGE, :NOPE] = hist_nope[s][j]
            flat[phys * PAGE + j % PAGE, 128 : 128 + PE] = hist_pe[s][j]
    cache = flat.reshape(cache.shape)

    total = sum(q_lens) + PAD_TOKENS
    qsl = np.concatenate([[0], np.cumsum(q_lens_all)]).astype(np.int32)
    seq_of, pos_of = [], []
    for s, (kl, ql) in enumerate(zip(kv_lens, q_lens, strict=True)):
        seq_of += [s] * ql
        pos_of += list(range(kl - ql, kl))
    q_nope = rng.standard_normal((total, HEADS, NOPE)).astype(np.float32)
    q_pe = rng.standard_normal((total, HEADS, PE)).astype(np.float32)
    k_nope = rng.standard_normal((total, NOPE)).astype(np.float32)
    k_pe = rng.standard_normal((total, PE)).astype(np.float32)
    for t, (s, p) in enumerate(zip(seq_of, pos_of, strict=True)):
        k_nope[t] = hist_nope[s][p]
        k_pe[t] = hist_pe[s][p]

    num_chunks = PAGES_PER_SEQ * PAGE // chunk_size
    selected = np.full((total, width), -1, np.int32)
    for t in range(total):
        n = rng.integers(0, width + 1)
        ids = rng.choice(num_chunks, size=n, replace=False)
        if t < len(pos_of) and rng.random() < 0.7:  # usually include the query's own chunk
            own = pos_of[t] // chunk_size
            ids = np.unique(np.concatenate([ids[: max(n - 1, 0)], [own]]))[:width]
        selected[t, : len(ids)] = rng.permutation(ids)
    return {
        "inputs": (
            jnp.asarray(q_nope),
            jnp.asarray(q_pe),
            jnp.asarray(k_nope),
            jnp.asarray(k_pe),
            jnp.asarray(cache, dtype),
            jnp.asarray(kv_lens_all, jnp.int32),
            jnp.asarray(block_tables),
            jnp.asarray(qsl),
            jnp.asarray([0, 0, num_live], jnp.int32),
            jnp.asarray(selected),
        ),
        "cache_np": cache,
        "hist": (hist_nope, hist_pe),
        "seq_of": seq_of,
        "pos_of": pos_of,
        "block_tables": block_tables,
        "selected": selected,
        "kv_lens": kv_lens,
    }


def _reference(case, *, chunk_size, scale, sliding_window=None, soft_cap=None):
    q_nope, q_pe = (np.asarray(x, np.float64) for x in case["inputs"][:2])
    hist_nope, hist_pe = case["hist"]
    out = np.zeros(q_nope.shape, np.float64)
    for t, (s, p) in enumerate(zip(case["seq_of"], case["pos_of"], strict=True)):
        chunks = set(int(c) for c in case["selected"][t] if c >= 0)
        js = [j for j in range(case["kv_lens"][s]) if j // chunk_size in chunks and j <= p]
        if sliding_window is not None:
            js = [j for j in js if j > p - sliding_window]
        if not js:
            continue
        kn, kp = hist_nope[s][js], hist_pe[s][js]
        logits = (q_nope[t] @ kn.T + q_pe[t] @ kp.T) * scale
        if soft_cap is not None:
            logits = soft_cap * np.tanh(logits / soft_cap)
        w = np.exp(logits - logits.max(-1, keepdims=True))
        out[t] = (w / w.sum(-1, keepdims=True)) @ kn
    return out


def _expected_cache(case):
    flat = case["cache_np"].reshape(-1, CACHE_DIM).copy()
    k_nope, k_pe = (np.asarray(x, np.float64) for x in case["inputs"][2:4])
    for t, (s, p) in enumerate(zip(case["seq_of"], case["pos_of"], strict=True)):
        row = case["block_tables"][s * PAGES_PER_SEQ + p // PAGE] * PAGE + p % PAGE
        flat[row] = 0.0
        flat[row, :NOPE] = k_nope[t]
        flat[row, 128 : 128 + PE] = k_pe[t]
    return flat


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("chunk_size,width", [(4, 6), (1, 20), (16, 2)])
def test_sparse_mla_matches_dense_masked_reference(dtype, chunk_size, width):
    # decode (1), partial prefill crossing a page boundary (5), full prefill from position 0 (17)
    case = _build(0, kv_lens=[40, 23, 17], q_lens=[1, 5, 17], chunk_size=chunk_size, width=width, dtype=dtype)
    scale = (NOPE + PE) ** -0.5
    with jax.default_matmul_precision("highest"):
        out, cache = sparse_multi_latent_ragged_page_attention(*case["inputs"], chunk_size=chunk_size)
    expected = _reference(case, chunk_size=chunk_size, scale=scale)
    np.testing.assert_allclose(np.asarray(out, np.float64), expected, atol=2e-4, rtol=2e-4)
    # padding tokens (beyond the live sequences) produce zeros
    assert np.all(np.asarray(out)[-PAD_TOKENS:] == 0)
    got_cache = np.asarray(jnp.asarray(cache, jnp.float32), np.float64).reshape(-1, CACHE_DIM)
    np.testing.assert_allclose(
        got_cache, np.asarray(jnp.asarray(_expected_cache(case), dtype).astype(jnp.float32)), atol=0, rtol=0
    )


def test_sparse_mla_sliding_window_and_soft_cap():
    case = _build(3, kv_lens=[40, 23, 17], q_lens=[1, 5, 17], chunk_size=4, width=8)
    scale = 0.3
    with jax.default_matmul_precision("highest"):
        out, _ = sparse_multi_latent_ragged_page_attention(
            *case["inputs"], chunk_size=4, softmax_scale=scale, sliding_window=12, logits_soft_cap=5.0
        )
    expected = _reference(case, chunk_size=4, scale=scale, sliding_window=12, soft_cap=5.0)
    np.testing.assert_allclose(np.asarray(out, np.float64), expected, atol=2e-4, rtol=2e-4)


def test_full_selection_equals_dense_v2_attention_and_cache():
    """Selecting every chunk must reproduce the dense MLA ragged kernel (output and cache write)."""
    case = _build(5, kv_lens=[40, 23, 17], q_lens=[1, 5, 17], chunk_size=4, width=16)
    inputs = list(case["inputs"])
    inputs[9] = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), inputs[9].shape)
    with jax.default_matmul_precision("highest"):
        sparse_out, sparse_cache = sparse_multi_latent_ragged_page_attention(
            *[jnp.array(x) for x in inputs], chunk_size=4
        )
        dense_out, dense_cache = multi_latent_ragged_page_attention_v2(*[jnp.array(x) for x in inputs[:9]])
    live = sum([1, 5, 17])
    np.testing.assert_allclose(np.asarray(sparse_out)[:live], np.asarray(dense_out)[:live], atol=2e-4, rtol=2e-4)
    np.testing.assert_array_equal(np.asarray(sparse_cache), np.asarray(dense_cache))


def test_all_padding_selection_gives_zero_output():
    case = _build(7, kv_lens=[40], q_lens=[3], chunk_size=4, width=4)
    inputs = list(case["inputs"])
    inputs[9] = jnp.full_like(inputs[9], -1)
    out, _ = sparse_multi_latent_ragged_page_attention(*inputs, chunk_size=4)
    assert np.all(np.asarray(out) == 0)


def test_tokens_before_first_query_start_are_ignored():
    """DP-local slices pass a nonzero ``query_start_loc[0]``; earlier tokens belong to other shards.

    They must produce zero output and must not be written into this shard's cache.
    """
    case = _build(9, kv_lens=[40, 23], q_lens=[1, 5], chunk_size=4, width=6)
    lead = 3
    with jax.default_matmul_precision("highest"):
        base_out, base_cache = sparse_multi_latent_ragged_page_attention(
            *[jnp.array(x) for x in case["inputs"]], chunk_size=4
        )
        inputs = [jnp.array(x) for x in case["inputs"]]
        for i in range(4):
            inputs[i] = jnp.concatenate([jnp.full((lead, *inputs[i].shape[1:]), 7.0, inputs[i].dtype), inputs[i]])
        inputs[7] = inputs[7] + lead
        inputs[9] = jnp.concatenate([jnp.zeros((lead, inputs[9].shape[1]), jnp.int32), inputs[9]])
        out, cache = sparse_multi_latent_ragged_page_attention(*inputs, chunk_size=4)
    assert np.all(np.asarray(out)[:lead] == 0)
    np.testing.assert_allclose(np.asarray(out)[lead:], np.asarray(base_out), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(cache), np.asarray(base_cache))


def test_registered_xla_fallback():
    impl = kernel_registry.get("sparse_multi_latent_ragged_page_attention", platform="xla", backend="any")
    assert impl is not None
