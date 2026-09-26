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

"""Smoke tests for XLA ragged_page_attention_v3_turboquant."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels import Platform, kernel_registry
from ejkernel.kernels._xla import ragged_page_attention_v3_turboquant


def test_registry_and_export_for_ragged_page_attention_v3_turboquant():
    impl = kernel_registry.get("ragged_page_attention_v3_turboquant", platform=Platform.XLA)
    assert callable(impl)
    assert callable(ragged_page_attention_v3_turboquant)


def test_ragged_page_attention_v3_turboquant_smoke_run():
    head_dim = 8
    qjl_dim = 8

    queries = jnp.arange(16, dtype=jnp.bfloat16).reshape(2, 1, head_dim) / 10
    keys = queries + jnp.asarray(0.5, dtype=jnp.bfloat16)
    values = queries - jnp.asarray(0.25, dtype=jnp.bfloat16)

    key_indices_pages = jnp.zeros((1, 2, 1, head_dim // 2), dtype=jnp.uint8)
    key_signs_pages = jnp.zeros((1, 2, 1, qjl_dim // 8), dtype=jnp.uint8)
    key_norms_pages = jnp.zeros((1, 2, 1, 2), dtype=jnp.bfloat16)
    value_indices_pages = jnp.zeros((1, 2, 1, head_dim // 2), dtype=jnp.uint8)
    value_norms_pages = jnp.zeros((1, 2, 1), dtype=jnp.bfloat16)

    kv_lens = jnp.array([2], dtype=jnp.int32)
    block_tables = jnp.array([0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 2], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    rotation_matrix = jnp.eye(head_dim, dtype=jnp.float32)
    qjl_projection = jnp.eye(qjl_dim, head_dim, dtype=jnp.float32)
    key_codebook = jnp.linspace(-1.0, 1.0, 2 ** (4 - 1), dtype=jnp.float32)
    value_codebook = jnp.linspace(-1.0, 1.0, 2**4, dtype=jnp.float32)

    output = ragged_page_attention_v3_turboquant(
        queries,
        keys,
        values,
        key_indices_pages,
        key_signs_pages,
        key_norms_pages,
        value_indices_pages,
        value_norms_pages,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        rotation_matrix,
        qjl_projection,
        key_codebook,
        value_codebook,
        qjl_dim=qjl_dim,
    )

    assert isinstance(output, tuple)
    assert len(output) == 6
    assert output[0].shape == queries.shape
    assert output[0].dtype == queries.dtype
    assert output[1].shape == key_indices_pages.shape
    assert output[1].dtype == key_indices_pages.dtype
    assert output[2].shape == key_signs_pages.shape
    assert output[2].dtype == key_signs_pages.dtype
    assert output[3].shape == key_norms_pages.shape
    assert output[4].shape == value_indices_pages.shape
    assert output[4].dtype == value_indices_pages.dtype
    assert output[5].shape == value_norms_pages.shape


def test_ragged_page_attention_v3_turboquant_accepts_interface_parity_kwargs():
    head_dim = 8
    qjl_dim = 8

    queries = jnp.arange(16, dtype=jnp.bfloat16).reshape(2, 1, head_dim) / 10
    keys = queries + jnp.asarray(0.5, dtype=jnp.bfloat16)
    values = queries - jnp.asarray(0.25, dtype=jnp.bfloat16)

    key_indices_pages = jnp.zeros((1, 2, 1, head_dim // 2), dtype=jnp.uint8)
    key_signs_pages = jnp.zeros((1, 2, 1, qjl_dim // 8), dtype=jnp.uint8)
    key_norms_pages = jnp.zeros((1, 2, 1, 2), dtype=jnp.bfloat16)
    value_indices_pages = jnp.zeros((1, 2, 1, head_dim // 2), dtype=jnp.uint8)
    value_norms_pages = jnp.zeros((1, 2, 1), dtype=jnp.bfloat16)

    kv_lens = jnp.array([2], dtype=jnp.int32)
    block_tables = jnp.array([0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 2], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    rotation_matrix = jnp.eye(head_dim, dtype=jnp.float32)
    qjl_projection = jnp.eye(qjl_dim, head_dim, dtype=jnp.float32)
    key_codebook = jnp.linspace(-1.0, 1.0, 2 ** (4 - 1), dtype=jnp.float32)
    value_codebook = jnp.linspace(-1.0, 1.0, 2**4, dtype=jnp.float32)

    output = ragged_page_attention_v3_turboquant(
        queries,
        keys,
        values,
        key_indices_pages,
        key_signs_pages,
        key_norms_pages,
        value_indices_pages,
        value_norms_pages,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        rotation_matrix,
        qjl_projection,
        key_codebook,
        value_codebook,
        chunk_prefill_size=16,
        vmem_limit_bytes=1 << 20,
        qjl_dim=qjl_dim,
    )

    assert isinstance(output, tuple)
    assert len(output) == 6
    assert output[0].shape == queries.shape
    assert output[1].shape == key_indices_pages.shape
    assert output[2].shape == key_signs_pages.shape
    assert output[3].shape == key_norms_pages.shape
    assert output[4].shape == value_indices_pages.shape
    assert output[5].shape == value_norms_pages.shape


def _tq_unpack_dense(ki, ks, kn, vi, vn, key_codebook, value_codebook, rotation):
    """Decode TurboQuant pages into per-token logit terms with plain NumPy (independent of the kernel).

    Returns ``(key_centroids * ||k||, signs * ||r_k||, values)`` so that the kernel's estimator is
    ``<R q, key_term> + sqrt(pi/2)/qjl_dim * <P q, sign_term>``.
    """
    ki, ks, vi = np.asarray(ki), np.asarray(ks), np.asarray(vi)
    kn, vn = np.asarray(kn, dtype=np.float64), np.asarray(vn, dtype=np.float64)

    def unpack_nibbles(packed):
        out = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.int64)
        out[..., 0::2], out[..., 1::2] = packed & 0x0F, packed >> 4
        return out

    bits = (ks[..., :, None] >> np.arange(8)) & 1
    signs = np.where(bits.reshape(*ks.shape[:-1], ks.shape[-1] * 8) > 0, 1.0, -1.0)
    key_term = np.asarray(key_codebook, dtype=np.float64)[unpack_nibbles(ki)] * kn[..., 0:1]
    sign_term = signs * kn[..., 1:2]
    values = (np.asarray(value_codebook, dtype=np.float64)[unpack_nibbles(vi)] @ np.asarray(rotation)) * vn[..., None]
    return key_term, sign_term, values


def _tq_dense_window_attention(q, key_term, sign_term, values, q_pos, sliding_window, sinks, scale, rotation, proj):
    """Dense TurboQuant attention for one sequence with an HF sliding window (``W`` keys incl. self)."""
    qjl_dim = proj.shape[0]
    q_rot = q @ np.asarray(rotation, dtype=np.float64).T
    q_proj = q @ np.asarray(proj, dtype=np.float64).T
    kv_pos = np.arange(key_term.shape[0])
    visible = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - sliding_window)
    group = q.shape[1] // key_term.shape[1]
    out = np.empty(q.shape, dtype=np.float64)
    for head in range(q.shape[1]):
        kv_head = head // group
        scores = q_rot[:, head] @ key_term[:, kv_head].T
        scores += np.sqrt(np.pi / 2.0) / qjl_dim * (q_proj[:, head] @ sign_term[:, kv_head].T)
        scores = np.where(visible, scores * scale, -np.inf)
        maximum = np.maximum(scores.max(axis=-1), sinks[head])
        weights = np.exp(scores - maximum[:, None])
        denominator = weights.sum(axis=-1) + np.exp(sinks[head] - maximum)
        out[:, head] = weights @ values[:, kv_head] / denominator[:, None]
    return out


@pytest.mark.parametrize("sliding_window", [1, 5, 8, 33])
@pytest.mark.parametrize("with_softmax_aux", [False, True])
def test_ragged_page_attention_v3_turboquant_sliding_window_matches_dense(sliding_window, with_softmax_aux):
    """Every query of a prefill chunk keeps its own ``W``-key window (not the last query's).

    Decode over a long context, ``q_len == kv_len`` and ``kv_len > q_len`` chunks with 8-token KV
    blocks. The reference decodes the returned pages (which hold the freshly compressed tokens).
    """
    rng = np.random.default_rng(sliding_window + 10 * int(with_softmax_aux))
    num_q_heads, num_kv_heads, head_dim, qjl_dim = 4, 2, 8, 8
    page_size, pages_per_seq = 2, 75
    q_lens = np.array([1, 20, 13], dtype=np.int32)
    kv_lens = np.array([150, 20, 140], dtype=np.int32)
    query_start_loc = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(q_lens, dtype=np.int32)])
    num_seqs = len(q_lens)
    total_q = int(q_lens.sum())
    total_pages = num_seqs * pages_per_seq
    page_shape = (total_pages, page_size, num_kv_heads)

    queries = rng.normal(size=(total_q, num_q_heads, head_dim)).astype(np.float32)
    keys = rng.normal(size=(total_q, num_kv_heads, head_dim)).astype(np.float32)
    values = rng.normal(size=(total_q, num_kv_heads, head_dim)).astype(np.float32)
    key_nibbles = rng.integers(0, 8, size=(2, *page_shape, head_dim // 2))  # key codebook has 8 levels
    ki = (key_nibbles[0] | (key_nibbles[1] << 4)).astype(np.uint8)
    ks = rng.integers(0, 256, size=(*page_shape, qjl_dim // 8)).astype(np.uint8)
    kn = jnp.asarray(rng.uniform(0.5, 2.0, size=(*page_shape, 2)), dtype=jnp.bfloat16)
    vi = rng.integers(0, 256, size=(*page_shape, head_dim // 2)).astype(np.uint8)
    vn = jnp.asarray(rng.uniform(0.5, 2.0, size=page_shape), dtype=jnp.bfloat16)
    block_tables = rng.permutation(total_pages).astype(np.int32).reshape(num_seqs, pages_per_seq)
    rotation = np.linalg.qr(rng.normal(size=(head_dim, head_dim)))[0].astype(np.float32)
    projection = rng.normal(size=(qjl_dim, head_dim)).astype(np.float32)
    key_codebook = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    value_codebook = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    sinks = rng.normal(size=(num_q_heads,)) if with_softmax_aux else np.full(num_q_heads, -np.inf)
    softmax_scale = head_dim**-0.5

    with jax.default_matmul_precision("highest"):
        out, ki_out, ks_out, kn_out, vi_out, vn_out = ragged_page_attention_v3_turboquant(
            jnp.asarray(queries),
            jnp.asarray(keys),
            jnp.asarray(values),
            jnp.asarray(ki),
            jnp.asarray(ks),
            kn,
            jnp.asarray(vi),
            vn,
            jnp.asarray(kv_lens),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray(query_start_loc),
            jnp.array([0, 0, num_seqs], dtype=jnp.int32),
            jnp.asarray(rotation),
            jnp.asarray(projection),
            jnp.asarray(key_codebook),
            jnp.asarray(value_codebook),
            jnp.asarray(sinks, dtype=jnp.float32) if with_softmax_aux else None,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            qjl_dim=qjl_dim,
            num_kv_pages_per_block=4,
        )

    key_term, sign_term, dense_values = _tq_unpack_dense(
        ki_out,
        ks_out,
        np.asarray(kn_out, np.float32),
        vi_out,
        np.asarray(vn_out, np.float32),
        key_codebook,
        value_codebook,
        rotation,
    )
    expected = np.empty(queries.shape, dtype=np.float64)
    for seq in range(num_seqs):
        q_start, q_len, kv_len = query_start_loc[seq], q_lens[seq], kv_lens[seq]

        def gather(x, seq=seq, kv_len=kv_len):
            return x[block_tables[seq]].reshape(pages_per_seq * page_size, *x.shape[2:])[:kv_len]

        expected[q_start : q_start + q_len] = _tq_dense_window_attention(
            queries[q_start : q_start + q_len].astype(np.float64),
            gather(key_term),
            gather(sign_term),
            gather(dense_values),
            kv_len - q_len + np.arange(q_len),
            sliding_window,
            sinks,
            softmax_scale,
            rotation,
            projection,
        )

    out = np.asarray(out, dtype=np.float32)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)
