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

"""TPU Pallas sparse MLA ragged paged attention vs the float64 numpy reference and the XLA op.

The reference (shared with the XLA test) keeps each sequence's latent history as a plain array;
the kernel must resolve the same tokens through a permuted page table.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.sparse_multi_latent_ragged_page_attention import (
    sparse_multi_latent_ragged_page_attention as pallas_sparse_mla,
)
from ejkernel.kernels._pallas.tpu.sparse_multi_latent_ragged_page_attention._pallas_impl_fwd import supports
from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.sparse_multi_latent_ragged_page_attention import (
    sparse_multi_latent_ragged_page_attention as xla_sparse_mla,
)

from ..._xla.test_sparse_multi_latent_ragged_page_attention import (
    CACHE_DIM,
    HEADS,
    NOPE,
    PAD_TOKENS,
    PE,
    _build,
    _expected_cache,
    _reference,
)

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

KV_LENS, Q_LENS = [40, 23, 17], [1, 5, 17]  # decode, partial prefill across a page, full prefill


def _copies(inputs):
    return [jnp.array(x) for x in inputs]


@pytest.mark.parametrize(
    "dtype,chunk_size,width,group",
    [
        (jnp.float32, 4, 6, None),
        (jnp.float32, 4, 6, 1),  # one chunk per step: exercises the double-buffered group loop
        (jnp.float32, 1, 20, None),
        (jnp.float32, 1, 64, None),  # wide row: 130 metadata ints -> 256-long SMEM block, fori chunk loop
        (jnp.float32, 16, 2, None),
        (jnp.bfloat16, 4, 6, None),
        (jnp.bfloat16, 4, 6, 2),
        (jnp.bfloat16, 16, 2, None),
        # bf16 packs 2 tokens per word: chunk 1 fetches the containing word and masks the other lane.
        (jnp.bfloat16, 1, 20, None),
        (jnp.bfloat16, 1, 20, 1),
        (jnp.bfloat16, 1, 64, None),
    ],
)
def test_pallas_matches_reference_and_xla(dtype, chunk_size, width, group):
    case = _build(0, kv_lens=KV_LENS, q_lens=Q_LENS, chunk_size=chunk_size, width=width, dtype=dtype)
    assert supports(tuple(case["inputs"][4].shape), chunk_size)
    scale = (NOPE + PE) ** -0.5
    out, cache = pallas_sparse_mla(*_copies(case["inputs"]), chunk_size=chunk_size, num_chunks_per_block=group)
    with jax.default_matmul_precision("highest"):
        xla_out, xla_cache = xla_sparse_mla(*_copies(case["inputs"]), chunk_size=chunk_size)
    expected = _reference(case, chunk_size=chunk_size, scale=scale)
    got = np.asarray(out, np.float64)
    np.testing.assert_allclose(got, expected, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(got, np.asarray(xla_out, np.float64), atol=2e-3, rtol=2e-3)
    assert np.all(np.asarray(out)[-PAD_TOKENS:] == 0)
    np.testing.assert_array_equal(np.asarray(cache), np.asarray(xla_cache))
    got_cache = np.asarray(jnp.asarray(cache, jnp.float32), np.float64).reshape(-1, CACHE_DIM)
    want_cache = np.asarray(jnp.asarray(_expected_cache(case), dtype).astype(jnp.float32))
    np.testing.assert_array_equal(got_cache, want_cache)


def test_pallas_sliding_window_and_soft_cap():
    case = _build(3, kv_lens=KV_LENS, q_lens=Q_LENS, chunk_size=4, width=8)
    out, _ = pallas_sparse_mla(
        *_copies(case["inputs"]), chunk_size=4, softmax_scale=0.3, sliding_window=12, logits_soft_cap=5.0
    )
    expected = _reference(case, chunk_size=4, scale=0.3, sliding_window=12, soft_cap=5.0)
    np.testing.assert_allclose(np.asarray(out, np.float64), expected, atol=2e-3, rtol=2e-3)


def test_pallas_all_padding_selection_gives_zero_output():
    case = _build(7, kv_lens=[40], q_lens=[3], chunk_size=4, width=4)
    inputs = _copies(case["inputs"])
    inputs[9] = jnp.full_like(inputs[9], -1)
    out, _ = pallas_sparse_mla(*inputs, chunk_size=4)
    assert np.all(np.asarray(out) == 0)


def test_pallas_tokens_before_first_query_start_are_ignored():
    case = _build(9, kv_lens=[40, 23], q_lens=[1, 5], chunk_size=4, width=6)
    lead = 3
    base_out, base_cache = pallas_sparse_mla(*_copies(case["inputs"]), chunk_size=4)
    inputs = _copies(case["inputs"])
    for i in range(4):
        inputs[i] = jnp.concatenate([jnp.full((lead, *inputs[i].shape[1:]), 7.0, inputs[i].dtype), inputs[i]])
    inputs[7] = inputs[7] + lead
    inputs[9] = jnp.concatenate([jnp.zeros((lead, inputs[9].shape[1]), jnp.int32), inputs[9]])
    out, cache = pallas_sparse_mla(*inputs, chunk_size=4)
    assert np.all(np.asarray(out)[:lead] == 0)
    np.testing.assert_allclose(np.asarray(out)[lead:], np.asarray(base_out), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(cache), np.asarray(base_cache))


def test_supports_sub_word_chunks_but_not_word_straddling_ones():
    """Chunks narrower than a packed word are served whole-word; chunks that straddle words are not."""
    bf16_shape = (4, 8, 2, 256)  # page_size 16, 2 tokens per word
    fp8_shape = (4, 4, 4, 256)  # page_size 16, 4 tokens per word
    assert supports(bf16_shape, 1)
    assert supports(fp8_shape, 1)
    assert supports(fp8_shape, 2)
    assert not supports(bf16_shape, 3)  # straddles words and does not divide the page
    assert not supports(fp8_shape, 6)


def test_registered_pallas_tpu_backend():
    impl = kernel_registry.get("sparse_multi_latent_ragged_page_attention", platform="pallas", backend="any")
    assert impl is pallas_sparse_mla
    assert HEADS > 0
