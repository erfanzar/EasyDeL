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
from ejkernel.kernels._pallas.tpu.sparse_multi_latent_ragged_page_attention._pallas_impl_fwd import (
    block_path_supported,
    sparse_mla_ragged_paged_attention_pallas,
    supports,
    write_token_rows,
)
from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.sparse_multi_latent_ragged_page_attention import (
    sparse_multi_latent_ragged_page_attention as xla_sparse_mla,
)
from ejkernel.kernels._xla.sparse_multi_latent_ragged_page_attention._xla_impl_fwd import (
    token_layout,
    write_tokens_to_cache,
)

from ..._xla.test_sparse_multi_latent_ragged_page_attention import (
    CACHE_DIM,
    HEADS,
    NOPE,
    PAD_TOKENS,
    PAGE,
    PAGES_PER_SEQ,
    PE,
    _build,
    _expected_cache,
    _reference,
)

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

KV_LENS, Q_LENS = [40, 23, 17], [1, 5, 17]  # decode, partial prefill across a page, full prefill


def _copies(inputs):
    return [jnp.array(x) for x in inputs]


def _pallas(inputs, *, chunk_size, group=None, translate_pages=None, write_rows=None):
    """Run the Pallas kernel on fresh copies of ``inputs``.

    With ``translate_pages`` and ``write_rows`` both ``None`` this calls the registered entry point (the production
    dispatch). Otherwise it calls the backend with the chunk page translation forced into the kernel / into XLA
    (``translate_pages`` ``True`` / ``False``) or the cache write forced to whole packed word rows / single tokens
    (``write_rows`` ``True`` / ``False``).
    """
    if translate_pages is None and write_rows is None:
        return pallas_sparse_mla(*_copies(inputs), chunk_size=chunk_size, num_chunks_per_block=group)
    return sparse_mla_ragged_paged_attention_pallas(
        *_copies(inputs),
        chunk_size=chunk_size,
        sm_scale=(NOPE + PE) ** -0.5,
        num_chunks_per_block=group,
        translate_pages=translate_pages,
        write_rows=write_rows,
    )


def _check_against_reference_and_xla(case, out, cache, *, chunk_size, dtype):
    with jax.default_matmul_precision("highest"):
        xla_out, xla_cache = xla_sparse_mla(*_copies(case["inputs"]), chunk_size=chunk_size)
    expected = _reference(case, chunk_size=chunk_size, scale=(NOPE + PE) ** -0.5)
    got = np.asarray(out, np.float64)
    np.testing.assert_allclose(got, expected, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(got, np.asarray(xla_out, np.float64), atol=2e-3, rtol=2e-3)
    assert np.all(np.asarray(out)[-PAD_TOKENS:] == 0)
    np.testing.assert_array_equal(np.asarray(cache), np.asarray(xla_cache))
    got_cache = np.asarray(jnp.asarray(cache, jnp.float32), np.float64).reshape(-1, CACHE_DIM)
    want_cache = np.asarray(jnp.asarray(_expected_cache(case), dtype).astype(jnp.float32))
    np.testing.assert_array_equal(got_cache, want_cache)


@pytest.mark.parametrize(
    "translate_pages,write_rows",
    [(None, None), (False, None), (None, True)],
    ids=["kernel_pages", "xla_pages", "word_rows"],
)
@pytest.mark.parametrize(
    "dtype,chunk_size,width,group",
    [
        (jnp.float32, 4, 6, None),
        (jnp.float32, 4, 6, 1),  # one chunk per step: exercises the double-buffered group loop
        (jnp.float32, 1, 20, None),
        (jnp.float32, 1, 64, None),  # wide row: one 64-chunk group
        (jnp.float32, 16, 2, None),
        (jnp.bfloat16, 4, 6, None),
        (jnp.bfloat16, 4, 6, 2),
        (jnp.bfloat16, 16, 2, None),
        # bf16 packs 2 tokens per word: chunk 1 fetches the containing word and masks the other lane.
        (jnp.bfloat16, 1, 20, None),
        (jnp.bfloat16, 1, 20, 1),
        (jnp.bfloat16, 1, 64, None),  # sub-word chunks carry starts: 131 metadata ints -> 256-long SMEM block
    ],
)
def test_pallas_matches_reference_and_xla(dtype, chunk_size, width, group, translate_pages, write_rows):
    case = _build(0, kv_lens=KV_LENS, q_lens=Q_LENS, chunk_size=chunk_size, width=width, dtype=dtype)
    assert supports(tuple(case["inputs"][4].shape), chunk_size)
    out, cache = _pallas(
        case["inputs"], chunk_size=chunk_size, group=group, translate_pages=translate_pages, write_rows=write_rows
    )
    _check_against_reference_and_xla(case, out, cache, chunk_size=chunk_size, dtype=dtype)


@pytest.mark.parametrize("translate_pages", [None, False], ids=["kernel_pages", "xla_pages"])
@pytest.mark.parametrize(
    "dtype,chunk_size,group",
    [
        # f32 pages are 16 words: chunks of 4 tokens in groups of 4 / 8 / 16 words tile them.
        (jnp.float32, 4, 1),
        (jnp.float32, 4, 2),
        (jnp.float32, 4, 4),
        # bf16 pages are 8 words: 4-token chunks (2 words) in groups of 4 / 8 words; one 16-token chunk is a page.
        (jnp.bfloat16, 4, 2),
        (jnp.bfloat16, 4, 4),
        (jnp.bfloat16, 16, 1),
    ],
)
def test_pallas_prefix_selections_match_reference(dtype, chunk_size, group, translate_pages):
    """Rows selecting every visible chunk (what the indexer returns while the context fits its top-k).

    The kernel fetches such a row one page-tiling group per copy instead of one chunk per copy; every third row keeps a
    random sparse selection so both kinds of rows share one call. Prefix rows also select the chunk after their own,
    which causality drops; with several chunks per group their last group's copy reaches past their last attendable
    chunk, and those keys must stay masked.
    """
    width = PAGES_PER_SEQ * PAGE // chunk_size
    case = _build(11, kv_lens=KV_LENS, q_lens=Q_LENS, chunk_size=chunk_size, width=width, dtype=dtype)
    rng = np.random.default_rng(1)
    selected = case["selected"].copy()
    for t, p in enumerate(case["pos_of"]):
        if t % 3 == 2:
            continue
        n = min(p // chunk_size + 2, width)
        selected[t] = -1
        selected[t, :n] = rng.permutation(n)
    case["selected"] = selected
    inputs = list(case["inputs"])
    inputs[9] = jnp.asarray(selected)
    case["inputs"] = tuple(inputs)
    out, cache = _pallas(case["inputs"], chunk_size=chunk_size, group=group, translate_pages=translate_pages)
    _check_against_reference_and_xla(case, out, cache, chunk_size=chunk_size, dtype=dtype)


def test_pallas_page_translation_in_kernel_equals_xla_translation():
    """Translating chunk pages inside the kernel or in XLA must give bitwise identical outputs and caches."""
    case = _build(13, kv_lens=KV_LENS, q_lens=Q_LENS, chunk_size=4, width=10, dtype=jnp.bfloat16)
    in_kernel = _pallas(case["inputs"], chunk_size=4, translate_pages=True)
    in_xla = _pallas(case["inputs"], chunk_size=4, translate_pages=False)
    np.testing.assert_array_equal(np.asarray(in_kernel[0]), np.asarray(in_xla[0]))
    np.testing.assert_array_equal(np.asarray(in_kernel[1]), np.asarray(in_xla[1]))


# Page sizes the kernel must address: shorter than a default 32-chunk group (64), not a power of two (96), the serving
# default (128), and pages holding two and four default groups (256, 512).
PAGE_SIZES = (64, 96, 128, 256, 512)


@pytest.mark.parametrize("page", PAGE_SIZES)
@pytest.mark.parametrize(
    "dtype,chunk_size,group",
    [
        (jnp.bfloat16, 4, None),  # default 32-chunk group = 64 words: whole-group copies where groups tile the page
        (jnp.bfloat16, 4, 2),  # 4-word groups: whole-group copies at every page size
        (jnp.bfloat16, 4, 8),
        (jnp.float32, 4, None),
        (jnp.float32, 4, 4),
        (jnp.bfloat16, 1, None),  # sub-word chunks: masked per key from the chunk starts, never whole-group copies
        (jnp.bfloat16, 16, None),
    ],
)
def test_pallas_page_sizes_match_reference(page, dtype, chunk_size, group):
    """Ragged batches spanning several pages, at every page size; page translation in the kernel and in XLA agree, and
    so does the cache write as whole packed word rows (the registered entry's default write differs by bucket size).

    Rows: a decode token whose context ends mid-chunk on a partly filled third page, a prefill chunk crossing a page
    boundary, and a fresh prefill ending just past the first page (mid-group). Tokens cycle through three selections:
    every visible chunk plus the next one (a prefix: one copy per group where groups tile pages), the same prefix
    without one visible chunk (per-chunk copies), and a random sparse set in random order.
    """
    kv_lens = [2 * page + page // 2 + 3, page + 37, page + 5]
    q_lens = [1, 40, page + 5]
    pages_per_seq = -(-max(kv_lens) // page) + 1
    width = (max(kv_lens) - 1) // chunk_size + 2
    case = _build(
        17,
        kv_lens=kv_lens,
        q_lens=q_lens,
        chunk_size=chunk_size,
        width=width,
        dtype=dtype,
        page=page,
        pages_per_seq=pages_per_seq,
    )
    assert supports(tuple(case["inputs"][4].shape), chunk_size)
    rng = np.random.default_rng(page)
    num_chunks = pages_per_seq * page // chunk_size
    selected = case["selected"].copy()
    for t, p in enumerate(case["pos_of"]):
        n = p // chunk_size + 2
        if t % 3 == 0:
            ids = np.arange(n)
        elif t % 3 == 1 and n > 2:
            ids = np.delete(np.arange(n), n // 2)
        else:
            ids = rng.choice(num_chunks, size=width // 2, replace=False)
        selected[t] = -1
        selected[t, : ids.size] = rng.permutation(ids)
    case["selected"] = selected
    case["inputs"] = (*case["inputs"][:9], jnp.asarray(selected))
    out, cache = _pallas(case["inputs"], chunk_size=chunk_size, group=group)
    _check_against_reference_and_xla(case, out, cache, chunk_size=chunk_size, dtype=dtype)
    xla_pages = _pallas(case["inputs"], chunk_size=chunk_size, group=group, translate_pages=False)
    np.testing.assert_array_equal(np.asarray(xla_pages[0]), np.asarray(out))
    np.testing.assert_array_equal(np.asarray(xla_pages[1]), np.asarray(cache))
    word_rows = _pallas(case["inputs"], chunk_size=chunk_size, group=group, write_rows=True)
    np.testing.assert_array_equal(np.asarray(word_rows[0]), np.asarray(out))
    np.testing.assert_array_equal(np.asarray(word_rows[1]), np.asarray(cache))


@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.int8], ids=["bf16", "f32", "int8"])
def test_word_row_cache_write_equals_per_token_scatter(dtype, page):
    """The Pallas path writes the step's tokens as whole packed word rows; the cache must be bitwise the XLA
    reference's per-token scatter (1, 2 and 4 tokens per word).

    Sequences: a fresh prefill ending mid-word, a prefill starting on an odd position and crossing a page, a decode
    token in a word's last lane, an empty live sequence, a prefill starting and ending mid-word, and an unused slot;
    plus tokens before ``query_start_loc[0]`` (another shard's) and padding after the last live sequence, which must
    not be written. The old cache content must survive in every lane the step does not own.
    """
    rng = np.random.default_rng(page)
    pack = 32 // (jnp.dtype(dtype).itemsize * 8)
    kv_lens = [7, 2 * page + 5, page + 4, 40, 9, 0]
    q_lens = [7, page + 2, 1, 0, 6, 0]
    lead, pad = 3, 4
    max_seqs = len(kv_lens)
    pages_per_seq = -(-max(kv_lens) // page) + 1
    num_pages = max_seqs * pages_per_seq + 2
    qsl = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]) + lead, jnp.int32)
    total = int(qsl[-1]) + pad
    block_tables = jnp.asarray(rng.permutation(num_pages)[: max_seqs * pages_per_seq], jnp.int32)
    cache = jnp.asarray(20 * rng.standard_normal((num_pages, page // pack, pack, CACHE_DIM)), jnp.float32)
    cache = cache.astype(dtype)
    keys_values = jnp.asarray(20 * rng.standard_normal((total, NOPE)), jnp.float32)
    keys_pe = jnp.asarray(20 * rng.standard_normal((total, PE)), jnp.float32)
    kv_lens_arr = jnp.asarray(kv_lens, jnp.int32)
    seq, pos, valid = token_layout(kv_lens_arr, qsl, jnp.asarray([0, 0, max_seqs - 1], jnp.int32), total)
    want = jax.jit(write_tokens_to_cache, static_argnames=("pages_per_seq",))(
        cache, keys_values, keys_pe, block_tables, seq, pos, valid, pages_per_seq=pages_per_seq
    )
    got = jax.jit(write_token_rows, static_argnames=("pages_per_seq",))(
        cache, keys_values, keys_pe, kv_lens_arr, block_tables, qsl, seq, pos, valid, pages_per_seq=pages_per_seq
    )
    assert int(jnp.sum(valid)) == sum(q_lens)
    want, got, before = (np.asarray(x).view(np.uint8) for x in (want, got, cache))
    np.testing.assert_array_equal(got, want)
    assert np.any(want != before)


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


# The query-block prefill kernel maps the rows of a block's [Q * H, L] query tile to its tokens by a shift and steps
# through 128 fetched keys at a time: it needs a power-of-two head count that is a multiple of 16, buckets of at least
# 256 tokens, and 4-token chunks in groups of 32 (selections at least 32 wide).
QB_HEADS, QB_CHUNK, QB_WIDTH, QB_TOKENS = 16, 4, 48, 16


def _prefill_selection(case, *, seed):
    """Selections shaped like a prefill's: neighbouring tokens share most of their chunks.

    A token that sees fewer chunks than the selection width selects all of them plus the next chunk (as the indexer
    does while the context fits its top-k; causality drops the next one), so such a block's union is a prefix. Longer
    contexts rank their visible chunks by one score shared by the whole batch plus per-token noise, and always keep
    their own chunk.
    """
    rng = np.random.default_rng(seed)
    num_chunks = case["pages_per_seq"] * case["page"] // QB_CHUNK
    shared = rng.random(num_chunks)
    selected = np.full_like(case["selected"], -1)
    for t, p in enumerate(case["pos_of"]):
        visible = p // QB_CHUNK + 1
        if visible < QB_WIDTH:
            ids = np.arange(visible + 1)
        else:
            score = shared[:visible] + 0.3 * rng.random(visible)
            score[visible - 1] = -1.0
            ids = np.argsort(score)[:QB_WIDTH]
        selected[t, : ids.size] = rng.permutation(ids)
    return selected


def _query_block_case(seed, *, kv_lens, q_lens, dtype, page, selection):
    """A 16-head ragged batch; ``selection`` is ``prefill`` (overlapping neighbours) or ``random`` (no overlap)."""
    pages_per_seq = -(-max(kv_lens) // page) + 1
    case = _build(
        seed,
        kv_lens=kv_lens,
        q_lens=q_lens,
        chunk_size=QB_CHUNK,
        width=QB_WIDTH,
        dtype=dtype,
        page=page,
        pages_per_seq=pages_per_seq,
        heads=QB_HEADS,
    )
    if selection == "prefill":
        selected = _prefill_selection(case, seed=seed)
        case["selected"] = selected
        case["inputs"] = (*case["inputs"][:9], jnp.asarray(selected))
    return case


@pytest.mark.parametrize("selection", ["prefill", "random"])
@pytest.mark.parametrize(
    "dtype,page",
    [(jnp.bfloat16, 16), (jnp.bfloat16, 128), (jnp.float32, 16)],
    ids=["bf16-p16", "bf16-p128", "f32-p16"],
)
def test_pallas_query_block_matches_reference(dtype, page, selection):
    """``num_queries_per_block=16`` on a prefill bucket: each block of 16 consecutive tokens fetches the union of its
    tokens' selected chunks once (the query-block kernel), yet every token attends exactly to its own chunks.

    Sequences start on multiples of 16 tokens, so every block holds one sequence and the whole step runs the
    query-block kernel: a fresh prefill (prefix unions; at page 128 one copy per page-aligned group), a chunk
    continuing a longer context and a short one, then padding tokens. ``prefill`` selections overlap between
    neighbours; ``random`` ones do not (wide unions in which most chunks belong to a single token).
    """
    case = _query_block_case(
        29, kv_lens=[208, 900, 40], q_lens=[208, 64, 16], dtype=dtype, page=page, selection=selection
    )
    cache = case["inputs"][4]
    assert block_path_supported(
        total_tokens=int(case["inputs"][0].shape[0]),
        chunk_size=QB_CHUNK,
        kv_packing=int(cache.shape[2]),
        sliding_window=None,
        query_block=QB_TOKENS,
        width=QB_WIDTH,
        num_heads=QB_HEADS,
        group=128 // QB_CHUNK,
        page_size=page,
        pages_per_seq=case["pages_per_seq"],
    )
    out, cache = pallas_sparse_mla(*_copies(case["inputs"]), chunk_size=QB_CHUNK, num_queries_per_block=QB_TOKENS)
    _check_against_reference_and_xla(case, out, cache, chunk_size=QB_CHUNK, dtype=dtype)


def test_pallas_query_block_falls_back_to_per_token_kernel_across_sequences():
    """A bucket in which some block of 16 tokens holds two sequences runs the per-token kernel for the whole step:
    outputs and cache are bitwise those of the call without ``num_queries_per_block``."""
    case = _query_block_case(
        31, kv_lens=[250, 900, 41], q_lens=[250, 70, 9], dtype=jnp.bfloat16, page=16, selection="prefill"
    )
    per_token = pallas_sparse_mla(*_copies(case["inputs"]), chunk_size=QB_CHUNK)
    blocked = pallas_sparse_mla(*_copies(case["inputs"]), chunk_size=QB_CHUNK, num_queries_per_block=QB_TOKENS)
    np.testing.assert_array_equal(np.asarray(blocked[0]), np.asarray(per_token[0]))
    np.testing.assert_array_equal(np.asarray(blocked[1]), np.asarray(per_token[1]))
    _check_against_reference_and_xla(case, blocked[0], blocked[1], chunk_size=QB_CHUNK, dtype=jnp.bfloat16)
