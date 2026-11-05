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

"""Fused Pallas paged pool top-k vs a float64 numpy reference and the XLA fallback."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._registry import kernel_registry

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

H, D, PPP, PPS = 32, 128, 32, 512  # 512 pages/seq covers the 10000-pool row (313 pages)


def _case(seed, npool, *, dim=D, ppp=PPP, tie=False):
    rng = np.random.default_rng(seed)
    T = len(npool)
    need = max(1, -(-max(npool) // ppp))
    P = T * need + 2
    bt = np.zeros((T, PPS), np.int32)
    perm = rng.permutation(P)
    for t in range(T):
        bt[t, :need] = perm[t * need : (t + 1) * need]
    pools = np.array(jnp.asarray(rng.standard_normal((P, ppp, dim)), jnp.bfloat16).astype(jnp.float32))
    if tie:
        pools[:] = 1.0
    q = rng.standard_normal((T, H, dim)).astype(np.float32)
    w = (rng.standard_normal((T, H)) * H**-0.5).astype(np.float32)
    if tie:
        q[:] = 1.0
        w[:] = 1.0
    args = tuple(
        jnp.asarray(a)
        for a in (q, w, jnp.asarray(pools, jnp.bfloat16), bt, np.arange(T, dtype=np.int32), np.asarray(npool, np.int32))
    )
    return args, (q, w, pools, bt, np.asarray(npool))


def _scores(ref, t, scale, ppp):
    q, w, pools, bt, npool = ref
    n = int(npool[t])
    keys = pools[bt[t, np.arange(n) // ppp], np.arange(n) % ppp].astype(np.float64)
    s = np.maximum(np.einsum("hd,cd->hc", q[t].astype(np.float64), keys) * scale, 0.0)
    return w[t].astype(np.float64) @ s


def _run(platform, args, k, scale):
    impl = kernel_registry.get("paged_pool_topk", platform=platform, backend="any")
    with jax.default_matmul_precision("highest"):
        return np.asarray(impl(*args, k=k, sm_scale=scale))


@pytest.mark.parametrize("npool", [[256] * 8, [7168] * 8, [75, 1250, 10000, 0, 4, 896, 1800, 511, 512, 513]])
@pytest.mark.parametrize("k", [512, 64])
def test_pallas_matches_reference_and_xla(npool, k):
    scale = D**-0.5
    args, ref = _case(0, npool)
    got = _run("pallas", args, k, scale)
    xla = _run("xla", args, k, scale)
    assert got.shape == (len(npool), k)
    for t, n in enumerate(npool):
        ids = got[t][got[t] >= 0]
        assert len(ids) == min(k, n) and np.all(np.diff(ids) > 0) and np.all(ids < max(n, 1))
        assert np.all(got[t][len(ids) :] == -1)
        if n > k:
            s = _scores(ref, t, scale, PPP)
            kth = np.sort(s)[::-1][k - 1]
            assert s[ids].min() >= kth - 1e-5 * max(abs(kth), 1.0), t
    np.testing.assert_array_equal(got, xla)


@pytest.mark.parametrize("k", [512, 64])
def test_default_kpool16_layout_eight_pools_per_page(k):
    """``index_kpool=16`` with 128-token pages gives 8 pools per page (bf16): the default config's shape."""
    ppp = 8
    npool = [75, 1250, 4000, 0, 4, 896, 511, 513]
    scale = D**-0.5
    args, ref = _case(3, npool, ppp=ppp)
    got = _run("pallas", args, k, scale)
    xla = _run("xla", args, k, scale)
    for t, n in enumerate(npool):
        ids = got[t][got[t] >= 0]
        assert len(ids) == min(k, n) and np.all(np.diff(ids) > 0) and np.all(ids < max(n, 1))
        if n > k:
            s = _scores(ref, t, scale, ppp)
            kth = np.sort(s)[::-1][k - 1]
            assert s[ids].min() >= kth - 1e-5 * max(abs(kth), 1.0), t
    np.testing.assert_array_equal(got, xla)


@pytest.mark.parametrize("ppp", [16, 64, 128], ids=["page64", "page256", "page512"])
def test_pools_per_page_of_other_page_sizes(ppp):
    """``index_kpool=4`` pools per page at page sizes 64 / 256 / 512 (the 128-token default gives 32)."""
    npool = [75, 1250, 7000, 0, 4, 896, 511, 512, 513]
    k = 512
    scale = D**-0.5
    args, ref = _case(5, npool, ppp=ppp)
    got = _run("pallas", args, k, scale)
    xla = _run("xla", args, k, scale)
    for t, n in enumerate(npool):
        ids = got[t][got[t] >= 0]
        assert len(ids) == min(k, n) and np.all(np.diff(ids) > 0) and np.all(ids < max(n, 1))
        assert np.all(got[t][len(ids) :] == -1)
        if n > k:
            s = _scores(ref, t, scale, ppp)
            kth = np.sort(s)[::-1][k - 1]
            assert s[ids].min() >= kth - 1e-5 * max(abs(kth), 1.0), t
    np.testing.assert_array_equal(got, xla)


def _packed_rows(seed, segments, *, num_seqs, shared_table=False, tie=False):
    """Packed query rows ``segments = [(seq, first_position, rows), ...]`` (kpool 4); position ``-1`` = padding rows.

    ``shared_table`` gives every sequence the same page list, so rows of different sequences read identical pools;
    ``tie`` makes every score equal.
    """
    rng = np.random.default_rng(seed)
    seq_ids, npool = [], []
    for s, p0, n in segments:
        seq_ids += [s] * n
        npool += [0] * n if p0 < 0 else [(p0 + i + 1) // 4 for i in range(n)]
    need = max(1, -(-max(npool) // PPP))
    num_pages = num_seqs * need + 3
    perm = rng.permutation(num_pages)
    if shared_table:
        bt = np.tile(perm[:need], (num_seqs, 1))
    else:
        bt = perm[: num_seqs * need].reshape(num_seqs, need)
    bt = np.pad(bt, ((0, 0), (0, PPS - need))).astype(np.int32)
    T = len(seq_ids)
    pools = np.array(jnp.asarray(rng.standard_normal((num_pages, PPP, D)), jnp.bfloat16).astype(jnp.float32))
    q = rng.standard_normal((T, H, D)).astype(np.float32)
    w = (rng.standard_normal((T, H)) * H**-0.5).astype(np.float32)
    if tie:
        pools[:], q[:], w[:] = 1.0, 1.0, 1.0
    seq_ids = np.asarray(seq_ids, np.int32)
    npool = np.asarray(npool, np.int32)
    args = tuple(jnp.asarray(a) for a in (q, w, jnp.asarray(pools, jnp.bfloat16), bt, seq_ids, npool))
    return args, (q, w, pools, bt, npool)


def _check_rows(got, ref, seq_ids, k, scale):
    q, w, pools, bt, npool = ref
    for t, n in enumerate(npool):
        ids = got[t][got[t] >= 0]
        assert len(ids) == min(k, n) and np.all(np.diff(ids) > 0) and np.all(ids < max(n, 1)), t
        assert np.all(got[t][len(ids) :] == -1), t
        if n > k:
            keys = pools[bt[seq_ids[t], np.arange(n) // PPP], np.arange(n) % PPP].astype(np.float64)
            s = w[t].astype(np.float64) @ np.maximum(np.einsum("hd,cd->hc", q[t].astype(np.float64), keys) * scale, 0)
            kth = np.sort(s)[::-1][k - 1]
            assert s[ids].min() >= kth - 1e-5 * max(abs(kth), 1.0), t


_DECODE_ROWS = [(1 + i, 3000 + 37 * i, 1) for i in range(12)] + [(13, 300, 1)]  # the last row has 75 pools
_PACKED_STEPS = {
    "rows45": [*_DECODE_ROWS, (0, 2500, 30), (0, -1, 2)],
    "rows205": [*_DECODE_ROWS, (0, 2500, 150), (14, 5000, 40), (0, -1, 2)],
    "rows392": [*_DECODE_ROWS, (0, 2500, 300), (14, 5000, 77), (0, -1, 2)],
    "rows1100": [*_DECODE_ROWS, (0, 2500, 1000), (14, 5000, 85), (0, -1, 2)],
}


@pytest.mark.parametrize("layout", list(_PACKED_STEPS))
@pytest.mark.parametrize("k", [512, 64])
def test_prefill_chunks_mixed_with_decode_rows_and_padding(layout, k):
    """eSurge-style packed step: decode rows, prompt chunks (some starting inside a row block), padding rows.

    The row counts give the kernel row blocks of every size it picks (8 rows for small calls, larger ones for long
    calls), with a pad block where the block count is odd. Blocks of one sequence fetch each page once for all their
    rows; blocks of several sequences (decode rows, a chunk boundary) fetch per row. With k = 512 the 75-pool decode
    row has fewer pools than k, so its block ranks the keys tied at the k-th value; other blocks keep every key at or
    above it.
    """
    args, ref = _packed_rows(6, _PACKED_STEPS[layout], num_seqs=15)
    scale = D**-0.5
    got = _run("pallas", args, k, scale)
    xla = _run("xla", args, k, scale)
    _check_rows(got, ref, np.asarray(args[4]), k, scale)
    np.testing.assert_array_equal(got, xla)


def test_page_shared_scoring_equals_per_row_scoring():
    """The same rows over identical pools: one sequence (pages fetched once per row block) == eight (per row)."""
    k = 512
    args, _ = _packed_rows(7, [(0, 6000, 64)], num_seqs=8, shared_table=True)
    scale = D**-0.5
    shared = _run("pallas", args, k, scale)
    per_row_args = (*args[:4], jnp.asarray(np.arange(64) % 8, jnp.int32), args[5])
    per_row = _run("pallas", per_row_args, k, scale)
    np.testing.assert_array_equal(shared, per_row)
    np.testing.assert_array_equal(shared, _run("xla", args, k, scale))


@pytest.mark.parametrize("one_sequence", [True, False])
def test_ties_select_lowest_ids_in_large_row_blocks(one_sequence):
    """All scores equal: each of 256 rows keeps its lowest pool ids, in row blocks of one sequence and of several.

    Rows have 50-114 pools against k = 64, so some keep all their pools and the others cut their ties.
    """
    k = 64
    args, ref = _packed_rows(8, [(0, 200, 256)], num_seqs=8, shared_table=True, tie=True)
    if not one_sequence:
        args = (*args[:4], jnp.asarray(np.arange(256) % 8, jnp.int32), args[5])
    got = _run("pallas", args, k, D**-0.5)
    for t, n in enumerate(ref[4]):
        m = min(k, int(n))
        np.testing.assert_array_equal(got[t][:m], np.arange(m))
        assert np.all(got[t][m:] == -1), t
    np.testing.assert_array_equal(got, _run("xla", args, k, D**-0.5))


def test_ties_select_lowest_ids():
    npool = [300, 64]
    args, _ = _case(1, npool, tie=True)
    got = _run("pallas", args, 128, D**-0.5)
    np.testing.assert_array_equal(got[0], np.arange(128))
    np.testing.assert_array_equal(got[1][:64], np.arange(64))
    assert np.all(got[1][64:] == -1)


def test_untileable_layout_routes_to_xla():
    args, _ = _case(2, [40, 9], dim=8, ppp=2)
    np.testing.assert_array_equal(_run("pallas", args, 8, 0.3), _run("xla", args, 8, 0.3))


def test_oversized_block_table_routes_to_xla_instead_of_overflowing_smem():
    """A 2 MiB block table (256 seqs x 2048 pages, ~262k context) exceeds TPU SMEM; it must not fail."""
    rng = np.random.default_rng(4)
    T, P = 16, 64
    q = jnp.asarray(rng.standard_normal((T, H, D)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((T, H)), jnp.float32)
    pools = jnp.asarray(rng.standard_normal((P, PPP, D)), jnp.bfloat16)
    bt = jnp.asarray(rng.integers(0, P, (256, 2048)), jnp.int32)
    seq = jnp.asarray(np.arange(T), jnp.int32)
    npool = jnp.asarray(rng.integers(0, P * PPP, T), jnp.int32)
    args = (q, w, pools, bt, seq, npool)
    np.testing.assert_array_equal(_run("pallas", args, 64, D**-0.5), _run("xla", args, 64, D**-0.5))


def test_registered_backends():
    assert kernel_registry.get("paged_pool_topk", platform="xla", backend="any") is not None
    assert kernel_registry.get("paged_pool_topk", platform="pallas", backend="any") is not None
