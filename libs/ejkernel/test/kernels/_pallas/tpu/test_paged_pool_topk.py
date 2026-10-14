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
