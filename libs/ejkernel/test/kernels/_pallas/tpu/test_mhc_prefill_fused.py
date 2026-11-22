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

"""Prefill mHC project / read / write kernels (Pallas TPU, APPROXIMATE class) against their XLA reference.

The projection statistics are checked against the XLA reference to float32 accumulation-order noise, the read (gate
table and collapse) on identical statistics to within one bf16 ulp, and the write on an identical gate table against a
float32 truth: XLA's own rounding of the write depends on its fusion on TPU, so the kernel must be within two bf16 ulps
of XLA and no less accurate than XLA (relative L2 against the float32 truth).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from ejkernel.modules import (
    mhc_prefill_project,
    mhc_prefill_read,
    mhc_prefill_stats_rows,
    mhc_prefill_supports,
    mhc_prefill_write,
)
from jax import numpy as jnp

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernels need a TPU")

HC, ITERS, EPS, NORM_EPS = 4, 20, 1e-6, 1e-5
MIX = (HC + 2) * HC


def _inputs(tokens, width, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((tokens, HC * width)), jnp.bfloat16)
    fn = jnp.asarray(0.02 * rng.standard_normal((MIX, HC * width)), jnp.bfloat16)
    base = jnp.asarray(0.5 * rng.standard_normal((MIX,)), jnp.float32)
    scale = jnp.asarray(1.0 + 0.1 * rng.standard_normal((3,)), jnp.float32)
    out = jnp.asarray(rng.standard_normal((tokens, width)), jnp.bfloat16)
    return x, fn, base, scale, out


def _ulp_close(got, want, max_fraction):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape
    assert (np.abs(got - want) <= 2.0**-7 * np.maximum(np.abs(got), np.abs(want)) + 1e-30).all()
    assert np.count_nonzero(got != want) <= max(1, int(got.size * max_fraction))


def test_supports_prefill_blocks_only():
    assert not mhc_prefill_supports(128, HC, 1024)
    assert mhc_prefill_supports(256, HC, 1024)
    assert not mhc_prefill_supports(320, HC, 1024)
    assert not mhc_prefill_supports(1024, HC, 1000)


@pytest.mark.parametrize("tokens,width", [(256, 512), (2048, 1024), (4096, 2048)])
def test_project_matches_reference(tokens, width):
    x, fn, *_ = _inputs(tokens, width)
    got = jax.jit(lambda x, f: mhc_prefill_project(x, f, platform="pallas"))(x, fn)
    want = jax.jit(lambda x, f: mhc_prefill_project(x, f, platform="xla"))(x, fn)
    assert got.shape == want.shape == (mhc_prefill_stats_rows(MIX), tokens) and got.dtype == jnp.float32
    got, want = np.asarray(got), np.asarray(want)
    np.testing.assert_allclose(got[:MIX], want[:MIX], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(got[MIX], want[MIX], rtol=1e-5)
    assert not got[MIX + 1 :].any()


@pytest.mark.parametrize("tokens,width", [(256, 512), (2048, 1024), (4096, 2048)])
def test_read_matches_reference(tokens, width):
    x, fn, base, scale, _ = _inputs(tokens, width)
    stats = jax.jit(lambda x, f: mhc_prefill_project(x, f, platform="xla"))(x, fn)
    kw = dict(global_width=HC * width, norm_eps=NORM_EPS, n_iters=ITERS, eps=EPS)

    def read(platform):
        return jax.jit(lambda s, b, c, x: mhc_prefill_read(s, b, c, x, **kw, platform=platform))(stats, base, scale, x)

    (table, collapsed), (table_ref, collapsed_ref) = read("pallas"), read("xla")
    np.testing.assert_allclose(np.asarray(table), np.asarray(table_ref), rtol=1e-5, atol=1e-6)
    _ulp_close(collapsed, collapsed_ref, 1e-3)


@pytest.mark.parametrize("tokens,width", [(256, 512), (2048, 1024), (4096, 2048)])
def test_write_matches_reference(tokens, width):
    x, fn, base, scale, out = _inputs(tokens, width)
    stats = mhc_prefill_project(x, fn, platform="xla")
    table, _ = mhc_prefill_read(stats, base, scale, x, HC * width, NORM_EPS, ITERS, EPS, platform="xla")

    def write(platform):
        return jax.jit(lambda x, o, t: mhc_prefill_write(x, o, t, platform=platform))(x, out, table)

    got, want = write("pallas"), write("xla")
    assert got.shape == want.shape == x.shape and got.dtype == x.dtype
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert (np.abs(got - want) <= 2.0**-6 * np.maximum(np.abs(got), np.abs(want)) + 2.0**-20).all()
    gates = np.asarray(table, np.float64)
    post = gates[:, 8:12].astype(np.float32).astype(jnp.bfloat16).astype(np.float64)
    comb = gates[:, 16:48].reshape(tokens, HC, 8)[:, :, :HC].astype(np.float32).astype(jnp.bfloat16).astype(np.float64)
    xs = np.asarray(x, np.float64).reshape(tokens, HC, width)
    truth = post[..., None] * np.asarray(out, np.float64)[:, None, :] + np.einsum("tji,tjd->tid", comb, xs)
    truth = truth.reshape(tokens, HC * width)

    def rel(a):
        return np.linalg.norm(a - truth) / np.linalg.norm(truth)

    assert rel(got) <= 1.05 * rel(want) + 1e-7
