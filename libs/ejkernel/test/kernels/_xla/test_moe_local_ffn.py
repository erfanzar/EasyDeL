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

"""XLA local expert FFN vs an independent dense per-expert reference (fp32, HIGHEST).

The reference runs every local expert densely over every token and weights the result by the routing mask, so it
shares none of the sort / window / gather machinery under test. Cases cover several chunks with token padding,
several row windows per chunk, routes owned by other devices (ids >= experts and negative ids), routing skew far
beyond one window (dropless), a chunk with no local route, and a non-default activation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.moe_local_ffn import moe_local_ffn, plan_local_ffn
from jax import lax

F32 = jnp.float32
HI = lax.Precision.HIGHEST
T, H, M, EL, K = 512, 128, 64, 4, 4


def _silu_glu(g, u):
    return jax.nn.silu(g) * u


def _clamped(g, u):
    return jax.nn.silu(jnp.minimum(g, 2.0)) * jnp.clip(u, -2.0, 2.0)


def reference(x, weights, experts, wg, wu, wd, activation=_silu_glu):
    x = x.astype(F32)
    out = jnp.zeros(x.shape, F32)
    for e in range(wg.shape[0]):
        coef = jnp.sum(jnp.where(experts == e, weights.astype(F32), 0.0), axis=1)
        h = activation(jnp.dot(x, wg[e].astype(F32), precision=HI), jnp.dot(x, wu[e].astype(F32), precision=HI))
        out = out + coef[:, None] * jnp.dot(h, wd[e].astype(F32), precision=HI)
    return out


def _inputs(seed, experts=None, low=-2, high=2 * EL):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((T, H)), F32)
    w = jnp.asarray(rng.uniform(0.05, 1.0, (T, K)), F32)
    if experts is None:
        experts = rng.integers(low, high, (T, K))
    e = jnp.asarray(experts, jnp.int32)
    wg = jnp.asarray(0.2 * rng.standard_normal((EL, H, M)), F32)
    wu = jnp.asarray(0.2 * rng.standard_normal((EL, H, M)), F32)
    wd = jnp.asarray(0.2 * rng.standard_normal((EL, M, H)), F32)
    ct = jnp.asarray(rng.standard_normal((T, H)), F32)
    return x, w, e, wg, wu, wd, ct


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _check(x, w, e, wg, wu, wd, ct, *, activation=None, tol=2e-5, **kw):
    act = activation or _silu_glu

    def l_ffn(x, w, wg, wu, wd):
        return jnp.sum(moe_local_ffn(x, w, e, wg, wu, wd, activation, **kw) * ct)

    def l_ref(x, w, wg, wu, wd):
        return jnp.sum(reference(x, w, e, wg, wu, wd, act) * ct)

    args = (x, w, wg, wu, wd)
    out = jax.jit(lambda *a: moe_local_ffn(a[0], a[1], e, *a[2:], activation, **kw))(*args)
    want = jax.jit(lambda *a: reference(a[0], a[1], e, *a[2:], act))(*args)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert _rel(out, want) < tol, _rel(out, want)
    g_ffn = jax.jit(jax.grad(l_ffn, argnums=range(5)))(*args)
    g_ref = jax.jit(jax.grad(l_ref, argnums=range(5)))(*args)
    for name, a, b in zip(("x", "weights", "w_gate", "w_up", "w_down"), g_ffn, g_ref, strict=True):
        assert np.isfinite(np.asarray(a)).all(), name
        assert _rel(a, b) < tol, (name, _rel(a, b))


@pytest.mark.parametrize(
    "tokens_per_chunk,rows_per_window",
    [(None, None), (192, 100), (512, 64), (100, 37)],
)
def test_matches_reference_over_chunks_and_windows(tokens_per_chunk, rows_per_window):
    _check(*_inputs(0), tokens_per_chunk=tokens_per_chunk, rows_per_window=rows_per_window)


def test_routing_skew_is_dropless():
    # Three of four routes per token go to local expert 1: 768 local rows per 256-token chunk vs 64-row windows.
    experts = np.full((T, K), 1)
    experts[:, 1] = 9  # one route per token is another device's
    _check(*_inputs(1, experts=experts), tokens_per_chunk=256, rows_per_window=64)


def test_chunk_without_local_routes_contributes_nothing():
    experts = np.random.default_rng(2).integers(0, EL, (T, K))
    experts[:256] = EL + 3  # the first chunk has no local route at all
    x, w, e, wg, wu, wd, ct = _inputs(2, experts=experts)
    out = moe_local_ffn(x, w, e, wg, wu, wd, tokens_per_chunk=256, rows_per_window=128)
    np.testing.assert_array_equal(np.asarray(out[:256]), 0.0)
    _check(x, w, e, wg, wu, wd, ct, tokens_per_chunk=256, rows_per_window=128)


def test_custom_activation():
    _check(*_inputs(3), activation=_clamped, tokens_per_chunk=128, rows_per_window=96)


@pytest.mark.cpu_ok
def test_plan_rounds_and_caps():
    assert plan_local_ffn(1000, 8, 300, 500) == (300, 500, 4, 5)
    assert plan_local_ffn(1000, 8, None, None) == (1000, 8000, 1, 1)
    assert plan_local_ffn(1000, 8, 300, 500, token_align=128, row_align=1024) == (384, 1024, 3, 3)
    assert plan_local_ffn(100, 8, 4096, 1 << 20, token_align=128) == (128, 1024, 1, 1)
    with pytest.raises(ValueError):
        plan_local_ffn(100, 8, 0, None)


def test_bad_shapes_raise():
    x, w, e, wg, wu, wd, _ = _inputs(4)
    with pytest.raises(ValueError, match="w_down"):
        moe_local_ffn(x, w, e, wg, wu, wd[:, :, :-1])
    with pytest.raises(ValueError, match="integer"):
        moe_local_ffn(x, w, e.astype(F32), wg, wu, wd)
    with pytest.raises(ValueError, match="weights"):
        moe_local_ffn(x, w[:, :-1], e, wg, wu, wd)
