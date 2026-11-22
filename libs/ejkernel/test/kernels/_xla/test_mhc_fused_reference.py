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

"""XLA mHC collapse / expand reference against independent float64 / flattened-stream equations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.mhc_fused._xla_impl_fwd import (
    gates_from_stats,
    mhc_collapse,
    mhc_expand,
    pack_gates,
    unpack_gates,
)

HC, W, S, ITERS, EPS, RMS_EPS = 4, 128, 16, 5, 1e-6, 1e-6
MIX = (2 + HC) * HC


def _inputs(dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, S, HC, W)).astype(dtype)
    fn = (rng.standard_normal((MIX, HC, W)) * 0.05).astype(np.float32)
    base = (rng.standard_normal((MIX,)) * 0.2).astype(np.float32)
    scale = np.asarray([0.8, 1.2, 0.9], np.float32)
    out = rng.standard_normal((2, S, W)).astype(dtype)
    return x, fn, base, scale, out


def _numpy_collapse(x, fn, base, scale, hidden_size=W):
    """Float64 equations on the flattened ``[B, S, hc * W]`` row (the unfused layer form)."""
    x64 = x.astype(np.float64)
    flat = x64.reshape(*x.shape[:2], -1)
    normalized = flat / np.sqrt((flat**2).sum(-1, keepdims=True) / (HC * hidden_size) + RMS_EPS)
    logits = normalized @ fn.reshape(MIX, -1).astype(np.float64).T
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    pre = sig(logits[..., :HC] * scale[0] + base[:HC]) + EPS
    post = 2 * sig(logits[..., HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
    c = (logits[..., 2 * HC :] * scale[2] + base[2 * HC :]).reshape(*x.shape[:2], HC, HC)
    c = np.exp(c - c.max(-1, keepdims=True))
    c = c / c.sum(-1, keepdims=True) + EPS
    for axis in [-2] + [-1, -2] * (ITERS - 1):
        c = c / (c.sum(axis, keepdims=True) + EPS)
    collapsed = sum(pre[..., h, None] * x64[..., h, :] for h in range(HC))
    return post, c, collapsed


def _jax_collapse_independent(x, fn, base, scale):
    """Independent JAX equations (flattened row, explicit stream loop) for gradient references."""
    xf = x.astype(jnp.float32)
    flat = xf.reshape(*x.shape[:2], -1)
    normalized = flat * jax.lax.rsqrt(jnp.mean(flat * flat, axis=-1, keepdims=True) + RMS_EPS)
    logits = jnp.matmul(normalized, fn.reshape(MIX, -1).T, precision=jax.lax.Precision.HIGHEST)
    pre = jax.nn.sigmoid(logits[..., :HC] * scale[0] + base[:HC]) + EPS
    post = 2 * jax.nn.sigmoid(logits[..., HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
    c = jax.nn.softmax((logits[..., 2 * HC :] * scale[2] + base[2 * HC :]).reshape(*x.shape[:2], HC, HC), -1) + EPS
    for axis in [-2] + [-1, -2] * (ITERS - 1):
        c = c / (jnp.sum(c, axis=axis, keepdims=True) + EPS)
    collapsed = sum(pre[..., h, None] * xf[..., h, :] for h in range(HC))
    return post, c, collapsed.astype(x.dtype)


def _kw():
    return dict(sinkhorn_iters=ITERS, eps=EPS, rms_eps=RMS_EPS)


def _assert_close(got, want, tol):
    got = np.asarray(jnp.asarray(got, jnp.float32), np.float64)
    want = np.asarray(want, np.float64)
    err = np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)
    assert err < tol, f"max-rel {err:.3e} >= {tol:.1e}"


def test_collapse_forward_matches_float64_equations():
    x, fn, base, scale, _ = _inputs()
    got = jax.jit(lambda *a: mhc_collapse(*a, **_kw()))(x, fn, base, scale)
    want = _numpy_collapse(x, fn, base, scale)
    assert [g.shape for g in got] == [(2, S, HC), (2, S, HC, HC), (2, S, W)]
    assert [g.dtype for g in got] == [jnp.float32, jnp.float32, jnp.float32]
    for g, w in zip(got, want, strict=True):
        _assert_close(g, w, 1e-5)
    np.testing.assert_allclose(np.asarray(got[1]).sum(-2), 1.0, atol=1e-5)  # trailing column normalisation


def test_explicit_hidden_size_changes_only_the_rms_denominator():
    x, fn, base, scale, _ = _inputs()
    got = jax.jit(lambda *a: mhc_collapse(*a, hidden_size=4 * W, **_kw()))(x, fn, base, scale)
    for g, w in zip(got, _numpy_collapse(x, fn, base, scale, hidden_size=4 * W), strict=True):
        _assert_close(g, w, 1e-5)
    default = jax.jit(lambda *a: mhc_collapse(*a, **_kw()))(x, fn, base, scale)
    assert not np.allclose(np.asarray(default[0]), np.asarray(got[0]))


def test_collapse_gradients_match_independent_equations():
    x, fn, base, scale, _ = _inputs()
    rng = np.random.default_rng(3)
    cts = [jnp.asarray(rng.standard_normal(s), jnp.float32) for s in ((2, S, HC), (2, S, HC, HC), (2, S, W))]

    def loss(f):
        return lambda *a: sum(jnp.sum(o.astype(jnp.float32) * c) for o, c in zip(f(*a), cts, strict=True))

    argnums = (0, 1, 2, 3)
    got = jax.jit(jax.grad(loss(lambda *a: mhc_collapse(*a, **_kw())), argnums=argnums))(x, fn, base, scale)
    want = jax.jit(jax.grad(loss(_jax_collapse_independent), argnums=argnums))(x, fn, base, scale)
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape and g.dtype == w.dtype
        _assert_close(g, np.asarray(w), 1e-4)


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
def test_expand_casts_gates_first_and_mixes_with_the_transposed_comb(dtype):
    x, _, _, _, out = _inputs()
    x, out = jnp.asarray(x, dtype), jnp.asarray(out, dtype)
    rng = np.random.default_rng(5)
    post = jnp.asarray(rng.uniform(0, 2, (2, S, HC)), jnp.float32)
    comb = jnp.asarray(rng.uniform(0, 1, (2, S, HC, HC)), jnp.float32)
    got = jax.jit(mhc_expand)(x, out, post, comb)
    assert got.dtype == x.dtype and got.shape == x.shape
    # independent: explicit per-target loop over sources, gates quantised to the stream dtype, fp64 accumulation
    xq = np.asarray(jnp.asarray(x, jnp.float32), np.float64)
    oq = np.asarray(jnp.asarray(out, jnp.float32), np.float64)
    pq = np.asarray(post.astype(dtype).astype(jnp.float32), np.float64)
    cq = np.asarray(comb.astype(dtype).astype(jnp.float32), np.float64)
    want = np.stack(
        [pq[..., i, None] * oq + sum(cq[..., j, i, None] * xq[..., j, :] for j in range(HC)) for i in range(HC)],
        axis=2,
    )
    _assert_close(got, want, 1e-5 if dtype == np.float32 else 1.6e-2)
    swapped = jax.jit(mhc_expand)(x, out, post, jnp.swapaxes(comb, -1, -2))
    assert not np.allclose(np.asarray(swapped, np.float32), np.asarray(got, np.float32))


def test_pack_unpack_roundtrip_and_stats_gates_match_logit_gates():
    rng = np.random.default_rng(7)
    pre, post = (jnp.asarray(rng.standard_normal((5, HC)), jnp.float32) for _ in range(2))
    comb = jnp.asarray(rng.standard_normal((5, HC, HC)), jnp.float32)
    packed = pack_gates(pre, post, comb)
    assert packed.shape == (5, 128)
    for a, b in zip(unpack_gates(packed, HC), (pre, post, comb), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    x, fn, base, scale, _ = _inputs()
    x2 = jnp.asarray(x[0])
    sumsq = jnp.sum(x2.astype(jnp.float32) ** 2, axis=(1, 2))
    mixp = jnp.einsum("shd,mhd->sm", x2, fn, precision=jax.lax.Precision.HIGHEST)
    stats = jnp.pad(jnp.concatenate([sumsq[:, None], mixp], -1), ((0, 0), (0, 128 - 1 - MIX)))
    _, post_s, comb_s = jax.jit(
        lambda s: gates_from_stats(s, base, scale, hc=HC, hidden_size=W, rms_eps=RMS_EPS, sinkhorn_iters=ITERS, eps=EPS)
    )(stats)
    post_ref, comb_ref, _ = _numpy_collapse(x[:1], fn, base, scale)
    _assert_close(post_s, post_ref[0], 1e-5)
    _assert_close(comb_s, comb_ref[0], 1e-5)


def test_hc_that_does_not_fit_the_packed_row_is_refused():
    with pytest.raises(ValueError, match="hc=12"):
        mhc_collapse(jnp.zeros((1, 2, 12, 8)), jnp.zeros((168, 12, 8)), jnp.zeros((168,)), jnp.ones((3,)))


def test_xla_registration_is_the_any_backend_fallback():
    for op in ("mhc_collapse", "mhc_expand"):
        assert kernel_registry.get(op, platform=Platform.XLA, backend=Backend.ANY) is not None
        assert kernel_registry.get(op, platform="xla", backend="tpu") is not None
