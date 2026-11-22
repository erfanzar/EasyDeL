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

"""Pallas TPU local expert FFN vs an independent fp32 dense per-expert reference, plus fallback/registry contracts."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.moe_local_ffn import _interface as pallas_entry
from ejkernel.kernels._pallas.tpu.moe_local_ffn import local_ffn_unsupported_reason, moe_local_ffn
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.moe_local_ffn import moe_local_ffn as moe_local_ffn_xla_entry
from jax import lax

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

BF16, F32 = jnp.bfloat16, jnp.float32
HI = lax.Precision.HIGHEST
T, H, M, EL, K = 2048, 2048, 256, 8, 8


def _act(g, u):
    return jax.nn.silu(jnp.minimum(g, 4.0)) * jnp.clip(u, -4.0, 4.0)


def reference(x, weights, experts, wg, wu, wd):
    x = x.astype(F32)
    out = jnp.zeros(x.shape, F32)
    for e in range(wg.shape[0]):
        coef = jnp.sum(jnp.where(experts == e, weights.astype(F32), 0.0), axis=1)
        h = _act(jnp.dot(x, wg[e].astype(F32), precision=HI), jnp.dot(x, wu[e].astype(F32), precision=HI))
        out = out + coef[:, None] * jnp.dot(h, wd[e].astype(F32), precision=HI)
    return out


def _inputs(seed, experts=None, dtype=BF16):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((T, H)), F32).astype(dtype)
    w = jnp.asarray(rng.uniform(0.05, 1.0, (T, K)), F32)
    if experts is None:
        experts = rng.integers(0, 4 * EL, (T, K))  # ~1/4 of the routes are local
    e = jnp.asarray(experts, jnp.int32)
    wg = jnp.asarray(0.03 * rng.standard_normal((EL, H, M)), F32).astype(dtype)
    wu = jnp.asarray(0.03 * rng.standard_normal((EL, H, M)), F32).astype(dtype)
    wd = jnp.asarray(0.03 * rng.standard_normal((EL, M, H)), F32).astype(dtype)
    ct = jnp.asarray(rng.standard_normal((T, H)), F32)
    return x, w, e, wg, wu, wd, ct


def _rel(a, b):
    a, b = np.asarray(jnp.asarray(a, F32), np.float64), np.asarray(jnp.asarray(b, F32), np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _check(x, w, e, wg, wu, wd, ct, *, tol=1.5e-2, **kw):
    def l_ffn(x, w, wg, wu, wd):
        return jnp.sum(moe_local_ffn(x, w, e, wg, wu, wd, _act, **kw).astype(F32) * ct)

    def l_ref(x, w, wg, wu, wd):
        return jnp.sum(reference(x, w, e, wg, wu, wd) * ct)

    args = (x, w, wg, wu, wd)
    assert (
        local_ffn_unsupported_reason(x, w, e, wg, wu, wd, kw.get("tokens_per_chunk"), kw.get("rows_per_window")) is None
    )
    out = jax.jit(lambda *a: moe_local_ffn(a[0], a[1], e, *a[2:], _act, **kw))(*args)
    assert out.dtype == BF16 and out.shape == (T, H)
    assert _rel(out, reference(x, w, e, wg, wu, wd)) < tol
    grad_fn = jax.jit(jax.grad(l_ffn, argnums=range(5)))
    hlo = grad_fn.lower(*args).compile().as_text()
    # pack + row gathers, gather-sums, grouped matmuls and in-place transposed grouped matmuls all run as Mosaic calls
    assert hlo.count("tpu_custom_call") >= 8, hlo.count("tpu_custom_call")
    g_ffn = grad_fn(*args)
    g_ref = jax.jit(jax.grad(l_ref, argnums=range(5)))(*args)
    for name, a, b in zip(("x", "weights", "w_gate", "w_up", "w_down"), g_ffn, g_ref, strict=True):
        assert a.dtype == b.dtype or name == "weights"
        assert np.isfinite(np.asarray(jnp.asarray(a, F32))).all(), name
        assert _rel(a, b) < tol, (name, _rel(a, b))


def test_matches_reference_and_runs_the_kernels():
    _check(*_inputs(0), tokens_per_chunk=1024, rows_per_window=1024)


def test_single_window_default_plan():
    _check(*_inputs(1))


def test_routing_skew_is_dropless():
    experts = np.random.default_rng(2).integers(0, 4 * EL, (T, K))
    experts[:, :6] = 3  # 6 of 8 routes per token hit local expert 3: at least 6144 local rows per 1024-token chunk
    _check(*_inputs(2, experts=experts), tokens_per_chunk=1024, rows_per_window=1024)


def test_unsupported_shapes_warn_once_and_run_xla():
    x, w, e, wg, wu, wd, _ = _inputs(3, dtype=F32)
    reason = local_ffn_unsupported_reason(x, w, e, wg, wu, wd)
    assert reason is not None and "bfloat16" in reason
    pallas_entry._WARNED.clear()
    fn = jax.jit(lambda *a: moe_local_ffn(*a, tokens_per_chunk=1024))
    with pytest.warns(RuntimeWarning, match="moe_local_ffn"):
        got = fn(x, w, e, wg, wu, wd)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jax.jit(lambda *a: moe_local_ffn(*a, tokens_per_chunk=512))(x, w, e, wg, wu, wd)
    want = jax.jit(lambda *a: moe_local_ffn_xla_entry(*a, tokens_per_chunk=1024))(x, w, e, wg, wu, wd)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_registry_entries_share_one_signature():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert kernel_registry.validate_signatures("moe_local_ffn")
    pallas = kernel_registry.get("moe_local_ffn", platform=Platform.PALLAS, backend=Backend.TPU)
    assert kernel_registry.get("moe_local_ffn", platform=Platform.XLA, backend=Backend.ANY) is not None
    assert kernel_registry.get("moe_local_ffn", backend=Backend.TPU) is pallas
