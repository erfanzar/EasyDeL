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

"""TileLang parity tests for quantized_matmul."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.quantization import prepack_quantized_weights

from ._helpers import (
    _FP16_BWD_TOL,
    _FP16_FWD_TOL,
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_quantized_matmul_fwd_bwd():
    M, N, K = 64, 128, 256
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    x = _randn(k1, (M, K), scale=0.5)
    w = jax.random.randint(k2, (N, K), -64, 64).astype(jnp.int8)
    s = (jax.random.uniform(k3, (N,)) * 0.01).astype(jnp.float16)
    tl = _tl("quantized_matmul")
    ref = ((x.astype(jnp.float32) @ w.astype(jnp.float32).T) * s.astype(jnp.float32)).astype(jnp.float16)
    out_tl = tl(x, w, s, mode="affine", bits=8)
    assert _max_abs(out_tl, ref) < _FP16_FWD_TOL
    g_tl = jax.grad(lambda x: jnp.sum(tl(x, w, s, mode="affine", bits=8).astype(jnp.float32)))(x)
    g_ref = jax.grad(
        lambda x: jnp.sum((x.astype(jnp.float32) @ w.astype(jnp.float32).T * s.astype(jnp.float32)).astype(jnp.float32))
    )(x)
    assert _max_abs(g_tl, g_ref) < _FP16_BWD_TOL


@pytest.mark.parametrize("axis", ["row", "col"])
@pytest.mark.parametrize("bits", range(1, 9))
def test_quantized_matmul_packed_affine_fwd_bwd(axis, bits):
    M, N, K = 8, 64, 64
    group_size = 32
    key = jax.random.PRNGKey(_SEED + 28)
    kx, kw = jax.random.split(key, 2)
    x = _randn(kx, (M, K), scale=0.5)
    w = _randn(kw, (N, K), scale=0.5)
    wq, scales, zeros = prepack_quantized_weights(
        w,
        mode="affine",
        bits=bits,
        group_size=group_size,
        axis=axis,
    )
    kwargs = {
        "mode": "affine",
        "bits": bits,
        "group_size": group_size,
        "axis": axis,
        "use_bf16": False,
        "allow_dense_fallback": False,
        "block_m": 32,
        "block_n": 64,
        "block_k": 32,
    }
    tl, xla = _tl("quantized_matmul"), _xla("quantized_matmul")
    out_tl = tl(x, wq, scales, zeros, **kwargs)
    xla_kwargs = kwargs | {"allow_dense_fallback": True}
    out_xla = xla(x, wq, scales, zeros, **xla_kwargs)
    assert _max_abs(out_tl, out_xla) < 3e-2

    def loss(fn, call_kwargs, x_, s_, z_):
        out = fn(x_, wq, s_, z_, **call_kwargs).astype(jnp.float32)
        return jnp.sum(out * out)

    g_tl = jax.grad(lambda x_, s_, z_: loss(tl, kwargs, x_, s_, z_), argnums=(0, 1, 2))(x, scales, zeros)
    g_xla = jax.grad(lambda x_, s_, z_: loss(xla, xla_kwargs, x_, s_, z_), argnums=(0, 1, 2))(x, scales, zeros)
    for a, b, name in zip(g_tl, g_xla, ("x", "scales", "zeros"), strict=True):
        limit = 5e-3 * max(float(jnp.abs(b.astype(jnp.float32)).max()), 1.0) + 2.5e-1
        assert _max_abs(a, b) < limit, f"packed affine grad d{name} diff too large"


@pytest.mark.parametrize("axis", ["row", "col"])
@pytest.mark.parametrize("mode", ["nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"])
def test_quantized_matmul_packed_nonaffine_modes_fwd_bwd_x(axis, mode):
    M, N, K = 16, 64, 64
    key = jax.random.PRNGKey(_SEED + 29)
    kx, kw = jax.random.split(key, 2)
    x = _randn(kx, (M, K), scale=0.25)
    w = _randn(kw, (N, K), scale=0.25)
    wq, scales = prepack_quantized_weights(w, mode=mode, axis=axis)
    kwargs = {
        "mode": mode,
        "axis": axis,
        "use_bf16": False,
        "allow_dense_fallback": False,
        "block_m": 32,
        "block_n": 64,
        "block_k": 64,
    }
    tl, xla = _tl("quantized_matmul"), _xla("quantized_matmul")
    out_tl = tl(x, wq, scales, None, **kwargs)
    out_xla = xla(x, wq, scales, None, **kwargs)
    assert _max_abs(out_tl, out_xla) < 3e-2

    def loss(fn, x_):
        out = fn(x_, wq, scales, None, **kwargs).astype(jnp.float32)
        return jnp.sum(out * out)

    gx_tl = jax.grad(lambda x_: loss(tl, x_))(x)
    gx_xla = jax.grad(lambda x_: loss(xla, x_))(x)
    assert _max_abs(gx_tl, gx_xla) < 3e-2


@pytest.mark.parametrize("axis", ["row", "col"])
def test_quantized_matmul_packed_nf4_scale_grad(axis):
    M, N, K = 16, 64, 64
    key = jax.random.PRNGKey(_SEED + 30)
    kx, kw = jax.random.split(key, 2)
    x = _randn(kx, (M, K), scale=0.25)
    w = _randn(kw, (N, K), scale=0.25)
    wq, scales = prepack_quantized_weights(w, mode="nf4", axis=axis)
    kwargs = {
        "mode": "nf4",
        "axis": axis,
        "use_bf16": False,
        "allow_dense_fallback": False,
        "block_m": 32,
        "block_n": 64,
        "block_k": 64,
    }
    tl, xla = _tl("quantized_matmul"), _xla("quantized_matmul")

    def loss(fn, x_, s_):
        out = fn(x_, wq, s_, None, **kwargs).astype(jnp.float32)
        return jnp.sum(out * out)

    g_tl = jax.grad(lambda x_, s_: loss(tl, x_, s_), argnums=(0, 1))(x, scales)
    g_xla = jax.grad(lambda x_, s_: loss(xla, x_, s_), argnums=(0, 1))(x, scales)
    for a, b, name in zip(g_tl, g_xla, ("x", "scales"), strict=True):
        limit = 5e-3 * max(float(jnp.abs(b.astype(jnp.float32)).max()), 1.0) + 2.5e-1
        assert _max_abs(a, b) < limit, f"packed nf4 grad d{name} diff too large"
