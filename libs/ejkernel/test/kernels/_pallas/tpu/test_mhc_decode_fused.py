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

"""Decode mHC read / write kernels (Pallas TPU, APPROXIMATE class) against the unfused XLA path.

The read is checked bit for bit against the registered packed ``mhc_coefficients`` kernel plus the float32 XLA
collapse. The write is checked against the XLA residual write to within one bf16 ulp per element (XLA's own rounding
of that write depends on its fusion context). One chained layer (two sites, flat stream carry) is checked against the
unfused chain within the stated APPROXIMATE tolerance.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from ejkernel.modules import (
    mhc_coefficients,
    mhc_decode_read,
    mhc_decode_supports,
    mhc_decode_table_gates,
    mhc_decode_write,
)
from jax import lax
from jax import numpy as jnp

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernels need a TPU")

HC, ITERS, EPS, NORM_EPS = 4, 20, 1e-6, 1e-5
MIX = (HC + 2) * HC


def _differing(got, want):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape and got.dtype == want.dtype, (got.shape, want.shape, got.dtype, want.dtype)
    view = np.uint16 if got.dtype.itemsize == 2 else np.uint32
    return int(np.count_nonzero(got.view(view) != want.view(view)))


def _gates_inputs(tokens, seed):
    rng = np.random.default_rng(seed)
    logits = jnp.asarray(2.5 * rng.standard_normal((tokens, MIX)), jnp.float32)
    base = jnp.asarray(0.5 * rng.standard_normal((MIX,)), jnp.bfloat16).astype(jnp.float32)
    scale = jnp.asarray(1.0 + 0.1 * rng.standard_normal((3,)), jnp.bfloat16).astype(jnp.float32)
    return logits, base, scale


def _xla_write(x, out, post, comb):
    dtype = x.dtype
    return post.astype(dtype)[..., None] * out[..., None, :] + jnp.einsum("tji,tjd->tid", comb.astype(dtype), x)


@pytest.mark.parametrize("tokens,width", [(8, 512), (32, 4096), (128, 1024)])
def test_read_bitwise(tokens, width):
    logits, base, scale = _gates_inputs(tokens, 0)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((tokens, HC, width)), jnp.bfloat16)

    @jax.jit
    def reference(logits, base, scale, x):
        pre, post, comb = mhc_coefficients(logits, base, scale, hc_mult=HC, n_iters=ITERS, eps=EPS)
        return pre, post, comb, jnp.sum(pre[..., None] * x.astype(jnp.float32), axis=1).astype(x.dtype)

    @jax.jit
    def fused(logits, base, scale, x):
        table, collapsed = mhc_decode_read(logits, base, scale, x.reshape(tokens, -1), n_iters=ITERS, eps=EPS)
        return (*mhc_decode_table_gates(table, HC), collapsed)

    names = ("pre", "post", "comb", "collapsed")
    got, want = fused(logits, base, scale, x), reference(logits, base, scale, x)
    diffs = {n: _differing(g, w) for n, g, w in zip(names, got, want, strict=True)}
    assert diffs == dict.fromkeys(names, 0), diffs


@pytest.mark.parametrize("tokens,width", [(8, 512), (32, 4096)])
def test_write_within_one_ulp(tokens, width):
    logits, base, scale = _gates_inputs(tokens, 2)
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal((tokens, HC, width)), jnp.bfloat16)
    out = jnp.asarray(rng.standard_normal((tokens, width)), jnp.bfloat16)
    table, _ = jax.jit(lambda *a: mhc_decode_read(*a, n_iters=ITERS, eps=EPS))(
        logits, base, scale, x.reshape(tokens, -1)
    )
    _, post, comb = mhc_decode_table_gates(table, HC)
    got = np.asarray(jax.jit(mhc_decode_write)(x.reshape(tokens, -1), out, table), np.float32)
    want = np.asarray(jax.jit(_xla_write)(x, out, post, comb).reshape(tokens, -1), np.float32)
    diff = np.abs(got - want)
    assert (diff <= 2.0**-7 * np.maximum(np.abs(got), np.abs(want))).all(), float(diff.max())
    assert np.count_nonzero(diff) <= max(1, got.size // 10000), np.count_nonzero(diff)


def _site_params(rng, width):
    return (
        jnp.asarray(0.02 * rng.standard_normal((MIX, HC * width)), jnp.bfloat16),
        jnp.asarray(0.5 * rng.standard_normal((MIX,)), jnp.bfloat16),
        jnp.asarray(1.0 + 0.1 * rng.standard_normal((3,)), jnp.bfloat16),
        jnp.asarray(1.0 + 0.2 * rng.standard_normal((width,)), jnp.bfloat16),
    )


def _project(flat, fn):
    scale = lax.rsqrt(jnp.mean(jnp.square(flat.astype(jnp.float32)), axis=-1, keepdims=True) + NORM_EPS)
    return jnp.matmul(flat, fn.T, preferred_element_type=jnp.float32) * scale


def _norm(c, w):
    c = c.astype(jnp.float32)
    return (w * (c * lax.rsqrt(jnp.mean(jnp.square(c), -1, keepdims=True) + NORM_EPS)).astype(jnp.bfloat16)).astype(
        jnp.bfloat16
    )


def test_one_layer_chain_within_approximate_tolerance():
    """Two sites: max |d| <= 0.25 on the streams and <= 0.0625 on the normed inputs (one-layer measurement on TPU
    v5p: 0.125 and 0.0156)."""
    tokens, width = 32, 4096
    rng = np.random.default_rng(5)
    params = [_site_params(rng, width) for _ in range(2)]
    x0 = jnp.asarray(rng.standard_normal((tokens, HC, width)), jnp.bfloat16)

    @jax.jit
    def reference(x, params):
        normed = []
        for fn, base, scale, w in params:
            logits = _project(x.reshape(tokens, -1), fn)
            pre, post, comb = mhc_coefficients(
                logits, base.astype(jnp.float32), scale.astype(jnp.float32), hc_mult=HC, n_iters=ITERS, eps=EPS
            )
            y = _norm(jnp.sum(pre[..., None] * x.astype(jnp.float32), axis=1).astype(x.dtype), w)
            normed.append(y)
            x = _xla_write(x, lax.optimization_barrier(y), post, comb)
        return x.reshape(tokens, -1), normed

    @jax.jit
    def fused(x, params):
        x, normed = x.reshape(tokens, -1), []
        for fn, base, scale, w in params:
            table, col = mhc_decode_read(_project(x, fn), base, scale, x, n_iters=ITERS, eps=EPS)
            y = _norm(col, w)
            normed.append(y)
            x = mhc_decode_write(x, lax.optimization_barrier(y), table)
        return x, normed

    got, want = fused(x0, params), reference(x0, params)
    stream_err = float(jnp.max(jnp.abs(got[0].astype(jnp.float32) - want[0].astype(jnp.float32))))
    normed_err = max(
        float(jnp.max(jnp.abs(g.astype(jnp.float32) - w.astype(jnp.float32))))
        for g, w in zip(got[1], want[1], strict=True)
    )
    assert np.isfinite(np.asarray(got[0], np.float32)).all()
    assert stream_err <= 0.25 and normed_err <= 0.0625, (stream_err, normed_err)


def test_rejects_unsupported_shapes():
    assert (
        not mhc_decode_supports(129, HC, 512)
        and not mhc_decode_supports(8, 3, 512)
        and not mhc_decode_supports(8, HC, 500)
    )
    logits, base, scale = _gates_inputs(8, 0)
    with pytest.raises(ValueError):
        mhc_decode_read(logits, base[:-1], scale, jnp.zeros((8, HC * 512), jnp.bfloat16))
    with pytest.raises(ValueError):
        mhc_decode_write(
            jnp.zeros((8, HC * 512), jnp.bfloat16), jnp.zeros((8, 384), jnp.bfloat16), jnp.zeros((1, 8, 128))
        )
