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

"""Independent numerical contracts for the public mHC coefficient operation."""

from functools import partial

import jax
import numpy as np
import pytest
from ejkernel.modules import mhc_coefficients
from jax import numpy as jnp


def reference(logits, base, scale, hc, iters, eps):
    # Split the three independent groups rather than share the production
    # broadcast-index implementation. Retain the specified normalization order.
    pre = jax.nn.sigmoid(logits[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * jax.nn.sigmoid(logits[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    z = logits[..., 2 * hc :] * scale[2] + base[2 * hc :]
    matrix = jax.nn.softmax(z.reshape(*z.shape[:-1], hc, hc), axis=-1) + eps
    matrix /= matrix.sum(-2, keepdims=True) + eps
    for _ in range(iters - 1):
        matrix /= matrix.sum(-1, keepdims=True) + eps
        matrix /= matrix.sum(-2, keepdims=True) + eps
    return pre, post, matrix


@pytest.mark.parametrize("platform", ["xla", "pallas"])
@pytest.mark.parametrize(
    "shape,hc,iters,eps",
    [
        ((1,), 4, 1, 1e-4),
        ((2, 8), 4, 2, 1e-6),
        ((129,), 4, 20, 1e-6),
        ((2, 127), 4, 7, 1e-3),
        ((0,), 4, 20, 1e-6),
        ((), 4, 20, 1e-6),
        ((3,), 2, 20, 1e-6),
        ((7,), 4, 21, 1e-6),
    ],
)
def test_public_forward_and_nonuniform_gradients(platform, shape, hc, iters, eps):
    if platform == "pallas" and jax.default_backend() != "tpu":
        pytest.skip("physical TPU required")
    rng = np.random.default_rng(918)
    width = hc * (hc + 2)
    args = (
        jnp.asarray(rng.normal(size=(*shape, width)), jnp.float32),
        jnp.asarray(rng.normal(size=width), jnp.float32),
        jnp.array([0.7, 1.3, -0.4], jnp.float32),
    )
    fn = partial(mhc_coefficients, hc_mult=hc, n_iters=iters, eps=eps, platform=platform)
    ref = partial(reference, hc=hc, iters=iters, eps=eps)
    expected = jax.jit(ref)(*args)
    actual = jax.jit(fn)(*args)
    weights = tuple(jnp.asarray(rng.normal(size=x.shape), jnp.float32) for x in expected)

    def loss(call, *xs):
        return sum(jnp.sum(a * b) for a, b in zip(call(*xs), weights, strict=True))

    expected_grad = jax.jit(jax.grad(partial(loss, ref), argnums=(0, 1, 2)))(*args)
    actual_grad = jax.jit(jax.grad(partial(loss, fn), argnums=(0, 1, 2)))(*args)
    for value, want in zip((*actual, *actual_grad), (*expected, *expected_grad), strict=True):
        assert value.shape == want.shape and value.dtype == jnp.float32
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, want, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("setting,value", [("hc_mult", 0), ("n_iters", 0), ("eps", 0.0), ("eps", float("nan"))])
def test_invalid_settings(setting, value):
    with pytest.raises(ValueError):
        mhc_coefficients(jnp.zeros((2, 24)), jnp.zeros(24), jnp.ones(3), platform="xla", **{setting: value})


def test_xla_jvp_and_second_derivative():
    x = jnp.linspace(-1, 1, 48).reshape(2, 24)
    base, scale = jnp.zeros(24), jnp.ones(3)

    def fn(z):
        return mhc_coefficients(z, base, scale, platform="xla")

    def ref(z):
        return reference(z, base, scale, 4, 20, 1e-6)

    def jvp(f):
        return jax.jvp(f, (x,), (jnp.ones_like(x),))

    def jvp_of_grad(f):
        return jax.jvp(jax.grad(lambda z: jnp.sum(f(z)[0] ** 2)), (x,), (jnp.ones_like(x),))

    for call in (jvp, jvp_of_grad):
        for a, b in zip(jax.tree.leaves(call(fn)), jax.tree.leaves(call(ref)), strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=3e-6)
