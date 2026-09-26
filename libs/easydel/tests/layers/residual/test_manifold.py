# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Independent mHC equation references, gradients, and checkpoint contracts."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.residual import (
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
    ManifoldHyperHead,
    MeanHyperHead,
    manifold_residual_write,
    mean_hyper_head,
)
from jax import numpy as jnp

CONFIG = ManifoldHyperConnectionConfig(hidden_size=5, hc_mult=3, eps=2e-5, norm_eps=3e-4, iters=5, initializer=0.07)


def _input(dtype):
    return jnp.asarray(np.random.default_rng(10).normal(size=(2, 3, 3, 5)), dtype)


def _module(cls, dtype):
    module = cls(CONFIG, dtype=dtype, param_dtype=dtype, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(42))
    names = ("fn", "base", "scale") if cls is ManifoldHyperConnection else ("hc_fn", "hc_base", "hc_scale")
    rng = np.random.default_rng(31)
    for name in names:
        parameter = getattr(module, name)
        # Nonzero bias and asymmetric projection/mixer weights expose swaps of
        # read/write gates, source/target axes, scale slots, or epsilon settings.
        parameter.value = jnp.asarray(rng.normal(size=parameter.value.shape) * 0.4 + 0.1, dtype)
    return module, tuple(getattr(module, name).value for name in names)


def _reference(x, weights, *, xp, head=False):
    """Equation reference using explicit stream sums and no production helpers.

    NumPy supplies forward expectations; JAX differentiates the same independent
    equations. Sinkhorn is expressed as a sequence of alternating normalization
    axes, starting and ending with column normalization.
    """
    fn, base, scale = (xp.asarray(w, dtype=xp.float32) for w in weights)
    xf = xp.asarray(x, dtype=xp.float32)
    flat = xf.reshape(x.shape[0], x.shape[1], -1)
    normalized = flat / xp.sqrt(xp.mean(flat * flat, axis=-1, keepdims=True) + CONFIG.norm_eps)
    logits = xp.stack([xp.sum(normalized * row, axis=-1) for row in fn], axis=-1)
    sigmoid = lambda z: 1 / (1 + xp.exp(-z))  # noqa: E731
    hc = CONFIG.hc_mult
    read = sigmoid(logits[..., :hc] * scale[0] + base[:hc]) + CONFIG.eps
    collapsed = sum(read[..., i, None] * xf[..., i, :] for i in range(hc)).astype(x.dtype)
    if head:
        return collapsed
    post = 2 * sigmoid(logits[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    scores = (logits[..., 2 * hc :] * scale[2] + base[2 * hc :]).reshape(*x.shape[:2], hc, hc)
    exp_scores = xp.exp(scores - xp.max(scores, axis=-1, keepdims=True))
    comb = exp_scores / xp.sum(exp_scores, axis=-1, keepdims=True) + CONFIG.eps
    for axis in [-2] + [-1, -2] * (CONFIG.iters - 1):
        comb = comb / (xp.sum(comb, axis=axis, keepdims=True) + CONFIG.eps)
    return post, comb, collapsed


def _objective(outputs):
    # Unequal cotangents ensure a doubly-stochastic matrix's constant total sum
    # cannot mask errors in its gradient or the stream orientation.
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    result = 0.0
    for output in outputs:
        weights = jnp.linspace(-0.7, 1.3, output.size).reshape(output.shape)
        result = result + jnp.sum(output.astype(jnp.float32) * weights)
    return result


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("cls", [ManifoldHyperConnection, ManifoldHyperHead])
def test_forward_and_jit_against_numpy(cls, dtype):
    module, weights = _module(cls, dtype)
    x = _input(dtype)
    expected = _reference(np.asarray(x), tuple(np.asarray(w) for w in weights), xp=np, head=cls is ManifoldHyperHead)
    actual = module(x)
    compiled = jax.jit(module)(x)
    if cls is ManifoldHyperHead:
        assert actual.shape == (*x.shape[:2], CONFIG.hidden_size)
        assert actual.dtype == dtype
        expected, actual, compiled = (expected,), (actual,), (compiled,)
    else:
        assert [a.shape for a in actual] == [(2, 3, 3), (2, 3, 3, 3), (2, 3, 5)]
        assert [a.dtype for a in actual] == [jnp.float32, jnp.float32, dtype]
        assert not np.allclose(actual[1], np.swapaxes(actual[1], -1, -2))
        np.testing.assert_allclose(np.asarray(actual[1]).sum(axis=-2), 1, atol=4e-5)
    for got, jit_got, want in zip(actual, compiled, expected, strict=True):
        tol = 8e-3 if got.dtype == jnp.bfloat16 else 3e-6
        np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), atol=tol, rtol=tol)
        np.testing.assert_allclose(np.asarray(jit_got, np.float32), np.asarray(want, np.float32), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("cls", [ManifoldHyperConnection, ManifoldHyperHead])
def test_input_and_parameter_gradients_against_independent_jax(cls, dtype):
    module, weights = _module(cls, dtype)
    x = _input(dtype)

    def reference_loss(x, weights):
        return _objective(_reference(x, weights, xp=jnp, head=cls is ManifoldHyperHead))

    expected_x, expected_weights = jax.grad(reference_loss, argnums=(0, 1))(x, weights)
    actual_x = jax.grad(lambda x: _objective(module(x)))(x)
    actual_weights = spx.grad(lambda m: _objective(m(x)))(module).flatten()
    names = ("fn", "base", "scale") if cls is ManifoldHyperConnection else ("hc_fn", "hc_base", "hc_scale")
    pairs = [(actual_x, expected_x)] + [
        (actual_weights[f"parameters/{name}"], want) for name, want in zip(names, expected_weights, strict=True)
    ]
    for got, want in pairs:
        assert got.dtype == dtype
        assert np.isfinite(np.asarray(got, np.float32)).all()
        np.testing.assert_allclose(
            np.asarray(got, np.float32),
            np.asarray(want, np.float32),
            rtol=3e-2 if dtype == jnp.bfloat16 else 2e-5,
            atol=3e-2 if dtype == jnp.bfloat16 else 3e-6,
        )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_residual_write_transpose_casts_and_gradients(dtype):
    x = _input(dtype)
    rng = np.random.default_rng(22)
    y = jnp.asarray(rng.normal(size=(2, 3, 5)), dtype)
    post = jnp.asarray(rng.normal(size=(2, 3, 3)), jnp.float32)
    comb = jnp.asarray(rng.normal(size=(2, 3, 3, 3)), jnp.float32)

    def reference(x, y, post, comb):
        # matmul (not the production einsum) provides independent orientation
        # and dot precision checks, with low-precision rounding at each step.
        mixer = jnp.swapaxes(comb.astype(x.dtype), -1, -2)
        mixed = jnp.matmul(mixer, x, precision=jax.lax.Precision.HIGHEST)
        return (post.astype(x.dtype)[..., None] * y[..., None, :]) + mixed

    def actual(x, y, post, comb):
        return manifold_residual_write(x, y, post, comb, precision=jax.lax.Precision.HIGHEST)

    np.testing.assert_array_equal(actual(x, y, post, comb), reference(x, y, post, comb))
    assert actual(x, y, post, comb).dtype == dtype
    got_grads = jax.grad(lambda *args: _objective(actual(*args)), argnums=(0, 1, 2, 3))(x, y, post, comb)
    want_grads = jax.grad(lambda *args: _objective(reference(*args)), argnums=(0, 1, 2, 3))(x, y, post, comb)
    for got, want in zip(got_grads, want_grads, strict=True):
        np.testing.assert_array_equal(got, want)
    if dtype == jnp.bfloat16:
        # This fixture distinguishes pre-multiply casts from an fp32 write and
        # late cast, which is a numerically different residual update.
        late_cast = reference(x.astype(jnp.float32), y.astype(jnp.float32), post, comb).astype(dtype)
        assert not np.array_equal(actual(x, y, post, comb), late_cast)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_mean_head_parameter_free_values_and_gradients(dtype):
    x = _input(dtype)
    module = MeanHyperHead()
    _, state = spx.export(module)
    assert state.flatten() == {}
    want = np.asarray(x, np.float32).mean(axis=2).astype(x.dtype)
    for fn in (module, mean_hyper_head):
        actual = fn(x)
        assert actual.dtype == dtype
        np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(want, np.float32), rtol=2e-7, atol=2e-7)
        gradient = jax.grad(lambda x, fn=fn: fn(x).astype(jnp.float32).sum())(x)
        np.testing.assert_array_equal(gradient, jnp.full_like(x, 1 / CONFIG.hc_mult))


@pytest.mark.parametrize("cls,prefix", [(ManifoldHyperConnection, ""), (ManifoldHyperHead, "hc_")])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_checkpoint_paths_shapes_rng_and_rebind(cls, prefix, dtype):
    rngs = spx.Rngs(17)
    module = cls(CONFIG, dtype=dtype, param_dtype=dtype, rngs=rngs)
    graphdef, state = spx.export(module)
    leaves = state.flatten()
    hc, hidden = CONFIG.hc_mult, CONFIG.hidden_size
    width = (2 + hc) * hc if cls is ManifoldHyperConnection else hc
    shapes = {
        f"{prefix}fn": (width, hc * hidden),
        f"{prefix}base": (width,),
        f"{prefix}scale": (3 if not prefix else 1,),
    }
    assert set(leaves) == {f"parameters/{name}" for name in shapes}
    for name, shape in shapes.items():
        assert leaves[f"parameters/{name}"].shape == shape
        assert leaves[f"parameters/{name}"].dtype == dtype
    expected_rngs = spx.Rngs(17)
    expected_fn = jax.nn.initializers.normal(CONFIG.initializer)(expected_rngs.param, (width, hc * hidden), dtype)
    np.testing.assert_array_equal(leaves[f"parameters/{prefix}fn"], expected_fn)
    np.testing.assert_array_equal(leaves[f"parameters/{prefix}base"], jnp.zeros((width,), dtype))
    np.testing.assert_array_equal(leaves[f"parameters/{prefix}scale"], jnp.ones(shapes[f"{prefix}scale"], dtype))
    _ = expected_rngs.param  # zeros still consume their parameter key
    _ = expected_rngs.param  # ones still consume their parameter key
    np.testing.assert_array_equal(jax.random.key_data(rngs.param), jax.random.key_data(expected_rngs.param))
    restored = spx.bind(graphdef, state)
    original_output, restored_output = module(_input(dtype)), restored(_input(dtype))
    for got, want in zip(jax.tree.leaves(restored_output), jax.tree.leaves(original_output), strict=True):
        np.testing.assert_array_equal(got, want)
