"""Actual dense integer arithmetic and the explicit activation STE contract."""

import jax
import numpy as np
import pytest
from ejkernel.kernels._xla.quantized_matmul._channelwise import channelwise_quantized_matmul
from jax import numpy as jnp


def _data(bits):
    rng = np.random.default_rng(512)
    x = rng.normal(size=(5, 16)).astype(np.float32) * 1.73
    x[0] = 0
    w = rng.integers(-7, 8, size=(16, 9)).astype(np.int8)
    s = rng.uniform(0.03, 0.71, size=(1, 9)).astype(np.float32)
    dx = rng.normal(size=x.shape).astype(np.float32)
    ds = rng.normal(size=s.shape).astype(np.float32)
    cot = rng.normal(size=(5, 9)).astype(np.float32)
    bound = 2 ** (bits - 1) - 1
    xs = np.max(np.abs(x), axis=1, keepdims=True) / bound
    codes = np.clip(np.rint(x / np.where(xs == 0, 1, xs)), -bound, bound).astype(np.int32)
    base = (codes @ w.astype(np.int32)).astype(np.float32) * xs
    return x, w, s, dx, ds, cot, base


@pytest.mark.parametrize("bits", [4, 8])
def test_integer_forward_and_jitted_ste(bits):
    x, w, s, dx, ds, cot, base = _data(bits)
    wq = jnp.asarray(w, dtype=jnp.int4 if bits == 4 else jnp.int8)
    def fn(a, b):
        return channelwise_quantized_matmul(
            a, wq, b, quantize_activations=True, activation_bits=bits, prefill_threshold=0
        )
    args = (jnp.asarray(x), jnp.asarray(s))
    for forward in (fn, jax.jit(fn)):
        np.testing.assert_allclose(forward(*args), base * s, rtol=3e-6, atol=3e-6)
    # A chosen surrogate, not the natural derivative of hard rounding. In
    # particular, a zero activation row must still transmit dx @ W.
    y, dy = jax.jit(lambda a, b, da, db: jax.jvp(fn, (a, b), (da, db)))(*args, dx, ds)
    np.testing.assert_allclose(y, base * s, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(dy, (dx @ w.astype(np.float32)) * s + base * ds, rtol=3e-6, atol=3e-6)

    def backward(a, b, c):
        _, pullback = jax.vjp(fn, a, b)
        return pullback(c)

    gx, gs = jax.jit(backward)(*args, cot)
    np.testing.assert_allclose(gx, (cot * s) @ w.astype(np.float32).T, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(gs, np.sum(cot * base, axis=0, keepdims=True), rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("quantize,threshold", [(False, 0), (True, 6)])
def test_weight_only_and_legacy_decode_unchanged(bits, quantize, threshold):
    x, w, s, dx, ds, _, _ = _data(bits)
    wq = jnp.asarray(w, dtype=jnp.int4 if bits == 4 else jnp.int8)
    def fn(a, b):
        return channelwise_quantized_matmul(
            a, wq, b, quantize_activations=quantize, activation_bits=bits, prefill_threshold=threshold
        )
    y, dy = jax.jit(lambda a, b, da, db: jax.jvp(fn, (a, b), (da, db)))(x, s, dx, ds)
    base = x @ w.astype(np.float32)
    np.testing.assert_allclose(y, base * s, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(dy, (dx @ w.astype(np.float32)) * s + base * ds, rtol=3e-6, atol=3e-6)
