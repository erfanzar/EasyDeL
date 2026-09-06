"""Compiled JVP/VJP on the actual v5p widened W4A4 prefill family."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.quantized_matmul._integer_quantization import quantize_rows
from ejkernel.modules import grouped_matmul_channelwise

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU widened prefill AD")


@pytest.mark.parametrize("platform", ["xla", "pallas"])
def test_widened_prefill_compiled_ad(platform):
    m, e, k, n = 1280, 128, 640, 2560
    rng = np.random.default_rng(84)
    hx = np.asarray(jnp.asarray(rng.normal(size=(m, k)), jnp.bfloat16)).astype(np.float32)
    hw = rng.integers(-8, 8, (e, k, n), dtype=np.int8)
    hs = rng.uniform(0.001, 0.02, (e, 1, n)).astype(np.float32)
    sizes = np.zeros(e, np.int32)
    sizes[3] = 11
    sizes[97] = 9
    valid = 20
    x = jnp.asarray(hx, jnp.bfloat16).at[valid:].set(jnp.nan)
    w = jnp.asarray(hw).astype(jnp.int4)
    s = jnp.asarray(hs)
    g = jnp.asarray(sizes)
    f = jax.jit(lambda a, b, c, d: grouped_matmul_channelwise(a, b, c, d, activation_bits=4, platform=platform))
    dx = jnp.full_like(x, 0.125)
    ds = jnp.full_like(s, 0.01)
    _, dy = jax.jvp(lambda a, c: f(a, w, c, g), (x, s), (dx, ds))
    gx, gs = jax.grad(lambda a, c: f(a, w, c, g).astype(jnp.float32).sum(), argnums=(0, 1))(x, s)
    hq, hrs = jax.device_get(jax.jit(lambda a: quantize_rows(a, 4))(jnp.asarray(hx[:valid], jnp.bfloat16)))
    hq = np.asarray(hq).astype(np.int32)
    hrs = np.asarray(hrs)
    expected_dy = np.zeros((m, n), np.float32)
    expected_gx = np.zeros((m, k), np.float32)
    expected_gs = np.zeros_like(hs)
    start = 0
    for expert, count in enumerate(sizes):
        if not count:
            continue
        end = start + count
        weight = hw[expert].astype(np.float32)
        base = (hq[start:end] @ hw[expert].astype(np.int32)).astype(np.float32) * hrs[start:end]
        expected_dy[start:end] = (np.full((count, k), 0.125, np.float32) @ weight) * hs[expert] + base * np.float32(0.01)
        expected_gx[start:end] = np.sum(weight * hs[expert], axis=1)
        expected_gs[expert, 0] = base.sum(axis=0)
        start = end
    np.testing.assert_allclose(dy, np.asarray(jnp.asarray(expected_dy, jnp.bfloat16)), rtol=0.008, atol=1e-5)
    np.testing.assert_allclose(gx, np.asarray(jnp.asarray(expected_gx, jnp.bfloat16)), rtol=0.008, atol=1e-5)
    np.testing.assert_allclose(gs, expected_gs, rtol=2e-5, atol=2e-4)
    assert np.count_nonzero(np.asarray(dy)[valid:]) == 0
    assert np.count_nonzero(np.asarray(gx)[valid:]) == 0
