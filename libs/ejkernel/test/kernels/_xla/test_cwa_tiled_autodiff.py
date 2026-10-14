"""CPU regressions for bounded-query compressed-attention autodiff."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
    compressed_window_attention_tpu,
)
from ejkernel.kernels._xla.compressed_window_attention._xla_impl_fwd import (
    compressed_window_attention_xla,
)


@pytest.mark.parametrize("use_sink", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_tiled_reference_primal_jvp_and_vjp(use_sink, dtype):
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import (
        _dense_reference,
        _tiled_reference,
    )

    keys = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(keys[0], (1, 2, 19, 8), dtype=dtype)
    kv = jax.random.normal(keys[1], (1, 25, 8), dtype=dtype)
    bias = jax.random.normal(keys[2], (1, 19, 25)) * 0.1
    sink = jax.random.normal(keys[3], (2,))
    args = (q, kv, bias, sink)
    tangents = tuple(jnp.ones_like(a) * 0.01 for a in args)

    def ref(a, b, c, d):
        return _dense_reference(a, b, c, d if use_sink else None, 0.3)

    def tiled(a, b, c, d):
        return _tiled_reference(a, b, c, d if use_sink else None, 0.3, block_q=8)

    atol = 0.02 if dtype == jnp.bfloat16 else 2e-5
    expected, expected_tan = jax.jvp(ref, args, tangents)
    actual, actual_tan = jax.jvp(tiled, args, tangents)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)
    np.testing.assert_allclose(actual_tan, expected_tan, atol=atol, rtol=atol)
    cotangent = jnp.sin(jnp.arange(q.size).reshape(q.shape)).astype(dtype)
    expected_grads = jax.vjp(ref, *args)[1](cotangent)
    actual_grads = jax.vjp(tiled, *args)[1](cotangent)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert np.all(np.isfinite(actual_grad))
        np.testing.assert_allclose(actual_grad, expected_grad, atol=atol, rtol=atol)


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="Run with JAX_PLATFORMS=cpu for Pallas interpret")
@pytest.mark.parametrize("use_sink", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("band", [False, True])
def test_custom_jvp_masked_rows_cpu_interpret(use_sink, dtype, band):
    """Exercise the actual Pallas primal, not just its reference AD surrogate."""
    # Force CPU placement: the wrapper itself enables Pallas interpret off TPU.
    with jax.default_device(jax.devices("cpu")[0]):
        _check_custom_jvp_masked_rows(use_sink, dtype, band)


def _check_custom_jvp_masked_rows(use_sink, dtype, band):
    token_len = 513 if band else 19  # five token tiles; only two are visited
    entries = 3 if band else 0  # independently padded token and entry regions
    kv_len = token_len + entries
    q_len, dim = 9, 8
    rng = np.random.default_rng(41)
    q = jnp.asarray(rng.normal(size=(1, 2, q_len, dim)) * 0.2, dtype)
    # Deliberately nonzero, nonuniform values expose both dilution by physical
    # zero padding and omission of distant, masked token tiles.
    kv = jnp.asarray(rng.normal(size=(1, kv_len, dim)) * 0.2 + np.arange(kv_len)[None, :, None] / kv_len + 1, dtype)
    qi, ki = np.arange(q_len)[:, None], np.arange(kv_len)[None, :]
    visible = (ki <= qi) & (ki > qi - 3)
    bias = np.where(visible, 0.0, np.finfo(np.float32).min).astype(np.float32)[None]
    bias[:, [0, 8], :] = np.finfo(np.float32).min
    bias = jnp.asarray(bias)
    sink = jnp.asarray([0.2, -0.3], jnp.float32)
    args = (q, kv, bias, sink)

    def actual(a, b, c, d):
        return compressed_window_attention_tpu(
            a,
            b,
            c,
            d if use_sink else None,
            softmax_scale=0.3,
            window=3 if band else 0,
            token_kv_len=token_len,
        )

    def reference(a, b, c, d):
        return compressed_window_attention_xla(
            a,
            b,
            c,
            d if use_sink else None,
            softmax_scale=0.3,
        )

    # All operands participate in reference AD, including the learnable bias.
    tangents = tuple(jnp.asarray(rng.normal(size=a.shape) * 0.01, a.dtype) for a in args)
    tol = 0.025 if dtype == jnp.bfloat16 else 3e-5
    # Check the public primal first so the unchanged source fails numerically,
    # before exercising its derivative implementation.
    np.testing.assert_allclose(actual(*args), reference(*args), atol=tol, rtol=tol)
    expected, expected_tangent = jax.jvp(reference, args, tangents)
    result, tangent = jax.jvp(actual, args, tangents)
    np.testing.assert_allclose(result, expected, atol=tol, rtol=tol)
    np.testing.assert_allclose(tangent, expected_tangent, atol=tol, rtol=tol)
    cot = jnp.asarray(rng.normal(size=q.shape), dtype)
    expected_grads = jax.vjp(reference, *args)[1](cot)
    actual_grads = jax.vjp(actual, *args)[1](cot)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, want, atol=tol, rtol=tol)
    assert np.any(np.asarray(expected_grads[2]) != 0), "bias cotangent must not be suppressed"

    if dtype == jnp.float32:
        # Finite differences on masked rows use zero queries: adding a tiny
        # score to finfo.min rounds away in the primal, whereas reference AD
        # still differentiates that addition. Isolate the genuine value path
        # so this check does not conflate that existing numerical AD contract.
        zero_q = jnp.zeros_like(q)
        direction = jnp.asarray(rng.normal(size=kv.shape) * 0.2, dtype)

        def values(b):
            return actual(zero_q, b, bias, sink)

        _, value_tangent = jax.jvp(values, (kv,), (direction,))
        eps = 0.01
        finite_difference = (values(kv + eps * direction) - values(kv - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(value_tangent, finite_difference, atol=4e-5, rtol=2e-3)


def test_tiled_reference_never_builds_full_query_head_scores():
    from ejkernel.kernels._pallas.tpu.compressed_window_attention._pallas_impl_fwd import _tiled_reference

    q = jnp.ones((1, 2, 129, 8))
    kv = jnp.ones((1, 137, 8))
    bias = jnp.zeros((1, 129, 137))

    def forward(a, b, c):
        return _tiled_reference(a, b, c, None, 0.3, block_q=16)

    graphs = [
        jax.make_jaxpr(forward)(q, kv, bias),
        jax.make_jaxpr(lambda a, b, c: jax.jvp(forward, (a, b, c), tuple(jnp.ones_like(x) for x in (a, b, c))))(
            q, kv, bias
        ),
        jax.make_jaxpr(jax.grad(lambda a, b, c: forward(a, b, c).sum(), argnums=(0, 1, 2)))(q, kv, bias),
    ]

    def visit(jaxpr):
        jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
        for eq in jaxpr.eqns:
            for var in eq.outvars:
                shape = getattr(getattr(var, "aval", None), "shape", ())
                assert shape != (1, 2, 129, 137)
                assert shape != (1, 2, 144, 137)
                assert shape != (9, 1, 2, 16, 137), "reverse AD retained every score tile"
            for value in eq.params.values():
                if hasattr(value, "jaxpr") or hasattr(value, "eqns"):
                    visit(value)

    for graph in graphs:
        visit(graph)
