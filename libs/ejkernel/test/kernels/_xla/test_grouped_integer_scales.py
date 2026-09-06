"""Integer grouped weights must retain fractional scales in forward/AD."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.grouped_matmulv3._interface import _apply_rhs_scale_bias


@pytest.mark.parametrize("dtype", [jnp.int4, jnp.int8])
@pytest.mark.parametrize("transpose", [False, True])
def test_fractional_integer_weight_scales(dtype, transpose):
    q = jnp.array([[[1, -2], [3, 4], [-1, 2], [5, -3]]], dtype)
    scales = jnp.array([[[[0.25, 0.5]], [[0.125, 0.0625]]]], jnp.float32)
    stored = q.swapaxes(1, 2) if transpose else q
    got, _ = _apply_rhs_scale_bias(stored, scales, None, transpose_rhs=transpose)
    want = np.asarray(q).astype(np.float32) * np.repeat(np.asarray(scales)[:, :, 0, :], 2, axis=1)
    np.testing.assert_array_equal(got, want)
    assert jnp.issubdtype(got.dtype, jnp.floating)
    grad = jax.grad(lambda s: _apply_rhs_scale_bias(stored, s, None, transpose_rhs=transpose)[0].sum())(scales)
    np.testing.assert_array_equal(grad, np.asarray(q).astype(np.float32).reshape(1, 2, 2, 2).sum(axis=2, keepdims=True))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_public_integer_weight_forward_and_input_gradient(dtype):
    from ejkernel.modules import grouped_matmul

    q = jnp.array([[[1, -2], [3, 4], [-1, 2], [5, -3]]], jnp.int4)
    scales = jnp.array([[[[0.25, 0.5]]]], jnp.float32)
    x = jnp.ones((2, 4), dtype)
    sizes = jnp.array([2], jnp.int32)
    def f(a):
        return grouped_matmul(
            a, q, sizes, rhs_scale=scales, use_v3=True, platform="xla", preferred_element_type=dtype
        )
    got = jax.jit(f)(x)
    np.testing.assert_array_equal(got, np.array([[2.0, 0.5], [2.0, 0.5]], np.float32))
    grad = jax.grad(lambda a: f(a).astype(jnp.float32).sum())(x)
    want = np.sum(np.asarray(q).astype(np.float32)[0] * np.array([0.25, 0.5]), axis=1)
    np.testing.assert_array_equal(grad, np.broadcast_to(want, (2, 4)))
