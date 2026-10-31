from __future__ import annotations

import importlib

import pytest

jnp = pytest.importorskip("jax.numpy")

_all_gather_module = importlib.import_module("ejkernel.kernels._pallas.tpu.all_gather_matmul._pallas_impl")
all_gather_matmul = _all_gather_module.all_gather_matmul
_reduce_scatter_module = importlib.import_module("ejkernel.kernels._pallas.tpu.reduce_scatter_matmul._pallas_impl")
reduce_scatter_matmul = _reduce_scatter_module.reduce_scatter_matmul


@pytest.mark.parametrize("rhs_transpose", [False, True])
def test_all_gather_matmul_tp_size_one_falls_back_to_local_matmul(rhs_transpose: bool):
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    if rhs_transpose:
        y = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)
        expected = jnp.dot(x, y.T, preferred_element_type=jnp.float32).astype(x.dtype)
    else:
        y = jnp.arange(20, dtype=jnp.float32).reshape(4, 5)
        expected = jnp.dot(x, y, preferred_element_type=jnp.float32).astype(x.dtype)

    out = all_gather_matmul(
        x,
        y,
        axis_name="tp",
        tp_size=1,
        rhs_transpose=rhs_transpose,
    )

    assert out.shape == expected.shape
    assert jnp.array_equal(out, expected)


def test_reduce_scatter_matmul_tp_size_one_falls_back_to_local_matmul():
    x = jnp.arange(18, dtype=jnp.float32).reshape(3, 6)
    y = jnp.arange(24, dtype=jnp.float32).reshape(4, 6)
    expected = jnp.dot(x, y.T, preferred_element_type=jnp.float32).astype(x.dtype)

    out = reduce_scatter_matmul(
        x,
        y,
        axis_name="tp",
        tp_size=1,
    )

    assert out.shape == expected.shape
    assert jnp.array_equal(out, expected)
