"""Pure-shape regression for the integer streaming RHS buffer budget."""

import importlib

import jax
import jax.numpy as jnp
import pytest

impl = importlib.import_module("ejkernel.kernels._pallas.tpu.grouped_matmul_channelwise._interface")


@pytest.mark.parametrize(
    "k,n,dtype",
    [
        (128, 131072, jnp.int8),
        (128, 262144, jnp.int4),
        (65536, 4096, jnp.int8),
        (2560, 1280, jnp.int4),
        (640, 2560, jnp.int4),
    ],
)
def test_integer_tile_budget(k, n, dtype):
    policy = getattr(impl, "_integer_streaming_tiles", None)
    assert callable(policy), "missing independently testable bounded policy"
    m, tk, tn = policy(k, n, jnp.dtype(dtype))
    assert m == 32 and tk >= 128 and tn >= 128 and tk % 128 == tn % 128 == 0
    assert tk * tn * jax.dtypes.itemsize_bits(dtype) // 8 <= 2 * 1024 * 1024
    if (k, n) == (2560, 1280):
        assert (tk, tn) == (2560, 640)
    if (k, n) == (640, 2560):
        assert (tk, tn) == (640, 1280)
