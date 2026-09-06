"""Conservative weight-only defaults for measured decode matrix families."""

import importlib

import jax.numpy as jnp
import pytest

impl = importlib.import_module("ejkernel.kernels._pallas.tpu.grouped_matmul_channelwise._interface")


@pytest.mark.parametrize(
    "m,e,k,n,dtype,expected",
    [
        (24, 128, 2560, 1280, jnp.int4, (16, 2560, 1280)),
        (24, 128, 2560, 1280, jnp.int8, (16, 2560, 640)),
        (80, 128, 640, 2560, jnp.int4, (16, 640, 2560)),
        (80, 128, 640, 2560, jnp.int8, (16, 640, 2560)),
        (256, 128, 640, 2560, jnp.int4, (16, 512, 1024)),
        (24, 3, 2560, 1280, jnp.int4, (16, 512, 1024)),
        (3, 2, 128, 128, jnp.int8, (16, 128, 128)),
    ],
)
def test_a16_decode_policy(m, e, k, n, dtype, expected):
    policy = getattr(impl, "_weight_only_streaming_tiles", None)
    assert callable(policy), "missing measured weight-only tile policy"
    assert policy(m, e, k, n, jnp.dtype(dtype)) == expected
