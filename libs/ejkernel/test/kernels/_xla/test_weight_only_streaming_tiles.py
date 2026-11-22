"""A16 tiles: full K and the widest N tile within the 2 MiB RHS buffer (codes counted at their bit width) at up to
12,288 rows for any 128-aligned shape; generic partial-K tiles elsewhere. The former measured families keep their
tiles (also pinned in ``test_grouped_general_tiles``)."""

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
        (256, 128, 640, 2560, jnp.int4, (64, 640, 2560)),
        (24, 3, 2560, 1280, jnp.int4, (16, 2560, 1280)),
        (3, 2, 128, 128, jnp.int8, (16, 128, 128)),
        # DeepSeek-V4.1 tp=4 per-device families (measured best tiles on TPU v5).
        (48, 96, 5120, 2304, jnp.int4, (16, 5120, 768)),
        (48, 96, 5120, 4608, jnp.int4, (16, 5120, 768)),
        (48, 96, 2304, 5120, jnp.int4, (16, 2304, 1280)),
        (48, 48, 5120, 2304, jnp.int4, (16, 5120, 768)),
        # DeepSeek-V4.1 prefill rows (measured 129..12288 rows; tile_m=64, same full-K / widest-N split).
        (256, 96, 5120, 2304, jnp.int4, (64, 5120, 768)),
        (1950, 96, 5120, 2304, jnp.int4, (64, 5120, 768)),
        (12288, 96, 2304, 5120, jnp.int4, (64, 2304, 1280)),
        (12289, 96, 2304, 5120, jnp.int4, (16, 512, 1024)),
        (1950, 96, 5120, 4608, jnp.int4, (64, 5120, 768)),
        (1950, 128, 2560, 1280, jnp.int4, (64, 2560, 1280)),
        (20480, 72, 4096, 512, jnp.int4, (16, 512, 128)),
        (16, 4, 160, 256, jnp.int4, (16, 128, 128)),
    ],
)
def test_a16_decode_policy(m, e, k, n, dtype, expected):
    policy = getattr(impl, "_weight_only_streaming_tiles", None)
    assert callable(policy), "missing weight-only tile policy"
    assert policy(m, e, k, n, jnp.dtype(dtype)) == expected
