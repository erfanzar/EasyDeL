"""Explicit dense backend contract, without starting a TPU backend."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import channelwise_quantized_matmul as matmul


def inputs(m=64, k=128, n=128, dtype=jnp.bfloat16):
    return jnp.ones((m, k), dtype), jnp.ones((k, n), jnp.int4), jnp.ones((1, n), jnp.float32)


def test_dense_default_remains_xla():
    assert inspect.signature(matmul).parameters["platform"].default == "xla"
    x, w, s = inputs()
    np.testing.assert_array_equal(matmul(x, w, s), matmul(x, w, s, platform="xla"))


@pytest.mark.parametrize("platform", ["auto", "cuda", None])
def test_invalid_dense_platform(platform):
    with pytest.raises(ValueError, match="platform must be"):
        matmul(*inputs(), platform=platform)


def test_dense_pallas_rejects_weight_only_and_below_threshold():
    with pytest.raises(ValueError, match="active integer"):
        matmul(*inputs(), platform="pallas")
    with pytest.raises(ValueError, match="active integer"):
        matmul(*inputs(), quantize_activations=True, activation_bits=4, platform="pallas")


def test_dense_pallas_rejects_cpu():
    if jax.default_backend() != "cpu":
        pytest.skip("CPU rejection")
    with pytest.raises(ValueError, match="TPU"):
        matmul(*inputs(), quantize_activations=True, activation_bits=4, prefill_threshold=0, platform="pallas")


@pytest.mark.parametrize(
    "shape,dtype,message",
    [
        ((3, 128, 128), jnp.bfloat16, "divisible"),
        ((64, 160, 128), jnp.bfloat16, "divisible"),
        ((64, 128, 129), jnp.bfloat16, "divisible"),
        ((64, 128, 128), jnp.float32, "BF16"),
        ((64, 4224, 128), jnp.bfloat16, "<= 4096"),
    ],
)
def test_dense_pallas_validation_before_compilation(monkeypatch, shape, dtype, message):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    with pytest.raises(ValueError, match=message):
        matmul(
            *inputs(*shape, dtype), quantize_activations=True, activation_bits=4, prefill_threshold=0, platform="pallas"
        )


@pytest.mark.parametrize(
    "xshape,wshape",
    [
        ((64, 128), (256, 128)),
        ((64, 256), (128, 128)),
        ((64, 0), (0, 128)),
        ((64, 128), (128, 0)),
        ((1, 64, 128), (128, 128)),
        ((64, 128), (1, 128, 128)),
    ],
)
def test_reject_bad_dense_shapes_before_pallas_call(monkeypatch, xshape, wshape):
    from jax.experimental import pallas as pl

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid shape reached compilation")

    monkeypatch.setattr(pl, "pallas_call", forbidden)
    with pytest.raises(ValueError, match=r"rank.two|contracting|positive"):
        matmul(
            jnp.ones(xshape, jnp.bfloat16),
            jnp.ones(wshape, jnp.int4),
            jnp.ones((1, wshape[-1]), jnp.float32),
            quantize_activations=True,
            activation_bits=4,
            prefill_threshold=0,
            platform="pallas",
        )
