"""CPU-safe explicit streaming platform contract."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import grouped_matmul_channelwise


def inputs(dtype=jnp.bfloat16):
    return (
        jnp.ones((3, 128), dtype),
        jnp.ones((2, 128, 128), jnp.int8),
        jnp.ones((2, 1, 128), jnp.float32),
        jnp.array([0, 3], jnp.int32),
    )


def test_platform_is_explicit_xla_default():
    assert inspect.signature(grouped_matmul_channelwise).parameters["platform"].default == "xla"
    args = inputs()
    np.testing.assert_array_equal(grouped_matmul_channelwise(*args), grouped_matmul_channelwise(*args, platform="xla"))


def test_registry_has_real_pallas_tpu_implementation():
    from ejkernel.kernels._registry import Backend, Platform, kernel_registry

    impl = kernel_registry.get("grouped_matmul_channelwise", platform=Platform.PALLAS, backend=Backend.TPU)
    assert "._pallas.tpu.grouped_matmul_channelwise." in impl.__module__


@pytest.mark.parametrize("bits", [2, 12])
def test_pallas_rejects_unsupported_activation_precision(bits):
    with pytest.raises(ValueError, match="activation_bits must be"):
        grouped_matmul_channelwise(*inputs(), activation_bits=bits, platform="pallas")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_pallas_rejects_non_bf16(dtype):
    with pytest.raises(TypeError, match="bfloat16"):
        grouped_matmul_channelwise(*inputs(dtype), platform="pallas")


def test_pallas_rejects_cpu():
    if jax.default_backend() != "cpu":
        pytest.skip("CPU rejection contract")
    with pytest.raises(ValueError, match="TPU"):
        grouped_matmul_channelwise(*inputs(), platform="pallas")


@pytest.mark.parametrize("platform", ["auto", None, "cuda"])
def test_unknown_platform_rejected(platform):
    with pytest.raises(ValueError, match="platform"):
        grouped_matmul_channelwise(*inputs(), platform=platform)


@pytest.mark.parametrize("bits", [4, 8])
def test_mocked_streaming_jit_ad_operands_and_exact_scales(monkeypatch, bits):
    """CPU-only AD plumbing test, NOT evidence of Pallas/TPU correctness."""
    import importlib

    impl = importlib.import_module("ejkernel.kernels._pallas.tpu.grouped_matmul_channelwise._interface")
    from ejkernel.kernels._xla.grouped_matmul_quant._channelwise import grouped_matmul_channelwise as reference

    x = (jnp.arange(24).reshape(8, 3) / 13).astype(jnp.bfloat16)
    q = (jnp.arange(18).reshape(3, 3, 2) % 7 - 3).astype(jnp.int4 if bits == 4 else jnp.int8)
    s = jnp.array([0.1234567, -0.2765432, 0.7123456, 0.3123456, 0.1919191, -0.4321234]).reshape(3, 1, 2)
    groups = jnp.array([3, 0, 5], jnp.int32)
    seen = []

    def fake_raw(a, b, sizes, **kwargs):
        assert b.dtype == q.dtype  # No full-bank floating cast before streaming.
        assert kwargs["maybe_quantize_lhs"] is False
        assert kwargs["preferred_element_type"] == jnp.float32
        assert kwargs["acc_dtype"] == jnp.float32
        tiles = kwargs["tile_info"]
        seen.append((tiles.tile_m, tiles.tile_k, tiles.tile_n))
        return jax.lax.ragged_dot(
            a.astype(jnp.float32), b.astype(jnp.float32), sizes, preferred_element_type=jnp.float32
        )

    monkeypatch.setattr(impl, "grouped_matmulv3_pallas_impl", fake_raw)
    monkeypatch.setattr(impl.jax, "default_backend", lambda: "tpu")
    f = jax.jit(
        lambda a, b, c, d: grouped_matmul_channelwise(a, b, c, d, platform="pallas", preferred_element_type=jnp.float32)
    )
    def ref(a, b, c, d):
        return reference(a, b, c, d, preferred_element_type=jnp.float32)
    np.testing.assert_allclose(f(x, q, s, groups), ref(x, q, s, groups), rtol=1e-6, atol=1e-6)
    assert seen and all(t == (16, 128, 128) for t in seen)
    dx, ds = jnp.ones_like(x), jnp.ones_like(s) * 0.17
    # First compile with all four dynamic operands; only then differentiate.
    _, got = jax.jvp(lambda a, c: f(a, q, c, groups), (x, s), (dx, ds))
    _, want = jax.jvp(lambda a, c: ref(a, q, c, groups), (x, s), (dx, ds))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    actual = jax.jit(jax.grad(lambda a, b, c, d: f(a, b, c, d).sum(), argnums=(0, 2)))
    expected = jax.grad(lambda a, b, c, d: ref(a, b, c, d).sum(), argnums=(0, 2))
    for got, want in zip(actual(x, q, s, groups), expected(x, q, s, groups), strict=False):
        np.testing.assert_allclose(got.astype(jnp.float32), want.astype(jnp.float32), rtol=1e-6, atol=1e-6)
