"""Quantized projections preserve scaling, placement, and the requested compute format."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.sharding import TensorLayout
from easydel.layers.linears import (
    ColumnParallelLinear,
    ColumnParallelLinearQuantized,
    RowParallelLinear,
)
from easydel.layers.quantization import QuantizationConfig, QuantizationType
from ejkernel.quantization import dequantize, prepack_quantized_weights
from jax import numpy as jnp

IN, OUT = 64, 64


def _x():
    return jnp.asarray(np.random.default_rng(0).normal(size=(4, IN)), dtype=jnp.float32)


@pytest.mark.parametrize("linear_class", [ColumnParallelLinear, RowParallelLinear])
@pytest.mark.parametrize("scale", ["fan_in", "fan_out", 0.5])
def test_quantized_weight_override_scales_before_bias(linear_class, scale):
    """An exact weight override isolates output scaling from quantization error."""
    dense = linear_class(
        IN, OUT, scale=scale, use_bias=True, dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    dense.bias.value = jnp.arange(OUT, dtype=jnp.float32) / OUT
    quant = dense.to_quantized(QuantizationConfig(dtype=QuantizationType.INT8))
    x = _x()
    expected_scale = IN**-0.5 if scale == "fan_in" else OUT**-0.5 if scale == "fan_out" else scale
    expected = np.asarray(x) @ np.asarray(dense.weight.value) * expected_scale + np.asarray(dense.bias.value)
    actual = quant(x, w=dense.weight.value)
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


def test_to_quantized_keeps_fan_in_scale():
    dense = ColumnParallelLinear(
        IN, OUT, scale="fan_in", use_bias=False, dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    quant = dense.to_quantized(QuantizationConfig(dtype=QuantizationType.INT8), qmm_platform="xla", qmm_fuse=False)
    want = np.asarray(dense(_x()))
    got = np.asarray(quant(_x()))
    # Without the scale the outputs would differ by sqrt(IN) = 8.
    np.testing.assert_allclose(got, want, rtol=0.1, atol=0.05 * np.abs(want).max())


@pytest.mark.parametrize("linear_class", [ColumnParallelLinear, RowParallelLinear])
@pytest.mark.parametrize("quant_type", [QuantizationType.INT8, QuantizationType.CHANNELWISE])
def test_roundtrip_preserves_scaled_outputs_and_layouts(linear_class, quant_type):
    weight_layout = TensorLayout((None, "tp"))
    bias_layout = TensorLayout(("tp",))
    dense = linear_class(
        IN,
        OUT,
        scale=0.5,
        use_bias=True,
        dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        sharding_layout=weight_layout,
        bias_sharding_layout=bias_layout,
        rngs=spx.Rngs(0),
    )
    dense.bias.value = jnp.arange(OUT, dtype=jnp.float32) / OUT
    quant = dense.to_quantized(QuantizationConfig(dtype=quant_type), qmm_platform="xla", qmm_fuse=False)
    back = quant.from_quantized()
    assert quant.sharding_layout == back.sharding_layout == weight_layout
    assert quant.bias_sharding_layout == back.bias_sharding_layout == bias_layout
    assert quant.quant_kernel.sharding.axis_names == back.weight.sharding.axis_names == weight_layout.axes
    assert quant.bias.sharding.axis_names == back.bias.sharding.axis_names == bias_layout.axes
    assert back.output_scale == 0.5
    # Compare to the independent dense formula using the dequantized weights.
    x = _x()
    expected = np.asarray(x) @ np.asarray(back.weight.value) * 0.5 + np.asarray(dense.bias.value)
    np.testing.assert_allclose(np.asarray(back(x)), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(quant(x)), expected, rtol=0.01, atol=0.01)


def test_runtime_dtype_matches_repacked_weight_reference_without_mutating_storage():
    """INT8 storage with NF4 compute must multiply by the re-quantized weights."""
    # A rectangular weight makes a transposed reference fail by shape rather than
    # silently comparing against a different square matrix.
    out_features = 2 * OUT
    layer = ColumnParallelLinearQuantized(
        IN,
        out_features,
        use_bias=True,
        scale=0.5,
        config=QuantizationConfig(dtype=QuantizationType.INT8, runtime_dtype=QuantizationType.NF4),
        dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        qmm_platform="xla",
        qmm_fuse=False,
        rngs=spx.Rngs(1),
    )
    layer.bias.value = jnp.arange(out_features, dtype=jnp.float32) / out_features
    stored_kernel = np.asarray(layer.quant_kernel.value).copy()
    stored_scales = np.asarray(layer.quant_scales.value).copy()
    stored_biases = np.asarray(layer.quant_biases.value).copy()
    stored_dense = layer.from_quantized().weight.value
    assert stored_dense.shape == (IN, out_features)
    # EasyDeL kernels are [in, out]. ejkernel's legacy prepack default instead
    # transposes its logical [out, in] input; match the layer's explicit layout.
    nf4_codes, nf4_scales = prepack_quantized_weights(stored_dense, mode="nf4", group_size=64, bits=4, transpose=False)
    nf4_dense = dequantize(nf4_codes, nf4_scales, mode="nf4", group_size=64, bits=4, axis="col")
    assert nf4_dense.shape == (IN, out_features)
    x = _x()
    expected = np.asarray(x) @ np.asarray(nf4_dense).astype(np.float32) * 0.5 + np.asarray(layer.bias.value)
    actual = np.asarray(layer(x))
    np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.01)
    stored_output = np.asarray(x) @ np.asarray(stored_dense) * 0.5 + np.asarray(layer.bias.value)
    assert not np.allclose(actual, stored_output, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(layer.quant_kernel.value), stored_kernel)
    np.testing.assert_array_equal(np.asarray(layer.quant_scales.value), stored_scales)
    np.testing.assert_array_equal(np.asarray(layer.quant_biases.value), stored_biases)


@pytest.mark.parametrize("runtime_dtype", [QuantizationType.BINARY, QuantizationType.TERNARY])
def test_runtime_integer_code_formats_match_dense_reference(runtime_dtype):
    layer = ColumnParallelLinearQuantized(
        IN,
        OUT,
        use_bias=False,
        config=QuantizationConfig(dtype=QuantizationType.INT8, runtime_dtype=runtime_dtype),
        dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(2),
    )
    dense_weights = np.asarray(layer.from_quantized().weight.value)
    if runtime_dtype == QuantizationType.BINARY:
        codes = np.where(dense_weights >= 0, 1.0, -1.0)
    else:
        threshold = 0.5 * dense_weights.std()
        codes = np.where(dense_weights > threshold, 1.0, np.where(dense_weights < -threshold, -1.0, 0.0))
    x = _x()
    expected = np.asarray(x) @ codes
    np.testing.assert_allclose(np.asarray(layer(x)), expected, rtol=1e-5, atol=1e-5)
