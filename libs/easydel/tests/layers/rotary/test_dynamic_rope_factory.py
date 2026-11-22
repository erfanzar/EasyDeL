"""Dynamic-cache consistency and training/deployment affine forward parity."""

from copy import deepcopy

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.quantization import QuantizationConfig, quantization_config_to_rule
from easydel.layers.rotary import DynamicNTKScalingRotaryEmbedding, get_frequencies, get_rope
from ejkernel.quantization import QuantRuntimeConfig
from ejkernel.quantization import dequantize as deployment_dequantize
from ejkernel.quantization import quantize as deployment_quantize
from jax import numpy as jnp


@pytest.mark.cpu_ok
@pytest.mark.parametrize("max_position", [48, 128])
@pytest.mark.parametrize("original", [None, 32])
def test_dynamic_factory_keeps_original_context_separate_from_cache_length(max_position, original):
    scaling = {"rope_type": "dynamic", "factor": 2.0}
    if original is not None:
        scaling["original_max_position_embeddings"] = original
    before = deepcopy(scaling)
    rope = get_rope(16, 12, max_position, 4096, rope_scaling=scaling)
    assert isinstance(rope, DynamicNTKScalingRotaryEmbedding)
    threshold = max_position if original is None else original
    assert rope.max_position_embeddings == threshold
    assert rope.max_positions == max(max_position, threshold * 2)
    assert scaling == before


@pytest.mark.cpu_ok
def test_direct_dynamic_constructor_keeps_legacy_context_and_cache_defaults():
    rope = DynamicNTKScalingRotaryEmbedding(2.0, 16, 12, 32, 4096, True, jnp.float32)
    assert rope.max_position_embeddings == 32
    assert rope.scaling_factor == 2.0
    assert rope.max_positions is None


@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("max_position", [48, 128])
def test_nonmultimodal_dynamic_automatic_and_supplied_cache_match_numpy(neox, max_position):
    scaling = {"rope_type": "dynamic", "factor": 2.0, "original_max_position_embeddings": 32}
    rope = get_rope(16, 12, max_position, 4096, is_neox_style=neox, rope_scaling=scaling)
    # Includes the original boundary, the transition interval just past it,
    # and a position beyond original*factor when the served cache is longer.
    positions = np.array([0, 30, 31, 39, max_position - 2], dtype=np.int32)
    shifted = positions + 1
    length = np.maximum(shifted + 1, 32).astype(np.float32)
    base = np.where(length > 32, 4096.0 * (2 * length / 32 - 1) ** (12 / 10), 4096.0)
    inverse = base[:, None] ** (-np.arange(0, 12, 2, dtype=np.float32) / 12)
    angles = shifted[:, None] * inverse
    cos, sin = np.cos(angles), np.sin(angles)
    host = np.random.default_rng(23).normal(size=(4, 5, 2, 16)).astype(np.float32)
    x = host[..., :12]
    if neox:
        cos, sin = np.concatenate([cos, cos], -1), np.concatenate([sin, sin], -1)
        rotated = np.concatenate([-x[..., 6:], x[..., :6]], -1)
    else:
        cos, sin = np.repeat(cos, 2, -1), np.repeat(sin, 2, -1)
        pairs = x.reshape(4, 5, 2, 6, 2)
        rotated = np.stack([-pairs[..., 1], pairs[..., 0]], -1).reshape(x.shape)
    expected = np.concatenate([x * cos[None, :, None] + rotated * sin[None, :, None], host[..., 12:]], -1)
    cache = get_frequencies(16, 12, max_position, 4096, scaling)
    assert cache.shape == (max(max_position, 64), 12)
    outputs = []
    for frequencies in (None, cache):
        query, key = rope(
            jnp.asarray(positions),
            jnp.asarray(host),
            jnp.asarray(host * 2),
            offsets=jnp.asarray(1),
            frequencies=frequencies,
        )
        assert query.dtype == key.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(query), expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(np.asarray(key), expected * 2, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(np.asarray(query[..., 12:]), host[..., 12:])
        outputs.append(np.asarray(query))
    np.testing.assert_allclose(*outputs, rtol=2e-5, atol=2e-5)


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "dtype,method",
    [("affine", "affine"), ("int8", "affine"), ("channelwise", "absmax"), ("nf4", "absmax")],
)
def test_bridge_selects_deployment_calibration_without_changing_other_formats(dtype, method):
    rule = quantization_config_to_rule(QuantizationConfig(dtype=dtype))
    assert rule.weight_calibration_method == method
    assert rule.act_calibration_method == "absmax"
    assert rule.act_qtype is None


def _affine_reference(weights, group_size, bits):
    grouped = weights.reshape(-1, group_size, weights.shape[-1])
    low, high = grouped.min(axis=1, keepdims=True), grouped.max(axis=1, keepdims=True)
    scale = (high - low) / np.float32(2**bits - 1)
    scale = np.where(scale == 0, np.float32(1), scale)
    zero = -low / scale
    codes = np.clip(np.round(grouped / scale + zero), 0, 2**bits - 1)
    return ((codes - zero) * scale).reshape(weights.shape)


@pytest.mark.parametrize(
    "dtype,bits,group_size",
    [("affine", None, None), ("int8", None, None), ("affine", 8, 16), ("int8", 4, 32)],
)
@pytest.mark.parametrize("constant", [False, True])
def test_affine_training_and_deployment_forward_parity(dtype, bits, group_size, constant):
    config = QuantizationConfig(dtype=dtype, bits=bits, group_size=group_size, simulate=True)
    rule = quantization_config_to_rule(config)
    resolved_group = 64 if group_size is None else group_size
    resolved_bits = (4 if dtype == "affine" else 8) if bits is None else bits
    # Shifted-positive/negative output channels and per-tile shifts expose
    # zero-clamping and rounded-zero-point bugs; constants expose zero ranges.
    weights = np.random.default_rng(41).uniform(-0.6, 0.6, size=(128, 8)).astype(np.float32)
    if constant:
        weights[:] = 0
    weights += np.array([2.25, -3.5, 0, 1.5, -1.25, 4, -5, 0], dtype=np.float32)
    weights += (np.arange(128) // resolved_group).astype(np.float32)[:, None] / 4
    expected = _affine_reference(weights, resolved_group, resolved_bits)
    array = jnp.asarray(weights)
    # Compare the canonical fp32 grid. The deployment library's implicit fast
    # policy stores fp16/bf16 metadata depending on backend; pin its public
    # precision config rather than loosening parity tolerances for that loss.
    runtime = QuantRuntimeConfig(affine_metadata_dtype="fp32", prefer_compute_dtype="fp32", dequant_output_dtype="fp32")
    packed = deployment_quantize(
        array, group_size=resolved_group, bits=resolved_bits, mode="affine", runtime_config=runtime
    )
    deployed = jnp.swapaxes(
        deployment_dequantize(
            *packed, group_size=resolved_group, bits=resolved_bits, mode="affine", runtime_config=runtime
        ),
        -2,
        -1,
    )
    trained = jax.jit(lambda w: spx.quantization.fake_quant(w, rule=rule, contracting_axes=(0,)))(array)
    np.testing.assert_allclose(np.asarray(deployed), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(trained), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(trained), np.asarray(deployed), rtol=2e-6, atol=2e-6)
    cotangent = jnp.asarray(np.random.default_rng(19).normal(size=weights.shape).astype(np.float32))
    _, pullback = jax.vjp(lambda w: spx.quantization.fake_quant(w, rule=rule, contracting_axes=(0,)), array)
    np.testing.assert_array_equal(np.asarray(pullback(cotangent)[0]), np.asarray(cotangent))
    inputs = np.random.default_rng(7).normal(size=(4, 128)).astype(np.float32)
    # Pin float32 matmul precision instead of hiding TPU bf16 contraction error.
    with jax.default_matmul_precision("highest"):
        output = spx.quantization.qdot_general(
            jnp.asarray(inputs), array, (((1,), (0,)), ((), ())), rule=rule, precision=jax.lax.Precision.HIGHEST
        )
        deployed_output = jnp.matmul(jnp.asarray(inputs), deployed, precision=jax.lax.Precision.HIGHEST)
    np.testing.assert_allclose(np.asarray(output), inputs @ expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(output), np.asarray(deployed_output), rtol=2e-5, atol=2e-5)
