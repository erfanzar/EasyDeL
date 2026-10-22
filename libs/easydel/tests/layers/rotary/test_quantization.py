"""Quantization: simulate layout, per-call simulate override, jit-safe channelwise STE, adapter registry."""

import re

import jax
import numpy as np
import pytest
from easydel.layers.quantization import (
    DEFAULT_QUANTIZED_MODULE_PATH,
    EasyQuantizer,
    QuantizationConfig,
    QuantizationType,
    quantize,
)
from easydel.layers.quantization._straight_through import straight_through_channelwise
from jax import numpy as jnp


def _weights(shape=(64, 32)):
    return jnp.asarray(np.random.default_rng(0).standard_normal(shape), jnp.float32)


def test_quantize_simulate_keeps_input_layout_for_non_square_weight():
    w = _weights((64, 32))
    out = quantize(w, dtype=QuantizationType.NF4, group_size=64, simulate=True)
    assert out.shape == w.shape
    assert float(jnp.abs(out - w).mean()) < 0.25 * float(jnp.abs(w).mean())


def test_easy_quantizer_simulate_argument_overrides_config():
    w = _weights()
    quantizer = EasyQuantizer(QuantizationConfig(dtype=QuantizationType.NF4, group_size=64, simulate=False))
    simulated = quantizer.quantize_array(w, simulate=True)
    assert isinstance(simulated, jax.Array)
    assert simulated.shape == w.shape
    packed = quantizer.quantize_array(w)
    assert isinstance(packed, tuple)
    assert quantizer.config.simulate is False

    simulate_by_default = EasyQuantizer(QuantizationConfig(dtype=QuantizationType.NF4, group_size=64, simulate=True))
    assert simulate_by_default.quantize_array(w).shape == w.shape
    assert isinstance(simulate_by_default.quantize_array(w, simulate=False), tuple)
    assert simulate_by_default.config.simulate is True


@pytest.mark.parametrize("bits", [8, 4])
def test_straight_through_channelwise_runs_under_jit_and_passes_gradient(bits):
    w = _weights((16, 8))
    jitted = jax.jit(straight_through_channelwise, static_argnames="bits")
    host = np.asarray(w)
    qmax = 2 ** (bits - 1) - 1
    scales = np.max(np.abs(host), axis=-2, keepdims=True) / qmax
    expected = np.clip(np.round(host / scales), -qmax, qmax) * scales
    np.testing.assert_allclose(np.asarray(jitted(w, bits=bits)), expected, rtol=1e-5, atol=1e-6)
    grad = jax.jit(jax.grad(lambda x: straight_through_channelwise(x, bits).sum()))(w)
    np.testing.assert_allclose(np.asarray(grad), 1.0)


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "path, selected",
    [("model.embedding", False), ("model.norm", False), ("lm_head", False), ("model.layers.0.q_proj", True)],
)
def test_training_and_serving_default_patterns_select_documented_layers(path, selected):
    assert bool(re.fullmatch(QuantizationConfig().pattern, path)) is selected
    assert bool(re.fullmatch(DEFAULT_QUANTIZED_MODULE_PATH, path)) is selected


@pytest.mark.cpu_ok
@pytest.mark.parametrize("expert_dtype", ["fp4", "float4", "mxfp4"])
def test_fp8_adapter_handles_deepseek_v4_mixed_layout_through_one_registration(expert_dtype):
    from easydel.layers.quantization.checkpoint import ScaledElementsAdapter, get_adapter
    from easydel.layers.quantization.checkpoint._formats import DeepseekV4MixedAdapter

    assert get_adapter("fp8") is ScaledElementsAdapter
    assert get_adapter("deepseek_v4_mixed") is ScaledElementsAdapter
    assert DeepseekV4MixedAdapter is ScaledElementsAdapter

    plain = ScaledElementsAdapter.parse_config({"quant_method": "fp8", "weight_block_size": [128, 128]})
    assert plain.overrides == ()
    assert ScaledElementsAdapter.source_suffixes(plain.default) == ("weight", "weight_scale_inv")

    mixed = ScaledElementsAdapter.parse_config(
        {"quant_method": "fp8", "weight_block_size": [128, 128], "expert_dtype": expert_dtype}
    )
    pattern, expert_source = mixed.overrides[0]
    assert re.search(pattern, "model.layers.0.mlp.experts.3.gate_proj")
    assert not re.search(pattern, "model.layers.0.mlp.shared_experts.gate_proj")
    assert ScaledElementsAdapter.target_spec(expert_source, expert_dim=True).bits == 4
    assert ScaledElementsAdapter.source_suffixes(mixed.default) == ("weight", "scale")
    assert ScaledElementsAdapter.source_suffixes(expert_source) == ("weight", "scale")
