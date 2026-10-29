"""Array-free training/deployment precision and RoPE normalization contracts."""

import pytest
from easydel.layers.quantization import QuantizationConfig, quantization_config_to_rule
from easydel.layers.quantization.checkpoint import get_adapter
from easydel.layers.rotary import rope_scaling_from_rope_parameters
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "dtype,bits,group,qtype,tile",
    [
        ("affine", None, None, jnp.int4, 64),
        ("int8", None, None, jnp.int8, 64),
        ("nf4", None, None, "nf4", 64),
        ("affine", 8, 128, jnp.int8, 128),
        ("nf4", 4, 32, "nf4", 32),
        ("channelwise", 4, None, jnp.int4, None),
    ],
)
def test_training_rule_resolves_storage_bit_and_group_defaults(dtype, bits, group, qtype, tile):
    rule = quantization_config_to_rule(QuantizationConfig(dtype=dtype, bits=bits, group_size=group))
    assert rule.weight_qtype == qtype
    assert rule.tile_size == tile


@pytest.mark.parametrize(
    "preset,weight,activation",
    [
        ("w4a8", jnp.int4, jnp.int8),
        ("w4a4", jnp.int4, jnp.int4),
        ("w8a8", jnp.int8, jnp.int8),
        ("w4a16", jnp.int4, None),
        ("w8a16", jnp.int8, None),
    ],
)
def test_training_rule_preserves_explicit_activation_precision(preset, weight, activation):
    rule = quantization_config_to_rule(QuantizationConfig(dtype=preset))
    assert rule.weight_qtype == weight
    assert rule.act_qtype == activation
    assert rule.tile_size is None


def test_hf_rope_conversion_keeps_correction_and_magnitude_overrides():
    fields = {
        "rope_type": "yarn",
        "factor": 4.0,
        "attention_factor": 2.0,
        "truncate": False,
        "short_mscale": 1.25,
        "long_mscale": 1.5,
        "repetition_style": False,
    }
    assert rope_scaling_from_rope_parameters(fields, None) == fields
    assert rope_scaling_from_rope_parameters({"type": "linear", "scaling_factor": 2}, None) == {
        "rope_type": "linear",
        "factor": 2,
    }


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_gptq_adapter_retains_source_bit_width_separately_from_target(bits):
    adapter = get_adapter("gptq")
    source = adapter.parse_config({"bits": bits, "group_size": 64}).default
    assert source.weight_dtype == f"uint{bits}"
    assert source.raw["bits"] == bits
    assert adapter.target_spec(source).bits == bits


@pytest.mark.parametrize("bits", [1, 5, 6, 7, 16])
def test_gptq_rejects_non_gptq_source_width_before_reading_weights(bits):
    with pytest.raises(ValueError, match="source bits"):
        get_adapter("gptq").parse_config({"bits": bits})
