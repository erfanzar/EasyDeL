"""Explicit precision must not silently fall into an incompatible format."""

import pytest
from easydel.layers.quantization import QuantizationConfig


@pytest.mark.parametrize("dtype", ["nf4", "affine", "int8", "mxfp4"])
def test_explicit_activation_contract_rejects_other_storage_formats(dtype):
    with pytest.raises(ValueError, match=r"explicit.*channelwise"):
        QuantizationConfig(dtype=dtype, activation_policy="explicit", activation_bits=8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime_dtype": "mxfp4"},
        {"simulate": True},
        {"jax_native": True},
        {"bits": 2},
        {"bits": 8, "activation_bits": 4},
    ],
)
def test_explicit_activation_contract_rejects_unsupported_dispatch(overrides):
    settings = dict(dtype="channelwise", bits=4, activation_policy="explicit", activation_bits=8)
    settings.update(overrides)
    with pytest.raises(ValueError):
        QuantizationConfig(**settings)


def test_channelwise_runtime_override_remains_supported():
    cfg = QuantizationConfig(
        dtype="channelwise", runtime_dtype="channelwise", bits=4, activation_policy="explicit", activation_bits=16
    )
    assert cfg.activation_bits == 16
