# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integer matmul presets must opt out of legacy activation auto-selection.

These are configuration contracts, not kernel dispatch or numerical tests.
In particular, legacy grouped auto-selection may quantize INT4 weights' LHS
as INT8; an A16 preset must not be represented by that legacy auto policy.
"""

import json

import pytest
from easydel.layers.quantization import QuantizationConfig, QuantizationType
from easydel.layers.quantization._configs import resolve_ejkernel_quant_params


@pytest.mark.parametrize(
    ("mode", "weight_bits", "activation_bits"),
    [("w4a16", 4, 16), ("w8a16", 8, 16), ("w4a4", 4, 4), ("w8a8", 8, 8)],
)
def test_integer_matmul_preset(mode, weight_bits, activation_bits):
    config = QuantizationConfig.for_matmul(mode)

    assert config.dtype == QuantizationType.CHANNELWISE
    assert config.bits == weight_bits
    assert config.activation_bits == activation_bits
    assert config.activation_policy == "explicit"
    assert config.runtime_dtype is None
    assert config.simulate is False
    assert config.jax_native is False
    assert config.pattern == QuantizationConfig().pattern
    assert resolve_ejkernel_quant_params(config) == ("channelwise", 0, weight_bits, False)


@pytest.mark.parametrize("mode", ["", "int4", "w4a8", "w8a4", "w4a32", "unknown"])
def test_invalid_integer_matmul_preset(mode):
    with pytest.raises(ValueError):
        QuantizationConfig.for_matmul(mode)


@pytest.mark.parametrize("activation_bits", [None, 4, 8, 16])
def test_valid_activation_bits(activation_bits):
    config = QuantizationConfig(dtype="channelwise", bits=4, activation_bits=activation_bits)
    assert config.activation_bits == activation_bits
    assert config.activation_policy == "auto"


@pytest.mark.parametrize("activation_bits", [-1, 0, 2, 3, 7, 32])
def test_invalid_activation_bits(activation_bits):
    with pytest.raises(ValueError, match="activation_bits"):
        QuantizationConfig(dtype="channelwise", bits=4, activation_bits=activation_bits)


@pytest.mark.parametrize("mode", ["w4a16", "w8a16", "w4a4", "w8a8"])
def test_integer_matmul_preset_json_roundtrip(mode):
    config = QuantizationConfig.for_matmul(mode)
    payload = json.loads(json.dumps(config.to_dict()))

    assert payload["dtype"] == "channelwise"
    assert payload["activation_bits"] == config.activation_bits
    assert payload["activation_policy"] == "explicit"
    assert QuantizationConfig.from_dict(payload) == config


def test_default_config_preserves_legacy_auto():
    config = QuantizationConfig()
    assert config.dtype == QuantizationType.NF4
    assert config.bits is None
    assert config.activation_bits is None
    assert config.activation_policy == "auto"
    assert resolve_ejkernel_quant_params(config) == ("nf4", 64, 4, False)
    assert QuantizationConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


@pytest.mark.parametrize("weight_bits", [4, 8])
@pytest.mark.parametrize("activation_bits", [None, 4, 8, 16])
def test_legacy_serialized_config_does_not_opt_into_explicit_policy(weight_bits, activation_bits):
    payload = {
        "dtype": "channelwise",
        "runtime_dtype": None,
        "group_size": None,
        "bits": weight_bits,
        "activation_bits": activation_bits,
        "simulate": False,
        "jax_native": False,
        "pattern": ".*attention.*",
    }
    config = QuantizationConfig.from_dict(payload)

    assert config.activation_policy == "auto"
    for key, value in payload.items():
        assert config.to_dict()[key] == value
    assert QuantizationConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_older_config_without_activation_fields_keeps_auto_and_ignores_unknown_keys():
    config = QuantizationConfig.from_dict({"dtype": "int8", "future_knob": True})
    assert config == QuantizationConfig(dtype="int8")
    assert config.activation_bits is None
    assert config.activation_policy == "auto"


def test_invalid_activation_policy():
    with pytest.raises(ValueError, match="activation_policy"):
        QuantizationConfig(activation_policy="unknown")


def test_explicit_activation_policy_requires_bits():
    with pytest.raises(ValueError, match="activation_bits"):
        QuantizationConfig(dtype="channelwise", bits=4, activation_policy="explicit")
