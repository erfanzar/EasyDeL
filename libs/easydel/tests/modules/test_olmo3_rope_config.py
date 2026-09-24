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

"""OLMo3's persisted local/global RoPE configuration contract."""

import copy
import json

import pytest
from easydel import Olmo3Config
from transformers import Olmo3Config as HFOlmo3Config


@pytest.mark.parametrize("rope_theta", [10000.0, 123456.0])
@pytest.mark.parametrize("rope_scaling", [None, {"type": "linear", "factor": 2.0}])
def test_legacy_shared_rope_roundtrip(rope_theta, rope_scaling):
    """Legacy shared settings keep their frequencies on both attention types."""
    config = Olmo3Config(num_hidden_layers=4, rope_theta=rope_theta, rope_scaling=rope_scaling)
    restored = Olmo3Config.from_dict(json.loads(config.to_json_string(use_diff=False)))
    assert restored.rope_parameters == config.rope_parameters
    assert set(restored.rope_parameters) == {"sliding_attention", "full_attention"}
    for params in restored.rope_parameters.values():
        assert params["rope_theta"] == rope_theta
        assert params["rope_type"] == ("default" if rope_scaling is None else "linear")
        if rope_scaling is not None:
            assert params["factor"] == 2.0


@pytest.mark.parametrize(
    "layer_types", [["sliding_attention"], ["full_attention"], ["sliding_attention", "full_attention"]]
)
def test_canonical_per_type_rope_roundtrip(layer_types):
    """Canonical settings survive even when one attention type is not active."""
    parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "linear", "rope_theta": 500000.0, "factor": 2.0},
    }
    original_parameters = copy.deepcopy(parameters)
    config = Olmo3Config(num_hidden_layers=len(layer_types), layer_types=layer_types, rope_parameters=parameters)
    restored = Olmo3Config.from_dict(json.loads(config.to_json_string(use_diff=False)))
    hf_config = HFOlmo3Config(
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        rope_parameters=copy.deepcopy(restored.rope_parameters),
    )
    assert parameters == original_parameters
    assert restored.rope_parameters == config.rope_parameters
    for layer_type, expected in parameters.items():
        for key, value in expected.items():
            assert restored.rope_parameters[layer_type][key] == value
            assert hf_config.rope_parameters[layer_type][key] == value
        layer_config = restored.get_layer_rope_config(layer_type)
        assert layer_config.rope_theta == expected["rope_theta"]
        assert layer_config.rope_scaling == restored.rope_parameters[layer_type]
        assert layer_config.rope_parameters == restored.rope_parameters[layer_type]
        # HF's alias must expose a flat builder view, not the owning config's
        # per-type map, including the factor consumed by the rotary builders.
        if "factor" in expected:
            assert layer_config.rope_scaling["factor"] == expected["factor"]
    assert restored.rope_parameters == config.rope_parameters


def test_flat_rope_parameters_are_migrated_without_changing_theta():
    """Older serialized flat mappings must not acquire HF's new default theta."""
    config = Olmo3Config(
        num_hidden_layers=4,
        rope_parameters={"rope_type": "linear", "rope_theta": 32100.0, "factor": 3.0},
    )
    for layer_type in ("sliding_attention", "full_attention"):
        params = config.rope_parameters[layer_type]
        assert params["rope_theta"] == 32100.0
        assert params["rope_type"] == "linear"
        assert params["factor"] == 3.0


def test_legacy_scaling_overrides_stale_flat_default_payload():
    """Old persisted compatibility fields must not erase actual shared scaling."""
    config = Olmo3Config(
        num_hidden_layers=4,
        rope_theta=32100.0,
        rope_scaling={"type": "linear", "factor": 3.0},
        rope_parameters={"rope_type": "default", "rope_theta": 32100.0},
    )
    for params in config.rope_parameters.values():
        assert params["rope_theta"] == 32100.0
        assert params["rope_type"] == "linear"
        assert params["factor"] == 3.0


@pytest.mark.parametrize(
    "rope_scaling",
    ["linear", {"type": "invalid", "factor": 2.0}, {"type": "linear", "factor": 0.5}],
)
def test_invalid_legacy_scaling_is_not_hidden_by_hf_alias(rope_scaling):
    """The original legacy input must be validated before canonical expansion."""
    with pytest.raises(ValueError, match="rope_scaling"):
        Olmo3Config(num_hidden_layers=4, rope_scaling=rope_scaling)


@pytest.mark.parametrize("override_path", ["assignment", "from_dict", "from_pretrained"])
@pytest.mark.parametrize("rope_scaling", [None, {"type": "linear", "factor": 2.0}])
def test_shared_theta_override_survives_serialization(override_path, rope_scaling, tmp_path):
    """Legacy theta overrides update both types, including after a local save."""
    config = Olmo3Config(num_hidden_layers=4, rope_scaling=rope_scaling)
    if override_path == "assignment":
        config.rope_theta = 32100.0
    elif override_path == "from_dict":
        config = Olmo3Config.from_dict(config.to_dict(), rope_theta=32100.0)
    else:
        config.save_pretrained(tmp_path)
        saved = json.loads((tmp_path / "config.json").read_text())
        assert saved["rope_theta_is_shared"] is True
        config = Olmo3Config.from_pretrained(tmp_path, rope_theta=32100.0, local_files_only=True)
    assert config.rope_theta == 32100.0
    assert config.rope_theta_is_shared is True
    for layer_type in ("sliding_attention", "full_attention"):
        assert config.rope_parameters[layer_type]["rope_theta"] == 32100.0
        assert config.get_layer_rope_config(layer_type).rope_theta == 32100.0
        if rope_scaling is not None:
            assert config.rope_parameters[layer_type]["factor"] == 2.0
    # The new value and its shared provenance both survive another roundtrip.
    restored = Olmo3Config.from_dict(json.loads(config.to_json_string()))
    restored.rope_theta = 65400.0
    for params in restored.rope_parameters.values():
        assert params["rope_theta"] == 65400.0


@pytest.mark.parametrize("override_path", ["assignment", "from_dict", "from_pretrained"])
@pytest.mark.parametrize("global_theta", [10000.0, 500000.0])
def test_explicit_per_type_theta_remains_authoritative(override_path, global_theta, tmp_path):
    """Even numerically identical explicit maps must not become legacy shared maps."""
    parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "default", "rope_theta": global_theta},
    }
    config = Olmo3Config(num_hidden_layers=4, rope_parameters=parameters)
    if override_path == "assignment":
        config.rope_theta = 32100.0
    elif override_path == "from_dict":
        payload = config.to_dict()
        # HF configs / older explicit maps have no EasyDeL provenance marker.
        payload.pop("rope_theta_is_shared")
        config = Olmo3Config.from_dict(payload, rope_theta=32100.0)
    else:
        config.save_pretrained(tmp_path)
        config = Olmo3Config.from_pretrained(tmp_path, rope_theta=32100.0, local_files_only=True)
    assert config.rope_theta == 32100.0
    assert config.rope_theta_is_shared is False
    for layer_type, expected in parameters.items():
        assert config.rope_parameters[layer_type]["rope_theta"] == expected["rope_theta"]
        assert config.get_layer_rope_config(layer_type).rope_theta == expected["rope_theta"]


def test_late_explicit_map_replaces_shared_theta_provenance():
    """Replacing a synthesized map explicitly disables scalar-theta propagation."""
    config = Olmo3Config(num_hidden_layers=4)
    config.rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
    }
    restored = Olmo3Config.from_dict(config.to_diff_dict(), rope_theta=32100.0)
    assert restored.rope_theta_is_shared is False
    assert restored.rope_parameters["sliding_attention"]["rope_theta"] == 10000.0
    assert restored.rope_parameters["full_attention"]["rope_theta"] == 500000.0
