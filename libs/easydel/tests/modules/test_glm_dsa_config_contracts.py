# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Array-free GLM configuration and sparse-training eligibility regressions."""

import importlib
from copy import deepcopy

import pytest

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("rope_parameters", [None, {}, {"rope_type": "default"}])
def test_dsa_rope_defaults_do_not_acquire_vision_mrope(rope_parameters):
    from easydel.modules.glm_moe_dsa.glm_moe_dsa_configuration import GlmMoeDsaConfig

    config = GlmMoeDsaConfig(num_hidden_layers=2, rope_parameters=rope_parameters)
    assert config.rope_scaling["rope_type"] == "default"
    assert "mrope_section" not in config.rope_scaling
    assert "mrope_interleaved" not in config.rope_scaling


def test_dsa_accepts_legacy_rope_parameter_alias_without_mutating_inputs():
    from easydel.modules.glm_moe_dsa.glm_moe_dsa_configuration import GlmMoeDsaConfig

    parameters = {"type": "yarn", "factor": 4.0, "beta_fast": 32, "beta_slow": 1}
    original = deepcopy(parameters)
    config = GlmMoeDsaConfig(num_hidden_layers=2, rope_parameters=parameters)
    assert config.rope_scaling["rope_type"] == "yarn"
    assert config.rope_scaling["factor"] == 4.0
    assert config.rope_scaling["beta_fast"] == 32
    assert config.rope_scaling["beta_slow"] == 1
    assert parameters == original


def test_dsa_layer_norm_original_public_name_remains_available():
    from easydel.layers.indexer import IndexerLayerNorm
    from easydel.modules.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaLayerNorm

    assert GlmMoeDsaLayerNorm is IndexerLayerNorm


def test_gpt_oss_rms_norm_original_public_name_remains_available():
    from easydel.layers import RMSNorm
    from easydel.modules.gpt_oss.modeling_gpt_oss import GptOssRMSNorm

    assert GptOssRMSNorm is RMSNorm


def test_dsa_exported_rope_normalizer_keeps_text_only_defaults():
    from easydel.modules.glm_moe_dsa.glm_moe_dsa_configuration import rope_scaling_from_rope_parameters

    assert rope_scaling_from_rope_parameters(None, None) is None
    parameters = {"rope_type": "yarn", "factor": 4.0}
    scaling = {"type": "linear", "factor": 2.0}
    original_parameters, original_scaling = deepcopy(parameters), deepcopy(scaling)
    result = rope_scaling_from_rope_parameters(parameters, scaling)
    assert result["rope_type"] == "linear"
    assert result["factor"] == 2.0
    assert "mrope_section" not in result
    assert parameters == original_parameters
    assert scaling == original_scaling


@pytest.fixture(params=["glm_moe_dsa", "glm5_next"])
def sparse_policy(request):
    family = request.param
    module = importlib.import_module(f"easydel.modules.{family}.modeling_{family}")
    return module.use_sparse_training_attention


def _eligible_options():
    # Opaque host objects deliberately avoid creating JAX arrays or probing hardware.
    return {
        "topk_indices": object(),
        "cache_view": None,
        "mask_info": object(),
        "output_attentions": False,
        "sliding_window": None,
        "softmax_aux": None,
    }


@pytest.mark.parametrize(
    ("sparse_training", "seq_len", "backend", "expected"),
    [
        (True, 32, "cpu", True),
        (False, 32768, "tpu", False),
        (None, 8191, "tpu", False),
        (None, 8192, "tpu", True),
        (None, 32768, "gpu", False),
        (None, None, "tpu", False),
    ],
)
def test_family_sparse_policy_preserves_config_and_auto_threshold(
    sparse_policy, sparse_training, seq_len, backend, expected
):
    assert (
        sparse_policy(**_eligible_options(), sparse_training=sparse_training, seq_len=seq_len, backend=backend)
        is expected
    )


@pytest.mark.parametrize(
    ("feature", "value"),
    [
        ("topk_indices", None),
        ("cache_view", object()),
        ("mask_info", None),
        ("output_attentions", True),
        ("sliding_window", 128),
        ("softmax_aux", object()),
    ],
)
def test_family_force_sparse_still_preserves_dense_only_features(sparse_policy, feature, value):
    options = _eligible_options()
    options[feature] = value
    assert sparse_policy(**options, sparse_training=True, seq_len=32768, backend="tpu") is False
