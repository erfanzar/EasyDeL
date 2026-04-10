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

"""Tests for OpenELM model."""

import pytest

import easydel as ed

try:
    from .test_utils import CausalLMTester, get_hf_model_from_hub
    from .test_utils.model_factory import _build_openelm_config_from_raw
except ImportError:
    from test_utils import CausalLMTester, get_hf_model_from_hub  # pyright: ignore[reportImplicitRelativeImport]
    from test_utils.model_factory import _build_openelm_config_from_raw  # pyright: ignore[reportImplicitRelativeImport]


class TestOpenELM:
    """Test suite for OpenELM model."""

    @pytest.fixture
    def openelm_config(self, small_model_config):
        """Create OpenELM-specific config."""
        _, conf = get_hf_model_from_hub(
            "apple/OpenELM-270M-Instruct",
            small_model_config,
        )
        config = ed.OpenELMConfig()
        for k, v in conf.__dict__.items():
            setattr(config, k, v)
        config.max_context_length = small_model_config["max_position_embeddings"]
        return config

    @pytest.fixture
    def hf_openelm_class(self, small_model_config):
        """Load OpenELM HF class from hub."""
        hf_class, _ = get_hf_model_from_hub(
            "apple/OpenELM-270M-Instruct",
            small_model_config,
        )
        return hf_class

    def test_causal_lm(self, openelm_config, small_model_config, hf_openelm_class):
        """Test OpenELMForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="openelm",
            hf_class=hf_openelm_class,
            task=ed.TaskType.CAUSAL_LM,
            config=openelm_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"OpenELM CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, openelm_config, small_model_config, hf_openelm_class):
        """Test OpenELM text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="openelm",
            hf_class=hf_openelm_class,
            config=openelm_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"OpenELM generation failed: {result.error_message}"

    def test_fallback_builder_recomputes_openelm_derived_fields(self, small_model_config):
        """The OpenELM fallback path should rebuild per-layer derived metadata."""
        raw_config = {
            "model_type": "openelm",
            "vocab_size": 32000,
            "num_transformer_layers": 12,
            "model_dim": 2048,
            "head_dim": 128,
            "qkv_multipliers": 1.0,
            "ffn_multipliers": [2.5 + 0.1 * i for i in range(12)],
            "num_gqa_groups": 1,
            "use_cache": True,
            "num_query_heads": [16] * 12,
            "num_kv_heads": [16] * 12,
        }

        conf = _build_openelm_config_from_raw(raw_config, small_model_config)

        assert conf.model_dim == small_model_config["hidden_size"]
        assert conf.num_transformer_layers == small_model_config["num_hidden_layers"]
        assert len(conf.num_query_heads) == small_model_config["num_hidden_layers"]
        assert len(conf.num_kv_heads) == small_model_config["num_hidden_layers"]
        assert len(conf.ffn_multipliers) == small_model_config["num_hidden_layers"]

    def test_ragged_geometry_uses_per_layer_num_kv_heads(self):
        """Paged-cache geometry should read OpenELM's per-layer KV-head schedule."""
        from easydel.infra.mixins.generation import (
            _has_mixed_standard_ragged_geometry,
            _resolve_standard_ragged_layer_geometries,
        )

        config = ed.OpenELMConfig(
            vocab_size=1024,
            max_context_length=128,
            num_transformer_layers=3,
            model_dim=128,
            head_dim=16,
            qkv_multipliers=[1.0, 2.0],
            num_gqa_groups=2,
        )

        geometries = _resolve_standard_ragged_layer_geometries(config)

        assert geometries == {
            0: (4, 16),
            1: (6, 16),
            2: (8, 16),
        }
        assert _has_mixed_standard_ragged_geometry(config)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
