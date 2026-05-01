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

"""Tests for PhiMoE model."""

import pytest
import transformers

import easydel as ed

try:
    from tests.modules.test_utils import CausalLMTester
except ImportError:
    from tests.modules.test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestPhiMoE:
    """Test suite for PhiMoE model."""

    @pytest.fixture
    def phimoe_config(self, small_model_config):
        """Create PhiMoE-specific config."""
        hidden_size = small_model_config["hidden_size"]
        num_attention_heads = small_model_config["num_attention_heads"]
        head_dim = hidden_size // num_attention_heads
        rope_dim = head_dim // 2  # 128 // 4 // 2 = 16

        # PhiMoE requires longrope scaling with specific fields
        rope_scaling = {
            "type": "longrope",
            "short_factor": [1.0] * rope_dim,
            "long_factor": [1.0] * rope_dim,
            "short_mscale": 1.0,
            "long_mscale": 1.0,
            "original_max_position_embeddings": small_model_config["max_position_embeddings"],
        }

        return ed.PhiMoeConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=hidden_size,
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=num_attention_heads,
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_local_experts=small_model_config.get("num_experts", 8),
            num_experts_per_tok=small_model_config.get("num_experts_per_tok", 2),
            rope_scaling=rope_scaling,
        )

    def test_causal_lm(self, phimoe_config, small_model_config):
        """Test PhiMoEForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="phimoe",
            hf_class=transformers.PhimoeForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=phimoe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"PhiMoE CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, phimoe_config, small_model_config):
        """Test PhiMoE text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="phimoe",
            hf_class=transformers.PhimoeForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=phimoe_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"PhiMoE generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
