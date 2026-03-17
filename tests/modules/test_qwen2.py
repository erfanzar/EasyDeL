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

"""Tests for Qwen2 model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestQwen2:
    """Test suite for Qwen2 model."""

    @pytest.fixture
    def qwen2_config(self, small_model_config):
        """Create Qwen2-specific config."""
        return ed.Qwen2Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, qwen2_config, small_model_config):
        """Test Qwen2ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen2",
            hf_class=transformers.Qwen2ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=qwen2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen2 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, qwen2_config, small_model_config):
        """Test Qwen2 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen2",
            hf_class=transformers.Qwen2ForCausalLM,
            config=qwen2_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Qwen2 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
