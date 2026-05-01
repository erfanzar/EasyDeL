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

"""Tests for Qwen3Next model."""

import pytest
import transformers

import easydel as ed
from tests.modules.mpmd._scheduler_utils import GENERATION_SCHEDULE_KIND, LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import CausalLMTester


class TestQwen3Next:
    """Test suite for Qwen3Next model."""

    @pytest.fixture
    def qwen3_next_config(self, small_model_config):
        """Create Qwen3Next-specific config."""
        return ed.Qwen3NextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_causal_lm(self, qwen3_next_config, small_model_config, mpmd_schedule_kind):
        """Test Qwen3NextForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen3_next",
            hf_class=transformers.Qwen3NextForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=qwen3_next_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3Next CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_generation(self, qwen3_next_config, small_model_config, mpmd_schedule_kind):
        """Test Qwen3Next text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen3_next",
            hf_class=transformers.Qwen3NextForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=qwen3_next_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Qwen3Next generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
