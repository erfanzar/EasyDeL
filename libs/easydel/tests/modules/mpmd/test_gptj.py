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

"""Tests for GPT-J model."""

import pytest
import transformers

import easydel as ed
from tests.modules.mpmd._scheduler_utils import GENERATION_SCHEDULE_KIND, LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import CausalLMTester


class TestGPTJ:
    """Test suite for GPT-J model."""

    @pytest.fixture
    def gptj_config(self, small_model_config):
        """Create GPT-J-specific config."""
        return ed.GPTJConfig(
            vocab_size=small_model_config["vocab_size"],
            n_positions=small_model_config["max_position_embeddings"],
            n_embd=small_model_config["hidden_size"],
            n_layer=small_model_config["num_hidden_layers"],
            n_head=small_model_config["num_attention_heads"],
            rotary_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_causal_lm(self, gptj_config, small_model_config, mpmd_schedule_kind):
        """Test GPTJForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gptj",
            hf_class=transformers.GPTJForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=gptj_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GPTJ CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_generation(self, gptj_config, small_model_config, mpmd_schedule_kind):
        """Test GPT-J text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gptj",
            hf_class=transformers.GPTJForCausalLM,
            config=gptj_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"GPTJ generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
