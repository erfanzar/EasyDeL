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

"""Tests for Gemma2 model."""

import pytest
import transformers

import easydel as ed
from tests.modules.mpmd._scheduler_utils import GENERATION_SCHEDULE_KIND, LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import CausalLMTester


class TestGemma2:
    """Test suite for Gemma2 model."""

    @pytest.fixture
    def gemma2_config(self, small_model_config):
        """Create Gemma2-specific config."""
        return ed.Gemma2Config(
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128 // 8,
            use_scan_mlp=False,
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_causal_lm(self, gemma2_config, small_model_config, mpmd_schedule_kind):
        """Test Gemma2ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gemma2",
            hf_class=transformers.Gemma2ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=gemma2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Gemma2 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_generation(self, gemma2_config, small_model_config, mpmd_schedule_kind):
        """Test Gemma2 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gemma2",
            hf_class=transformers.Gemma2ForCausalLM,
            config=gemma2_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Gemma2 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
