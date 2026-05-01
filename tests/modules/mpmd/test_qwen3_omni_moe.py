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

"""Tests for Qwen3OmniMoe model."""

import pytest

import easydel as ed
from tests.modules.mpmd._scheduler_utils import LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import EasyDeLOnlyTester


class TestQwen3OmniMoe:
    """Test suite for Qwen3OmniMoe model.

    Note: Qwen3OmniMoe is registered with TaskType.ANY_TO_ANY (multimodal)
    and TaskType.BASE_MODULE (thinker). No CAUSAL_LM registration exists.
    """

    @pytest.fixture
    def qwen3_omni_moe_thinker_config(self, small_model_config):
        """Create Qwen3OmniMoeThinker-specific config for BASE_MODULE testing."""
        return ed.Qwen3OmniMoeThinkerConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_experts=small_model_config.get("num_experts", 8),
            num_experts_per_tok=small_model_config.get("num_experts_per_tok", 2),
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_causal_lm(self, qwen3_omni_moe_thinker_config, small_model_config, mpmd_schedule_kind):
        """Test Qwen3OmniMoeThinker with BASE_MODULE task."""
        tester = EasyDeLOnlyTester()
        result = tester.run(
            module_name="qwen3_omni_moe",
            task=ed.TaskType.BASE_MODULE,  # Thinker is registered as BASE_MODULE
            config=qwen3_omni_moe_thinker_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3OmniMoe BASE_MODULE failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
