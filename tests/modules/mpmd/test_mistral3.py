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

"""Tests for Mistral3 model."""

import jax.numpy as jnp
import pytest
import transformers

import easydel as ed
from tests.modules.mpmd._scheduler_utils import GENERATION_SCHEDULE_KIND, LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import CausalLMTester

# Check if Mistral3ForConditionalGeneration is available in transformers
HAS_MISTRAL3 = hasattr(transformers, "Mistral3ForConditionalGeneration")


class TestMistral3:
    """Test suite for Mistral3 model."""

    @pytest.fixture
    def mistral3_config(self, small_model_config):
        """Create Mistral3-specific config."""
        image_token_id = small_model_config["vocab_size"] - 1
        text_config = ed.MistralConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            sliding_window=None,
        )
        vision_config = ed.PixtralVisionConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=224,
            patch_size=14,
        )
        return ed.Mistral3Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_index=image_token_id,
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    @pytest.mark.skipif(not HAS_MISTRAL3, reason="transformers.Mistral3ForConditionalGeneration not available")
    def test_causal_lm(self, mistral3_config, small_model_config, mpmd_schedule_kind):
        """Test Mistral3ForConditionalGeneration."""
        local_cfg = small_model_config.copy()
        local_cfg["attn_dtype"] = jnp.float32
        local_cfg["attn_softmax_dtype"] = jnp.float32

        tester = CausalLMTester()
        result = tester.run(
            module_name="mistral3",
            hf_class=transformers.Mistral3ForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=mistral3_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"Mistral3 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    @pytest.mark.skipif(not HAS_MISTRAL3, reason="transformers.Mistral3ForConditionalGeneration not available")
    def test_generation(self, mistral3_config, small_model_config, mpmd_schedule_kind):
        """Test Mistral3 text generation."""
        local_cfg = small_model_config.copy()
        local_cfg["attn_dtype"] = jnp.float32
        local_cfg["attn_softmax_dtype"] = jnp.float32

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mistral3",
            hf_class=transformers.Mistral3ForConditionalGeneration,
            config=mistral3_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
        )
        assert result.success, f"Mistral3 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
