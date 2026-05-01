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

"""Tests for Gemma4 model — text, vision-language, and generation."""

import numpy as np
import pytest
import transformers

import easydel as ed
from tests.modules.mpmd._scheduler_utils import GENERATION_SCHEDULE_KIND, LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import CausalLMTester, VisionLanguageTester


class TestGemma4:
    """Test suite for Gemma4 text and vision-language model."""

    @pytest.fixture
    def gemma4_config(self, small_model_config):
        """Create Gemma4 VLM config with mixed sliding/global attention."""
        image_token_id = small_model_config["vocab_size"] - 1
        boi_token_id = image_token_id - 1
        eoi_token_id = image_token_id - 2

        text_config = ed.Gemma4TextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=32,
            global_head_dim=32,
            sliding_window=64,
            hidden_size_per_layer_input=0,
        )
        vision_config = ed.Gemma4VisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            patch_size=4,
            pooling_kernel_size=2,
            position_embedding_size=16,
        )
        return ed.Gemma4Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            boi_token_id=boi_token_id,
            eoi_token_id=eoi_token_id,
        )

    @pytest.fixture
    def vlm_config(self, gemma4_config, small_model_config):
        """Create VLM-specific config for Gemma4."""
        batch_size = small_model_config["batch_size"]
        num_images = 1
        image_size = 8
        # Gemma4 patches: (image_size / patch_size)^2 / pooling^2
        patch_size = gemma4_config.vision_config.patch_size
        pooling = gemma4_config.vision_config.pooling_kernel_size
        grid = image_size // patch_size
        num_image_tokens = (grid // pooling) ** 2
        grid_y, grid_x = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
        image_position_ids = np.stack((grid_x, grid_y), axis=-1).reshape(1, grid * grid, 2)
        image_position_ids = np.broadcast_to(image_position_ids, (batch_size * num_images, grid * grid, 2)).copy()

        return {
            "image_token_id": gemma4_config.image_token_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (batch_size * num_images, grid * grid, 3 * patch_size * patch_size),
            "num_images": num_images,
            "use_token_type_ids": True,
            "image_position_ids": image_position_ids,
        }

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_vision_language(self, gemma4_config, small_model_config, vlm_config, mpmd_schedule_kind):
        """Test Gemma4ForConditionalGeneration with vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 128), tokens_per_image + 32)

        gemma4_config.text_config.max_position_embeddings = 2048

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="gemma4",
            hf_class=transformers.Gemma4ForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=gemma4_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"Gemma4 VLM failed: {result.error_message or result.comparison.details}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_generation(self, gemma4_config, small_model_config, mpmd_schedule_kind):
        """Test Gemma4 text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        gemma4_config.text_config.max_position_embeddings = 2048

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gemma4",
            hf_class=transformers.Gemma4ForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=gemma4_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"Gemma4 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
