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

"""Tests for the InternVL vision-language model."""

import pytest
import transformers

import easydel as ed

try:
    from tests.modules.test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from tests.modules.test_utils import (  # pyright: ignore[reportImplicitRelativeImport]
        CausalLMTester,
        VisionLanguageTester,
    )


class TestInternVL:
    """Test suite for the InternVL vision-language model."""

    @pytest.fixture
    def internvl_config(self, small_model_config):
        """Create an InternVL VLM config with a Qwen2 text backbone."""
        vision_config = ed.InternVLVisionConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=(56, 56),
            patch_size=(14, 14),
            attention_bias=True,
            use_qk_norm=True,
        )
        text_config = ed.Qwen2Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=max(small_model_config["max_position_embeddings"], 2048),
        )
        # Use a valid image token ID within vocab range
        image_token_id = small_model_config["vocab_size"] - 1
        return ed.InternVLConfig(
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=image_token_id,
            tie_word_embeddings=False,
        )

    @pytest.fixture
    def vlm_config(self, internvl_config, small_model_config):
        """Create VLM-specific config for InternVL."""
        batch_size = small_model_config["batch_size"]
        num_images = 1
        image_size = internvl_config.vision_config.image_size[0]
        patch_size = internvl_config.vision_config.patch_size[0]
        downsample_ratio = internvl_config.downsample_ratio
        # InternVL drops the CLS token, then pixel shuffle scales the number of
        # patch tokens by downsample_ratio**2.
        num_patches = int((image_size // patch_size) ** 2 * downsample_ratio**2)

        return {
            "image_token_id": internvl_config.image_token_id,
            "num_image_tokens": num_patches,
            "pixel_values_shape": (batch_size * num_images, 3, image_size, image_size),
            "num_images": num_images,
        }

    def test_vision_language(self, internvl_config, small_model_config, vlm_config):
        """Test InternVLForConditionalGeneration with vision inputs against HF."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(
            local_cfg.get("sequence_length", 128),
            tokens_per_image + 32,
        )

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="internvl",
            hf_class=transformers.InternVLForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=internvl_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"InternVL VLM failed: {result.error_message or result.comparison.details}"

    def test_text_only(self, internvl_config, small_model_config):
        """Test InternVL text-only forward pass against HF."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.run(
            module_name="internvl",
            hf_class=transformers.InternVLForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=internvl_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"InternVL text-only failed: {result.error_message or result.comparison.details}"

    def test_generation(self, internvl_config, small_model_config):
        """Test InternVL text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="internvl",
            hf_class=transformers.InternVLForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=internvl_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"InternVL generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
