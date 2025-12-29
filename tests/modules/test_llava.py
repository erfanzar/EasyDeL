"""Tests for LLaVA vision-language model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from test_utils import CausalLMTester, VisionLanguageTester


class TestLLaVA:
    """Test suite for LLaVA vision-language model."""

    @pytest.fixture
    def llava_config(self, small_model_config):
        """Create LLaVA VLM config."""
        vision_config = ed.CLIPVisionConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=224,
            patch_size=14,
        )
        text_config = ed.LlamaConfig(
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
        return ed.LlavaConfig(
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=image_token_id,
        )

    @pytest.fixture
    def vlm_config(self, llava_config, small_model_config):
        """Create VLM-specific config for LLaVA."""
        batch_size = small_model_config["batch_size"]
        num_images = 1
        image_size = llava_config.vision_config.image_size
        patch_size = llava_config.vision_config.patch_size
        # Llava's default vision feature selection drops the CLS token.
        num_patches = (image_size // patch_size) ** 2

        return {
            "image_token_id": llava_config.image_token_id,
            "num_image_tokens": num_patches,
            "pixel_values_shape": (batch_size * num_images, 3, image_size, image_size),
            "num_images": num_images,
        }

    def test_vision_language(self, llava_config, small_model_config, vlm_config):
        """Test LlavaForConditionalGeneration with vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(
            local_cfg.get("sequence_length", 128),
            tokens_per_image + 32,
        )

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="llava",
            hf_class=transformers.LlavaForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=llava_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"LLaVA VLM failed: {result.error_message or result.comparison.details}"

    def test_text_only(self, llava_config, small_model_config):
        """Test LLaVA text-only forward pass."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.run(
            module_name="llava",
            hf_class=transformers.LlavaForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=llava_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"LLaVA text-only failed: {result.error_message or result.comparison.details}"

    def test_generation(self, llava_config, small_model_config):
        """Test LLaVA text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="llava",
            hf_class=transformers.LlavaForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=llava_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"LLaVA generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
