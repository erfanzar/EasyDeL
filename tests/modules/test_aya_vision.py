"""Tests for Aya Vision model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from test_utils import CausalLMTester, VisionLanguageTester


class TestAyaVision:
    """Test suite for Aya Vision model."""

    @pytest.fixture
    def aya_vision_config(self, small_model_config):
        """Create Aya Vision config."""
        # AyaVision uses pixel_shuffle with downsample_factor=2, so patch grid must be even
        # 224 / 14 = 16 patches per side (even, works with pixel shuffle)
        vision_config = ed.SiglipVisionConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=224,
            patch_size=14,
        )
        # AyaVision uses image_token_index=255001, so vocab_size must be larger
        text_config = ed.CohereConfig(
            vocab_size=256000,  # Must be > image_token_index (255001)
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )
        return ed.AyaVisionConfig(vision_config=vision_config, text_config=text_config)

    @pytest.fixture
    def vlm_config(self, aya_vision_config, small_model_config):
        """Create VLM-specific config for Aya Vision."""
        batch_size = small_model_config["batch_size"]
        num_images = 1
        image_size = aya_vision_config.vision_config.image_size
        patch_size = aya_vision_config.vision_config.patch_size
        num_patches = (image_size // patch_size) ** 2
        # AyaVision uses pixel_shuffle with downsample_factor=2, which reduces tokens by 4x
        downsample_factor = 2
        num_image_tokens = num_patches // (downsample_factor**2)

        return {
            "image_token_id": aya_vision_config.image_token_index,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (batch_size * num_images, 3, image_size, image_size),
            "num_images": num_images,
        }

    def test_vision_language(self, aya_vision_config, small_model_config, vlm_config):
        """Test AyaVisionForConditionalGeneration with vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 128), tokens_per_image + 32)
        # Use vocab_size from config to ensure image_token_id is valid
        local_cfg["vocab_size"] = aya_vision_config.text_config.vocab_size

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="aya_vision",
            hf_class=transformers.AyaVisionForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=aya_vision_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"Aya Vision VLM failed: {result.error_message or result.comparison.details}"

    def test_text_only(self, aya_vision_config, small_model_config):
        """Test Aya Vision text-only forward pass."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        # Use vocab_size from config
        local_cfg["vocab_size"] = aya_vision_config.text_config.vocab_size

        tester = CausalLMTester()
        result = tester.run(
            module_name="aya_vision",
            hf_class=transformers.AyaVisionForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=aya_vision_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"Aya Vision text-only failed: {result.error_message or result.comparison.details}"

    def test_generation(self, aya_vision_config, small_model_config):
        """Test Aya Vision text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        local_cfg["vocab_size"] = aya_vision_config.text_config.vocab_size

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="aya_vision",
            hf_class=transformers.AyaVisionForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=aya_vision_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"Aya Vision generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
