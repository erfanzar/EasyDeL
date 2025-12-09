"""Tests for Gemma3 vision-language model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import VisionLanguageTester
except ImportError:
    from test_utils import VisionLanguageTester


class TestGemma3:
    """Test suite for Gemma3 vision-language model."""

    @pytest.fixture
    def gemma3_config(self, small_model_config):
        """Create Gemma3 VLM config."""
        # Keep the image token inside the small test vocab to avoid embedding index errors
        image_token_id = small_model_config["vocab_size"] - 1
        boi_token_index = image_token_id - 1
        eoi_token_index = image_token_id - 2

        text_config = ed.Gemma3TextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=128,
        )
        vision_config = {
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": 224,
            "patch_size": 14,
        }
        return ed.Gemma3Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            boi_token_index=boi_token_index,
            eoi_token_index=eoi_token_index,
        )

    @pytest.fixture
    def vlm_config(self, gemma3_config, small_model_config):
        """Create VLM-specific config for Gemma3."""
        batch_size = small_model_config["batch_size"]
        num_images = 1
        image_size = gemma3_config.vision_config.image_size
        num_image_tokens = gemma3_config.mm_tokens_per_image

        return {
            "image_token_id": gemma3_config.image_token_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (batch_size * num_images, 3, image_size, image_size),
            "num_images": num_images,
            "use_token_type_ids": True,
        }

    def test_vision_language(self, gemma3_config, small_model_config, vlm_config):
        """Test Gemma3ForConditionalGeneration with vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 128), tokens_per_image + 32)

        # Update the config's max_position_embeddings to match
        gemma3_config.text_config.max_position_embeddings = 2048

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="gemma3",
            hf_class=transformers.Gemma3ForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=gemma3_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"Gemma3 VLM failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
