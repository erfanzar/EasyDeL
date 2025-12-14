"""Tests for Qwen3-VL model."""

import numpy as np
import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from test_utils import CausalLMTester, VisionLanguageTester


class TestQwen3VL:
    """Test suite for Qwen3-VL vision-language model."""

    @pytest.fixture
    def qwen3_vl_config(self, small_model_config):
        """Create Qwen3-VL-specific config."""
        org_config = ed.Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")
        org_config.text_config.hidden_size = 512
        org_config.text_config.intermediate_size = 1024
        org_config.text_config.num_attention_heads = 4
        org_config.text_config.num_key_value_heads = 2
        org_config.text_config.num_hidden_layers = 2
        org_config.text_config.head_dim = 128
        org_config.text_config.rope_scaling = {"rope_type": "default", "mrope_section": [24, 20, 20]}
        org_config.vision_config.out_hidden_size = org_config.text_config.hidden_size
        return org_config

    @pytest.fixture
    def vlm_config(self, qwen3_vl_config, small_model_config):
        """Create VLM-specific config for Qwen3-VL."""
        num_images_per_batch = 1
        batch_size = small_model_config["batch_size"]
        grid_h, grid_w = 16, 16
        spatial_merge_size = qwen3_vl_config.vision_config.spatial_merge_size
        merged_h = grid_h // spatial_merge_size
        merged_w = grid_w // spatial_merge_size
        num_image_tokens = merged_h * merged_w

        total_patches = batch_size * num_images_per_batch * grid_h * grid_w
        in_channels = getattr(qwen3_vl_config.vision_config, "in_chans", None) or getattr(
            qwen3_vl_config.vision_config, "in_channels", 3
        )
        patch_size = qwen3_vl_config.vision_config.patch_size
        temporal_patch_size = qwen3_vl_config.vision_config.temporal_patch_size or 2
        patch_features = in_channels * temporal_patch_size * patch_size * patch_size

        image_grid_thw = np.tile(
            np.array([[1, grid_h, grid_w]], dtype=np.int64),
            (batch_size * num_images_per_batch, 1),
        )

        return {
            "image_token_id": qwen3_vl_config.image_token_id,
            "vision_start_token_id": qwen3_vl_config.vision_start_token_id,
            "vision_end_token_id": qwen3_vl_config.vision_end_token_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (total_patches, patch_features),
            "image_grid_thw": image_grid_thw,
            "num_images": num_images_per_batch,
            "is_qwen_vl": True,
        }

    def test_vision_language(self, qwen3_vl_config, small_model_config, vlm_config):
        """Test Qwen3VLForConditionalGeneration with vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(
            local_cfg.get("sequence_length", 128),
            tokens_per_image + 32,
        )

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="qwen3_vl",
            hf_class=transformers.Qwen3VLForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=qwen3_vl_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"Qwen3-VL VLM failed: {result.error_message or result.comparison.details}"

    def test_text_only(self, qwen3_vl_config, small_model_config):
        """Test Qwen3-VL text-only forward pass."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen3_vl",
            hf_class=transformers.Qwen3VLForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=qwen3_vl_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"Qwen3-VL text-only failed: {result.error_message or result.comparison.details}"

    def test_generation(self, qwen3_vl_config, small_model_config):
        """Test Qwen3-VL text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen3_vl",
            hf_class=transformers.Qwen3VLForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=qwen3_vl_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"Qwen3-VL generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
