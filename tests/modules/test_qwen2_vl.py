"""Tests for Qwen2-VL model."""

# pyright: reportPrivateLocalImportUsage=false

import numpy as np
import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from test_utils import CausalLMTester, VisionLanguageTester  # pyright: ignore[reportImplicitRelativeImport]


class TestQwen2VL:
    """Test suite for Qwen2-VL vision-language model."""

    @pytest.fixture
    def qwen2_vl_config(self, small_model_config):
        """Create Qwen2-VL-specific config."""
        org_config = ed.Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        hf_config = transformers.Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        # Keep configs small for unit tests.
        org_config.text_config.vocab_size = small_model_config["vocab_size"]
        mrope_section = None
        if hf_config.text_config.rope_scaling is not None:
            mrope_section = hf_config.text_config.rope_scaling.get("mrope_section")
        required_head_dim = int(sum(mrope_section) * 2) if mrope_section else small_model_config["head_dim"]

        hidden_size = small_model_config["hidden_size"]
        if hidden_size % required_head_dim != 0:
            hidden_size = required_head_dim

        org_config.text_config.hidden_size = hidden_size
        org_config.text_config.intermediate_size = max(small_model_config["intermediate_size"], hidden_size * 2)
        org_config.text_config.num_attention_heads = hidden_size // required_head_dim
        org_config.text_config.num_key_value_heads = org_config.text_config.num_attention_heads
        org_config.text_config.num_hidden_layers = small_model_config["num_hidden_layers"]
        org_config.text_config.head_dim = required_head_dim

        # Ensure special tokens are in-range for embedding lookup.
        vocab_size = org_config.text_config.vocab_size
        org_config.video_token_id = vocab_size - 4
        org_config.vision_start_token_id = vocab_size - 3
        org_config.vision_end_token_id = vocab_size - 2
        org_config.image_token_id = vocab_size - 1

        org_config.text_config.rope_scaling = hf_config.text_config.rope_scaling
        org_config.text_config.layer_types = [
            "sliding_attention"
            if org_config.text_config.sliding_window is not None and i >= org_config.text_config.max_window_layers
            else "full_attention"
            for i in range(org_config.text_config.num_hidden_layers)
        ]

        org_config.vision_config.hidden_size = org_config.text_config.hidden_size
        org_config.vision_config.embed_dim = org_config.text_config.hidden_size
        org_config.vision_config.depth = small_model_config["num_hidden_layers"]
        # Keep original vision head count to match rotary embedding dims.
        org_config.vision_config.num_heads = hf_config.vision_config.num_heads
        return org_config

    @pytest.fixture
    def vlm_config(self, qwen2_vl_config, small_model_config):
        """Create VLM-specific config for Qwen2-VL."""
        num_images_per_batch = 1
        batch_size = small_model_config["batch_size"]
        grid_h, grid_w = 14, 14
        spatial_merge_size = qwen2_vl_config.vision_config.spatial_merge_size
        merged_h = grid_h // spatial_merge_size
        merged_w = grid_w // spatial_merge_size
        num_image_tokens = merged_h * merged_w

        total_patches = batch_size * num_images_per_batch * grid_h * grid_w
        in_channels = getattr(qwen2_vl_config.vision_config, "in_chans", None) or getattr(
            qwen2_vl_config.vision_config, "in_channels", 3
        )
        patch_size = qwen2_vl_config.vision_config.patch_size
        temporal_patch_size = qwen2_vl_config.vision_config.temporal_patch_size or 2
        patch_features = in_channels * temporal_patch_size * patch_size * patch_size

        image_grid_thw = np.tile(
            np.array([[1, grid_h, grid_w]], dtype=np.int64),
            (batch_size * num_images_per_batch, 1),
        )

        return {
            "image_token_id": qwen2_vl_config.image_token_id,
            "vision_start_token_id": qwen2_vl_config.vision_start_token_id,
            "vision_end_token_id": qwen2_vl_config.vision_end_token_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (total_patches, patch_features),
            "image_grid_thw": image_grid_thw,
            "num_images": num_images_per_batch,
            "is_qwen_vl": True,
        }

    def test_vision_language(self, qwen2_vl_config, small_model_config, vlm_config):
        """Test Qwen2VLForConditionalGeneration with vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(
            local_cfg.get("sequence_length", 128),
            tokens_per_image + 32,
        )

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="qwen2_vl",
            hf_class=transformers.Qwen2VLForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=qwen2_vl_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"Qwen2VL VLM failed: {result.error_message or result.comparison.details}"

    def test_text_only(self, qwen2_vl_config, small_model_config):
        """Test Qwen2VL text-only forward pass."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen2_vl",
            hf_class=transformers.Qwen2VLForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=qwen2_vl_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"Qwen2VL text-only failed: {result.error_message or result.comparison.details}"

    def test_generation(self, qwen2_vl_config, small_model_config):
        """Test Qwen2VL text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen2_vl",
            hf_class=transformers.Qwen2VLForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=qwen2_vl_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"Qwen2VL generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
