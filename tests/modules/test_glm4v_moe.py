"""Tests for GLM4V-MoE model."""

# pyright: reportPrivateLocalImportUsage=false

import numpy as np
import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from test_utils import CausalLMTester, VisionLanguageTester  # pyright: ignore[reportImplicitRelativeImport]


class TestGlm4vMoe:
    """Test suite for GLM4V-MoE vision-language model."""

    @pytest.fixture
    def hf_glm4v_moe_class(self):
        return getattr(transformers, "Glm4vMoeForConditionalGeneration", None)

    @pytest.fixture
    def glm4v_moe_config(self, small_model_config):
        cfg = ed.Glm4vMoeConfig()

        vocab_size = small_model_config["vocab_size"]
        cfg.text_config.vocab_size = vocab_size

        cfg.text_config.hidden_size = 512
        cfg.text_config.intermediate_size = 1024
        cfg.text_config.num_attention_heads = 4
        cfg.text_config.num_key_value_heads = 2
        cfg.text_config.num_hidden_layers = 2
        cfg.text_config.head_dim = 128
        cfg.text_config.rope_scaling = {"rope_type": "default", "mrope_section": [8, 12, 12]}

        cfg.text_config.moe_intermediate_size = 128
        cfg.text_config.n_routed_experts = small_model_config.get("num_experts", 8)
        cfg.text_config.num_experts_per_tok = small_model_config.get("num_experts_per_tok", 2)
        cfg.text_config.n_group = 1
        cfg.text_config.topk_group = 1
        cfg.text_config.first_k_dense_replace = 1
        cfg.text_config.norm_topk_prob = True

        cfg.image_token_id = vocab_size - 1
        cfg.video_token_id = vocab_size - 2
        cfg.image_start_token_id = vocab_size - 3
        cfg.image_end_token_id = vocab_size - 4
        cfg.video_start_token_id = vocab_size - 5
        cfg.video_end_token_id = vocab_size - 6

        cfg.vision_config.hidden_size = 256
        cfg.vision_config.intermediate_size = 512
        cfg.vision_config.num_heads = 4
        cfg.vision_config.num_attention_heads = 4
        cfg.vision_config.depth = 2
        cfg.vision_config.out_hidden_size = cfg.text_config.hidden_size
        return cfg

    @pytest.fixture
    def vlm_config(self, glm4v_moe_config, small_model_config):
        """Create VLM-specific config for GLM4V-MoE."""
        num_images_per_batch = 1
        batch_size = small_model_config["batch_size"]
        grid_h, grid_w = 14, 14

        spatial_merge_size = glm4v_moe_config.vision_config.spatial_merge_size
        merged_h = grid_h // spatial_merge_size
        merged_w = grid_w // spatial_merge_size
        num_image_tokens = merged_h * merged_w

        total_patches = batch_size * num_images_per_batch * grid_h * grid_w
        in_channels = glm4v_moe_config.vision_config.in_channels
        patch_size = glm4v_moe_config.vision_config.patch_size
        temporal_patch_size = glm4v_moe_config.vision_config.temporal_patch_size or 2
        patch_features = in_channels * temporal_patch_size * patch_size * patch_size

        image_grid_thw = np.tile(
            np.array([[1, grid_h, grid_w]], dtype=np.int64),
            (batch_size * num_images_per_batch, 1),
        )

        return {
            "image_token_id": glm4v_moe_config.image_token_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (total_patches, patch_features),
            "image_grid_thw": image_grid_thw,
            "num_images": num_images_per_batch,
        }

    def test_vision_language(self, glm4v_moe_config, small_model_config, vlm_config, hf_glm4v_moe_class):
        """Test Glm4vMoeForConditionalGeneration with vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        local_cfg["sequence_length"] = max(
            local_cfg.get("sequence_length", 128),
            vlm_config["num_image_tokens"] + 32,
        )

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="glm4v_moe",
            hf_class=hf_glm4v_moe_class,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=glm4v_moe_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"GLM4V-MoE VLM failed: {result.error_message or result.comparison.details}"

    def test_text_only(self, glm4v_moe_config, small_model_config, hf_glm4v_moe_class):
        """Test GLM4V-MoE text-only forward pass."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.run(
            module_name="glm4v_moe",
            hf_class=hf_glm4v_moe_class,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=glm4v_moe_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"GLM4V-MoE text-only failed: {result.error_message or result.comparison.details}"

    def test_generation(self, glm4v_moe_config, small_model_config, hf_glm4v_moe_class):
        """Test GLM4V-MoE text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="glm4v_moe",
            hf_class=hf_glm4v_moe_class,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=glm4v_moe_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"GLM4V-MoE generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
