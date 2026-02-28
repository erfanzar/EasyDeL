"""Tests for Qwen3.5-MoE text and multimodal models."""

import numpy as np
import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from test_utils import CausalLMTester, VisionLanguageTester  # pyright: ignore[reportImplicitRelativeImport]


def _resolve_hf_class(top_level_name: str, module_path: str, class_name: str):
    cls = getattr(transformers, top_level_name, None)
    if cls is not None:
        return cls
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except Exception:
        return None


class TestQwen3_5Moe:
    """Test suite for Qwen3.5-MoE model family."""

    @pytest.fixture
    def hf_qwen3_5_moe_causal_class(self):
        return _resolve_hf_class(
            top_level_name="Qwen3_5MoeForCausalLM",
            module_path="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
            class_name="Qwen3_5MoeForCausalLM",
        ) or getattr(transformers, "Qwen3NextForCausalLM", None)

    @pytest.fixture
    def hf_qwen3_5_moe_conditional_class(self):
        return _resolve_hf_class(
            top_level_name="Qwen3_5MoeForConditionalGeneration",
            module_path="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
            class_name="Qwen3_5MoeForConditionalGeneration",
        )

    @pytest.fixture
    def qwen3_5_moe_text_config(self, small_model_config):
        """Create Qwen3.5-MoE text config."""
        config = ed.Qwen3_5MoeTextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["head_dim"],
            rope_theta=small_model_config["rope_theta"],
            attention_bias=small_model_config["attention_bias"],
            attention_dropout=small_model_config["attention_dropout"],
            num_experts=small_model_config.get("num_experts", 8),
            num_experts_per_tok=small_model_config.get("num_experts_per_tok", 2),
            moe_intermediate_size=small_model_config["intermediate_size"] // 2,
            shared_expert_intermediate_size=small_model_config["intermediate_size"] // 2,
            output_router_logits=False,
        )
        config.moe_force_xla_gmm = True
        return config

    @pytest.fixture
    def qwen3_5_moe_config(self, qwen3_5_moe_text_config):
        """Create Qwen3.5-MoE multimodal config."""
        qwen3_5_moe_text_config.rope_scaling = {
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        }
        vision_config = ed.Qwen3_5MoeVisionConfig(
            depth=2,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=qwen3_5_moe_text_config.hidden_size,
            num_position_embeddings=512,
            deepstack_visual_indexes=[],
        )
        vocab_size = qwen3_5_moe_text_config.vocab_size
        config = ed.Qwen3_5MoeConfig(
            text_config=qwen3_5_moe_text_config,
            vision_config=vision_config,
            video_token_id=vocab_size - 4,
            vision_start_token_id=vocab_size - 3,
            vision_end_token_id=vocab_size - 2,
            image_token_id=vocab_size - 1,
        )
        config.moe_force_xla_gmm = True
        return config

    @pytest.fixture
    def vlm_config(self, qwen3_5_moe_config, small_model_config):
        """Create VLM-specific config for Qwen3.5-MoE."""
        num_images_per_batch = 1
        batch_size = small_model_config["batch_size"]
        grid_h, grid_w = 8, 8
        spatial_merge_size = qwen3_5_moe_config.vision_config.spatial_merge_size
        merged_h = grid_h // spatial_merge_size
        merged_w = grid_w // spatial_merge_size
        num_image_tokens = merged_h * merged_w

        total_patches = batch_size * num_images_per_batch * grid_h * grid_w
        in_channels = getattr(qwen3_5_moe_config.vision_config, "in_chans", None) or getattr(
            qwen3_5_moe_config.vision_config,
            "in_channels",
            3,
        )
        patch_size = qwen3_5_moe_config.vision_config.patch_size
        temporal_patch_size = qwen3_5_moe_config.vision_config.temporal_patch_size or 2
        patch_features = in_channels * temporal_patch_size * patch_size * patch_size

        image_grid_thw = np.tile(
            np.array([[1, grid_h, grid_w]], dtype=np.int64),
            (batch_size * num_images_per_batch, 1),
        )

        return {
            "image_token_id": qwen3_5_moe_config.image_token_id,
            "vision_start_token_id": qwen3_5_moe_config.vision_start_token_id,
            "vision_end_token_id": qwen3_5_moe_config.vision_end_token_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (total_patches, patch_features),
            "image_grid_thw": image_grid_thw,
            "num_images": num_images_per_batch,
            "is_qwen_vl": True,
        }

    def test_causal_lm(self, qwen3_5_moe_text_config, small_model_config, hf_qwen3_5_moe_causal_class):
        """Test Qwen3_5MoeForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen3_5_moe",
            hf_class=hf_qwen3_5_moe_causal_class,
            task=ed.TaskType.CAUSAL_LM,
            config=qwen3_5_moe_text_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3.5-MoE CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, qwen3_5_moe_text_config, small_model_config, hf_qwen3_5_moe_causal_class):
        """Test Qwen3.5-MoE text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen3_5_moe",
            hf_class=hf_qwen3_5_moe_causal_class,
            task=ed.TaskType.CAUSAL_LM,
            config=qwen3_5_moe_text_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Qwen3.5-MoE generation failed: {result.error_message}"

    def test_vision_language(
        self,
        qwen3_5_moe_config,
        small_model_config,
        vlm_config,
        hf_qwen3_5_moe_conditional_class,
    ):
        """Test Qwen3_5MoeForConditionalGeneration with image inputs."""
        if hf_qwen3_5_moe_conditional_class is None:
            pytest.skip("transformers.Qwen3_5MoeForConditionalGeneration not available")

        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 128), tokens_per_image + 32)

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="qwen3_5_moe",
            hf_class=hf_qwen3_5_moe_conditional_class,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=qwen3_5_moe_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"Qwen3.5-MoE VLM failed: {result.error_message or result.comparison.details}"

    def test_multimodal_generation(self, qwen3_5_moe_config, small_model_config, hf_qwen3_5_moe_conditional_class):
        """Test Qwen3.5-MoE multimodal text-only generation path."""
        if hf_qwen3_5_moe_conditional_class is None:
            pytest.skip("transformers.Qwen3_5MoeForConditionalGeneration not available")

        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen3_5_moe",
            hf_class=hf_qwen3_5_moe_conditional_class,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=qwen3_5_moe_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"Qwen3.5-MoE multimodal generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
