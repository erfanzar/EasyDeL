"""Tests for LLaMA4 vision (conditional generation) model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from test_utils import CausalLMTester, VisionLanguageTester


class TestLlama4Vision:
    """Test suite for LLaMA4 vision model."""

    @pytest.fixture
    def llama4_vision_config(self, small_model_config):
        """Create LLaMA4 vision config."""
        header_config = ed.Llama4Config(
            boi_token_index=200080,
            eoi_token_index=200081,
            image_token_index=200092,
        )

        text_config = ed.Llama4TextConfig(
            hidden_size=512,
            intermediate_size=2048,
            intermediate_size_mlp=2048,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            use_qk_norm=False,
            vocab_size=202048,
            bos_token_id=200000,
            eos_token_id=[200001, 200007, 200008],
            pad_token_id=200018,
            hidden_act="silu",
            max_position_embeddings=4096,
            initializer_range=0.02,
            rms_norm_eps=1e-05,
            use_cache=True,
            attention_bias=False,
            attention_dropout=0.0,
            rope_theta=500000.0,
            rope_scaling=None,
            num_experts_per_tok=2,
            num_local_experts=8,
            output_router_logits=False,
            router_aux_loss_coef=0.0,
            router_jitter_noise=0.0,
        )

        vision_config = ed.Llama4VisionConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            vision_output_dim=512,
            projector_input_dim=512,
            projector_output_dim=512,
            image_size=336,
            patch_size=14,
            num_channels=3,
            hidden_act="gelu",
            initializer_range=0.02,
            norm_eps=1e-05,
            attention_dropout=0.0,
            rope_theta=10000,
            pixel_shuffle_ratio=0.5,
            projector_dropout=0.0,
            multi_modal_projector_bias=False,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        )

        header_config.text_config = text_config
        header_config.vision_config = vision_config
        return header_config

    @pytest.fixture
    def vlm_config(self, llama4_vision_config, small_model_config):
        """Create VLM-specific config."""
        vision_config = llama4_vision_config.vision_config
        image_size = vision_config.image_size  # 336
        patch_size = vision_config.patch_size  # 14
        pixel_shuffle_ratio = vision_config.pixel_shuffle_ratio  # 0.5
        patches_per_side = image_size // patch_size  # 24
        num_image_tokens = int((patches_per_side * pixel_shuffle_ratio) ** 2)  # 144
        num_images_per_batch = 1
        batch_size = small_model_config["batch_size"]

        return {
            "image_token_id": llama4_vision_config.image_token_index,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (batch_size * num_images_per_batch, 3, image_size, image_size),
            "num_images": num_images_per_batch,
            "use_token_type_ids": False,
        }

    def test_vision_language(self, llama4_vision_config, small_model_config, vlm_config):
        """Test Llama4ForConditionalGeneration."""
        # Ensure sequence_length is large enough
        local_cfg = small_model_config.copy()
        local_cfg["sequence_length"] = max(
            small_model_config["sequence_length"],
            vlm_config["num_image_tokens"] + 32,
        )

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="llama4",
            hf_class=transformers.Llama4ForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=llama4_vision_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"Llama4 VLM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, llama4_vision_config, small_model_config):
        """Test Llama4 text-only generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="llama4",
            hf_class=transformers.Llama4ForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=llama4_vision_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Llama4 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
