"""Tests for Kimi-VL model."""

# pyright: reportPrivateLocalImportUsage=false

import numpy as np
import pytest
import transformers
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import easydel as ed

try:
    from .test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from test_utils import CausalLMTester, VisionLanguageTester  # pyright: ignore[reportImplicitRelativeImport]


class TestKimiVL:
    """Test suite for MoonshotAI Kimi-VL."""

    @pytest.fixture
    def hf_kimi_vl_class(self):
        """Load Kimi-VL HF class from hub without instantiating the large checkpoint."""
        repo_id = "moonshotai/Kimi-VL-A3B-Instruct"
        conf = transformers.AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        class_ref = conf.auto_map["AutoModelForCausalLM"]
        return get_class_from_dynamic_module(class_ref, repo_id)

    @pytest.fixture
    def kimi_vl_config(self, small_model_config):
        """Create a small Kimi-VL config for testing."""
        media_placeholder_token_id = 42
        if media_placeholder_token_id >= small_model_config["vocab_size"]:
            raise ValueError("media_placeholder_token_id must be < vocab_size for HF embedding lookup.")

        vision_config = ed.MoonViTConfig(
            patch_size=2,
            init_pos_emb_height=8,
            init_pos_emb_width=8,
            num_attention_heads=4,
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            merge_kernel_size=(2, 2),
        )

        text_config = ed.DeepseekV3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=max(1024, small_model_config["max_position_embeddings"]),
            moe_intermediate_size=128,
            n_shared_experts=1,
            n_routed_experts=small_model_config["num_experts"],
            num_experts_per_tok=small_model_config["num_experts_per_tok"],
            topk_method="noaux_tc",
            n_group=8,
            topk_group=4,
            scoring_func="sigmoid",
            norm_topk_prob=True,
            aux_loss_alpha=0.001,
            seq_aux=True,
            q_lora_rank=None,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=32,
            rope_scaling={"type": "linear", "factor": 1.0},
            attention_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )

        return ed.KimiVLConfig(
            vision_config=vision_config,
            text_config=text_config,
            media_placeholder_token_id=media_placeholder_token_id,
            pad_token_id=0,
            attn_implementation="eager",
            tie_word_embeddings=False,
        )

    @pytest.fixture
    def vlm_config(self, kimi_vl_config, small_model_config):
        """Create VLM-specific config for Kimi-VL."""
        batch_size = small_model_config["batch_size"]
        num_images = 1
        grid_h, grid_w = 8, 8
        merge_h, merge_w = tuple(kimi_vl_config.vision_config.merge_kernel_size)
        patch_size = kimi_vl_config.vision_config.patch_size

        num_image_tokens = (grid_h // merge_h) * (grid_w // merge_w)
        total_images = batch_size * num_images
        total_patches = total_images * grid_h * grid_w

        image_grid_hws = np.tile(
            np.array([[grid_h, grid_w]], dtype=np.int64),
            (total_images, 1),
        )

        return {
            "image_token_id": kimi_vl_config.media_placeholder_token_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (total_patches, 3, patch_size, patch_size),
            "num_images": num_images,
            "image_grid_hws": image_grid_hws,
        }

    def test_vision_language(self, hf_kimi_vl_class, kimi_vl_config, small_model_config, vlm_config):
        """Test KimiVLForConditionalGeneration with vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        local_cfg["vocab_size"] = kimi_vl_config.text_config.vocab_size
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 128), vlm_config["num_image_tokens"] + 32)

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="kimi_vl",
            hf_class=hf_kimi_vl_class,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=kimi_vl_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"Kimi-VL VLM failed: {result.error_message or result.comparison.details}"

    def test_text_only(self, hf_kimi_vl_class, kimi_vl_config, small_model_config):
        """Test Kimi-VL text-only forward pass."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        local_cfg["vocab_size"] = kimi_vl_config.text_config.vocab_size

        tester = CausalLMTester()
        result = tester.run(
            module_name="kimi_vl",
            hf_class=hf_kimi_vl_class,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=kimi_vl_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"Kimi-VL text-only failed: {result.error_message or result.comparison.details}"

    def test_generation(self, hf_kimi_vl_class, kimi_vl_config, small_model_config):
        """Test Kimi-VL text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        local_cfg["vocab_size"] = kimi_vl_config.text_config.vocab_size

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="kimi_vl",
            hf_class=hf_kimi_vl_class,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=kimi_vl_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"Kimi-VL generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
