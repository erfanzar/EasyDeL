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

"""Tests for the HunYuanVL vision-language model.

Configs are built locally (no hub access). HF parity runs against
``transformers.HunYuanVLForConditionalGeneration`` constructed from the same
config; if a transformers build without the class is installed the tests run
EasyDeL-only.
"""

import easydel as ed
import numpy as np
import pytest
import transformers

try:
    from tests.modules.test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from tests.modules.test_utils import (  # pyright: ignore[reportImplicitRelativeImport]
        CausalLMTester,
        VisionLanguageTester,
    )

HF_HUNYUAN_VL_CLASS = getattr(transformers, "HunYuanVLForConditionalGeneration", None)

# Small but non-trivial geometry: 8x8 patch grid, merge 2 -> 4x4 merged grid,
# 4 * (4 + 1) newline-augmented grid tokens + begin/end = 22 tokens per image.
_GRID_H = 8
_GRID_W = 8


def _force_eager_attention(config) -> None:
    """Pin the HF reference to eager attention for deterministic parity."""
    for cfg in (config, config.text_config, config.vision_config):
        try:
            cfg._attn_implementation = "eager"
        except Exception:
            pass


class TestHunYuanVL:
    """Test suite for the HunYuanVL vision-language model."""

    @pytest.fixture
    def hunyuan_vl_config(self, small_model_config):
        """Create a small HunYuanVL composite config (no hub access)."""
        hidden_size = small_model_config["hidden_size"]
        num_heads = small_model_config["num_attention_heads"]
        head_dim = hidden_size // num_heads
        half = head_dim // 2
        mrope_section = [half // 4, (half - half // 4) // 2, half - half // 4 - (half - half // 4) // 2]

        vocab_size = small_model_config["vocab_size"]
        config = ed.HunYuanVLConfig(
            text_config=dict(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=small_model_config["intermediate_size"],
                num_hidden_layers=small_model_config["num_hidden_layers"],
                num_attention_heads=num_heads,
                num_key_value_heads=small_model_config["num_key_value_heads"],
                head_dim=head_dim,
                max_position_embeddings=small_model_config["max_position_embeddings"],
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
                rope_scaling={"rope_type": "default", "mrope_section": mrope_section},
                tie_word_embeddings=True,
            ),
            vision_config=dict(
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_hidden_layers=2,
                num_channels=3,
                patch_size=4,
                spatial_merge_size=2,
                temporal_patch_size=1,
                max_image_size=64,
                rms_norm_eps=1e-5,
            ),
            # Keep special token ids in-range for the reduced vocabulary.
            image_token_id=vocab_size - 1,
            im_start_id=vocab_size - 3,
            im_end_id=vocab_size - 2,
            im_newline_id=vocab_size - 4,
        )
        _force_eager_attention(config)
        return config

    @pytest.fixture
    def vlm_config(self, hunyuan_vl_config, small_model_config):
        """Create VLM test inputs description for HunYuanVL."""
        num_images_per_batch = 1
        batch_size = small_model_config["batch_size"]
        vision_config = hunyuan_vl_config.vision_config

        merge = vision_config.spatial_merge_size
        llm_h = _GRID_H // merge
        llm_w = _GRID_W // merge
        # Grid tokens (with one newline per merged row) plus begin/end tokens.
        num_image_tokens = llm_h * (llm_w + 1) + 2

        total_patches = batch_size * num_images_per_batch * _GRID_H * _GRID_W
        patch_features = vision_config.num_channels * vision_config.temporal_patch_size * vision_config.patch_size**2

        image_grid_thw = np.tile(
            np.array([[1, _GRID_H, _GRID_W]], dtype=np.int64),
            (batch_size * num_images_per_batch, 1),
        )

        return {
            "image_token_id": hunyuan_vl_config.image_token_id,
            "vision_start_token_id": hunyuan_vl_config.im_start_id,
            "vision_end_token_id": hunyuan_vl_config.im_end_id,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (total_patches, patch_features),
            "image_grid_thw": image_grid_thw,
            "num_images": num_images_per_batch,
            "is_qwen_vl": True,
            "use_mm_token_type_ids": True,
        }

    def test_vision_language(self, hunyuan_vl_config, small_model_config, vlm_config):
        """Test HunYuanVLForConditionalGeneration logits parity with vision inputs."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 128), tokens_per_image + 32)

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="hunyuan_vl",
            hf_class=HF_HUNYUAN_VL_CLASS,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=hunyuan_vl_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"HunYuanVL VLM failed: {result.error_message or result.comparison.details}"

    def test_text_only(self, hunyuan_vl_config, small_model_config):
        """Test HunYuanVL text-only logits parity."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.run(
            module_name="hunyuan_vl",
            hf_class=HF_HUNYUAN_VL_CLASS,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=hunyuan_vl_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"HunYuanVL text-only failed: {result.error_message or result.comparison.details}"

    def test_generation(self, hunyuan_vl_config, small_model_config):
        """Test HunYuanVL text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="hunyuan_vl",
            hf_class=HF_HUNYUAN_VL_CLASS,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=hunyuan_vl_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"HunYuanVL generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
