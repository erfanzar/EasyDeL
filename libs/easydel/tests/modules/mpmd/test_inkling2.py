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

"""MPMD pipeline-parallelism tests for Inkling 2 and Inkling 2 Loop (text & VLM)."""

from __future__ import annotations

import easydel as ed
import numpy as np
import pytest
import transformers

from tests.modules.mpmd._scheduler_utils import GENERATION_SCHEDULE_KIND, LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import CausalLMTester, VisionLanguageTester


def _make_text_config(small_model_config: dict) -> ed.Inkling2TextConfig:
    vocab_size = int(small_model_config["vocab_size"])
    num_layers = int(small_model_config["num_hidden_layers"])
    return ed.Inkling2TextConfig(
        vocab_size=vocab_size,
        unpadded_vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=32,
        sliding_window_size=16,
        d_rel=8,
        rel_extent=16,
        log_scaling_n_floor=8,
        log_scaling_alpha=0.1,
        max_position_embeddings=512,
        conv_kernel_size=4,
        moe_intermediate_size=64,
        n_routed_experts=small_model_config.get("num_experts", 8),
        num_experts_per_tok=small_model_config.get("num_experts_per_tok", 2),
        route_scale=2.0,
        n_layers_per_attn_res_block=2,
    )


def _make_loop_text_config(small_model_config: dict) -> ed.Inkling2LoopTextConfig:
    # 6 physical layers with looped core [2, 4) gives 2 pre-loop + 1 atomic loop core + 2 post-loop = 5 acyclic units,
    # satisfying logical_pp <= 5 across GPipe (logical_pp=2), KimiK2 (logical_pp=4), and DualPipeV (logical_pp=4).
    vocab_size = int(small_model_config["vocab_size"])
    return ed.Inkling2LoopTextConfig(
        vocab_size=vocab_size,
        unpadded_vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=32,
        sliding_window_size=16,
        d_rel=8,
        rel_extent=16,
        log_scaling_n_floor=8,
        log_scaling_alpha=0.1,
        max_position_embeddings=512,
        conv_kernel_size=4,
        moe_intermediate_size=64,
        n_routed_experts=small_model_config.get("num_experts", 8),
        num_experts_per_tok=small_model_config.get("num_experts_per_tok", 2),
        route_scale=2.0,
        n_layers_per_attn_res_block=2,
        loop_layer_index_start=2,
        loop_layer_index_end=4,
        loop_num_passes=2,
        loop_gate_rank=16,
    )


def _make_vision_config(text_hidden_size: int) -> ed.Inkling2VisionConfig:
    return ed.Inkling2VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        patch_size=4,
        spatial_merge_size=2,
        posemb_grid_size=4,
        projection_hidden_dim=64,
        projector_bias=False,
        text_hidden_size=text_hidden_size,
    )


def _make_vlm_inputs_config(vlm_cfg: ed.Inkling2Config, small_model_config: dict) -> dict:
    batch_size = int(small_model_config["batch_size"])
    num_images_per_batch = 1
    # Inkling 2 `image_grid_thw` counts merged LM image tokens per spatial axis.
    merged_h, merged_w = 2, 2
    num_image_tokens = merged_h * merged_w
    merged_side = vlm_cfg.vision_config.patch_size * vlm_cfg.vision_config.spatial_merge_size
    total_image_tokens = batch_size * num_images_per_batch * num_image_tokens
    image_grid_thw = np.tile(
        np.array([[1, merged_h, merged_w]], dtype=np.int64),
        (batch_size * num_images_per_batch, 1),
    )
    return {
        "image_token_id": vlm_cfg.image_token_id,
        "num_image_tokens": num_image_tokens,
        "pixel_values_shape": (total_image_tokens, merged_side, merged_side, vlm_cfg.vision_config.num_channels),
        "image_grid_thw": image_grid_thw,
        "num_images": num_images_per_batch,
        "use_mm_token_type_ids": False,
    }


class TestInkling2MPMD:
    """MPMD pipeline-parallel test suite for flat Inkling 2 (text + VLM)."""

    @pytest.fixture
    def hf_causal_cls(self):
        return getattr(transformers, "Inkling2ForCausalLM", None)

    @pytest.fixture
    def hf_vlm_cls(self):
        return getattr(transformers, "Inkling2ForConditionalGeneration", None)

    @pytest.fixture
    def text_config(self, small_model_config):
        return _make_text_config(small_model_config)

    @pytest.fixture
    def vlm_model_config(self, small_model_config):
        tcfg = _make_text_config(small_model_config)
        return ed.Inkling2Config(
            text_config=tcfg,
            vision_config=_make_vision_config(tcfg.hidden_size),
            image_token_id=tcfg.vocab_size - 1,
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_causal_lm_loss(self, text_config, small_model_config, hf_causal_cls, mpmd_schedule_kind):
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 512
        result = CausalLMTester().run(
            module_name="inkling2_text",
            hf_class=hf_causal_cls,
            task=ed.TaskType.CAUSAL_LM,
            config=text_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"Inkling2 CausalLM MPMD failed: {result.error_message}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_causal_lm_generation(self, text_config, small_model_config, hf_causal_cls, mpmd_schedule_kind):
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 512
        result = CausalLMTester().test_generation(
            module_name="inkling2_text",
            hf_class=hf_causal_cls,
            task=ed.TaskType.CAUSAL_LM,
            config=text_config,
            small_model_config=local_cfg,
            max_new_tokens=8,
        )
        assert result.success, f"Inkling2 CausalLM MPMD generation failed: {result.error_message}"

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_vision_language_loss(self, vlm_model_config, small_model_config, hf_vlm_cls, mpmd_schedule_kind):
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 512
        vlm_inputs = _make_vlm_inputs_config(vlm_model_config, local_cfg)
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 64), vlm_inputs["num_image_tokens"] + 16)
        result = VisionLanguageTester().run(
            module_name="inkling2",
            hf_class=hf_vlm_cls,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=vlm_model_config,
            small_model_config=local_cfg,
            vlm_config=vlm_inputs,
        )
        assert result.success, f"Inkling2 VLM MPMD failed: {result.error_message}"

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_vlm_text_only_loss(self, vlm_model_config, small_model_config, hf_vlm_cls, mpmd_schedule_kind):
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 512
        result = CausalLMTester().run(
            module_name="inkling2",
            hf_class=hf_vlm_cls,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=vlm_model_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"Inkling2 VLM text-only MPMD failed: {result.error_message}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_vlm_generation(self, vlm_model_config, small_model_config, hf_vlm_cls, mpmd_schedule_kind):
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 512
        result = CausalLMTester().test_generation(
            module_name="inkling2",
            hf_class=hf_vlm_cls,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=vlm_model_config,
            small_model_config=local_cfg,
            max_new_tokens=8,
        )
        assert result.success, f"Inkling2 VLM MPMD generation failed: {result.error_message}"


class TestInkling2LoopMPMD:
    """MPMD pipeline-parallel test suite for Inkling 2 Loop (atomic looped core, text + VLM)."""

    @pytest.fixture
    def hf_loop_causal_cls(self):
        return getattr(transformers, "Inkling2LoopForCausalLM", None)

    @pytest.fixture
    def hf_loop_vlm_cls(self):
        return getattr(transformers, "Inkling2LoopForConditionalGeneration", None)

    @pytest.fixture
    def loop_text_config(self, small_model_config):
        return _make_loop_text_config(small_model_config)

    @pytest.fixture
    def loop_vlm_model_config(self, small_model_config):
        tcfg = _make_loop_text_config(small_model_config)
        return ed.Inkling2LoopConfig(
            text_config=tcfg,
            vision_config=_make_vision_config(tcfg.hidden_size),
            image_token_id=tcfg.vocab_size - 1,
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_loop_causal_lm_loss(self, loop_text_config, small_model_config, hf_loop_causal_cls, mpmd_schedule_kind):
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 512
        local_cfg["num_hidden_layers"] = 6
        local_cfg["num_layers"] = 6
        result = CausalLMTester().run(
            module_name="inkling2_loop_text",
            hf_class=hf_loop_causal_cls,
            task=ed.TaskType.CAUSAL_LM,
            config=loop_text_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"Inkling2Loop CausalLM MPMD failed: {result.error_message}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_loop_causal_lm_generation(
        self, loop_text_config, small_model_config, hf_loop_causal_cls, mpmd_schedule_kind
    ):
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 512
        local_cfg["num_hidden_layers"] = 6
        local_cfg["num_layers"] = 6
        result = CausalLMTester().test_generation(
            module_name="inkling2_loop_text",
            hf_class=hf_loop_causal_cls,
            task=ed.TaskType.CAUSAL_LM,
            config=loop_text_config,
            small_model_config=local_cfg,
            max_new_tokens=8,
        )
        assert result.success, f"Inkling2Loop CausalLM MPMD generation failed: {result.error_message}"

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_loop_vision_language_loss(
        self, loop_vlm_model_config, small_model_config, hf_loop_vlm_cls, mpmd_schedule_kind
    ):
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 512
        local_cfg["num_hidden_layers"] = 6
        local_cfg["num_layers"] = 6
        vlm_inputs = _make_vlm_inputs_config(loop_vlm_model_config, local_cfg)
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 64), vlm_inputs["num_image_tokens"] + 16)
        result = VisionLanguageTester().run(
            module_name="inkling2_loop",
            hf_class=hf_loop_vlm_cls,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=loop_vlm_model_config,
            small_model_config=local_cfg,
            vlm_config=vlm_inputs,
        )
        assert result.success, f"Inkling2Loop VLM MPMD failed: {result.error_message}"

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_loop_vlm_text_only_loss(
        self, loop_vlm_model_config, small_model_config, hf_loop_vlm_cls, mpmd_schedule_kind
    ):
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 512
        local_cfg["num_hidden_layers"] = 6
        local_cfg["num_layers"] = 6
        result = CausalLMTester().run(
            module_name="inkling2_loop",
            hf_class=hf_loop_vlm_cls,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=loop_vlm_model_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"Inkling2Loop VLM text-only MPMD failed: {result.error_message}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_loop_vlm_generation(self, loop_vlm_model_config, small_model_config, hf_loop_vlm_cls, mpmd_schedule_kind):
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 512
        local_cfg["num_hidden_layers"] = 6
        local_cfg["num_layers"] = 6
        result = CausalLMTester().test_generation(
            module_name="inkling2_loop",
            hf_class=hf_loop_vlm_cls,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=loop_vlm_model_config,
            small_model_config=local_cfg,
            max_new_tokens=8,
        )
        assert result.success, f"Inkling2Loop VLM MPMD generation failed: {result.error_message}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
