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

"""Pure-Python config, registry, checkpoint-key, and NumPy tensor-reform tests for Inkling 2 / Inkling 2 Loop."""

from __future__ import annotations

import types

import numpy as np
import pytest
from easydel.caching import ConvKVCache, ConvKVCacheConfig, ConvKVCacheView
from easydel.infra.factory import TaskType, registry
from easydel.infra.utils import AttnMaskType
from easydel.layers import (
    BlockAttentionResidual,
    LoopMessageAttentionResidual,
    LoopMessageGate,
    softcapped_swiglu,
)
from easydel.layers.linear_attention import DynamicResidualShortConvolution, ResidualShortConvolution
from easydel.layers.rotary import (
    vision_grid_pos_embed_interpolation_indices_and_weights,
    vision_grid_position_ids,
)
from easydel.modules._base import BaseTaskModule, BaseVisionLanguageModule
from easydel.modules.auto.auto_configuration import infer_task_from_config
from easydel.modules.inkling2 import (
    INKLING2_CAUSAL_LM_KEY_RULES,
    INKLING2_LOOP_CAUSAL_LM_KEY_RULES,
    INKLING2_LOOP_TEXT_KEY_RULES,
    INKLING2_LOOP_VLM_KEY_RULES,
    INKLING2_LOOP_VLM_MODEL_KEY_RULES,
    INKLING2_TEXT_KEY_RULES,
    INKLING2_VISION_KEY_RULES,
    INKLING2_VLM_KEY_RULES,
    INKLING2_VLM_MODEL_KEY_RULES,
    Inkling2Cache,
    Inkling2CacheConfig,
    Inkling2CacheView,
    Inkling2Config,
    Inkling2ForCausalLM,
    Inkling2ForConditionalGeneration,
    Inkling2LoopConfig,
    Inkling2LoopForCausalLM,
    Inkling2LoopForConditionalGeneration,
    Inkling2LoopModel,
    Inkling2LoopTextConfig,
    Inkling2LoopTextModel,
    Inkling2Model,
    Inkling2TextConfig,
    Inkling2TextModel,
    Inkling2VisionConfig,
    Inkling2VisionModel,
)
from easydel.modules.inkling2._checkpoint import (
    deinterleave_inkling2_gate_up,
    dequantize_inkling2_experts,
    interleave_inkling2_gate_up,
    merge_inkling2_loop_message_queries,
    pad_inkling2_lm_head,
    split_inkling2_loop_message_queries,
    unpad_inkling2_lm_head,
)
from easydel.modules.inkling2.modeling_inkling2 import (
    Inkling2BlockAttentionResidual,
    Inkling2DynamicShortConvolution,
    Inkling2LoopMessageAttentionResidual,
    Inkling2LoopMessageGate,
    Inkling2ShortConvolution,
    _inkling2_loop_pipeline_layer_position,
    _inkling2_loop_unit_count,
)
from easydel.scripts.convert_hf_to_easydel import _import_easydel_family
from easydel.trainers.process_encoder import resolve_encoder_binding

pytestmark = pytest.mark.cpu_ok


# Representative key sets covering every pattern from:
# - thinkingmachines/Inkling-2-Small (66 patterns)
# - thinkingmachines/Inkling-2-Small-MXFP8 (62 patterns)
# - thinkingmachines/Inkling-2-Mini (80 patterns)
_SMALL_VENDOR_KEYS: dict[str, str] = {
    "model.llm.attn_res.attn_queries.0": "model.language_model.layers.1.attention_input.query",
    "model.llm.attn_res.attn_queries.46": "model.language_model.layers.47.attention_input.query",
    "model.llm.attn_res.mlp_queries.0": "model.language_model.layers.0.mlp_input.query",
    "model.llm.attn_res.mlp_queries.47": "model.language_model.layers.47.mlp_input.query",
    "model.llm.attn_res.output_query": "model.language_model.final_input.query",
    "model.llm.embed.weight": "model.language_model.embed_tokens.weight",
    "model.llm.embed_norm.weight": "model.language_model.embed_tokens.embed_norm.weight",
    "model.llm.norm.weight": "model.language_model.norm.weight",
    "model.llm.unembed.weight": "lm_head.weight",
    # Sliding layer (idx 0)
    "model.llm.layers.0.attn.attn_sink_proj.weight": "model.language_model.layers.0.self_attn.sink_proj.weight",
    "model.llm.layers.0.attn.gate_attn.weight": "model.language_model.layers.0.self_attn.o_proj.gate_proj.weight",
    "model.llm.layers.0.attn.k_sconv.weight": "model.language_model.layers.0.self_attn.k_proj.sconv.conv1d.weight",
    "model.llm.layers.0.attn.v_sconv.weight": "model.language_model.layers.0.self_attn.v_proj.sconv.conv1d.weight",
    "model.llm.layers.0.attn.q_l2_clip.scale": "model.language_model.layers.0.self_attn.q_proj.norm.scale",
    "model.llm.layers.0.attn.rel_l2_clip.scale": "model.language_model.layers.0.self_attn.relative_position_bias.norm.scale",
    "model.llm.layers.0.attn.rel_logits_proj.proj": "model.language_model.layers.0.self_attn.relative_position_bias.proj",
    "model.llm.layers.0.attn.wk_dv.weight": "model.language_model.layers.0.self_attn.k_proj.linear.weight",
    "model.llm.layers.0.attn.wo_ud.weight": "model.language_model.layers.0.self_attn.o_proj.linear.weight",
    "model.llm.layers.0.attn.wq_du.weight": "model.language_model.layers.0.self_attn.q_proj.linear.weight",
    "model.llm.layers.0.attn.wr_du.weight": "model.language_model.layers.0.self_attn.relative_position_bias.linear.weight",
    "model.llm.layers.0.attn.wv_dv.weight": "model.language_model.layers.0.self_attn.v_proj.linear.weight",
    "model.llm.layers.0.attn_norm.weight": "model.language_model.layers.0.input_layernorm.weight",
    "model.llm.layers.0.attn_sconv.weight": "model.language_model.layers.0.post_attention_convolution.conv1d.weight",
    "model.llm.layers.0.mlp_norm.weight": "model.language_model.layers.0.post_attention_layernorm.weight",
    "model.llm.layers.0.mlp_sconv.weight": "model.language_model.layers.0.post_mlp_convolution.conv1d.weight",
    "model.llm.layers.0.mlp.gate.bias": "model.language_model.layers.0.mlp.gate.e_score_correction_bias",
    "model.llm.layers.0.mlp.gate.weight": "model.language_model.layers.0.mlp.gate.weight",
    "model.llm.layers.0.mlp.experts.w13_weight": "model.language_model.layers.0.mlp.experts.w13_weight",
    "model.llm.layers.0.mlp.experts.w13_weight_scale": "model.language_model.layers.0.mlp.experts.w13_weight_scale",
    "model.llm.layers.0.mlp.experts.w13_weight_amax": "model.language_model.layers.0.mlp.experts.w13_weight_amax",
    "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator": (
        "model.language_model.layers.0.mlp.experts.w13_weight_outer_scale_denominator"
    ),
    "model.llm.layers.0.mlp.experts.w2_weight": "model.language_model.layers.0.mlp.experts.w2_weight",
    "model.llm.layers.0.mlp.experts.w2_weight_scale": "model.language_model.layers.0.mlp.experts.w2_weight_scale",
    "model.llm.layers.0.mlp.experts.w2_weight_amax": "model.language_model.layers.0.mlp.experts.w2_weight_amax",
    "model.llm.layers.0.mlp.experts.w2_weight_outer_scale_denominator": (
        "model.language_model.layers.0.mlp.experts.w2_weight_outer_scale_denominator"
    ),
    # Full layer (idx 5)
    "model.llm.layers.5.attn.k_sconv.static_w": "model.language_model.layers.5.self_attn.k_proj.sconv.static_weight",
    "model.llm.layers.5.attn.k_sconv.weight_proj": "model.language_model.layers.5.self_attn.k_proj.sconv.weight_proj.weight",
    "model.llm.layers.5.attn.v_sconv.static_w": "model.language_model.layers.5.self_attn.v_proj.sconv.static_weight",
    "model.llm.layers.5.attn.v_sconv.weight_proj": "model.language_model.layers.5.self_attn.v_proj.sconv.weight_proj.weight",
    "model.llm.layers.5.q_input_norm.weight": "model.language_model.layers.5.q_input_layernorm.weight",
    "model.llm.layers.5.k_input_norm.weight": "model.language_model.layers.5.k_input_layernorm.weight",
    "model.llm.layers.5.v_input_norm.weight": "model.language_model.layers.5.v_input_layernorm.weight",
    # Vision tower
    "model.visual.backbone.embeddings.class_embedding": "model.vision_tower.embeddings.class_embedding",
    "model.visual.backbone.embeddings.cls_position_embedding": "model.vision_tower.embeddings.cls_position_embedding",
    "model.visual.backbone.embeddings.patch_embedding.weight": "model.vision_tower.embeddings.patch_embedding.weight",
    "model.visual.backbone.embeddings.position_embedding": "model.vision_tower.embeddings.position_embedding.weight",
    "model.visual.backbone.ln_pre.bias": "model.vision_tower.pre_layernorm.bias",
    "model.visual.backbone.ln_pre.weight": "model.vision_tower.pre_layernorm.weight",
    "model.visual.backbone.layers.0.ln_1.bias": "model.vision_tower.layers.0.layernorm_before.bias",
    "model.visual.backbone.layers.0.ln_1.weight": "model.vision_tower.layers.0.layernorm_before.weight",
    "model.visual.backbone.layers.0.ln_2.bias": "model.vision_tower.layers.0.layernorm_after.bias",
    "model.visual.backbone.layers.0.ln_2.weight": "model.vision_tower.layers.0.layernorm_after.weight",
    "model.visual.backbone.layers.0.ls_1.gamma": "model.vision_tower.layers.0.lambda_1",
    "model.visual.backbone.layers.0.ls_2.gamma": "model.vision_tower.layers.0.lambda_2",
    "model.visual.backbone.layers.0.mlp.c_fc.bias": "model.vision_tower.layers.0.mlp.fc1.bias",
    "model.visual.backbone.layers.0.mlp.c_fc.weight": "model.vision_tower.layers.0.mlp.fc1.weight",
    "model.visual.backbone.layers.0.mlp.c_proj.bias": "model.vision_tower.layers.0.mlp.fc2.bias",
    "model.visual.backbone.layers.0.mlp.c_proj.weight": "model.vision_tower.layers.0.mlp.fc2.weight",
    "model.visual.backbone.layers.0.self_attn.k_proj.bias": "model.vision_tower.layers.0.self_attn.k_proj.bias",
    "model.visual.backbone.layers.0.self_attn.k_proj.weight": "model.vision_tower.layers.0.self_attn.k_proj.weight",
    "model.visual.backbone.layers.0.self_attn.out_proj.bias": "model.vision_tower.layers.0.self_attn.out_proj.bias",
    "model.visual.backbone.layers.0.self_attn.out_proj.weight": "model.vision_tower.layers.0.self_attn.out_proj.weight",
    "model.visual.backbone.layers.0.self_attn.q_proj.bias": "model.vision_tower.layers.0.self_attn.q_proj.bias",
    "model.visual.backbone.layers.0.self_attn.q_proj.weight": "model.vision_tower.layers.0.self_attn.q_proj.weight",
    "model.visual.backbone.layers.0.self_attn.v_proj.bias": "model.vision_tower.layers.0.self_attn.v_proj.bias",
    "model.visual.backbone.layers.0.self_attn.v_proj.weight": "model.vision_tower.layers.0.self_attn.v_proj.weight",
    "model.visual.proj_in.weight": "model.vision_tower.merger.linear_1.weight",
    "model.visual.proj_out.weight": "model.vision_tower.merger.linear_2.weight",
}

_MINI_EXTRA_VENDOR_KEYS: dict[str, str] = {
    "model.llm.layers.6.controllers.attn.down_weight": "model.language_model.layers.6.q_message_gate.down_proj.weight",
    "model.llm.layers.6.controllers.attn.forget_weight": (
        "model.language_model.layers.6.q_message_gate.forget_gate_proj.weight"
    ),
    "model.llm.layers.6.controllers.attn.gain_weight": (
        "model.language_model.layers.6.q_message_gate.message_gate_proj.weight"
    ),
    "model.llm.layers.6.controllers.k.down_weight": "model.language_model.layers.6.k_message_gate.down_proj.weight",
    "model.llm.layers.6.controllers.k.forget_weight": (
        "model.language_model.layers.6.k_message_gate.forget_gate_proj.weight"
    ),
    "model.llm.layers.6.controllers.k.gain_weight": (
        "model.language_model.layers.6.k_message_gate.message_gate_proj.weight"
    ),
    "model.llm.layers.6.controllers.v.down_weight": "model.language_model.layers.6.v_message_gate.down_proj.weight",
    "model.llm.layers.6.controllers.v.forget_weight": (
        "model.language_model.layers.6.v_message_gate.forget_gate_proj.weight"
    ),
    "model.llm.layers.6.controllers.v.gain_weight": (
        "model.language_model.layers.6.v_message_gate.message_gate_proj.weight"
    ),
    "model.llm.layers.6.controllers.mlp.down_weight": "model.language_model.layers.6.mlp_message_gate.down_proj.weight",
    "model.llm.layers.6.controllers.mlp.forget_weight": (
        "model.language_model.layers.6.mlp_message_gate.forget_gate_proj.weight"
    ),
    "model.llm.layers.6.controllers.mlp.gain_weight": (
        "model.language_model.layers.6.mlp_message_gate.message_gate_proj.weight"
    ),
    "model.llm.middle_cycle_cross_pass_queries": "model.language_model.middle_cycle_cross_pass_queries",
    "model.llm.middle_cycle_cross_pass_kv_queries": "model.language_model.middle_cycle_cross_pass_kv_queries",
}


class TestInkling2ConfigsAndRegistry:
    """Configuration parsing, validation, mask metadata, and registry tests."""

    def test_hub_small_config_payload(self):
        local_ids = [i for i in range(48) if (i + 1) % 6 != 0]
        cfg = Inkling2Config(
            eos_token_id=200006,
            text_config={
                "model_max_length": 1048576,
                "hidden_size": 4608,
                "num_hidden_layers": 48,
                "vocab_size": 201024,
                "unpadded_vocab_size": 200058,
                "num_attention_heads": 48,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "d_rel": 16,
                "rel_extent": 1024,
                "log_scaling_n_floor": 8192,
                "log_scaling_alpha": 0.1,
                "local_layer_ids": local_ids,
                "sconv_kernel_size": 4,
                "sliding_window_size": 1024,
                "n_layers_per_attn_res_block": 6,
                "n_routed_experts": 384,
                "num_experts_per_tok": 8,
                "intermediate_size": 2304,
                "route_scale": 8.0,
            },
            vision_config={
                "native_patch_size": 20,
                "adapter_merge_size": 2,
                "projection_hidden_dim": 4096,
                "projector_bias": False,
                "hidden_size": 1024,
                "num_hidden_layers": 23,
                "num_attention_heads": 16,
                "mlp_ratio": 4.0,
                "posemb_grid_size": 32,
                "layer_norm_eps": 1e-5,
            },
            quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"},
            mtp_config={"num_hidden_layers": 5},
        )
        assert cfg.text_config.max_position_embeddings == 1048576
        assert cfg.text_config.conv_kernel_size == 4
        assert cfg.text_config.moe_intermediate_size == 2304
        assert cfg.text_config.unpadded_vocab_size == 200058
        assert cfg.text_config.total_num_hidden_layers == 48
        assert cfg.text_config.layer_types[5] == "hybrid"
        assert cfg.text_config.layer_types[0] == "hybrid_sliding"
        assert cfg.vision_config.patch_size == 20
        assert cfg.vision_config.spatial_merge_size == 2
        assert cfg.vision_config.intermediate_size == 4096
        assert cfg.vision_config.text_hidden_size == 4608
        assert cfg.vision_config.rope_parameters["rope_type"] == "axial"
        assert cfg.vision_config.rope_parameters["rope_theta"] == 10000.0
        assert cfg.quantization_config == {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        mask_details = cfg.get_mask_details()
        assert len(mask_details) == 48
        assert mask_details[0].mask_type == AttnMaskType.SLIDING
        assert mask_details[0].size == 1024
        assert mask_details[5].mask_type == AttnMaskType.FULL

    def test_hub_mini_loop_config_payload(self):
        local_ids = [i for i in range(24) if (i + 1) % 6 != 0]
        cfg = Inkling2LoopConfig(
            text_config={
                "model_type": "inkling2_loop_text",
                "hidden_size": 2304,
                "num_hidden_layers": 24,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "local_layer_ids": local_ids,
                "n_routed_experts": 128,
                "intermediate_size": 1152,
                "loop_config": {
                    "core_start": 6,
                    "core_end": 18,
                    "num_passes": 3,
                    "gate_rank": 64,
                    "is_varloop": False,
                },
            },
            vision_config={"text_hidden_size": 2304, "native_patch_size": 20, "adapter_merge_size": 2},
        )
        tcfg = cfg.text_config
        assert isinstance(tcfg, Inkling2LoopTextConfig)
        assert tcfg.loop_layer_index_start == 6
        assert tcfg.loop_layer_index_end == 18
        assert tcfg.loop_num_passes == 3
        assert tcfg.loop_gate_rank == 64
        assert tcfg.num_loop_layers == 12
        assert tcfg.total_num_hidden_layers == 48
        assert len(tcfg.effective_layer_types) == 48
        assert tcfg.effective_layer_types[24:36] == tcfg.layer_types[6:18]
        assert tcfg.effective_layer_types[36:48] == tcfg.layer_types[6:18]
        mask_details = cfg.get_mask_details()
        assert len(mask_details) == 48
        assert mask_details[29].mask_type == AttnMaskType.FULL  # slot 24 + (11 - 6) = 29

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="unpadded_vocab_size"):
            Inkling2TextConfig(vocab_size=64, unpadded_vocab_size=65)
        with pytest.raises(ValueError, match="Unknown Inkling 2 layer type"):
            Inkling2TextConfig(num_hidden_layers=2, layer_types=["hybrid", "bogus"])
        with pytest.raises(ValueError, match="must match"):
            Inkling2Config(
                text_config=Inkling2TextConfig(hidden_size=128),
                vision_config=Inkling2VisionConfig(text_hidden_size=256),
            )
        with pytest.raises(ValueError, match="axial"):
            Inkling2VisionConfig(rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
        with pytest.raises(ValueError, match="loop_layer_index_start"):
            Inkling2LoopTextConfig(
                num_hidden_layers=12, n_layers_per_attn_res_block=2, loop_layer_index_start=0, loop_layer_index_end=4
            )
        with pytest.raises(ValueError, match="multiple of"):
            Inkling2LoopTextConfig(
                num_hidden_layers=12, n_layers_per_attn_res_block=3, loop_layer_index_start=2, loop_layer_index_end=6
            )
        with pytest.raises(ValueError, match="multiple of"):
            Inkling2LoopTextConfig(
                num_hidden_layers=12, n_layers_per_attn_res_block=3, loop_layer_index_start=3, loop_layer_index_end=7
            )
        with pytest.raises(ValueError, match="loop_num_passes"):
            Inkling2LoopTextConfig(
                num_hidden_layers=12,
                n_layers_per_attn_res_block=2,
                loop_layer_index_start=2,
                loop_layer_index_end=6,
                loop_num_passes=0,
            )

    def test_factory_and_autoconfig_registrations(self):
        assert registry.get_module_registration(TaskType.BASE_MODULE, "inkling2_text").module is Inkling2TextModel
        assert registry.get_module_registration(TaskType.CAUSAL_LM, "inkling2_text").module is Inkling2ForCausalLM
        assert registry.get_module_registration(TaskType.CAUSAL_LM, "inkling2").module is Inkling2ForCausalLM
        assert registry.get_module_registration(TaskType.BASE_VISION, "inkling2_vision").module is Inkling2VisionModel
        assert registry.get_module_registration(TaskType.VISION_LM, "inkling2").module is Inkling2Model
        assert (
            registry.get_module_registration(TaskType.IMAGE_TEXT_TO_TEXT, "inkling2").module
            is Inkling2ForConditionalGeneration
        )
        assert (
            registry.get_module_registration(TaskType.BASE_MODULE, "inkling2_loop_text").module is Inkling2LoopTextModel
        )
        assert (
            registry.get_module_registration(TaskType.CAUSAL_LM, "inkling2_loop_text").module is Inkling2LoopForCausalLM
        )
        assert registry.get_module_registration(TaskType.CAUSAL_LM, "inkling2_loop").module is Inkling2LoopForCausalLM
        assert registry.get_module_registration(TaskType.VISION_LM, "inkling2_loop").module is Inkling2LoopModel
        assert (
            registry.get_module_registration(TaskType.IMAGE_TEXT_TO_TEXT, "inkling2_loop").module
            is Inkling2LoopForConditionalGeneration
        )
        assert infer_task_from_config({"model_type": "inkling2"}) == TaskType.IMAGE_TEXT_TO_TEXT
        assert infer_task_from_config({"model_type": "inkling2_loop"}) == TaskType.IMAGE_TEXT_TO_TEXT

        from transformers import AutoConfig

        assert isinstance(AutoConfig.for_model("inkling2"), Inkling2Config)
        assert isinstance(AutoConfig.for_model("inkling2_loop"), Inkling2LoopConfig)


class TestInkling2CheckpointKeyRules:
    """Exhaustive key-normalization coverage for raw Thinking Machines and canonical HF layouts."""

    def test_vlm_small_vendor_keys_exact_and_collision_free(self):
        mapped = {raw: INKLING2_VLM_KEY_RULES(raw) for raw in _SMALL_VENDOR_KEYS}
        assert mapped == _SMALL_VENDOR_KEYS
        assert len(set(mapped.values())) == len(mapped)
        # Canonical HF keys pass through verbatim
        for canonical in _SMALL_VENDOR_KEYS.values():
            assert INKLING2_VLM_KEY_RULES(canonical) == canonical
        # MTP keys are dropped
        assert INKLING2_VLM_KEY_RULES("mtp.layers.0.self_attn.q_proj.weight") is None
        assert INKLING2_VLM_KEY_RULES("model.mtp.head.weight") is None

    def test_vlm_loop_mini_vendor_keys_exact_and_collision_free(self):
        all_mini = {**_SMALL_VENDOR_KEYS, **_MINI_EXTRA_VENDOR_KEYS}
        mapped = {raw: INKLING2_LOOP_VLM_KEY_RULES(raw) for raw in all_mini}
        assert mapped == all_mini
        assert len(set(mapped.values())) == len(mapped)
        for canonical in all_mini.values():
            assert INKLING2_LOOP_VLM_KEY_RULES(canonical) == canonical

    def test_causal_lm_and_base_text_rules_scope_and_drop_vision(self):
        for raw, vlm_target in _SMALL_VENDOR_KEYS.items():
            clm_out = INKLING2_CAUSAL_LM_KEY_RULES(raw)
            txt_out = INKLING2_TEXT_KEY_RULES(raw)
            vlm_base_out = INKLING2_VLM_MODEL_KEY_RULES(raw)
            if raw.startswith("model.visual."):
                assert clm_out is None
                assert txt_out is None
                assert vlm_base_out == vlm_target.removeprefix("model.")
            elif raw == "model.llm.unembed.weight":
                assert clm_out == "lm_head.weight"
                assert txt_out is None
                assert vlm_base_out is None
            else:
                expected_clm = vlm_target.replace("model.language_model.", "model.", 1)
                expected_txt = vlm_target.removeprefix("model.language_model.")
                expected_vlm_base = vlm_target.removeprefix("model.")
                assert clm_out == expected_clm
                assert txt_out == expected_txt
                assert vlm_base_out == expected_vlm_base

        # Looped text-only / base-VLM spot checks
        assert (
            INKLING2_LOOP_CAUSAL_LM_KEY_RULES("model.llm.layers.6.controllers.attn.down_weight")
            == "model.layers.6.q_message_gate.down_proj.weight"
        )
        assert (
            INKLING2_LOOP_TEXT_KEY_RULES("model.llm.middle_cycle_cross_pass_queries")
            == "middle_cycle_cross_pass_queries"
        )
        assert (
            INKLING2_LOOP_VLM_MODEL_KEY_RULES("model.llm.middle_cycle_cross_pass_kv_queries")
            == "language_model.middle_cycle_cross_pass_kv_queries"
        )
        assert INKLING2_LOOP_CAUSAL_LM_KEY_RULES("model.visual.proj_in.weight") is None

    def test_standalone_vision_key_rules(self):
        assert (
            INKLING2_VISION_KEY_RULES("model.visual.backbone.embeddings.position_embedding")
            == "embeddings.position_embedding.weight"
        )
        assert INKLING2_VISION_KEY_RULES("model.visual.backbone.layers.2.ls_1.gamma") == "layers.2.lambda_1"
        assert INKLING2_VISION_KEY_RULES("model.visual.proj_out.weight") == "merger.linear_2.weight"
        assert INKLING2_VISION_KEY_RULES("embeddings.class_embedding") == "embeddings.class_embedding"


class TestInkling2ConversionHelpers:
    """Pure-NumPy validation of expert dequantization, interleaving, query merge, unpadding, and vision grid math."""

    def test_gate_up_interleave_roundtrip(self):
        rng = np.random.default_rng(0)
        gate = rng.standard_normal((4, 6, 8)).astype(np.float32)
        up = rng.standard_normal((4, 6, 8)).astype(np.float32)
        concat = np.concatenate([gate, up], axis=1)
        interleaved = np.empty_like(concat)
        interleaved[:, 0::2, :] = gate
        interleaved[:, 1::2, :] = up

        np.testing.assert_array_equal(deinterleave_inkling2_gate_up(interleaved), concat)
        np.testing.assert_array_equal(interleave_inkling2_gate_up(concat), interleaved)

    def test_dequantize_mxfp8_and_nvfp4_experts(self):
        # MXFP8: weight * 2 ** (scale - 127) over groups of 32
        rng = np.random.default_rng(1)
        w = rng.integers(-4, 5, size=(2, 4, 64)).astype(np.float32)
        scale = np.array([[[127, 128]] * 4, [[126, 129]] * 4], dtype=np.uint8)
        deq = dequantize_inkling2_experts(w, scale, deinterleave=False)
        expected = np.concatenate(
            [
                w[:, :, :32] * (2.0 ** (scale[:, :, 0:1].astype(np.float32) - 127.0)),
                w[:, :, 32:] * (2.0 ** (scale[:, :, 1:2].astype(np.float32) - 127.0)),
            ],
            axis=-1,
        )
        np.testing.assert_allclose(deq, expected, rtol=0, atol=0)

        # NVFP4: two E2M1 codes per uint8 byte, scale per 16 values, amax / denom per expert
        # Low nibble code 2 -> +1.0, high nibble code 11 -> -1.5 => byte = (11 << 4) | 2 = 0xB2
        packed = np.full((2, 4, 16), 0xB2, dtype=np.uint8)  # unpacks to (2, 4, 32)
        nv_scale = np.full((2, 4, 2), 0.5, dtype=np.float32)
        amax = np.array([4.0, 8.0], dtype=np.float32)
        denom = np.array([2.0, 2.0], dtype=np.float32)
        nv_deq = dequantize_inkling2_experts(packed, nv_scale, amax, denom, deinterleave=True)
        assert nv_deq.shape == (2, 4, 32)
        # Expert 0 factor = 2.0 * 0.5 = 1.0 -> even cols +1.0, odd cols -1.5
        np.testing.assert_allclose(nv_deq[0, :, 0::2], 1.0)
        np.testing.assert_allclose(nv_deq[0, :, 1::2], -1.5)
        # Expert 1 factor = 4.0 * 0.5 = 2.0 -> even cols +2.0, odd cols -3.0
        np.testing.assert_allclose(nv_deq[1, :, 0::2], 2.0)
        np.testing.assert_allclose(nv_deq[1, :, 1::2], -3.0)

    def test_loop_message_queries_merge_and_split_roundtrip(self):
        rng = np.random.default_rng(2)
        num_loop_layers, hidden = 3, 16
        q_mlp = rng.standard_normal((2 * num_loop_layers, hidden)).astype(np.float32)
        k_v = rng.standard_normal((2 * num_loop_layers, hidden)).astype(np.float32)
        merged = merge_inkling2_loop_message_queries(q_mlp, k_v)
        assert merged.shape == (num_loop_layers, 4, hidden)
        for idx in range(num_loop_layers):
            np.testing.assert_array_equal(merged[idx, 0], q_mlp[2 * idx])
            np.testing.assert_array_equal(merged[idx, 1], k_v[2 * idx])
            np.testing.assert_array_equal(merged[idx, 2], k_v[2 * idx + 1])
            np.testing.assert_array_equal(merged[idx, 3], q_mlp[2 * idx + 1])
        q_mlp_rt, k_v_rt = split_inkling2_loop_message_queries(merged)
        np.testing.assert_array_equal(q_mlp_rt, q_mlp)
        np.testing.assert_array_equal(k_v_rt, k_v)

    def test_lm_head_unpad_and_pad_roundtrip(self):
        rng = np.random.default_rng(3)
        padded = np.zeros((20, 8), dtype=np.float32)
        padded[:15] = rng.standard_normal((15, 8)).astype(np.float32)
        easydel_w = unpad_inkling2_lm_head(padded, 15)
        assert easydel_w.shape == (8, 15)
        np.testing.assert_array_equal(easydel_w, padded[:15].T)
        restored = pad_inkling2_lm_head(easydel_w, 20)
        assert restored.shape == (20, 8)
        np.testing.assert_array_equal(restored, padded)

    def test_vision_grid_position_ids_and_bilinear_interpolation(self):
        patch_grid = np.array([[1, 4, 4], [1, 2, 4]], dtype=np.int64)
        pos = vision_grid_position_ids(patch_grid, spatial_merge_size=2)
        assert pos.shape == (16 + 8, 2)
        # First 2x2 block of image 0 must be (0,0), (0,1), (1,0), (1,1)
        np.testing.assert_array_equal(pos[:4], [[0, 0], [0, 1], [1, 0], [1, 1]])
        # Second 2x2 block of image 0 must be (0,2), (0,3), (1,2), (1,3)
        np.testing.assert_array_equal(pos[4:8], [[0, 2], [0, 3], [1, 2], [1, 3]])

        indices, weights = vision_grid_pos_embed_interpolation_indices_and_weights(
            patch_grid, num_grid_per_side=8, spatial_merge_size=2, align_corners=False
        )
        assert indices.shape == (24, 4)
        assert weights.shape == (24, 4)
        assert indices.min() >= 0 and indices.max() < 64
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)


class TestInkling2ModularIntegrationsAndMPMDMapping:
    """Pure-Python checks for shared exports, converter discovery, process_encoder, and MPMD loop units."""

    def test_shared_cache_and_layer_exports(self):
        assert Inkling2Cache is ConvKVCache
        assert Inkling2CacheConfig is ConvKVCacheConfig
        assert Inkling2CacheView is ConvKVCacheView
        assert Inkling2BlockAttentionResidual is BlockAttentionResidual
        assert Inkling2LoopMessageAttentionResidual is LoopMessageAttentionResidual
        assert Inkling2LoopMessageGate is LoopMessageGate
        assert Inkling2ShortConvolution is ResidualShortConvolution
        assert Inkling2DynamicShortConvolution is DynamicResidualShortConvolution
        assert callable(softcapped_swiglu)
        assert hasattr(BaseTaskModule, "prepare_lm_head_inputs")
        assert hasattr(BaseVisionLanguageModule, "prepare_lm_head_inputs")
        assert hasattr(Inkling2ForConditionalGeneration, "prepare_lm_head_inputs")

    def test_convert_hf_family_prefix_import(self):
        for model_type in ("inkling2", "inkling2_loop", "inkling2_text", "inkling2_loop_text"):
            _import_easydel_family(model_type)

    def test_process_encoder_binding_resolution(self):
        for vlm_cls, cfg in (
            (Inkling2ForConditionalGeneration, Inkling2Config()),
            (Inkling2LoopForConditionalGeneration, Inkling2LoopConfig()),
        ):
            # Lightweight bound stub carrying the real forward signature & methods without allocating arrays.
            stub = object.__new__(vlm_cls)
            object.__setattr__(stub, "config", cfg)
            object.__setattr__(stub, "_base_model_name", "model")
            object.__setattr__(stub, "model", types.SimpleNamespace(vision_tower=object()))
            binding = resolve_encoder_binding(stub)
            assert binding is not None
            assert binding.tower_attr == "vision_tower"
            assert binding.feature_kwarg == "image_features"
            assert binding.feature_valid_length_kwarg == "image_features_valid_length"
            assert binding.feature_fn_name == "get_image_features"
            assert {"pixel_values", "image_grid_thw"}.issubset(binding.accepted_kwargs)

    def test_loop_mpmd_unit_mapping_and_validation(self):
        # 6 physical layers: pre=[0, 1], loop=[2, 3], post=[4, 5] => 2 + 1 + 2 = 5 acyclic units
        cfg = Inkling2LoopTextConfig(
            num_hidden_layers=6,
            n_layers_per_attn_res_block=2,
            loop_layer_index_start=2,
            loop_layer_index_end=4,
            loop_num_passes=3,
        )
        assert _inkling2_loop_unit_count(cfg) == 5

        dummy = types.SimpleNamespace(
            config=cfg,
            _pipeline_effective_layer_total=lambda total: int(total),
        )
        mapped = [_inkling2_loop_pipeline_layer_position(dummy, cfg, i) for i in range(6)]
        # Layers 2 and 3 collapse onto atomic unit 2; post-loop layers shift cleanly to 3 and 4.
        assert mapped == [(0, 5), (1, 5), (2, 5), (2, 5), (3, 5), (4, 5)]

        # Validate logical_pp > total_units raises a clear ValueError.
        stub = object.__new__(Inkling2LoopTextModel)
        object.__setattr__(stub, "config", cfg)
        bad_cfg = types.SimpleNamespace(
            loop_layer_index_start=2,
            loop_layer_index_end=4,
            num_hidden_layers=6,
            pipeline_virtual_stages=3,
            mesh=types.SimpleNamespace(shape={"pp": 2}),  # logical_pp = 6 > 5
        )
        with pytest.raises(ValueError, match="acyclic pipeline units"):
            stub._validate_mpmd_loop_units(bad_cfg)

        # Cache-slot to physical-layer mapping for extra passes.
        assert [stub._cache_slot_physical_layer(s) for s in range(cfg.total_num_hidden_layers)] == [
            0,
            1,
            2,
            3,
            4,
            5,
            2,
            3,
            2,
            3,
        ]
