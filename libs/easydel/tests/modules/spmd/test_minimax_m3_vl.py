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

"""Tests for the MiniMax M3 VL family (composite VLM + self-contained text decoder).

The text parity config covers all four per-layer variants in two layers:
dense clamped-SwiGLU-OAI MLP + full attention on layer 0, sigmoid-routed MoE
+ block-sparse Lightning-Indexer attention on layer 1, with partial RoPE
(``partial_rotary_factor=0.5``). The VLM parity config exercises the Conv3d /
3D-RoPE vision tower, the two-stage patch-merge projector and the placeholder
scatter against the HF reference built from the same config.
"""

import numpy as np
import pytest
import transformers

import easydel as ed

try:
    from tests.modules.test_utils import CausalLMTester, VisionLanguageTester
except ImportError:
    from tests.modules.test_utils import (  # pyright: ignore[reportImplicitRelativeImport]
        CausalLMTester,
        VisionLanguageTester,
    )


HF_TEXT_CLASS = getattr(transformers, "MiniMaxM3VLForCausalLM", None)
HF_VLM_CLASS = getattr(transformers, "MiniMaxM3SparseForConditionalGeneration", None)


class TestMiniMaxM3VL:
    """Test suite for the MiniMax M3 VL model family."""

    @pytest.fixture(autouse=True)
    def _seed_torch(self):
        """Pin the torch global RNG so the HF reference init is reproducible.

        The model mixes discrete top-k selections (MoE router, Lightning
        Indexer blocks) whose near-tie flips depend on the weight draw;
        seeding keeps the parity margins identical across invocations.
        """
        import torch

        torch.manual_seed(42)

    @pytest.fixture
    def m3_small_config(self, small_model_config):
        """Return a test-config dict with the ``ep`` mesh axis folded into ``fsdp``.

        XLA:CPU cannot lower the ``ragged-all-to-all`` collective the fused MoE
        path emits when the expert mesh has ``ep > 1`` (the same pre-existing
        limit that affects the other MoE model tests on CPU). Together with
        ``fsdp_is_ep_bound = False`` on the model config this keeps the expert
        mesh at ``ep=1`` while still exercising TP sharding on the 8-fake-device
        CPU mesh; ep > 1 routing itself is hardware-validated.
        """
        local_cfg = dict(small_model_config)
        dims = local_cfg["sharding_axis_dims"]
        local_cfg["sharding_axis_dims"] = (dims[0], dims[1], dims[3], 1, dims[4], dims[5])
        return local_cfg

    @pytest.fixture
    def text_config(self, small_model_config):
        """Text config covering dense+full and MoE+block-sparse layers with partial RoPE."""
        num_hidden_layers = small_model_config["num_hidden_layers"]
        layer_types = ["full_attention"] * num_hidden_layers
        mlp_layer_types = ["dense"] + ["sparse"] * (num_hidden_layers - 1)
        layer_types[-1] = "minimax_m3_sparse"
        return ed.MiniMaxM3VLTextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
            intermediate_size=64,
            dense_intermediate_size=small_model_config["intermediate_size"],
            shared_intermediate_size=64,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_local_experts=small_model_config.get("num_local_experts", 8),
            num_experts_per_tok=2,
            routed_scaling_factor=2.0,
            partial_rotary_factor=0.5,
            layer_types=layer_types,
            mlp_layer_types=mlp_layer_types,
            index_n_heads=small_model_config["num_key_value_heads"],
            index_head_dim=16,
            index_block_size=8,
            index_topk_blocks=4,
            index_local_blocks=1,
            # Sparse layers have no decode cache (Lightning-Indexer key history
            # is not cached); keep the HF reference stateless too.
            use_cache=False,
        )

    @pytest.fixture
    def full_attention_text_config(self, small_model_config):
        """All-full-attention text config (cache-compatible) for generation."""
        num_hidden_layers = small_model_config["num_hidden_layers"]
        return ed.MiniMaxM3VLTextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
            intermediate_size=64,
            dense_intermediate_size=small_model_config["intermediate_size"],
            shared_intermediate_size=64,
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_local_experts=small_model_config.get("num_local_experts", 8),
            num_experts_per_tok=2,
            routed_scaling_factor=2.0,
            partial_rotary_factor=0.5,
            mlp_layer_types=["dense"] + ["sparse"] * (num_hidden_layers - 1),
        )

    @pytest.fixture
    def vlm_model_config(self, small_model_config):
        """Composite VLM config with a tiny vision tower and MoE text decoder."""
        vocab_size = small_model_config["vocab_size"]
        num_hidden_layers = small_model_config["num_hidden_layers"]
        vision_config = ed.MiniMaxM3VLVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=56,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
        )
        text_config = ed.MiniMaxM3VLTextConfig(
            vocab_size=vocab_size,
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            head_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
            intermediate_size=64,
            dense_intermediate_size=small_model_config["intermediate_size"],
            shared_intermediate_size=64,
            max_position_embeddings=2048,
            num_local_experts=small_model_config.get("num_local_experts", 8),
            num_experts_per_tok=2,
            routed_scaling_factor=2.0,
            partial_rotary_factor=0.5,
            mlp_layer_types=["dense"] + ["sparse"] * (num_hidden_layers - 1),
            use_cache=False,
        )
        return ed.MiniMaxM3VLConfig(
            vision_config=vision_config,
            text_config=text_config,
            image_token_index=vocab_size - 1,
            video_token_index=vocab_size - 2,
            projector_hidden_size=128,
        )

    @pytest.fixture
    def vlm_config(self, vlm_model_config, small_model_config):
        """VLM tester inputs: flattened patches + per-image (t, h, w) grids."""
        batch_size = small_model_config["batch_size"]
        num_images_per_batch = 1
        grid_t, grid_h, grid_w = 1, 4, 4
        vision_cfg = vlm_model_config.vision_config
        merge = vision_cfg.spatial_merge_size
        num_image_tokens = (grid_t * grid_h * grid_w) // (merge**2)
        total_patches = batch_size * num_images_per_batch * grid_t * grid_h * grid_w
        patch_features = vision_cfg.num_channels * vision_cfg.temporal_patch_size * vision_cfg.patch_size**2
        image_grid_thw = np.tile(
            np.array([[grid_t, grid_h, grid_w]], dtype=np.int64),
            (batch_size * num_images_per_batch, 1),
        )
        return {
            "image_token_id": vlm_model_config.image_token_index,
            "num_image_tokens": num_image_tokens,
            "pixel_values_shape": (total_patches, patch_features),
            "image_grid_thw": image_grid_thw,
            "num_images": num_images_per_batch,
        }

    def test_causal_lm(self, text_config, m3_small_config):
        """Logits parity vs HF for the text decoder (dense+full / MoE+sparse mix)."""
        import torch

        # The top-2-of-8 sigmoid router and the indexer's top-k block selection
        # flip on genuine near-ties (score gaps below the ~1e-3 tp=2 numeric
        # floor; HF itself is backend-dependent there). Seed a weight draw
        # without near-ties so parity measures the numeric floor, not discrete
        # tie ordering.
        torch.manual_seed(0)
        text_config.moe_force_xla_gmm = True
        text_config.fsdp_is_ep_bound = False
        tester = CausalLMTester()
        result = tester.run(
            module_name="minimax_m3_vl_text",
            hf_class=HF_TEXT_CLASS,
            task=ed.TaskType.CAUSAL_LM,
            config=text_config,
            small_model_config=m3_small_config,
        )
        assert result.success, f"MiniMaxM3VL text CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_vision_language(self, vlm_model_config, m3_small_config, vlm_config):
        """Logits parity vs HF for the composite VLM with image inputs."""
        # Set on the composite: ``add_basic_configurations`` re-reads basics
        # into the sub-configs from the parent, so text-config-level flags
        # would be overwritten during ``setup_config``.
        vlm_model_config.moe_force_xla_gmm = True
        vlm_model_config.fsdp_is_ep_bound = False
        local_cfg = dict(m3_small_config)
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(local_cfg.get("sequence_length", 128), tokens_per_image + 32)

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="minimax_m3_vl",
            hf_class=HF_VLM_CLASS,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=vlm_model_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"MiniMaxM3VL VLM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, full_attention_text_config, m3_small_config):
        """Cached generation smoke for the full-attention text configuration."""
        full_attention_text_config.moe_force_xla_gmm = True
        full_attention_text_config.fsdp_is_ep_bound = False
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="minimax_m3_vl_text",
            hf_class=HF_TEXT_CLASS,
            config=full_attention_text_config,
            small_model_config=m3_small_config,
            max_new_tokens=16,
        )
        assert result.success, f"MiniMaxM3VL generation failed: {result.error_message}"

    def test_router_bias_and_norm_weights_live(self, text_config, m3_small_config):
        """Parity with non-zero ``e_score_correction_bias`` and norm weights.

        HF initializes the router's selection bias and every Gemma-style
        RMSNorm weight to zeros, so the plain parity test cannot distinguish a
        live-threaded bias from a dropped one (nor a norm that ignores its
        loaded weight). Randomize them on the HF reference before the weight
        transfer and re-check logits parity.
        """
        import torch

        from tests.modules.test_utils.comparators import compare_outputs
        from tests.modules.test_utils.input_generators import make_text_inputs
        from tests.modules.test_utils.model_factory import create_ed_model, create_hf_model, setup_config

        text_config.moe_force_xla_gmm = True
        text_config.fsdp_is_ep_bound = False
        config = setup_config(text_config, m3_small_config)

        hf_model = create_hf_model(HF_TEXT_CLASS, config)
        mutated_bias = 0
        mutated_norms = 0
        with torch.no_grad():
            for name, buf in hf_model.named_buffers():
                if name.endswith("e_score_correction_bias"):
                    buf.copy_(torch.randn_like(buf) * 0.5)
                    mutated_bias += 1
            for name, param in hf_model.named_parameters():
                if name.endswith("norm.weight") or "layernorm" in name:
                    param.copy_(torch.randn_like(param) * 0.2)
                    mutated_norms += 1
        assert mutated_bias > 0 and mutated_norms > 0

        with config.mesh:
            ed_model = create_ed_model(
                module_name="minimax_m3_vl_text",
                task=ed.TaskType.CAUSAL_LM,
                config=config,
                small_model_config=m3_small_config,
                hf_model=hf_model,
            ).shard_model()
            inputs = make_text_inputs(
                vocab_size=m3_small_config["vocab_size"],
                batch_size=m3_small_config["batch_size"],
                seq_len=m3_small_config["sequence_length"],
            )
            with torch.no_grad():
                hf_logits = hf_model(**inputs["torch"]).logits.cpu().numpy()
            ed_logits = ed_model(**inputs["jax"]).logits

        comparison = compare_outputs(
            name="minimax_m3_vl_text[live-bias]",
            hf_output=hf_logits,
            ed_output=ed_logits,
        )
        assert comparison.success, f"live-bias parity failed: {comparison.details}"

    def test_sparse_config_rejects_cache(self, text_config, m3_small_config):
        """Sparse-attention configs must refuse to build a decode cache."""
        from tests.modules.test_utils.model_factory import create_ed_model, setup_config

        text_config.moe_force_xla_gmm = True
        text_config.fsdp_is_ep_bound = False
        config = setup_config(text_config, m3_small_config)
        with config.mesh:
            model = create_ed_model(
                module_name="minimax_m3_vl_text",
                task=ed.TaskType.CAUSAL_LM,
                config=config,
                small_model_config=m3_small_config,
            )
            with pytest.raises(NotImplementedError, match="minimax_m3_sparse"):
                model.init_cache(batch_size=1, max_length=32)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
