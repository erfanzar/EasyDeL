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

"""Tests for PaliGemma vision-language model."""

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


class TestPaliGemma:
    """Test suite for PaliGemma vision-language model."""

    @pytest.fixture
    def paligemma_config(self, small_model_config):
        """Create PaliGemma VLM config (SigLIP tower + Gemma text + linear projector)."""
        vision_config = {
            "model_type": "siglip_vision_model",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": 224,
            "patch_size": 14,
            "vision_use_head": False,
        }
        text_config = ed.GemmaConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=2048,
            head_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
            tie_word_embeddings=True,
        )
        # Keep the image token inside the small test vocab to avoid embedding index errors
        image_token_index = small_model_config["vocab_size"] - 1
        return ed.PaliGemmaConfig(
            vision_config=vision_config,
            text_config=text_config,
            image_token_index=image_token_index,
            vocab_size=small_model_config["vocab_size"],
            projection_dim=small_model_config["hidden_size"],
            hidden_size=small_model_config["hidden_size"],
        )

    @pytest.fixture
    def vlm_config(self, paligemma_config, small_model_config):
        """Create VLM-specific config for PaliGemma."""
        batch_size = small_model_config["batch_size"]
        num_images = 1
        image_size = paligemma_config.vision_config.image_size
        patch_size = paligemma_config.vision_config.patch_size
        # SigLIP has no CLS token; PaliGemma uses every patch token.
        num_patches = (image_size // patch_size) ** 2

        return {
            "image_token_id": paligemma_config.image_token_id,
            "num_image_tokens": num_patches,
            "pixel_values_shape": (batch_size * num_images, 3, image_size, image_size),
            "num_images": num_images,
            # Exercises the PaliGemma bidirectional-prefix/causal-suffix mask path
            # (token_type_ids == 0 tokens form one bidirectional block in HF).
            "use_token_type_ids": True,
        }

    def test_vision_language(self, paligemma_config, small_model_config, vlm_config):
        """Test PaliGemmaForConditionalGeneration with vision inputs and token_type_ids."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048
        tokens_per_image = vlm_config["num_image_tokens"] + 2
        local_cfg["sequence_length"] = max(
            local_cfg.get("sequence_length", 128),
            tokens_per_image + 32,
        )

        tester = VisionLanguageTester()
        result = tester.run(
            module_name="paligemma",
            hf_class=transformers.PaliGemmaForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=paligemma_config,
            small_model_config=local_cfg,
            vlm_config=vlm_config,
        )
        assert result.success, f"PaliGemma VLM failed: {result.error_message or result.comparison.details}"

    def test_text_only(self, paligemma_config, small_model_config):
        """Test PaliGemma text-only forward pass (causal + 1-indexed positions)."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.run(
            module_name="paligemma",
            hf_class=transformers.PaliGemmaForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=paligemma_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"PaliGemma text-only failed: {result.error_message or result.comparison.details}"

    def test_generation(self, paligemma_config, small_model_config):
        """Test PaliGemma text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="paligemma",
            hf_class=transformers.PaliGemmaForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=paligemma_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"PaliGemma generation failed: {result.error_message}"

    def test_prefix_bidirectional_suffix_causal_mask(self):
        """Parity with the real PaliGemma token_type_ids convention.

        The PaliGemma processor emits ``token_type_ids == 0`` for image + text
        prefix tokens (one bidirectional attention block) and ``1`` for suffix
        tokens (strictly causal). This test checks, against the HF reference:

        1. the token-type mask path is live (logits change when it is applied),
        2. logits match HF both with and without ``token_type_ids``.
        """
        import copy

        import jax.numpy as jnp
        import numpy as np
        import torch

        try:
            from tests.modules.test_utils.model_factory import create_ed_model, setup_config
        except ImportError:
            from tests.modules.test_utils.model_factory import (  # pyright: ignore[reportImplicitRelativeImport]
                create_ed_model,
                setup_config,
            )

        text_config = ed.GemmaConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=512,
            head_dim=16,
            tie_word_embeddings=True,
        )
        config = ed.PaliGemmaConfig(
            vision_config={
                "model_type": "siglip_vision_model",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 28,
                "patch_size": 14,
                "vision_use_head": False,
            },
            text_config=text_config,
            image_token_index=999,
            vocab_size=1000,
            projection_dim=64,
            hidden_size=64,
        )
        hf_model = transformers.PaliGemmaForConditionalGeneration(copy.deepcopy(config)).eval()

        torch.manual_seed(0)
        batch_size, seq_len, num_image_tokens = 1, 16, 4  # (28 / 14) ** 2 image tokens
        input_ids = torch.randint(0, 998, (batch_size, seq_len))
        input_ids[0, 1 : 1 + num_image_tokens] = 999
        pixel_values = torch.randn(batch_size, 3, 28, 28) * 0.5
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        token_type_ids[0, 8:] = 1  # suffix

        with torch.no_grad():
            hf_with_tt = hf_model(
                input_ids=input_ids, pixel_values=pixel_values, token_type_ids=token_type_ids, use_cache=False
            ).logits
            hf_no_tt = hf_model(input_ids=input_ids, pixel_values=pixel_values, use_cache=False).logits

        small = {
            "dtype": jnp.float32,
            "precision": None,
            "sharding_axis_dims": (1, 1, 1, -1, 1, 1),
            "attn_mechanism": "vanilla",
            "blocksize_k": 64,
            "blocksize_q": 64,
            "attn_dtype": jnp.float32,
        }
        config = setup_config(config, {**small, "vocab_size": 1000})
        with config.mesh:
            ed_model = create_ed_model("paligemma", ed.TaskType.IMAGE_TEXT_TO_TEXT, config, small, hf_model=hf_model)
            jax_ids = jnp.asarray(input_ids.numpy(), dtype="i4")
            jax_pixels = jnp.asarray(pixel_values.numpy(), dtype="f4")
            jax_tt = jnp.asarray(token_type_ids.numpy(), dtype="i4")
            ed_with_tt = ed_model(input_ids=jax_ids, pixel_values=jax_pixels, token_type_ids=jax_tt).logits
            ed_no_tt = ed_model(input_ids=jax_ids, pixel_values=jax_pixels).logits

        # The bidirectional-prefix mask must actually change the logits...
        assert float(jnp.abs(ed_with_tt - ed_no_tt).max()) > 1e-3, "token_type_ids mask path is dead in EasyDeL"
        assert float((hf_with_tt - hf_no_tt).abs().max()) > 1e-3, "token_type_ids mask path is dead in HF"
        # ...and match HF in both regimes.
        max_err_tt = float(np.abs(hf_with_tt.numpy() - np.asarray(ed_with_tt)).max())
        max_err_causal = float(np.abs(hf_no_tt.numpy() - np.asarray(ed_no_tt)).max())
        assert max_err_tt < 2e-4, f"prefix-bidirectional parity failed: max|d|={max_err_tt}"
        assert max_err_causal < 2e-4, f"causal parity failed: max|d|={max_err_causal}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
