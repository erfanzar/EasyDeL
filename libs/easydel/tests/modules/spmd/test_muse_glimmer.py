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

"""HuggingFace-comparison tests for the Muse-Glimmer vision-language model.

``muse_glimmer`` only exists in ``transformers`` releases from the one that
introduced the architecture onward, so this module skips wholesale on older
pins and has therefore **not been executed** yet. The executable coverage for
this family lives in ``tests/modules/test_muse_glimmer_parity.py``, which
compares both towers and the checkpoint-conversion path against a NumPy
transcription of the reference implementation.

Only the text path is compared here: Muse-Glimmer's tower consumes pre-packed
patches plus a ``grid_thw`` table rather than the ``(batch, channels, height,
width)`` pixel tensor the shared vision testers generate, so a vision
comparison needs input-generator support that should be written against a
working HF class rather than guessed at.
"""

import easydel as ed
import pytest
import transformers

try:
    from tests.modules.test_utils import CausalLMTester
except ImportError:
    from tests.modules.test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]

pytestmark = pytest.mark.skipif(
    not hasattr(transformers, "MuseGlimmerForConditionalGeneration"),
    reason="installed transformers does not ship the muse_glimmer architecture",
)


class TestMuseGlimmer:
    """Test suite for the Muse-Glimmer vision-language model."""

    @pytest.fixture
    def muse_glimmer_config(self, small_model_config):
        """Create a small Muse-Glimmer VLM config."""
        vision_config = ed.MuseGlimmerVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            patch_size=14,
            patch_temporal=2,
            merge_size=2,
            pos_emb_height=8,
            pos_emb_width=8,
        )
        text_config = ed.MuseGlimmerTextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            head_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
            max_position_embeddings=max(small_model_config["max_position_embeddings"], 2048),
            sliding_window=64,
        )
        return ed.MuseGlimmerConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=small_model_config["vocab_size"] - 1,
            video_token_id=small_model_config["vocab_size"] - 2,
            out_hidden_size=vision_config.out_hidden_size,
            projector_hidden_size=128,
            tie_word_embeddings=False,
        )

    def test_text_only(self, muse_glimmer_config, small_model_config):
        """Test the Muse-Glimmer text-only forward pass against HF."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.run(
            module_name="muse_glimmer",
            hf_class=transformers.MuseGlimmerForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=muse_glimmer_config,
            small_model_config=local_cfg,
        )
        assert result.success, f"Muse-Glimmer text-only failed: {result.error_message or result.comparison.details}"

    def test_generation(self, muse_glimmer_config, small_model_config):
        """Test Muse-Glimmer text-only generation."""
        local_cfg = small_model_config.copy()
        local_cfg["max_position_embeddings"] = 2048

        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="muse_glimmer",
            hf_class=transformers.MuseGlimmerForConditionalGeneration,
            task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
            config=muse_glimmer_config,
            small_model_config=local_cfg,
            max_new_tokens=16,
        )
        assert result.success, f"Muse-Glimmer generation failed: {result.error_message}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
