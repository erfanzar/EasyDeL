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

"""Tests for Gemma3 text-only model."""

import pytest
import spectrax as spx
import transformers
from jax import numpy as jnp

import easydel as ed

try:
    from tests.modules.test_utils import CausalLMTester
except ImportError:
    from tests.modules.test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestGemma3Text:
    """Test suite for Gemma3 text-only model."""

    @pytest.fixture
    def gemma3_text_config(self, small_model_config):
        """Create Gemma3 text config."""
        return ed.Gemma3TextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=128,
        )

    def test_causal_lm(self, gemma3_text_config, small_model_config):
        """Test Gemma3ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gemma3_text",
            hf_class=transformers.Gemma3ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=gemma3_text_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Gemma3 text CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, gemma3_text_config, small_model_config):
        """Test Gemma3 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gemma3_text",
            hf_class=transformers.Gemma3ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=gemma3_text_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Gemma3 text generation failed: {result.error_message}"

    def test_compute_lm_logits_applies_final_softcapping(self, gemma3_text_config):
        """Chunked loss should use the same final-logit transform as forward."""
        gemma3_text_config.final_logit_softcapping = 7.5
        model = ed.Gemma3ForCausalLM(
            config=gemma3_text_config,
            rngs=spx.Rngs(0),
        )
        hidden_states = jnp.ones((1, 2, gemma3_text_config.hidden_size), dtype=jnp.float32)

        actual = model.compute_lm_logits(hidden_states)
        raw = model.apply_lm_head(hidden_states)
        cap = jnp.array(gemma3_text_config.final_logit_softcapping, dtype=raw.dtype)
        expected = cap * jnp.tanh(raw / cap)

        assert jnp.allclose(actual, expected, atol=1e-5)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
