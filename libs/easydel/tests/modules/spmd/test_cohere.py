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

"""Tests for Cohere model."""

import pytest
import spectrax as spx
import transformers
from jax import numpy as jnp

import easydel as ed

try:
    from tests.modules.test_utils import CausalLMTester
except ImportError:
    from tests.modules.test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestCohere:
    """Test suite for Cohere model."""

    @pytest.fixture
    def cohere_config(self, small_model_config):
        """Create Cohere-specific config."""
        return ed.CohereConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, cohere_config, small_model_config):
        """Test CohereForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="cohere",
            hf_class=transformers.CohereForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=cohere_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Cohere CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, cohere_config, small_model_config):
        """Test Cohere text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="cohere",
            hf_class=transformers.CohereForCausalLM,
            config=cohere_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Cohere generation failed: {result.error_message}"

    def test_compute_lm_logits_applies_logit_scale(self, cohere_config):
        """Chunked loss should use Cohere's scaled logits path."""
        cohere_config.logit_scale = 0.5
        model = ed.CohereForCausalLM(
            config=cohere_config,
            rngs=spx.Rngs(0),
        )
        hidden_states = jnp.ones((1, 2, cohere_config.hidden_size), dtype=jnp.float32)

        actual = model.compute_lm_logits(hidden_states)
        expected = model.apply_lm_head(hidden_states) * cohere_config.logit_scale

        assert jnp.allclose(actual, expected, atol=1e-5)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
