"""Tests for Cohere model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


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


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
