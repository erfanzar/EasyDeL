"""Tests for Cohere2 model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestCohere2:
    """Test suite for Cohere2 model."""

    @pytest.fixture
    def cohere2_config(self, small_model_config):
        """Create Cohere2-specific config."""
        return ed.Cohere2Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, cohere2_config, small_model_config):
        """Test Cohere2ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="cohere2",
            hf_class=transformers.Cohere2ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=cohere2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Cohere2 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, cohere2_config, small_model_config):
        """Test Cohere2 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="cohere2",
            hf_class=transformers.Cohere2ForCausalLM,
            config=cohere2_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Cohere2 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
