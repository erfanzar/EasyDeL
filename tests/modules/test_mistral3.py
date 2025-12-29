"""Tests for Mistral3 model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


# Check if Mistral3ForConditionalGeneration is available in transformers
HAS_MISTRAL3 = hasattr(transformers, "Mistral3ForConditionalGeneration")


class TestMistral3:
    """Test suite for Mistral3 model."""

    @pytest.fixture
    def mistral3_config(self, small_model_config):
        """Create Mistral3-specific config."""
        return ed.Mistral3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    @pytest.mark.skipif(not HAS_MISTRAL3, reason="transformers.Mistral3ForConditionalGeneration not available")
    def test_causal_lm(self, mistral3_config, small_model_config):
        """Test Mistral3ForConditionalGeneration."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="mistral3",
            hf_class=transformers.Mistral3ForConditionalGeneration,
            task=ed.TaskType.CAUSAL_LM,
            config=mistral3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Mistral3 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    @pytest.mark.skipif(not HAS_MISTRAL3, reason="transformers.Mistral3ForConditionalGeneration not available")
    def test_generation(self, mistral3_config, small_model_config):
        """Test Mistral3 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mistral3",
            hf_class=transformers.Mistral3ForConditionalGeneration,
            config=mistral3_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Mistral3 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
