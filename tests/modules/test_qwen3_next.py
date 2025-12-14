"""Tests for Qwen3Next model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestQwen3Next:
    """Test suite for Qwen3Next model."""

    @pytest.fixture
    def qwen3_next_config(self, small_model_config):
        """Create Qwen3Next-specific config."""
        return ed.Qwen3NextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, qwen3_next_config, small_model_config):
        """Test Qwen3NextForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen3_next",
            hf_class=transformers.Qwen3NextForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=qwen3_next_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3Next CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, qwen3_next_config, small_model_config):
        """Test Qwen3Next text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen3_next",
            hf_class=transformers.Qwen3NextForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=qwen3_next_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Qwen3Next generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
