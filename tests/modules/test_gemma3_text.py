"""Tests for Gemma3 text-only model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


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


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
