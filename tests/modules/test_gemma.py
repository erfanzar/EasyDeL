"""Tests for Gemma model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestGemma:
    """Test suite for Gemma model."""

    @pytest.fixture
    def gemma_config(self, small_model_config):
        """Create Gemma-specific config."""
        return ed.GemmaConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            tie_word_embeddings=True,
        )

    def test_causal_lm(self, gemma_config, small_model_config):
        """Test GemmaForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gemma",
            hf_class=transformers.GemmaForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=gemma_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Gemma CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, gemma_config, small_model_config):
        """Test Gemma text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gemma",
            hf_class=transformers.GemmaForCausalLM,
            config=gemma_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Gemma generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
