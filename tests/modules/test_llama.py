"""Tests for LLaMA model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestLlama:
    """Test suite for LLaMA model variants."""

    @pytest.fixture
    def llama_config(self, small_model_config):
        """Create LLaMA-specific config with rope scaling."""
        config_dict = small_model_config.copy()
        config_dict["rope_scaling"] = {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
        return ed.LlamaConfig(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            intermediate_size=config_dict["intermediate_size"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            rope_scaling=config_dict["rope_scaling"],
        )

    def test_causal_lm(self, llama_config, small_model_config):
        """Test LlamaForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="llama",
            hf_class=transformers.LlamaForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=llama_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Llama CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, llama_config, small_model_config):
        """Test LLaMA text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="llama",
            hf_class=transformers.LlamaForCausalLM,
            config=llama_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Llama generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
