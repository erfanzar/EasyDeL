"""Tests for Falcon model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestFalcon:
    """Test suite for Falcon model."""

    @pytest.fixture
    def falcon_config(self, small_model_config):
        """Create Falcon-specific config."""
        return ed.FalconConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_kv_heads=small_model_config["num_key_value_heads"],
            ffn_hidden_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            new_decoder_architecture=True,
            num_ln_in_parallel_attn=2,
            parallel_attn=True,
        )

    def test_causal_lm(self, falcon_config, small_model_config):
        """Test FalconForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="falcon",
            hf_class=transformers.FalconForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=falcon_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Falcon CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, falcon_config, small_model_config):
        """Test Falcon text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="falcon",
            hf_class=transformers.FalconForCausalLM,
            config=falcon_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Falcon generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
