"""Tests for StableLM model."""

import copy

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestStableLM:
    """Test suite for StableLM model."""

    @pytest.fixture
    def stablelm_config(self, small_model_config):
        """Create StableLM-specific config."""
        config_dict = copy.copy(small_model_config)
        # qk_layernorm=True causes HF transformers bug where _init_weights tries to access
        # LayerNorm.bias.data when bias is None. Setting to False works around this.
        config_dict.update({"attention_bias": True, "qk_layernorm": False})
        return ed.StableLmConfig(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            intermediate_size=config_dict["intermediate_size"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            use_qk_layernorm=config_dict["qk_layernorm"],
        )

    def test_causal_lm(self, stablelm_config, small_model_config):
        """Test StableLmForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="stablelm",
            hf_class=transformers.StableLmForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=stablelm_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"StableLM CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, stablelm_config, small_model_config):
        """Test StableLM text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="stablelm",
            hf_class=transformers.StableLmForCausalLM,
            config=stablelm_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"StableLM generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
