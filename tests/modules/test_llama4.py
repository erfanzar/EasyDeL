"""Tests for LLaMA4 text model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestLlama4:
    """Test suite for LLaMA4 text model."""

    @pytest.fixture
    def llama4_config(self, small_model_config):
        """Create LLaMA4 text-specific config."""
        return ed.Llama4TextConfig(
            hidden_size=128,
            intermediate_size=512,
            intermediate_size_mlp=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            use_qk_norm=False,
        )

    def test_causal_lm(self, llama4_config, small_model_config):
        """Test Llama4ForCausalLM."""
        tester = CausalLMTester()
        llama4_config.moe_force_xla_gmm = True
        result = tester.run(
            module_name="llama4_text",
            hf_class=transformers.Llama4ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=llama4_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Llama4 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, llama4_config, small_model_config):
        """Test LLaMA4 text generation."""
        tester = CausalLMTester()
        llama4_config.moe_force_xla_gmm = True
        result = tester.test_generation(
            module_name="llama4_text",
            hf_class=transformers.Llama4ForCausalLM,
            config=llama4_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Llama4 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
