"""Tests for Mamba model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestMamba:
    """Test suite for Mamba model."""

    @pytest.fixture
    def mamba_config(self, small_model_config):
        """Create Mamba-specific config."""
        return ed.MambaConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            intermediate_size=small_model_config["intermediate_size"],
        )

    def test_causal_lm(self, mamba_config, small_model_config):
        """Test MambaForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="mamba",
            hf_class=transformers.MambaForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=mamba_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Mamba CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, mamba_config, small_model_config):
        """Test Mamba text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mamba",
            hf_class=transformers.MambaForCausalLM,
            config=mamba_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Mamba generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
