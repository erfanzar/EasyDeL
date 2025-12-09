"""Tests for Mamba2 model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestMamba2:
    """Test suite for Mamba2 model."""

    @pytest.fixture
    def mamba2_config(self, small_model_config):
        """Create Mamba2-specific config."""
        return ed.Mamba2Config(
            hidden_size=256,
            num_hidden_layers=16,
            num_heads=8,
        )

    def test_causal_lm(self, mamba2_config, small_model_config):
        """Test Mamba2ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="mamba2",
            hf_class=transformers.Mamba2ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=mamba2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Mamba2 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, mamba2_config, small_model_config):
        """Test Mamba2 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mamba2",
            hf_class=transformers.Mamba2ForCausalLM,
            config=mamba2_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Mamba2 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
