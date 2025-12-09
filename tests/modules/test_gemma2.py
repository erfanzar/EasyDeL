"""Tests for Gemma2 model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestGemma2:
    """Test suite for Gemma2 model."""

    @pytest.fixture
    def gemma2_config(self, small_model_config):
        """Create Gemma2-specific config."""
        return ed.Gemma2Config(
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128 // 8,
            use_scan_mlp=False,
        )

    def test_causal_lm(self, gemma2_config, small_model_config):
        """Test Gemma2ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gemma2",
            hf_class=transformers.Gemma2ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=gemma2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Gemma2 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, gemma2_config, small_model_config):
        """Test Gemma2 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gemma2",
            hf_class=transformers.Gemma2ForCausalLM,
            config=gemma2_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Gemma2 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
