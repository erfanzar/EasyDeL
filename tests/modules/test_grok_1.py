"""Tests for Grok-1 model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestGrok1:
    """Test suite for Grok-1 model."""

    @pytest.fixture
    def grok1_config(self, small_model_config):
        """Create Grok-1-specific config."""
        return ed.Grok1Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_experts=small_model_config.get("num_experts", 8),
            num_experts_per_tok=small_model_config.get("num_experts_per_tok", 2),
        )

    def test_causal_lm(self, grok1_config, small_model_config):
        """Test Grok1ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="grok-1",  # Note: uses hyphen, not underscore
            hf_class=None,  # Grok-1 has no HF equivalent
            task=ed.TaskType.CAUSAL_LM,
            config=grok1_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Grok-1 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, grok1_config, small_model_config):
        """Test Grok-1 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="grok-1",
            hf_class=None,
            task=ed.TaskType.CAUSAL_LM,
            config=grok1_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Grok-1 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
