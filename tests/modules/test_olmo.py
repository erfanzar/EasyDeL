"""Tests for OLMo model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestOLMo:
    """Test suite for OLMo model."""

    @pytest.fixture
    def olmo_config(self, small_model_config):
        """Create OLMo-specific config."""
        return ed.OlmoConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
        )

    def test_causal_lm(self, olmo_config, small_model_config):
        """Test OlmoForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="olmo",
            hf_class=transformers.OlmoForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=olmo_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"OLMo CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, olmo_config, small_model_config):
        """Test OLMo text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="olmo",
            hf_class=transformers.OlmoForCausalLM,
            config=olmo_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"OLMo generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
