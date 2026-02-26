"""Tests for OLMo2 model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestOLMo2:
    """Test suite for OLMo2 model."""

    @pytest.fixture
    def olmo2_config(self, small_model_config):
        """Create OLMo2-specific config."""
        return ed.Olmo2Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
        )

    def test_causal_lm(self, olmo2_config, small_model_config):
        """Test Olmo2ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="olmo2",
            hf_class=transformers.Olmo2ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=olmo2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"OLMo2 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, olmo2_config, small_model_config):
        """Test OLMo2 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="olmo2",
            hf_class=transformers.Olmo2ForCausalLM,
            config=olmo2_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"OLMo2 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
