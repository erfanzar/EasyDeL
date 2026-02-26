"""Tests for SmolLM3 model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestSmolLM3:
    """Test suite for SmolLM3 model."""

    @pytest.fixture
    def smollm3_config(self, small_model_config):
        """Create SmolLM3-specific config."""
        return ed.SmolLM3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, smollm3_config, small_model_config):
        """Test SmolLM3ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="smollm3",
            hf_class=transformers.SmolLM3ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=smollm3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"SmolLM3 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, smollm3_config, small_model_config):
        """Test SmolLM3 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="smollm3",
            hf_class=transformers.SmolLM3ForCausalLM,
            config=smollm3_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"SmolLM3 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
