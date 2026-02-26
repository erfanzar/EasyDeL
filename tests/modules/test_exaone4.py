"""Tests for EXAONE4 model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestEXAONE4:
    """Test suite for EXAONE4 model."""

    @pytest.fixture
    def exaone4_config(self, small_model_config):
        """Create EXAONE4-specific config."""
        return ed.Exaone4Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, exaone4_config, small_model_config):
        """Test Exaone4ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="exaone4",
            hf_class=transformers.Exaone4ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=exaone4_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"EXAONE4 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, exaone4_config, small_model_config):
        """Test EXAONE4 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="exaone4",
            hf_class=transformers.Exaone4ForCausalLM,
            config=exaone4_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"EXAONE4 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
