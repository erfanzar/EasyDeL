"""Tests for GPT-J model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestGPTJ:
    """Test suite for GPT-J model."""

    @pytest.fixture
    def gptj_config(self, small_model_config):
        """Create GPT-J-specific config."""
        return ed.GPTJConfig(
            vocab_size=small_model_config["vocab_size"],
            n_positions=small_model_config["max_position_embeddings"],
            n_embd=small_model_config["hidden_size"],
            n_layer=small_model_config["num_hidden_layers"],
            n_head=small_model_config["num_attention_heads"],
            rotary_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
        )

    def test_causal_lm(self, gptj_config, small_model_config):
        """Test GPTJForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gptj",
            hf_class=transformers.GPTJForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=gptj_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GPTJ CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, gptj_config, small_model_config):
        """Test GPT-J text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gptj",
            hf_class=transformers.GPTJForCausalLM,
            config=gptj_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"GPTJ generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
