"""Tests for GPT-2 model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestGPT2:
    """Test suite for GPT-2 model."""

    @pytest.fixture
    def gpt2_config(self, small_model_config):
        """Create GPT-2-specific config."""
        return ed.GPT2Config(
            vocab_size=small_model_config["vocab_size"],
            n_embd=small_model_config["hidden_size"],
            n_layer=small_model_config["num_hidden_layers"],
            n_head=small_model_config["num_attention_heads"],
            n_positions=small_model_config["max_position_embeddings"],
            n_inner=small_model_config["intermediate_size"],
        )

    def test_causal_lm(self, gpt2_config, small_model_config):
        """Test GPT2LMHeadModel."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gpt2",
            hf_class=transformers.GPT2LMHeadModel,
            task=ed.TaskType.CAUSAL_LM,
            config=gpt2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GPT2 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, gpt2_config, small_model_config):
        """Test GPT-2 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gpt2",
            hf_class=transformers.GPT2LMHeadModel,
            config=gpt2_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"GPT2 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
