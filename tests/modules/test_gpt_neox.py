"""Tests for GPT-NeoX model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestGPTNeoX:
    """Test suite for GPT-NeoX model."""

    @pytest.fixture
    def gpt_neox_config(self, small_model_config):
        """Create GPT-NeoX-specific config."""
        return ed.GPTNeoXConfig(
            vocab_size=small_model_config["vocab_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            rotary_pct=1,
            rope_scaling=None,
        )

    def test_causal_lm(self, gpt_neox_config, small_model_config):
        """Test GPTNeoXForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gpt_neox",
            hf_class=transformers.GPTNeoXForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=gpt_neox_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GPTNeoX CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, gpt_neox_config, small_model_config):
        """Test GPT-NeoX text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="gpt_neox",
            hf_class=transformers.GPTNeoXForCausalLM,
            config=gpt_neox_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"GPTNeoX generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
