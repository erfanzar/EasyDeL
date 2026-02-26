"""Tests for MiniMax Text V1 model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed
from easydel.modules.minimax import MiniMaxConfig

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestMiniMaxTextV1:
    """Test suite for MiniMax Text V1 model."""

    @pytest.fixture
    def minimax_text_v1_config(self, small_model_config):
        """Create MiniMax Text V1-specific config."""
        return MiniMaxConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, minimax_text_v1_config, small_model_config):
        """Test MiniMaxForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="minimax",
            hf_class=transformers.MiniMaxForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=minimax_text_v1_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"MiniMax Text V1 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, minimax_text_v1_config, small_model_config):
        """Test MiniMax Text V1 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="minimax",
            hf_class=transformers.MiniMaxForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=minimax_text_v1_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"MiniMax Text V1 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
