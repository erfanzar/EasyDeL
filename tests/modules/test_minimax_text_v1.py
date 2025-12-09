"""Tests for MiniMax Text V1 model."""

import pytest

import easydel as ed
from easydel.modules.minimax_text_v1 import MiniMaxText01Config

try:
    from .test_utils import CausalLMTester, get_hf_model_from_hub
except ImportError:
    from test_utils import CausalLMTester, get_hf_model_from_hub


class TestMiniMaxTextV1:
    """Test suite for MiniMax Text V1 model."""

    @pytest.fixture
    def hf_minimax_class(self, small_model_config):
        """Load MiniMax Text V1 HF class from hub."""
        hf_class, _ = get_hf_model_from_hub(
            "MiniMaxAI/MiniMax-Text-01",
            small_model_config,
        )
        return hf_class

    @pytest.fixture
    def minimax_text_v1_config(self, small_model_config):
        """Create MiniMax Text V1-specific config."""
        return MiniMaxText01Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, minimax_text_v1_config, small_model_config, hf_minimax_class):
        """Test MiniMaxText01ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="minimax_text_01",
            hf_class=hf_minimax_class,
            task=ed.TaskType.CAUSAL_LM,
            config=minimax_text_v1_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"MiniMax Text V1 CAUSAL_LM failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
