"""Tests for Xerxes model."""

import pytest

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestXerxes:
    """Test suite for Xerxes model."""

    @pytest.fixture
    def xerxes_config(self, small_model_config):
        """Create Xerxes-specific config."""
        return ed.XerxesConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, xerxes_config, small_model_config):
        """Test XerxesForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="xerxes",
            hf_class=None,  # Custom model
            task=ed.TaskType.CAUSAL_LM,
            config=xerxes_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Xerxes CAUSAL_LM failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
