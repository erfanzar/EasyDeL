"""Tests for EXAONE model."""

import pytest

import easydel as ed

try:
    from .test_utils import CausalLMTester, get_hf_model_from_hub
except ImportError:
    from test_utils import CausalLMTester, get_hf_model_from_hub


class TestEXAONE:
    """Test suite for EXAONE model."""

    @pytest.fixture
    def exaone_config(self, small_model_config):
        """Create EXAONE-specific config."""
        return ed.ExaoneConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    @pytest.fixture
    def hf_exaone_class(self, small_model_config):
        """Load exaone HF class from hub."""
        hf_class, _ = get_hf_model_from_hub("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", small_model_config)
        return hf_class

    def test_causal_lm(self, exaone_config, hf_exaone_class, small_model_config):
        """Test ExaoneForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="exaone",
            hf_class=hf_exaone_class,
            task=ed.TaskType.CAUSAL_LM,
            config=exaone_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"EXAONE CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, exaone_config, hf_exaone_class, small_model_config):
        """Test EXAONE text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="exaone",
            hf_class=hf_exaone_class,
            task=ed.TaskType.CAUSAL_LM,
            config=exaone_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"EXAONE generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
