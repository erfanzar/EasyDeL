"""Tests for Seed OSS model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestSeedOSS:
    """Test suite for Seed OSS model."""

    @pytest.fixture
    def seed_oss_config(self, small_model_config):
        """Create Seed OSS-specific config."""
        return ed.SeedOssConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, seed_oss_config, small_model_config):
        """Test SeedOSSForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="seed_oss",
            hf_class=transformers.SeedOssForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=seed_oss_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Seed OSS CAUSAL_LM failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
