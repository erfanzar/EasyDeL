"""Tests for Mixtral MoE model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestMixtral:
    """Test suite for Mixtral MoE model."""

    @pytest.fixture
    def mixtral_config(self, small_model_config):
        """Create Mixtral-specific config."""
        return ed.MixtralConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_local_experts=small_model_config.get("num_local_experts", 8),
            num_experts_per_tok=small_model_config.get("num_experts_per_tok", 2),
        )

    def test_causal_lm(self, mixtral_config, small_model_config):
        """Test MixtralForCausalLM."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        mixtral_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.run(
            module_name="mixtral",
            hf_class=transformers.MixtralForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=mixtral_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Mixtral CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, mixtral_config, small_model_config):
        """Test Mixtral text generation."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        mixtral_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mixtral",
            hf_class=transformers.MixtralForCausalLM,
            config=mixtral_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Mixtral generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
