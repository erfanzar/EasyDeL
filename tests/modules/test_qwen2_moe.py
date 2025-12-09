"""Tests for Qwen2-MoE model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestQwen2Moe:
    """Test suite for Qwen2-MoE model."""

    @pytest.fixture
    def qwen2_moe_config(self, small_model_config):
        """Create Qwen2-MoE-specific config."""
        return ed.Qwen2MoeConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_experts=small_model_config.get("num_experts", 8),
            num_experts_per_tok=small_model_config.get("num_experts_per_tok", 2),
            head_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
        )

    def test_causal_lm(self, qwen2_moe_config, small_model_config):
        """Test Qwen2MoeForCausalLM."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        qwen2_moe_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen2_moe",
            hf_class=transformers.Qwen2MoeForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=qwen2_moe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen2Moe CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, qwen2_moe_config, small_model_config):
        """Test Qwen2-MoE text generation."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        qwen2_moe_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen2_moe",
            hf_class=transformers.Qwen2MoeForCausalLM,
            config=qwen2_moe_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Qwen2Moe generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
