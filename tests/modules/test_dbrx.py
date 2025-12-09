"""Tests for DBRX MoE model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestDBRX:
    """Test suite for DBRX MoE model."""

    @pytest.fixture
    def dbrx_config(self, small_model_config):
        """Create DBRX-specific config."""
        return ed.DbrxConfig(
            d_model=small_model_config["hidden_size"],
            n_heads=small_model_config["num_attention_heads"],
            n_layers=small_model_config["num_hidden_layers"],
            ffn_config=ed.DbrxFFNConfig(
                ffn_hidden_size=small_model_config["intermediate_size"],
                moe_top_k=small_model_config.get("num_experts_per_tok", 4),
                moe_num_experts=small_model_config.get("num_local_experts", 16),
            ),
            attn_config=ed.DbrxAttentionConfig(),
            max_seq_len=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, dbrx_config, small_model_config):
        """Test DbrxForCausalLM."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        dbrx_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.run(
            module_name="dbrx",
            hf_class=transformers.DbrxForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=dbrx_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"DBRX CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, dbrx_config, small_model_config):
        """Test DBRX text generation."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        dbrx_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="dbrx",
            hf_class=transformers.DbrxForCausalLM,
            config=dbrx_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"DBRX generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
