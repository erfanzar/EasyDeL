"""Tests for Arctic model."""

import pytest

import easydel as ed

try:
    from .test_utils import CausalLMTester, get_hf_model_from_hub
except ImportError:
    from test_utils import CausalLMTester, get_hf_model_from_hub


class TestArctic:
    """Test suite for Arctic model variants."""

    @pytest.fixture
    def arctic_config(self, small_model_config):
        """Create Arctic-specific config."""
        return ed.ArcticConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            # MoE settings
            num_local_experts=8,
            num_experts_per_tok=1,
            moe_layer_frequency=2,
        )

    @pytest.fixture
    def hf_arctic_class(self, small_model_config):
        """Load arctic HF class from hub."""
        hf_class, _ = get_hf_model_from_hub("Snowflake/snowflake-arctic-instruct", small_model_config)
        return hf_class

    def test_causal_lm(self, arctic_config, hf_arctic_class, small_model_config):
        """Test ArcticForCausalLM."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        arctic_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.run(
            module_name="arctic",
            hf_class=hf_arctic_class,
            task=ed.TaskType.CAUSAL_LM,
            config=arctic_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Arctic CAUSAL_LM failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
