"""Tests for GLM4-MoE model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestGLM4Moe:
    """Test suite for GLM4-MoE model."""

    @pytest.fixture
    def glm4_moe_config(self, small_model_config):
        """Create GLM4-MoE-specific config."""
        return ed.Glm4MoeConfig(
            vocab_size=151552,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            partial_rotary_factor=0.5,
            num_key_value_heads=4,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            moe_intermediate_size=1408,
            num_experts_per_tok=4,
            n_shared_experts=4,
            n_routed_experts=4,
            routed_scaling_factor=1.0,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            norm_topk_prob=True,
            use_qk_norm=False,
        )

    def test_causal_lm(self, glm4_moe_config, small_model_config):
        """Test Glm4MoeForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="glm4_moe",
            hf_class=transformers.Glm4MoeForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=glm4_moe_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GLM4-MoE CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, glm4_moe_config, small_model_config):
        """Test GLM4-MoE text generation."""
        tester = CausalLMTester()
        glm4_moe_config.moe_force_xla_gmm = True
        result = tester.test_generation(
            module_name="glm4_moe",
            hf_class=transformers.Glm4MoeForCausalLM,
            config=glm4_moe_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"GLM4-MoE generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
