"""Tests for GPT-OSS MoE model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestGptOss:
    """Test suite for GPT-OSS MoE model."""

    @pytest.fixture
    def gpt_oss_config(self, small_model_config):
        """Create GPT-OSS-specific config."""
        return ed.GptOssConfig(
            num_hidden_layers=8,
            num_local_experts=8,
            vocab_size=201088,
            hidden_size=128,
            intermediate_size=256,
            head_dim=64,
            num_attention_heads=8,
            num_key_value_heads=4,
            sliding_window=128,
            rope_theta=150000.0,
            tie_word_embeddings=False,
            hidden_act="silu",
            initializer_range=0.02,
            max_position_embeddings=2048,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            attention_dropout=0.0,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.9,
            output_router_logits=False,
            use_cache=True,
            layer_types=None,
        )

    def test_causal_lm(self, gpt_oss_config, small_model_config):
        """Test GptOssForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="gpt_oss",
            hf_class=transformers.GptOssForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=gpt_oss_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GPT-OSS CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, gpt_oss_config, small_model_config):
        """Test GPT-OSS text generation."""
        tester = CausalLMTester()
        gpt_oss_config.moe_force_xla_gmm = True
        result = tester.test_generation(
            module_name="gpt_oss",
            hf_class=transformers.GptOssForCausalLM,
            config=gpt_oss_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"GPT-OSS generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
