"""Tests for GLM4-MoE-Lite model."""

import types

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestGLM4MoeLite:
    """Test suite for GLM4-MoE-Lite model."""

    @pytest.fixture
    def glm4_moe_lite_config(self, small_model_config):
        """Create GLM4-MoE-Lite-specific config."""
        return ed.Glm4MoeLiteConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            moe_intermediate_size=64,
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_attention_heads"],
            n_shared_experts=1,
            n_routed_experts=4,
            routed_scaling_factor=1.0,
            kv_lora_rank=16,
            q_lora_rank=16,
            qk_rope_head_dim=16,
            v_head_dim=32,
            qk_nope_head_dim=16,
            n_group=1,
            topk_group=1,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            hidden_act=small_model_config["hidden_act"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            initializer_range=small_model_config["initializer_range"],
            rms_norm_eps=1e-5,
            use_cache=small_model_config["use_cache"],
            pad_token_id=small_model_config["pad_token_id"],
            bos_token_id=small_model_config["bos_token_id"],
            eos_token_id=small_model_config["eos_token_id"],
            pretraining_tp=1,
            tie_word_embeddings=small_model_config["tie_word_embeddings"],
            rope_theta=small_model_config["rope_theta"],
            rope_interleave=True,
            attention_bias=small_model_config["attention_bias"],
            attention_dropout=small_model_config["attention_dropout"],
        )

    def test_causal_lm(self, glm4_moe_lite_config, small_model_config):
        """Test Glm4MoeLiteForCausalLM."""
        glm4_moe_lite_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.run(
            module_name="glm4_moe_lite",
            hf_class=transformers.Glm4MoeLiteForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=glm4_moe_lite_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GLM4-MoE-Lite CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, glm4_moe_lite_config, small_model_config):
        """Test GLM4-MoE-Lite text generation."""
        glm4_moe_lite_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="glm4_moe_lite",
            hf_class=transformers.Glm4MoeLiteForCausalLM,
            config=glm4_moe_lite_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"GLM4-MoE-Lite generation failed: {result.error_message}"

    def test_ragged_cache_uses_mla_q_head_dim(self, glm4_moe_lite_config, monkeypatch):
        """Paged-cache config must use MLA q_head_dim, not rope-only head_dim."""
        captured = {}
        from easydel.caching import RaggedPagesCacheConfig

        original_create = RaggedPagesCacheConfig.create

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(**kwargs)

        monkeypatch.setattr(RaggedPagesCacheConfig, "create", staticmethod(_capture_create))
        try:
            dummy = types.SimpleNamespace(config=glm4_moe_lite_config, mesh=glm4_moe_lite_config.mesh)
            ed.Glm4MoeLiteForCausalLM.create_ragged_page_cache_config(dummy, max_length=1024)
        finally:
            monkeypatch.setattr(RaggedPagesCacheConfig, "create", original_create)

        expected_q_head_dim = glm4_moe_lite_config.qk_nope_head_dim + glm4_moe_lite_config.qk_rope_head_dim
        assert captured["kv_head_dim_size"] == expected_q_head_dim
        assert captured["k_headdim"] == expected_q_head_dim


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
