"""Tests for GLM-MoE-DSA model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


def _resolve_hf_class(top_level_name: str, module_path: str, class_name: str):
    cls = getattr(transformers, top_level_name, None)
    if cls is not None:
        return cls
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except Exception:
        return None


class TestGLMMoeDSA:
    """Test suite for GLM-MoE-DSA model."""

    @pytest.fixture
    def hf_glm_moe_dsa_class(self):
        return _resolve_hf_class(
            top_level_name="GlmMoeDsaForCausalLM",
            module_path="transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
            class_name="GlmMoeDsaForCausalLM",
        )

    @pytest.fixture
    def glm_moe_dsa_config(self, small_model_config):
        """Create GLM-MoE-DSA config."""
        return ed.GlmMoeDsaConfig(
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
            qk_nope_head_dim=16,
            v_head_dim=32,
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
            rope_interleave=False,
            mlp_layer_types=None,
            attention_bias=small_model_config["attention_bias"],
            attention_dropout=small_model_config["attention_dropout"],
            index_topk=128,
            index_head_dim=16,
            index_n_heads=4,
            indexer_rope_interleave=False,
        )

    def test_causal_lm(self, glm_moe_dsa_config, small_model_config, hf_glm_moe_dsa_class):
        """Test GlmMoeDsaForCausalLM forward/compare path."""
        if hf_glm_moe_dsa_class is None:
            pytest.skip("HF GlmMoeDsaForCausalLM not available")
        glm_moe_dsa_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.run(
            module_name="glm_moe_dsa",
            hf_class=hf_glm_moe_dsa_class,
            task=ed.TaskType.CAUSAL_LM,
            config=glm_moe_dsa_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GLM-MoE-DSA CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, glm_moe_dsa_config, small_model_config, hf_glm_moe_dsa_class):
        """Test GLM-MoE-DSA text generation."""
        if hf_glm_moe_dsa_class is None:
            pytest.skip("HF GlmMoeDsaForCausalLM not available")
        glm_moe_dsa_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="glm_moe_dsa",
            hf_class=hf_glm_moe_dsa_class,
            config=glm_moe_dsa_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"GLM-MoE-DSA generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
