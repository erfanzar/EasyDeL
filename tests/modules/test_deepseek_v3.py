"""Tests for DeepSeek V3 model."""

import pytest

import easydel as ed

try:
    from .test_utils import CausalLMTester, get_hf_model_from_hub
except ImportError:
    from test_utils import CausalLMTester, get_hf_model_from_hub


class TestDeepSeekV3:
    """Test suite for DeepSeek V3 MoE model."""

    @pytest.fixture
    def deepseek_v3_config(self, small_model_config):
        """Create DeepSeek V3-specific config."""
        return ed.DeepseekV3Config(
            aux_loss_alpha=0.001,
            bos_token_id=0,
            eos_token_id=1,
            ep_size=1,
            first_k_dense_replace=3,
            hidden_act="silu",
            hidden_size=128,
            initializer_range=0.02,
            intermediate_size=256,
            kv_lora_rank=512,
            max_position_embeddings=1024,
            moe_intermediate_size=128,
            moe_layer_freq=1,
            n_group=8,
            n_routed_experts=32,
            n_shared_experts=1,
            norm_topk_prob=True,
            num_attention_heads=128,
            num_experts_per_tok=8,
            num_hidden_layers=4,
            num_key_value_heads=128,
            num_nextn_predict_layers=1,
            pretraining_tp=1,
            q_lora_rank=1536,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            rms_norm_eps=1e-06,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
            rope_theta=10000,
            routed_scaling_factor=2.5,
            scoring_func="sigmoid",
            seq_aux=True,
            tie_word_embeddings=False,
            topk_group=4,
            topk_method="noaux_tc",
            use_cache=True,
            v_head_dim=128,
            vocab_size=129280,
        )

    @pytest.fixture
    def hf_deepseek_v3_class(self, small_model_config):
        """Load DeepSeek V3 HF class from hub."""
        hf_class, _ = get_hf_model_from_hub("deepseek-ai/DeepSeek-V3", small_model_config)
        return hf_class

    def test_causal_lm(self, deepseek_v3_config, small_model_config, hf_deepseek_v3_class):
        """Test DeepSeekV3ForCausalLM."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        deepseek_v3_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.run(
            module_name="deepseek_v3",
            hf_class=hf_deepseek_v3_class,
            task=ed.TaskType.CAUSAL_LM,
            config=deepseek_v3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"DeepSeekV3 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, deepseek_v3_config, small_model_config, hf_deepseek_v3_class):
        """Test DeepSeek V3 text generation."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        deepseek_v3_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="deepseek_v3",
            hf_class=hf_deepseek_v3_class,
            config=deepseek_v3_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"DeepSeekV3 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
