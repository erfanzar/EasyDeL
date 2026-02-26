"""Tests for DeepSeek V2 model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest

import easydel as ed
from easydel.infra.etils import EasyDeLGradientCheckPointers

try:
    from .test_utils import CausalLMTester, get_hf_model_from_hub
except ImportError:
    from test_utils import CausalLMTester, get_hf_model_from_hub  # pyright: ignore[reportImplicitRelativeImport]


class TestDeepSeekV2:
    """Test suite for DeepSeek V2 MoE model."""

    @pytest.fixture
    def deepseek_v2_config(self, small_model_config):
        """Create DeepSeek V2-specific config from HuggingFace Hub."""
        # Load config from Hub - this gets the proper MLA parameters
        _, conf = get_hf_model_from_hub("deepseek-ai/DeepSeek-V2", small_model_config)
        return ed.DeepseekV2Config(
            vocab_size=conf.vocab_size,
            hidden_size=conf.hidden_size,
            intermediate_size=conf.intermediate_size,
            moe_intermediate_size=conf.moe_intermediate_size,
            num_hidden_layers=conf.num_hidden_layers,
            num_attention_heads=conf.num_attention_heads,
            num_key_value_heads=conf.num_key_value_heads,
            n_shared_experts=conf.n_shared_experts,
            n_routed_experts=conf.n_routed_experts,
            ep_size=conf.ep_size,
            routed_scaling_factor=conf.routed_scaling_factor,
            kv_lora_rank=conf.kv_lora_rank,
            q_lora_rank=conf.q_lora_rank,
            qk_rope_head_dim=conf.qk_rope_head_dim,
            v_head_dim=conf.v_head_dim,
            qk_nope_head_dim=conf.qk_nope_head_dim,
            topk_method=conf.topk_method,
            n_group=conf.n_group,
            topk_group=conf.topk_group,
            num_experts_per_tok=conf.num_experts_per_tok,
            moe_layer_freq=conf.moe_layer_freq,
            first_k_dense_replace=conf.first_k_dense_replace,
            norm_topk_prob=conf.norm_topk_prob,
            scoring_func=conf.scoring_func,
            aux_loss_alpha=conf.aux_loss_alpha,
            seq_aux=conf.seq_aux,
            hidden_act=conf.hidden_act,
            max_position_embeddings=conf.max_position_embeddings,
            initializer_range=conf.initializer_range,
            rms_norm_eps=conf.rms_norm_eps,
            use_cache=conf.use_cache,
            pad_token_id=conf.pad_token_id,
            bos_token_id=conf.bos_token_id,
            eos_token_id=conf.eos_token_id,
            pretraining_tp=conf.pretraining_tp,
            tie_word_embeddings=conf.tie_word_embeddings,
            rope_theta=conf.rope_theta,
            attention_bias=conf.attention_bias,
            attention_dropout=conf.attention_dropout,
            gradient_checkpointing=EasyDeLGradientCheckPointers.NONE,
            rope_scaling=conf.rope_scaling,
        )

    @pytest.fixture
    def hf_deepseek_v2_class(self, small_model_config):
        """Load DeepSeek V2 HF class from hub."""
        hf_class, _ = get_hf_model_from_hub("deepseek-ai/DeepSeek-V2", small_model_config)
        return hf_class

    def test_causal_lm(self, deepseek_v2_config, small_model_config, hf_deepseek_v2_class):
        """Test DeepSeekV2ForCausalLM."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        deepseek_v2_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.run(
            module_name="deepseek_v2",
            hf_class=hf_deepseek_v2_class,
            task=ed.TaskType.CAUSAL_LM,
            config=deepseek_v2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"DeepSeekV2 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, deepseek_v2_config, small_model_config, hf_deepseek_v2_class):
        """Test DeepSeek V2 text generation."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        deepseek_v2_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="deepseek_v2",
            hf_class=hf_deepseek_v2_class,
            config=deepseek_v2_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"DeepSeekV2 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
