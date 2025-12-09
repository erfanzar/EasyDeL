"""Tests for GLM model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestGLM:
    """Test suite for GLM model."""

    @pytest.fixture
    def glm_config(self, small_model_config):
        """Create GLM-specific config."""
        return ed.GlmConfig(
            vocab_size=151552,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            partial_rotary_factor=0.5,
            head_dim=128,
            hidden_act="silu",
            attention_dropout=0.0,
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=0.00000015625,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            pad_token_id=151329,
            eos_token_id=None,
            bos_token_id=None,
            attention_bias=True,
        )

    def test_causal_lm(self, glm_config, small_model_config):
        """Test GlmForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="glm",
            hf_class=transformers.GlmForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=glm_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"GLM CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, glm_config, small_model_config):
        """Test GLM text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="glm",
            hf_class=transformers.GlmForCausalLM,
            config=glm_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"GLM generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
