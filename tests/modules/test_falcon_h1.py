"""Tests for FalconH1 model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestFalconH1:
    """Test suite for FalconH1 model."""

    @pytest.fixture
    def falcon_h1_config(self, small_model_config):
        """Create FalconH1-specific config."""
        return ed.FalconH1Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            rms_norm_eps=small_model_config["rms_norm_eps"],
            rope_theta=small_model_config["rope_theta"],
            attention_dropout=small_model_config["attention_dropout"],
            # Keep the Mamba mixer small for unit tests
            mamba_d_ssm=small_model_config["hidden_size"],
            mamba_n_heads=small_model_config["num_attention_heads"],
            mamba_d_head="auto",
            mamba_n_groups=1,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_chunk_size=64,
        )

    def test_causal_lm(self, falcon_h1_config, small_model_config):
        """Test FalconH1ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="falcon_h1",
            hf_class=transformers.FalconH1ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=falcon_h1_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"FalconH1 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, falcon_h1_config, small_model_config):
        """Test FalconH1 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="falcon_h1",
            hf_class=transformers.FalconH1ForCausalLM,
            config=falcon_h1_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"FalconH1 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
