"""Tests for Phi model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestPhi:
    """Test suite for Phi model."""

    @pytest.fixture
    def phi_config(self, small_model_config):
        """Create Phi-specific config."""
        return ed.PhiConfig(
            vocab_size=51200,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=None,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attention_dropout=0.0,
            hidden_act="gelu_new",
            max_position_embeddings=small_model_config["max_position_embeddings"],
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            partial_rotary_factor=0.5,
            qk_layernorm=False,
            bos_token_id=1,
            eos_token_id=2,
        )

    def test_causal_lm(self, phi_config, small_model_config):
        """Test PhiForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="phi",
            hf_class=transformers.PhiForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=phi_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Phi CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, phi_config, small_model_config):
        """Test Phi text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="phi",
            hf_class=transformers.PhiForCausalLM,
            config=phi_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Phi generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
