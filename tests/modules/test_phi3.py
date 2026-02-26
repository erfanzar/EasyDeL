"""Tests for Phi3 model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestPhi3:
    """Test suite for Phi3 model."""

    @pytest.fixture
    def phi3_config(self, small_model_config):
        """Create Phi3-specific config with LongRoPE scaling."""
        config_dict = small_model_config.copy()
        # LongRoPE requires factors of length head_dim / 2 = 32 / 2 = 16
        config_dict["rope_scaling"] = {
            "long_factor": [1.0] * 16,
            "long_mscale": 1.8,
            "original_max_position_embeddings": 128,
            "short_factor": [1.0] * 16,
            "short_mscale": 1.1,
            "type": "longrope",
        }
        return ed.Phi3Config(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            intermediate_size=config_dict["intermediate_size"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            rope_scaling=config_dict["rope_scaling"],
        )

    def test_causal_lm(self, phi3_config, small_model_config):
        """Test Phi3ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="phi3",
            hf_class=transformers.Phi3ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=phi3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Phi3 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, phi3_config, small_model_config):
        """Test Phi3 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="phi3",
            hf_class=transformers.Phi3ForCausalLM,
            config=phi3_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Phi3 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
