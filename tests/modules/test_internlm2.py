"""Tests for InternLM2 model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest

import easydel as ed

try:
    from .test_utils import CausalLMTester, get_hf_model_from_hub
except ImportError:
    from test_utils import CausalLMTester, get_hf_model_from_hub  # pyright: ignore[reportImplicitRelativeImport]


class TestInternLM2:
    """Test suite for InternLM2 model."""

    @pytest.fixture
    def internlm2_config(self, small_model_config):
        """Create InternLM2-specific config."""
        return ed.InternLM2Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    @pytest.fixture
    def hf_internlm2_class(self, small_model_config):
        """Load InternLM2 HF class from hub."""
        hf_class, _ = get_hf_model_from_hub(
            "internlm/internlm2_5-7b-chat",
            small_model_config,
        )
        return hf_class

    def test_causal_lm(self, internlm2_config, small_model_config, hf_internlm2_class):
        """Test InternLM2ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="internlm2",
            hf_class=hf_internlm2_class,
            task=ed.TaskType.CAUSAL_LM,
            config=internlm2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"InternLM2 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, internlm2_config, small_model_config, hf_internlm2_class):
        """Test InternLM2 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="internlm2",
            hf_class=hf_internlm2_class,
            config=internlm2_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"InternLM2 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
