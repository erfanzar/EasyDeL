"""Tests for OpenELM model."""

import pytest

import easydel as ed

try:
    from .test_utils import CausalLMTester, get_hf_model_from_hub
except ImportError:
    from test_utils import CausalLMTester, get_hf_model_from_hub


class TestOpenELM:
    """Test suite for OpenELM model."""

    @pytest.fixture
    def openelm_config(self, small_model_config):
        """Create OpenELM-specific config."""
        _, conf = get_hf_model_from_hub(
            "apple/OpenELM-270M-Instruct",
            small_model_config,
        )
        config = ed.OpenELMConfig()
        for k, v in conf.__dict__.items():
            setattr(config, k, v)
        config.max_context_length = small_model_config["max_position_embeddings"]
        return config

    @pytest.fixture
    def hf_openelm_class(self, small_model_config):
        """Load OpenELM HF class from hub."""
        hf_class, _ = get_hf_model_from_hub(
            "apple/OpenELM-270M-Instruct",
            small_model_config,
        )
        return hf_class

    def test_causal_lm(self, openelm_config, small_model_config, hf_openelm_class):
        """Test OpenELMForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="openelm",
            hf_class=hf_openelm_class,
            task=ed.TaskType.CAUSAL_LM,
            config=openelm_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"OpenELM CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, openelm_config, small_model_config, hf_openelm_class):
        """Test OpenELM text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="openelm",
            hf_class=hf_openelm_class,
            config=openelm_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"OpenELM generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
