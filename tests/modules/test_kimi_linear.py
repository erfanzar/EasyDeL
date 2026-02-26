"""Tests for Kimi Linear model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest

import easydel as ed

try:
    from .test_utils import CausalLMTester, get_hf_model_from_hub
except ImportError:
    from test_utils import CausalLMTester, get_hf_model_from_hub  # pyright: ignore[reportImplicitRelativeImport]


# Check if fla-core is available (required for Kimi Linear)
_has_fla = False
try:
    import fla  # noqa: F401  # pyright: ignore[reportMissingImports,reportUnusedImport]

    _has_fla = True
except ImportError:
    pass
HAS_FLA: bool = _has_fla


@pytest.mark.skipif(not HAS_FLA, reason="fla-core package required: pip install -U fla-core")
class TestKimiLinear:
    """Test suite for Kimi Linear model."""

    @pytest.fixture
    def hf_kimi_linear_class(self, small_model_config):
        """Load Kimi Linear HF class from hub."""
        hf_class, _ = get_hf_model_from_hub(
            "moonshotai/Kimi-Linear-48B-A3B-Instruct",
            small_model_config,
        )
        return hf_class

    @pytest.fixture
    def kimi_linear_config(self, small_model_config):
        """Create Kimi Linear-specific config."""
        return ed.KimiLinearConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_causal_lm(self, kimi_linear_config, small_model_config, hf_kimi_linear_class):
        """Test KimiLinearForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="kimi_linear",
            hf_class=hf_kimi_linear_class,
            task=ed.TaskType.CAUSAL_LM,
            config=kimi_linear_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Kimi Linear CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, kimi_linear_config, small_model_config, hf_kimi_linear_class):
        """Test Kimi Linear text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="kimi_linear",
            hf_class=hf_kimi_linear_class,
            task=ed.TaskType.CAUSAL_LM,
            config=kimi_linear_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Kimi Linear generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
