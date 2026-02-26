"""Tests for Qwen3 model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestQwen3:
    """Test suite for Qwen3 model."""

    @pytest.fixture
    def qwen3_config(self, small_model_config):
        """Create Qwen3-specific config."""
        return ed.Qwen3Config(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
        )

    def test_causal_lm(self, qwen3_config, small_model_config):
        """Test Qwen3ForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="qwen3",
            hf_class=transformers.Qwen3ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=qwen3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, qwen3_config, small_model_config):
        """Test Qwen3 text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="qwen3",
            hf_class=transformers.Qwen3ForCausalLM,
            config=qwen3_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Qwen3 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
