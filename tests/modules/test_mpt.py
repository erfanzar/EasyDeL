"""Tests for MPT model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestMPT:
    """Test suite for MPT model."""

    @pytest.fixture
    def mpt_config(self, small_model_config):
        """Create MPT-specific config."""
        return ed.MptConfig(
            d_model=small_model_config["hidden_size"],
            n_heads=small_model_config["num_attention_heads"],
            n_layers=4,
            attn_config=ed.MptAttentionConfig(),
            sharding_axis_dims=(1, 1, 1, 1, -1),
        )

    def test_causal_lm(self, mpt_config, small_model_config):
        """Test MptForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="mpt",
            hf_class=transformers.MptForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=mpt_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"MPT CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, mpt_config, small_model_config):
        """Test MPT text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mpt",
            hf_class=transformers.MptForCausalLM,
            config=mpt_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"MPT generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
