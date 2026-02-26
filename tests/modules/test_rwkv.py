"""Tests for RWKV model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed
from easydel.modules.rwkv import RwkvConfig

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestRWKV:
    """Test suite for RWKV model."""

    @pytest.fixture
    def rwkv_config(self, small_model_config):
        """Create RWKV-specific config."""
        return RwkvConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            attention_hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
        )

    def test_causal_lm(self, rwkv_config, small_model_config):
        """Test RwkvForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="rwkv",
            hf_class=transformers.RwkvForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=rwkv_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"RWKV CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, rwkv_config, small_model_config):
        """Test RWKV text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="rwkv",
            hf_class=transformers.RwkvForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=rwkv_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"RWKV generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
