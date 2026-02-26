"""Tests for Qwen3OmniMoe model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest

import easydel as ed

try:
    from .test_utils import EasyDeLOnlyTester
except ImportError:
    from test_utils import EasyDeLOnlyTester  # pyright: ignore[reportImplicitRelativeImport]


class TestQwen3OmniMoe:
    """Test suite for Qwen3OmniMoe model.

    Note: Qwen3OmniMoe is registered with TaskType.ANY_TO_ANY (multimodal)
    and TaskType.BASE_MODULE (thinker). No CAUSAL_LM registration exists.
    """

    @pytest.fixture
    def qwen3_omni_moe_thinker_config(self, small_model_config):
        """Create Qwen3OmniMoeThinker-specific config for BASE_MODULE testing."""
        return ed.Qwen3OmniMoeThinkerConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            num_experts=small_model_config.get("num_experts", 8),
            num_experts_per_tok=small_model_config.get("num_experts_per_tok", 2),
        )

    def test_causal_lm(self, qwen3_omni_moe_thinker_config, small_model_config):
        """Test Qwen3OmniMoeThinker with BASE_MODULE task."""
        tester = EasyDeLOnlyTester()
        result = tester.run(
            module_name="qwen3_omni_moe",
            task=ed.TaskType.BASE_MODULE,  # Thinker is registered as BASE_MODULE
            config=qwen3_omni_moe_thinker_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3OmniMoe BASE_MODULE failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
