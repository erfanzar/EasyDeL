"""Tests for Pixtral vision model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import BaseModuleTester
except ImportError:
    from test_utils import BaseModuleTester


class TestPixtral:
    """Test suite for Pixtral vision model."""

    @pytest.fixture
    def pixtral_config(self, small_model_config):
        """Create Pixtral vision config."""
        return ed.PixtralVisionConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=1024,
            patch_size=16,
        )

    def test_vision_model(self, pixtral_config, small_model_config):
        """Test PixtralVisionModel."""
        tester = BaseModuleTester()
        result = tester.run(
            module_name="pixtral",
            hf_class=transformers.PixtralVisionModel,
            task=ed.TaskType.BASE_VISION,
            config=pixtral_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Pixtral vision failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
