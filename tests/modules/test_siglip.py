"""Tests for SigLIP model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import BaseModuleTester
except ImportError:
    from test_utils import BaseModuleTester


class TestSigLIP:
    """Test suite for SigLIP model."""

    @pytest.fixture
    def siglip_vision_config(self, small_model_config):
        """Create SigLIP vision config."""
        return ed.SiglipVisionConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=384,
            patch_size=14,
        )

    @pytest.fixture
    def siglip_text_config(self, small_model_config):
        """Create SigLIP text config."""
        return ed.SiglipTextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,  # Must be >= sequence_length (128)
        )

    @pytest.fixture
    def siglip_config(self, siglip_vision_config, siglip_text_config):
        """Create SigLIP config."""
        return ed.SiglipConfig(
            vision_config=siglip_vision_config,
            text_config=siglip_text_config,
        )

    def test_vision_model(self, siglip_vision_config, small_model_config):
        """Test SiglipVisionModel."""
        tester = BaseModuleTester()
        result = tester.run(
            module_name="siglip_vision_model",
            hf_class=transformers.SiglipVisionModel,
            task=ed.TaskType.BASE_VISION,  # Registered as BASE_VISION, not BASE_MODULE
            config=siglip_vision_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"SigLIP vision BASE_VISION failed: {result.error_message or result.comparison.details}"

    def test_text_model(self, siglip_text_config, small_model_config):
        """Test SiglipTextModel."""
        tester = BaseModuleTester()
        result = tester.run(
            module_name="siglip_text_model",
            hf_class=transformers.SiglipTextModel,
            task=ed.TaskType.BASE_MODULE,
            config=siglip_text_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"SigLIP text BASE_MODULE failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
