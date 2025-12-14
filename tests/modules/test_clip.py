"""Tests for CLIP model."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import transformers

import easydel as ed

try:
    from .test_utils import compare_hidden_states, setup_config
    from .test_utils.model_factory import cleanup_models, create_ed_model, create_hf_model
except ImportError:
    from test_utils import compare_hidden_states, setup_config
    from test_utils.model_factory import cleanup_models, create_ed_model, create_hf_model


class TestCLIP:
    """Test suite for CLIP model."""

    @pytest.fixture
    def clip_vision_config(self, small_model_config):
        """Create CLIP vision config."""
        return ed.CLIPVisionConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=224,
            patch_size=14,
        )

    @pytest.fixture
    def clip_text_config(self, small_model_config):
        """Create CLIP text config."""
        return ed.CLIPTextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=77,
        )

    @pytest.fixture
    def clip_config(self, clip_vision_config, clip_text_config):
        """Create CLIP config."""
        return ed.CLIPConfig(
            vision_config=clip_vision_config,
            text_config=clip_text_config,
        )

    def test_vision_model(self, clip_vision_config, small_model_config):
        """Test CLIPVisionModel with pixel_values input."""
        config = setup_config(clip_vision_config, small_model_config)

        # Create models
        hf_model = create_hf_model(transformers.CLIPVisionModel, config)

        with config.mesh:
            ed_model = create_ed_model(
                module_name="clip_vision_model",
                task=ed.TaskType.BASE_VISION,
                config=config,
                small_model_config=small_model_config,
                hf_model=hf_model,
            )

            # Generate pixel_values input (not input_ids)
            batch_size = small_model_config["batch_size"]
            image_size = config.image_size
            rng = np.random.default_rng(42)
            pixel_values_np = rng.standard_normal((batch_size, 3, image_size, image_size), dtype=np.float32)

            # Run HF forward
            hf_output = hf_model(
                pixel_values=torch.from_numpy(pixel_values_np),
                output_hidden_states=True,
            )

            # Run ED forward
            ed_output = ed_model(
                pixel_values=jnp.asarray(pixel_values_np),
                output_hidden_states=True,
            )

            # Compare hidden states
            hf_hidden = hf_output.last_hidden_state.cpu().detach().numpy()
            ed_hidden = np.asarray(ed_output.last_hidden_state)

            comparison = compare_hidden_states(
                name="clip_vision_model",
                hf_hidden=hf_hidden,
                ed_hidden=ed_hidden,
            )

        cleanup_models(hf_model)
        assert comparison.success, f"CLIP vision failed: {comparison.details}"

    def test_text_model(self, clip_text_config, small_model_config):
        """Test CLIPTextModel with input_ids."""
        config = setup_config(clip_text_config, small_model_config)

        # Create models
        hf_model = create_hf_model(transformers.CLIPTextModel, config)

        with config.mesh:
            ed_model = create_ed_model(
                module_name="clip_text_model",
                task=ed.TaskType.BASE_MODULE,
                config=config,
                small_model_config=small_model_config,
                hf_model=hf_model,
            )

            # Generate text inputs
            batch_size = small_model_config["batch_size"]
            seq_len = min(small_model_config["sequence_length"], config.max_position_embeddings)
            rng = np.random.default_rng(42)
            input_ids_np = rng.integers(0, config.vocab_size, size=(batch_size, seq_len), dtype=np.int64)
            attention_mask_np = np.ones((batch_size, seq_len), dtype=np.int64)

            # Run HF forward
            hf_output = hf_model(
                input_ids=torch.from_numpy(input_ids_np).long(),
                attention_mask=torch.from_numpy(attention_mask_np).long(),
                output_hidden_states=True,
            )

            # Run ED forward
            ed_output = ed_model(
                input_ids=jnp.asarray(input_ids_np),
                attention_mask=jnp.asarray(attention_mask_np, dtype=jnp.bool_),
                output_hidden_states=True,
            )

            # Compare hidden states
            hf_hidden = hf_output.last_hidden_state.cpu().detach().numpy()
            ed_hidden = np.asarray(ed_output.last_hidden_state)

            comparison = compare_hidden_states(
                name="clip_text_model",
                hf_hidden=hf_hidden,
                ed_hidden=ed_hidden,
            )

        cleanup_models(hf_model)
        assert comparison.success, f"CLIP text failed: {comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
