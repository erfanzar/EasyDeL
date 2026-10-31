# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for CLIP model."""

import easydel as ed
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
import torch
import transformers

try:
    from tests.modules.test_utils import compare_hidden_states, setup_config
    from tests.modules.test_utils.model_factory import cleanup_models, create_ed_model, create_hf_model
except ImportError:
    from tests.modules.test_utils import (  # pyright: ignore[reportImplicitRelativeImport]
        compare_hidden_states,
        setup_config,
    )
    from tests.modules.test_utils.model_factory import (  # pyright: ignore[reportImplicitRelativeImport]
        cleanup_models,
        create_ed_model,
        create_hf_model,
    )


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

    def test_conversion_accepts_both_hf_key_layouts(self, clip_vision_config, small_model_config):
        """Old (wrapper-prefixed) and new (flattened) HF key layouts convert identically.

        transformers >= 5.13 dropped the inner ``vision_model.`` wrapper from
        standalone ``CLIPVisionModel.state_dict()`` keys, while real hub
        checkpoints keep the old prefixed layout. Conversion must accept both
        and produce the same EasyDeL parameter tree.
        """
        config = setup_config(clip_vision_config, small_model_config)
        hf_model = create_hf_model(transformers.CLIPVisionModel, config)
        state_dict = hf_model.state_dict()

        flattened = {key.removeprefix("vision_model."): value for key, value in state_dict.items()}
        prefixed = {f"vision_model.{key}": value for key, value in flattened.items()}

        _, module_class = ed.get_modules_by_type("clip_vision_model", ed.TaskType.BASE_VISION)
        with config.mesh:
            ed_model = module_class.lazy_init(
                config=config,
                dtype=small_model_config["dtype"],
                param_dtype=small_model_config["dtype"],
                precision=small_model_config["precision"],
                rngs=spx.Rngs(0),
            )
            tree_from_new = ed_model.pure_transform_fn(flattened)
            tree_from_old = ed_model.pure_transform_fn(prefixed)

        flat_new = ed.traversals.flatten_dict(tree_from_new)
        flat_old = ed.traversals.flatten_dict(tree_from_old)

        assert flat_new, "conversion produced an empty parameter tree"
        assert set(flat_new.keys()) == set(flat_old.keys())
        assert all(path[0] == "vision_model" for path in flat_new), (
            "flattened-layout keys were not renamed under the vision_model wrapper"
        )
        for path, value in flat_new.items():
            np.testing.assert_array_equal(
                np.asarray(value),
                np.asarray(flat_old[path]),
                err_msg=f"converted leaf mismatch at {path}",
            )

        cleanup_models(hf_model)

    def test_clip_model_forward_with_attention_mask(self, small_model_config):
        """CLIPModel.forward / get_text_features accept a padded ``attention_mask`` and match HF.

        Regression: both passed ``attention_mask=`` to ``CLIPTextTransformer.forward``
        (no such parameter -> TypeError) and built positions as
        ``cumsum(mask) - 1`` instead of HF's ``arange``.
        """
        config = ed.CLIPConfig(
            text_config=dict(
                vocab_size=small_model_config["vocab_size"],
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                max_position_embeddings=77,
                eos_token_id=small_model_config["vocab_size"] - 1,
            ),
            vision_config=dict(
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                image_size=56,
                patch_size=14,
            ),
            projection_dim=64,
        )
        setup_config(config.text_config, small_model_config)
        setup_config(config.vision_config, small_model_config)
        config = setup_config(config, small_model_config)
        hf_model = create_hf_model(transformers.CLIPModel, config)

        batch_size = small_model_config["batch_size"]
        seq_len = 16
        rng = np.random.default_rng(42)
        vocab_size = config.text_config.vocab_size
        eos_marker = config.text_config.eos_token_id
        assert eos_marker == vocab_size - 1
        input_ids_np = rng.integers(3, vocab_size - 1, size=(batch_size, seq_len), dtype=np.int64)
        attention_mask_np = np.ones((batch_size, seq_len), dtype=np.int64)
        attention_mask_np[1, 10:] = 0  # right-padded second row
        input_ids_np[1, 10:] = 0
        input_ids_np[0, -1] = eos_marker
        input_ids_np[1, 9] = eos_marker  # pooled token sits on a valid (unpadded) position
        pixel_values_np = rng.standard_normal(
            (batch_size, 3, config.vision_config.image_size, config.vision_config.image_size), dtype=np.float32
        )

        with torch.no_grad():
            hf_out = hf_model(
                input_ids=torch.from_numpy(input_ids_np),
                attention_mask=torch.from_numpy(attention_mask_np),
                pixel_values=torch.from_numpy(pixel_values_np),
            )
            hf_text_features = hf_model.get_text_features(
                input_ids=torch.from_numpy(input_ids_np),
                attention_mask=torch.from_numpy(attention_mask_np),
            )
        hf_text_features = getattr(hf_text_features, "pooler_output", hf_text_features).numpy()

        with config.mesh:
            ed_model = create_ed_model(
                module_name="clip",
                task=ed.TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION,
                config=config,
                small_model_config=small_model_config,
                hf_model=hf_model,
            )
            ed_out = ed_model(
                input_ids=jnp.asarray(input_ids_np),
                attention_mask=jnp.asarray(attention_mask_np, dtype=jnp.bool_),
                pixel_values=jnp.asarray(pixel_values_np),
            )
            ed_text_features = ed_model.get_text_features(
                input_ids=jnp.asarray(input_ids_np),
                attention_mask=jnp.asarray(attention_mask_np, dtype=jnp.bool_),
            )

            _, position_ids = ed_model._prepare_text_inputs(
                jnp.asarray(input_ids_np), jnp.asarray(attention_mask_np, dtype=jnp.bool_), None, None
            )
            np.testing.assert_array_equal(
                np.asarray(position_ids), np.broadcast_to(np.arange(seq_len), (batch_size, seq_len))
            )

            valid = attention_mask_np.astype(bool)
            np.testing.assert_allclose(
                np.asarray(ed_out.text_model_output.last_hidden_state)[valid],
                hf_out.text_model_output.last_hidden_state.numpy()[valid],
                atol=5e-2,
                rtol=0,
            )
            np.testing.assert_allclose(np.asarray(ed_out.text_embeds), hf_out.text_embeds.numpy(), atol=2e-2, rtol=0)
            np.testing.assert_allclose(np.asarray(ed_out.image_embeds), hf_out.image_embeds.numpy(), atol=2e-2, rtol=0)
            np.testing.assert_allclose(
                np.asarray(ed_out.logits_per_image), hf_out.logits_per_image.numpy(), atol=0.25, rtol=0
            )
            np.testing.assert_allclose(np.asarray(ed_text_features), hf_text_features, atol=5e-2, rtol=0)

        cleanup_models(hf_model)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
