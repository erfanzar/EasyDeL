"""Cached DeepStack parity tests (accelerator only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.modules.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModel
from jax import numpy as jnp

from .test_qwen3_vl_multimodal_numerics import _model


@pytest.mark.parametrize("cls", [Qwen3VLModel, Qwen3VLMoeModel])
@pytest.mark.parametrize("cached_modalities", [("image",), ("video",), ("image", "video")])
def test_cached_deepstack_matches_raw_mixed_media_and_reuses_encoder_outputs(cls, cached_modalities):
    model = _model(cls)
    ids = jnp.array([[0, 2, 1, 2, 1, 0]], dtype=jnp.int32)
    text = jnp.arange(12, dtype=jnp.float32).reshape(1, 6, 2)
    grid = ((1, 1, 2),)
    image = jnp.array([[10.0, 11.0], [12.0, 13.0]])
    video = jnp.array([[20.0, 21.0], [22.0, 23.0]])
    raw_kwargs = {"pixel_values": image, "pixel_values_videos": video, "image_grid_thw": grid, "video_grid_thw": grid}
    expected, expected_info = model.compute_embedding_with_info(ids, inputs_embeds=text, **raw_kwargs)
    image_features, image_deepstack = model.get_image_features(image, grid)
    video_features, video_deepstack = model.get_video_features(video, grid)
    cache = {"image": (image_features[0], image_deepstack), "video": (video_features[0], video_deepstack)}
    calls = []
    real_visual = model.visual

    class CountingVision:
        spatial_merge_size = 1

        def get_dtype(self):
            return jnp.float32

        def __call__(self, pixels, **kwargs):
            calls.append("encode")
            return real_visual(pixels, **kwargs)

    object.__setattr__(model, "visual", CountingVision())
    cached_kwargs = dict(raw_kwargs)
    for modality in cached_modalities:
        cached_kwargs[f"{modality}_embeds"], cached_kwargs[f"deepstack_{modality}_embeds"] = cache[modality]
        # Raw data can be absent or still present; cached features take priority.
        if modality == "image":
            cached_kwargs.pop("pixel_values")

    for _ in range(2):
        result, info = model.compute_embedding_with_info(ids, inputs_embeds=text, **cached_kwargs)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
        np.testing.assert_array_equal(np.asarray(info.position_ids), np.asarray(expected_info.position_ids))
        np.testing.assert_array_equal(np.asarray(info.rope_deltas), np.asarray(expected_info.rope_deltas))
        np.testing.assert_array_equal(np.asarray(info.visual_pos_masks), np.asarray(expected_info.visual_pos_masks))
        for actual, reference in zip(info.deepstack_visual_embeds, expected_info.deepstack_visual_embeds, strict=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
        for modality in cached_modalities:
            assert getattr(info, f"deepstack_{modality}_embeds") is cache[modality][1]
    assert len(calls) == 2 * (2 - len(cached_modalities))

    # The embedding pass is traceable with cached mixed-media tables, and decoder
    # consumption preserves the same residuals as the native encoder path.
    def embed(token_ids, embeddings):
        return model.compute_embedding_with_info(token_ids, inputs_embeds=embeddings, **cached_kwargs)

    result, info = jax.jit(embed)(ids, text)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
    options = {"output_router_logits": True} if cls is Qwen3VLMoeModel else {}
    actual_output = model.forward(
        inputs_embeds=result,
        position_ids=info.position_ids,
        visual_pos_masks=info.visual_pos_masks,
        deepstack_visual_embeds=info.deepstack_visual_embeds,
        **options,
    )
    expected_hidden = np.asarray(expected).copy()
    expected_hidden[np.asarray(ids) == 1] += np.asarray(image) * 0.3
    expected_hidden[np.asarray(ids) == 2] += np.asarray(video) * 0.3
    np.testing.assert_allclose(np.asarray(actual_output.last_hidden_state), expected_hidden, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cls", [Qwen3VLModel, Qwen3VLMoeModel])
@pytest.mark.parametrize("cached_modalities", [("image",), ("video",), ("image", "video")])
def test_cached_deepstack_preserves_production_mrope_and_jitted_decoder_inputs(cls, cached_modalities):
    model = _model(cls)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(
            image_token_id=1,
            video_token_id=2,
            vision_start_token_id=3,
            vision_config=SimpleNamespace(spatial_merge_size=1, tokens_per_second=2.0),
        ),
    )
    # Use the family method, not the embedding-only fixture's all-zero position
    # stub. Host-side mRoPE preprocessing intentionally stays outside the JIT;
    # the decoder consumes its actual positions, as in eSurge prefill.
    object.__setattr__(model, "get_rope_index", cls.get_rope_index.__get__(model, cls))
    ids = jnp.array([[0, 3, 2, 2, 0, 3, 1, 1, 1, 1, 0]], dtype=jnp.int32)
    text = jnp.arange(22, dtype=jnp.float32).reshape(1, 11, 2)
    image = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) + 10
    video = jnp.array([[20.0, 21.0], [22.0, 23.0]])
    raw_kwargs = {
        "pixel_values": image,
        "pixel_values_videos": video,
        "image_grid_thw": ((1, 2, 2),),
        "video_grid_thw": ((1, 1, 2),),
    }
    expected, expected_info = model.compute_embedding_with_info(ids, inputs_embeds=text, **raw_kwargs)
    # Hand-derived t/h/w coordinates: video spans a width-2 row starting at
    # position 2; the image spans a 2x2 grid starting at 6. Text resumes at 8,
    # so the compressed positional length is 9 for an eleven-token prompt.
    positions = np.array(
        [
            [[0, 1, 2, 2, 4, 5, 6, 6, 6, 6, 8]],
            [[0, 1, 2, 2, 4, 5, 6, 6, 7, 7, 8]],
            [[0, 1, 2, 3, 4, 5, 6, 7, 6, 7, 8]],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(np.asarray(expected_info.position_ids), positions)
    np.testing.assert_array_equal(np.asarray(expected_info.rope_deltas), [[-2]])
    cache = {}
    for modality, pixels in (("image", image), ("video", video)):
        features, residuals = getattr(model, f"get_{modality}_features")(pixels, raw_kwargs[f"{modality}_grid_thw"])
        cache[modality] = (features[0], residuals)
    cached_kwargs = dict(raw_kwargs)
    for modality in cached_modalities:
        cached_kwargs[f"{modality}_embeds"], cached_kwargs[f"deepstack_{modality}_embeds"] = cache[modality]
        cached_kwargs.pop("pixel_values" if modality == "image" else "pixel_values_videos")

    options = {"output_router_logits": True} if cls is Qwen3VLMoeModel else {}
    # Return the decoder's consumed positions too, so the assertion observes the
    # public forward result rather than merely checking the preparation output.
    decoder = model.language_model

    def decoder_with_positions(**kwargs):
        output = decoder(**kwargs)
        output.hidden_states = (kwargs["position_ids"],)
        return output

    object.__setattr__(model, "language_model", decoder_with_positions)

    def decode(embeddings, position_ids, mask, residuals):
        return model.forward(
            inputs_embeds=embeddings,
            position_ids=position_ids,
            visual_pos_masks=mask,
            deepstack_visual_embeds=residuals,
            **options,
        )

    compiled_decode = jax.jit(decode)
    for _ in range(2):
        result, info = model.compute_embedding_with_info(ids, inputs_embeds=text, **cached_kwargs)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
        np.testing.assert_array_equal(np.asarray(info.position_ids), positions)
        np.testing.assert_array_equal(np.asarray(info.rope_deltas), [[-2]])
        np.testing.assert_array_equal(np.asarray(info.visual_pos_masks), np.asarray(expected_info.visual_pos_masks))
        for actual, reference in zip(info.deepstack_visual_embeds, expected_info.deepstack_visual_embeds, strict=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
        output = compiled_decode(result, info.position_ids, info.visual_pos_masks, info.deepstack_visual_embeds)
        np.testing.assert_array_equal(np.asarray(output.hidden_states[0]), positions)
        expected_hidden = np.asarray(expected).copy()
        expected_hidden[np.asarray(ids) == 1] += np.asarray(image) * 0.3
        expected_hidden[np.asarray(ids) == 2] += np.asarray(video) * 0.3
        np.testing.assert_allclose(np.asarray(output.last_hidden_state), expected_hidden, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cls", [Qwen3VLModel, Qwen3VLMoeModel])
def test_cached_deepstack_rejects_mismatched_image_video_layer_counts(cls):
    model = _model(cls)
    ids = jnp.array([[0, 2, 1, 2, 1, 0]], dtype=jnp.int32)
    text = jnp.zeros((1, 6, 2))
    table = jnp.ones((2, 2))
    with pytest.raises(ValueError, match="same number of layers"):
        model.compute_embedding_with_info(
            ids,
            inputs_embeds=text,
            image_embeds=table,
            video_embeds=table,
            deepstack_image_embeds=[table],
            deepstack_video_embeds=[table, table],
        )
