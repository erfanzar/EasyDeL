"""Accelerator-only numerical coverage for real dense/MoE Qwen cached-media prefill.

No downloads, fake towers, CPU numerics, or substituted JIT functions. The native
all-raw model is the reference for eSurge's cached/native assembly and KV decode.
"""

from types import SimpleNamespace

import easydel as ed
import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.multimodal.cache import VisionEncoderCache
from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange
from easydel.inference.esurge.runners.states import CachedRequestState
from easydel.inference.esurge.runners.vlm_prefill import VlmPrefillHelper
from easydel.inference.sampling_params import SamplingParams
from easydel.utils.graph_jit import module_jit
from jax import numpy as jnp


@pytest.fixture(autouse=True)
def _highest_attention_precision():
    # Model precision configures projections, not the vanilla attention dots.
    # Pin their ambient precision too: reduced-precision dots need not agree
    # at 1e-5 across single-media and batched reference encoder shapes.
    with jax.default_matmul_precision("highest"):
        yield


def _model(family):
    config_cls = ed.Qwen3VLConfig if family == "dense" else ed.Qwen3VLMoeConfig
    text = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=128,
        pad_token_id=0,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        kvdtype=jnp.float32,
    )
    if family == "moe":
        text.update(moe_intermediate_size=64, num_experts=4, num_experts_per_tok=2, moe_force_xla_gmm=True)
    config = config_cls(
        text_config=text,
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=4,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=64,
            num_position_embeddings=64,
            deepstack_visual_indexes=[0, 1],
            attn_dtype=jnp.float32,
            attn_softmax_dtype=jnp.float32,
        ),
        image_token_id=90,
        video_token_id=91,
        vision_start_token_id=92,
        vision_end_token_id=93,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        moe_force_xla_gmm=True,
    )
    return ed.AutoEasyDeLModelForImageTextToText.from_config(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(7),
    )


def _feature(pixels, modality, offset):
    factory = MultiModalFeature.from_image if modality == "image" else MultiModalFeature.from_video
    return factory(pixels.copy(), np.array([[1, 4, 4]], np.int32), PlaceholderRange(offset, 4, modality))


def _state(media, raw_only=False):
    ids = [7]
    features = []
    for pixels_or_feature, modality in media:
        ids.append(92)
        feature = (
            pixels_or_feature
            if isinstance(pixels_or_feature, MultiModalFeature)
            else _feature(pixels_or_feature, modality, len(ids))
        )
        features.append(feature)
        ids.extend([90 if modality == "image" else 91] * 4)
        ids.append(93)
    ids.extend([8, 9, 10])
    kwargs = {}
    for modality, pixel_key, grid_key in (
        ("image", "pixel_values", "image_grid_thw"),
        ("video", "pixel_values_videos", "video_grid_thw"),
    ):
        rows = [f for f in features if f.modality == modality and f.pixel_values is not None]
        if rows:
            kwargs[pixel_key] = np.concatenate([f.pixel_values for f in rows])
            kwargs[grid_key] = np.concatenate([f.grid_thw for f in rows])
    return CachedRequestState(
        req_id="raw" if raw_only else "cached",
        prompt_token_ids=ids,
        sampling_params=SamplingParams(),
        generator=None,
        page_ids=([],),
        num_computed_tokens=0,
        output_token_ids=[],
        mm_features=features,
        **kwargs,
    )


def _raw_kwargs(state, batch=1):
    kwargs = {"input_ids": jnp.tile(jnp.asarray(state.prompt_token_ids)[None], (batch, 1))}
    for name in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        value = getattr(state, name)
        if value is not None:
            kwargs[name] = (
                jnp.asarray(np.tile(value, (batch, 1)))
                if "pixel" in name
                else tuple(map(tuple, np.tile(value, (batch, 1)).tolist()))
            )
    return kwargs


def _prepared_kwargs(state, batch):
    return dict(
        inputs_embeds=jnp.tile(jnp.asarray(state.prefill_inputs_embeds)[None], (batch, 1, 1)),
        position_ids=jnp.tile(jnp.asarray(state.prefill_position_ids)[:, None], (1, batch, 1)),
        visual_pos_masks=jnp.tile(jnp.asarray(state.prefill_visual_pos_masks)[None], (batch, 1)),
        deepstack_visual_embeds=[
            jnp.tile(jnp.asarray(value), (batch, 1)) for value in state.prefill_deepstack_visual_embeds
        ],
    )


def _forward(model, kwargs, cache):
    output = model.base_model(**kwargs, past_key_values=cache)
    return output.last_hidden_state, output.past_key_values


def _assert_metadata(model, raw, assembled):
    native_embeds, info = model.base_model.compute_embedding_with_info(**_raw_kwargs(raw))
    np.testing.assert_allclose(assembled.prefill_inputs_embeds, np.asarray(native_embeds[0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(assembled.prefill_position_ids, np.asarray(info.position_ids[:, 0]))
    np.testing.assert_array_equal(assembled.prefill_visual_pos_masks, np.asarray(info.visual_pos_masks[0]))
    np.testing.assert_array_equal(assembled.prefill_rope_deltas, np.asarray(info.rope_deltas))
    assert len(assembled.prefill_deepstack_visual_embeds) == 2
    for got, expected in zip(assembled.prefill_deepstack_visual_embeds, info.deepstack_visual_embeds, strict=True):
        np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("family", ["dense", "moe"])
@pytest.mark.parametrize("compiled_vision", [False, True])
def test_real_qwen_mixed_cached_native_prefill_and_decode(family, compiled_vision):
    model = _model(family)
    rng = np.random.default_rng(12)
    pixels = [rng.normal(size=(16, 48)).astype(np.float32) for _ in range(4)]
    helper = VlmPrefillHelper(
        model_getter=lambda: model,
        metadata=SimpleNamespace(data_parallel_size=1),
        compile_vision_encoder=compiled_vision,
        vision_patch_buckets=None,
        max_num_batched_tokens=64,
        max_num_tokens=64,
        vision_cache=VisionEncoderCache(capacity_mb=8),
    )
    # Warm both modalities through the actual eager/compiled encoder route.
    warm = _state([(pixels[0], "image"), (pixels[1], "video")])
    helper.precompute_prefill(warm)
    assert all(f.has_cached_embeddings and f.pixel_values is None for f in warm.mm_features)
    if compiled_vision:
        assert not helper.vision_jit_disabled
        assert helper.image_features_jit is not None and helper.video_features_jit is not None
    helper.vision_cache.clear()
    mixed = _state(
        [(warm.mm_features[0], "image"), (pixels[2], "video"), (pixels[3], "image"), (warm.mm_features[1], "video")]
    )
    raw = _state([(pixels[0], "image"), (pixels[2], "video"), (pixels[3], "image"), (pixels[1], "video")], True)
    helper.precompute_prefill(mixed)
    _assert_metadata(model, raw, mixed)
    assert all(f.has_cached_embeddings and f.pixel_values is None for f in mixed.mm_features)

    # A distinct request reorders retained media and rebuilds masks and mRoPE.
    helper.vision_cache.clear()
    reused = _state(
        [
            (mixed.mm_features[3], "video"),
            (mixed.mm_features[2], "image"),
            (mixed.mm_features[1], "video"),
            (mixed.mm_features[0], "image"),
        ]
    )
    reused_raw = _state([(pixels[1], "video"), (pixels[3], "image"), (pixels[2], "video"), (pixels[0], "image")], True)
    helper.precompute_prefill(reused)
    _assert_metadata(model, reused_raw, reused)

    # Batch 4 is divisible by the TPU mesh; compact DeepStack rows are repeated
    # in prompt order, and the cache has independent slots for every example.
    batch = 4
    forward_jit = module_jit(_forward, lambda: model)
    for reference, assembled in ((raw, mixed), (reused_raw, reused)):
        raw_kwargs = _raw_kwargs(reference, batch)
        prepared = _prepared_kwargs(assembled, batch)
        # Validate the actual batch too, not just each single-request warmup:
        # modality tables and compact residuals must be in flattened batch order.
        native_embeds, native_info = model.base_model.compute_embedding_with_info(**raw_kwargs)
        np.testing.assert_allclose(
            np.asarray(prepared["inputs_embeds"]), np.asarray(native_embeds), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_array_equal(np.asarray(prepared["position_ids"]), np.asarray(native_info.position_ids))
        np.testing.assert_array_equal(np.asarray(prepared["visual_pos_masks"]), np.asarray(native_info.visual_pos_masks))
        for got, expected in zip(prepared["deepstack_visual_embeds"], native_info.deepstack_visual_embeds, strict=True):
            np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-5, atol=1e-5)
        seq = len(reference.prompt_token_ids)
        expected, expected_cache = _forward(model, raw_kwargs, model.init_cache(batch, 64))
        eager, _ = _forward(model, prepared, model.init_cache(batch, 64))
        compiled, compiled_cache = forward_jit(prepared, model.init_cache(batch, 64))
        np.testing.assert_allclose(np.asarray(eager), np.asarray(expected), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(compiled), np.asarray(expected), rtol=1e-5, atol=1e-5)
        # Text decode must use the warmed KV state, with no stale vision residuals.
        for step in range(2):
            position = seq + step + jnp.asarray(assembled.prefill_rope_deltas).reshape(1, 1, 1)
            decode = dict(
                input_ids=jnp.full((batch, 1), 11 + step, jnp.int32),
                position_ids=jnp.broadcast_to(position, (3, batch, 1)),
            )
            expected_decode, expected_cache = _forward(model, decode, expected_cache)
            actual_decode, compiled_cache = forward_jit(decode, compiled_cache)
            np.testing.assert_allclose(np.asarray(actual_decode), np.asarray(expected_decode), rtol=1e-5, atol=1e-5)
