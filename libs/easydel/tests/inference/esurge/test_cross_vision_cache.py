"""Encoder cache routing tests with NumPy-only fake towers; no JAX array computation."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.multimodal.cache import VisionEncoderCache
from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange
from easydel.inference.esurge.runners import vlm_prefill
from easydel.inference.esurge.runners.states import CachedRequestState
from easydel.inference.esurge.runners.vlm_prefill import VlmPrefillHelper
from easydel.inference.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_ok


class Tower:
    def __init__(self, offset=0):
        self.offset = offset
        self.calls = 0
        self.config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=1))

    def compute_embedding(self, input_ids, *, image_embeds=None, video_embeds=None):
        raise AssertionError("Embedding merge is not needed in these routing tests")

    def get_image_features(self, pixels, image_grid_thw=None):
        self.calls += 1
        return pixels + self.offset + np.asarray(image_grid_thw).sum()

    def get_video_features(self, pixels, video_grid_thw=None):
        self.calls += 1
        return pixels + self.offset + np.asarray(video_grid_thw).sum() + 100


def helper(monkeypatch, models, *, compiled=False, cache=True):
    # The fake tower consumes NumPy, not JAX; never allocate device arrays in these cpu_ok tests.
    monkeypatch.setattr(vlm_prefill, "jnp", np)
    return VlmPrefillHelper(
        model_getter=lambda: models[0],
        metadata=SimpleNamespace(data_parallel_size=1),
        compile_vision_encoder=compiled,
        vision_patch_buckets=None,
        max_num_batched_tokens=8,
        max_num_tokens=8,
        vision_cache=VisionEncoderCache(capacity_mb=32) if cache else None,
    )


def state(pixels=None, grid=None, features=(), *, video=False):
    return SimpleNamespace(
        pixel_values=None if video else pixels,
        pixel_values_videos=pixels if video else None,
        image_grid_thw=None if video else grid,
        video_grid_thw=grid if video else None,
        mm_features=list(features),
    )


def test_eager_reuses_actual_encoder_outputs_and_clearing_forces_reencode(monkeypatch):
    model = Tower()
    h = helper(monkeypatch, [model])
    pixels = np.ones((2, 3), dtype=np.float32)
    grid = np.array([[1, 1, 2]])
    request = state(pixels, grid)
    first = h.eager_vision_embed_kwargs(request)
    second = h.eager_vision_embed_kwargs(state(pixels.copy(), grid.copy()))
    np.testing.assert_array_equal(first["image_embeds"], pixels + 4)
    np.testing.assert_array_equal(second["image_embeds"], first["image_embeds"])
    assert model.calls == 1
    assert h.vision_cache.hits == 1
    h.vision_cache.clear()
    h.eager_vision_embed_kwargs(request)
    assert model.calls == 2


def test_cache_separates_all_pixels_grids_modalities_and_model_replacements(monkeypatch):
    models = [Tower()]
    h = helper(monkeypatch, models)
    pixels = np.zeros((1_000_001, 1), dtype=np.float32)
    grid = np.array([[1, 1, 1]])
    h.eager_vision_embed_kwargs(state(pixels, grid))
    modified = pixels.copy()
    modified[-2] = 1
    changed = h.eager_vision_embed_kwargs(state(modified, grid))
    assert changed["image_embeds"][-2, 0] == 4
    assert models[0].calls == 2
    h.eager_vision_embed_kwargs(state(pixels, grid + 1))
    video = h.eager_vision_embed_kwargs(state(pixels, grid, video=True))
    assert video["video_embeds"][0, 0] == 103
    assert models[0].calls == 4
    models[0] = Tower(offset=10)
    replaced = h.eager_vision_embed_kwargs(state(pixels, grid))
    assert replaced["image_embeds"][0, 0] == 13
    assert models[0].calls == 1
    models[0].offset = 20
    h.reset_vision_cache()
    updated = h.eager_vision_embed_kwargs(state(pixels, grid))
    assert updated["image_embeds"][0, 0] == 23


def test_feature_embeddings_survive_pixel_clearing_and_split_in_order(monkeypatch):
    model = Tower()
    h = helper(monkeypatch, [model])
    pixels = np.arange(12, dtype=np.float32).reshape(4, 3)
    grid = np.array([[1, 1, 2], [1, 1, 2]])
    features = [
        MultiModalFeature.from_image(pixels[i : i + 2], grid[i // 2 : i // 2 + 1], PlaceholderRange(i, 2))
        for i in (0, 2)
    ]
    result = h.eager_vision_embed_kwargs(state(pixels, grid, features))
    assert all(f.has_cached_embeddings for f in features)
    for f in features:
        f.clear_pixel_values()
    reused = h.eager_vision_embed_kwargs(state(features=features))
    np.testing.assert_array_equal(reused["image_embeds"], result["image_embeds"])
    np.testing.assert_array_equal(reused["image_grid_thw"], grid)
    assert model.calls == 1
    h.vision_cache.clear()
    # Request-owned features retain their documented reuse contract even after LRU eviction/clear.
    retained = h.eager_vision_embed_kwargs(state(features=features))
    np.testing.assert_array_equal(retained["image_embeds"], result["image_embeds"])
    h.reset_vision_cache()
    with pytest.raises(ValueError, match="earlier model"):
        h.eager_vision_embed_kwargs(state(features=features))


@pytest.mark.parametrize("shared_cache", [True, False])
def test_mixed_same_modality_keeps_unequal_cached_feature_rows(monkeypatch, shared_cache):
    model = Tower()
    h = helper(monkeypatch, [model])
    cached = MultiModalFeature.from_image(
        np.full((2, 3), 1, dtype=np.float32), np.array([[1, 1, 2]]), PlaceholderRange(0, 2)
    )
    encoded = h.eager_vision_embed_kwargs(state(cached.pixel_values, cached.grid_thw, [cached]))
    cached.clear_pixel_values()
    raw = MultiModalFeature.from_image(
        np.full((3, 3), 2, dtype=np.float32), np.array([[1, 1, 3]]), PlaceholderRange(2, 3)
    )
    if not shared_cache:
        h.vision_cache = None
    result = h.eager_vision_embed_kwargs(state(raw.pixel_values, raw.grid_thw, [cached, raw]))
    np.testing.assert_array_equal(result["image_embeds"], np.concatenate([np.full((2, 3), 5), np.full((3, 3), 7)]))
    np.testing.assert_array_equal(result["image_grid_thw"], [[1, 1, 2], [1, 1, 3]])
    np.testing.assert_array_equal(cached.cached_embeddings, encoded["image_embeds"])
    np.testing.assert_array_equal(raw.cached_embeddings, np.full((3, 3), 7))
    raw.clear_pixel_values()
    reused = h.eager_vision_embed_kwargs(state(features=[raw, cached]))
    np.testing.assert_array_equal(reused["image_embeds"], np.concatenate([np.full((3, 3), 7), np.full((2, 3), 5)]))
    assert model.calls == 2


def test_compiled_route_shares_outputs_with_eager_route(monkeypatch):
    models = [Tower()]
    h = helper(monkeypatch, models, compiled=True)
    pixels = np.ones((2, 3), dtype=np.float32)
    grid = np.array([[1, 1, 2]])
    # Replace only the JIT executable, not the cache or request preparation.
    h.image_features_jit = lambda pixels, **kwargs: (models[0].get_image_features(pixels, kwargs["image_grid_thw"]), ())
    compiled = h.compiled_vision_embed_kwargs(state(pixels, grid))
    eager = h.eager_vision_embed_kwargs(state(pixels, grid))
    np.testing.assert_array_equal(compiled["image_embeds"], eager["image_embeds"])
    assert models[0].calls == 1
    assert h.vision_cache.hits == 1


def test_deepstack_models_keep_native_path_instead_of_losing_auxiliary_outputs(monkeypatch):
    model = Tower()
    model.config.vision_config.deepstack_visual_indexes = [1, 2]
    h = helper(monkeypatch, [model], compiled=True)
    request = state(np.ones((2, 3)), np.array([[1, 1, 2]]))
    assert h.eager_vision_embed_kwargs(request) is None
    assert h.compiled_vision_embed_kwargs(request) is None
    assert model.calls == 0


def test_prefill_rebuilds_prompt_from_retained_features_without_reencoding(monkeypatch):
    model = Tower()
    h = helper(monkeypatch, [model])
    monkeypatch.setattr(vlm_prefill, "model_needs_token_types", lambda _: False)
    monkeypatch.setattr(vlm_prefill, "model_uses_mrope", lambda _: True)

    def merge(ids, *, image_embeds, **kwargs):
        # A prompt-dependent merge and position computation; only the tower output is reusable.
        assert kwargs["image_grid_thw"] is not None
        return image_embeds[None] + ids[..., None], SimpleNamespace(position_ids=np.stack([ids] * 3))

    model.compute_embedding_with_info = merge
    pixels = np.ones((2, 3), dtype=np.float32)
    grid = np.array([[1, 1, 2]])
    feature = MultiModalFeature.from_image(pixels, grid)

    def request(ids, *, raw=False):
        return CachedRequestState(
            req_id="test",
            prompt_token_ids=ids,
            sampling_params=SamplingParams(),
            generator=None,
            page_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
            pixel_values=pixels if raw else None,
            image_grid_thw=grid if raw else None,
            mm_features=[feature],
        )

    first = request([1, 2], raw=True)
    h.precompute_prefill(first)
    assert first.vision_processed
    assert first.pixel_values is None
    assert feature.pixel_values is None
    second = request([3, 4])
    h.precompute_prefill(second)
    np.testing.assert_array_equal(second.prefill_inputs_embeds, first.prefill_inputs_embeds + 2)
    np.testing.assert_array_equal(second.prefill_position_ids, [[3, 4]] * 3)
    assert second.vision_processed
    assert model.calls == 1


def test_disabling_shared_cache_keeps_native_eager_path(monkeypatch):
    model = Tower()
    h = helper(monkeypatch, [model], cache=False)
    assert h.eager_vision_embed_kwargs(state(np.ones((2, 3)), np.array([[1, 1, 2]]))) is None
    assert model.calls == 0
