"""Real prefill orchestration with NumPy model doubles, without JAX computation."""

from contextlib import nullcontext
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.multimodal.cache import VisionEncoderCache
from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange
from easydel.inference.esurge.runners import vlm_prefill
from easydel.inference.esurge.runners.states import CachedRequestState
from easydel.inference.esurge.runners.vlm_prefill import VlmPrefillHelper
from easydel.inference.sampling_params import SamplingParams
from easydel.modules._base import qwen_vl
from easydel.modules.deepseek_v41.processing_deepseek_v41 import compute_prefill_embeddings

pytestmark = pytest.mark.cpu_ok


def helper(monkeypatch, model, compiled=False):
    monkeypatch.setattr(vlm_prefill, "jnp", np)
    monkeypatch.setattr(jax, "device_get", lambda value: value)
    monkeypatch.setattr(vlm_prefill, "model_uses_mrope", lambda model: True)
    monkeypatch.setattr(vlm_prefill, "model_needs_token_types", lambda model: isinstance(model, TokenTypeModel))
    return VlmPrefillHelper(
        model_getter=lambda: model,
        metadata=SimpleNamespace(data_parallel_size=1),
        compile_vision_encoder=compiled,
        vision_patch_buckets=None,
        max_num_batched_tokens=16,
        max_num_tokens=16,
        vision_cache=VisionEncoderCache(capacity_mb=1),
    )


def request(ids, features, *, raw=False):
    images = [f for f in features if f.modality == "image"]
    videos = [f for f in features if f.modality == "video"]
    return CachedRequestState(
        req_id="prefill",
        prompt_token_ids=ids,
        sampling_params=SamplingParams(),
        generator=None,
        page_ids=([],),
        num_computed_tokens=0,
        output_token_ids=[],
        pixel_values=np.concatenate([f.pixel_values for f in images]) if raw and images else None,
        image_grid_thw=np.concatenate([f.grid_thw for f in images]) if raw and images else None,
        pixel_values_videos=np.concatenate([f.pixel_values for f in videos]) if raw and videos else None,
        video_grid_thw=np.concatenate([f.grid_thw for f in videos]) if raw and videos else None,
        mm_features=features,
    )


class DeepStackModel:
    def __init__(self):
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(deepstack_visual_indexes=[0, 2], spatial_merge_size=1)
        )
        self.calls = 0
        self.grids = []

    def compute_embedding(self, ids, *, image_embeds=None, video_embeds=None):
        raise AssertionError("Only the metadata-aware embedding pass should run")

    def get_image_features(self, pixels, image_grid_thw=None):
        self.calls += 1
        return ((pixels + 10,), (pixels + 20, pixels + 30))

    def get_video_features(self, pixels, video_grid_thw=None):
        self.calls += 1
        return ((pixels + 100,), (pixels + 200, pixels + 300))

    def compute_embedding_with_info(
        self,
        ids,
        *,
        image_embeds=None,
        video_embeds=None,
        deepstack_image_embeds=None,
        deepstack_video_embeds=None,
        **kwargs,
    ):
        # Independently merge compact per-modality outputs in prompt order.
        self.grids.append(
            {name: np.array(value, copy=True) for name, value in kwargs.items() if name.endswith("grid_thw")}
        )
        embeds = np.repeat(ids[..., None], 2, axis=-1).astype(np.float32)
        mask = (ids == 90) | (ids == 91)
        layers = [np.zeros((int(mask.sum()), 2), dtype=np.float32) for _ in range(2)]
        visual_ids = ids[mask]
        for token, rows, auxiliary in (
            (90, image_embeds, deepstack_image_embeds),
            (91, video_embeds, deepstack_video_embeds),
        ):
            if rows is not None:
                embeds[ids == token] = rows
                for layer, residual in zip(layers, auxiliary, strict=True):
                    layer[visual_ids == token] = residual
        return embeds, SimpleNamespace(
            position_ids=np.stack([ids] * 3), visual_pos_masks=mask, deepstack_visual_embeds=layers
        )


def media(value, modality="image", offset=1):
    factory = MultiModalFeature.from_image if modality == "image" else MultiModalFeature.from_video
    return factory(np.full((1, 2), value, np.float32), np.array([[1, 1, 1]]), PlaceholderRange(offset, 1, modality))


@pytest.mark.parametrize("compiled", [False, True])
def test_deepstack_prefill_reuses_every_layer_and_rebuilds_prompt_masks(monkeypatch, compiled):
    model = DeepStackModel()
    h = helper(monkeypatch, model, compiled)
    if compiled:
        h.image_features_jit = lambda pixels, **kwargs: (pixels + 10, (pixels + 20, pixels + 30))
        h.video_features_jit = lambda pixels, **kwargs: (pixels + 100, (pixels + 200, pixels + 300))
    features = [media(1), media(2, "video", 3)]
    first = request([7, 90, 8, 91], features, raw=True)
    h.precompute_prefill(first)
    np.testing.assert_array_equal(first.prefill_inputs_embeds[:, 0], [7, 11, 8, 102])
    np.testing.assert_array_equal(first.prefill_visual_pos_masks, [False, True, False, True])
    np.testing.assert_array_equal(first.prefill_deepstack_visual_embeds[0][:, 0], [21, 202])
    np.testing.assert_array_equal(first.prefill_deepstack_visual_embeds[1][:, 0], [31, 302])
    assert all(f.pixel_values is None and f.has_cached_embeddings for f in features)
    h.vision_cache.clear()
    second = request([91, 9, 90, 10], features)
    h.precompute_prefill(second)
    np.testing.assert_array_equal(second.prefill_inputs_embeds[:, 0], [102, 9, 11, 10])
    np.testing.assert_array_equal(second.prefill_visual_pos_masks, [True, False, True, False])
    np.testing.assert_array_equal(second.prefill_deepstack_visual_embeds[0][:, 0], [202, 21])
    np.testing.assert_array_equal(second.prefill_deepstack_visual_embeds[1][:, 0], [302, 31])
    np.testing.assert_array_equal(second.prefill_position_ids, [[91, 9, 90, 10]] * 3)
    assert second.vision_processed
    assert model.calls == (0 if compiled else 2)


def test_deepstack_cache_shared_across_distinct_requests_and_mixed_raw_cached_media(monkeypatch):
    model = DeepStackModel()
    h = helper(monkeypatch, model, compiled=True)
    # Deliberately use the eager path on the first request; both routes share encoder outputs.
    h.compile_vision_encoder = False
    first = request([90], [media(4)], raw=True)
    h.precompute_prefill(first)
    h.compile_vision_encoder = True
    h.image_features_jit = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Cache miss"))
    fresh = request([90], [media(4)], raw=True)
    h.precompute_prefill(fresh)
    np.testing.assert_array_equal(fresh.prefill_deepstack_visual_embeds[1], [[34, 34]])
    assert model.calls == 1
    # A cached-only image and a raw video must both be assembled, not just the raw modality.
    video = media(6, "video", 2)
    mixed = request([90, 7, 91], [first.mm_features[0], video])
    mixed.pixel_values_videos = video.pixel_values
    mixed.video_grid_thw = video.grid_thw
    h.precompute_prefill(mixed)
    np.testing.assert_array_equal(mixed.prefill_inputs_embeds[:, 0], [14, 7, 106])
    np.testing.assert_array_equal(mixed.prefill_deepstack_visual_embeds[0][:, 0], [24, 206])
    assert model.calls == 2


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_same_modality_mixed_rows_preserve_cached_media_and_all_residuals(monkeypatch, compiled, modality):
    model = DeepStackModel()
    h = helper(monkeypatch, model, compiled)
    if compiled:
        h.image_features_jit = lambda pixels, **kwargs: (pixels + 10, (pixels + 20, pixels + 30))
        h.video_features_jit = lambda pixels, **kwargs: (pixels + 100, (pixels + 200, pixels + 300))
    token = 90 if modality == "image" else 91
    base, layer0, layer1 = (10, 20, 30) if modality == "image" else (100, 200, 300)
    retained = [media(value, modality, offset) for value, offset in ((1, 0), (3, 2))]
    initial = request([token, token], retained, raw=True)
    h.precompute_prefill(initial)
    before = [f.cached_embeddings.copy() for f in retained]
    h.vision_cache.clear()
    raw = media(2, modality, 1)
    mixed = request([token, token, token], [retained[0], raw, retained[1]])
    pixel_name, grid_name = (
        ("pixel_values", "image_grid_thw") if modality == "image" else ("pixel_values_videos", "video_grid_thw")
    )
    # This is the real request representation: aggregate arrays include only raw media.
    setattr(mixed, pixel_name, raw.pixel_values)
    setattr(mixed, grid_name, raw.grid_thw)
    h.precompute_prefill(mixed)
    np.testing.assert_array_equal(mixed.prefill_inputs_embeds[:, 0], np.array([1, 2, 3]) + base)
    np.testing.assert_array_equal(mixed.prefill_deepstack_visual_embeds[0][:, 0], np.array([1, 2, 3]) + layer0)
    np.testing.assert_array_equal(mixed.prefill_deepstack_visual_embeds[1][:, 0], np.array([1, 2, 3]) + layer1)
    np.testing.assert_array_equal(model.grids[-1][grid_name], [[1, 1, 1]] * 3)
    for feature, original in zip(retained, before, strict=True):
        np.testing.assert_array_equal(feature.cached_embeddings, original)
    assert raw.has_cached_embeddings and raw.pixel_values is None
    h.vision_cache.clear()
    reordered = request([token, 7, token, token], [retained[1], raw, retained[0]])
    h.precompute_prefill(reordered)
    np.testing.assert_array_equal(reordered.prefill_inputs_embeds[:, 0], [3 + base, 7, 2 + base, 1 + base])
    np.testing.assert_array_equal(reordered.prefill_deepstack_visual_embeds[1][:, 0], np.array([3, 2, 1]) + layer1)
    np.testing.assert_array_equal(reordered.prefill_visual_pos_masks, [True, False, True, True])
    np.testing.assert_array_equal(reordered.prefill_position_ids, [[token, 7, token, token]] * 3)
    assert model.calls == (1 if compiled else 2)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("grid_shape", [(3,), (1, 3)])
def test_mixed_feature_grids_reach_real_qwen_encoder_as_triples(monkeypatch, compiled, modality, grid_shape):
    # Execute Qwen's actual host validation/splitting; only the vision tower and
    # array operations are NumPy doubles, so this is not a model-numerics test.
    monkeypatch.setattr(qwen_vl, "jnp", np)

    class VisionTower:
        spatial_merge_size = 1

        def __init__(self):
            self.grids = []

        def get_dtype(self):
            return np.float32

        def __call__(self, pixels, *, grid_thw, max_grid_size):
            self.grids.append(np.array(grid_thw, copy=True))
            return pixels + 10, [pixels + 20, pixels + 30]

    class QwenGridModel(DeepStackModel):
        get_image_features = qwen_vl.QwenVLMultimodalMixin.get_image_features
        get_video_features = qwen_vl.QwenVLMultimodalMixin.get_video_features

    model = QwenGridModel()
    model.visual = VisionTower()
    h = helper(monkeypatch, model)
    token = 90 if modality == "image" else 91
    pixel_name, grid_name = (
        ("pixel_values", "image_grid_thw") if modality == "image" else ("pixel_values_videos", "video_grid_thw")
    )
    cached = media(1, modality, 0)
    cached.grid_thw = np.array([1, 1, 1])
    initial = request([token], [cached])
    # Real request aggregates are [N, 3], even when feature metadata is [3].
    setattr(initial, pixel_name, cached.pixel_values)
    setattr(initial, grid_name, cached.grid_thw.reshape(1, 3))
    h.precompute_prefill(initial)
    assert cached.pixel_values is None and cached.has_cached_embeddings
    h.vision_cache.clear()
    h.compile_vision_encoder = compiled
    raw = media(2, modality, 2)
    raw.grid_thw = raw.grid_thw.reshape(grid_shape)
    mixed = request([token, 7, token], [cached, raw])
    setattr(mixed, pixel_name, raw.pixel_values)
    setattr(mixed, grid_name, raw.grid_thw.reshape(1, 3))
    h.precompute_prefill(mixed)
    np.testing.assert_array_equal(mixed.prefill_inputs_embeds[:, 0], [11, 7, 12])
    np.testing.assert_array_equal(mixed.prefill_deepstack_visual_embeds[0][:, 0], [21, 22])
    np.testing.assert_array_equal(mixed.prefill_deepstack_visual_embeds[1][:, 0], [31, 32])
    np.testing.assert_array_equal(model.grids[-1][grid_name], [[1, 1, 1], [1, 1, 1]])
    assert mixed.vision_processed and not h.vision_jit_disabled
    assert len(model.visual.grids) == 2
    for grid in model.visual.grids:
        np.testing.assert_array_equal(grid, [[1, 1, 1]])
    assert cached.grid_thw is None and raw.grid_thw is None
    assert cached.pixel_values is None and raw.pixel_values is None
    h.vision_cache.clear()
    reused = request([token, token], [raw, cached])
    h.precompute_prefill(reused)
    np.testing.assert_array_equal(reused.prefill_inputs_embeds[:, 0], [12, 11])
    np.testing.assert_array_equal(reused.prefill_deepstack_visual_embeds[1][:, 0], [32, 31])
    assert len(model.visual.grids) == 2


def test_same_modality_mixed_rows_reject_stale_cached_weights_and_grid(monkeypatch):
    model = DeepStackModel()
    h = helper(monkeypatch, model, compiled=True)
    cached = media(1)
    h.compile_vision_encoder = False
    h.precompute_prefill(request([90], [cached], raw=True))
    h.compile_vision_encoder = True
    raw = media(2, offset=2)
    mixed = request([90, 7, 90], [cached, raw])
    mixed.pixel_values, mixed.image_grid_thw = raw.pixel_values, raw.grid_thw
    cached.grid_thw = np.array([[1, 1, 2]])
    with pytest.raises(RuntimeError, match="different grid"):
        h.precompute_prefill(mixed)
    assert not mixed.vision_processed and not raw.has_cached_embeddings
    cached.grid_thw = None
    h.reset_vision_cache()
    with pytest.raises(RuntimeError, match="earlier model"):
        h.precompute_prefill(mixed)
    np.testing.assert_array_equal(cached.cached_embeddings, [[11, 11]])
    assert model.calls == 1


def test_deepstack_cached_only_rejects_missing_auxiliaries_and_stale_weights(monkeypatch):
    model = DeepStackModel()
    h = helper(monkeypatch, model)
    feature = media(1)
    h.precompute_prefill(request([90], [feature], raw=True))
    layers = feature._esurge_deepstack_embeddings
    feature._esurge_deepstack_embeddings = ()
    with pytest.raises(RuntimeError, match="missing DeepStack residuals"):
        h.precompute_prefill(request([90], [feature]))
    feature._esurge_deepstack_embeddings = layers
    h.reset_vision_cache()
    with pytest.raises(RuntimeError, match="earlier model"):
        h.precompute_prefill(request([90], [feature]))


class TokenTypeModel:
    def __init__(self):
        self.config = SimpleNamespace(vision_params=lambda: {"downsample_ratio": 3})
        self.mesh = nullcontext()
        self.dtype = np.float32
        self.base_model = self
        self.calls = 0

    def get_image_features(self, pixels, image_grid):
        self.calls += 1
        return np.full((1, 1, 2), pixels.flat[0], dtype=np.float32)

    def compute_embedding(self, ids):
        return np.repeat(ids[..., None], 2, axis=-1).astype(np.float32)

    def merge_image_embeddings(self, embeds, types, rows):
        result = embeds.copy()
        result[types == 1] = rows.reshape(-1, 2)
        return result

    def esurge_compute_prefill_embeddings(self, ids, features, encoder_cache_provider=None):
        return compute_prefill_embeddings(self, ids, features, encoder_cache_provider=encoder_cache_provider)


def token_media(value=5, offset=1):
    return MultiModalFeature.from_image(
        np.full((9, 3, 1, 1), value, np.float32), np.array([[1, 3, 3]]), PlaceholderRange(offset, 4)
    )


def test_token_type_real_hook_rebuilds_cached_only_prefill_and_uses_shared_provider(monkeypatch):
    model = TokenTypeModel()
    h = helper(monkeypatch, model)
    # The model hook imports jax.numpy locally; all numeric work stays in NumPy doubles.
    monkeypatch.setattr(jax, "numpy", np)
    feature = token_media()
    first = request([7, 99, 99, 99, 99], [feature])
    h.precompute_prefill(first)
    np.testing.assert_array_equal(first.token_types, [-1, 0, 1, 2, 3])
    np.testing.assert_array_equal(first.prefill_inputs_embeds[:, 0], [7, 99, 5, 99, 99])
    fresh = request([8, 99, 99, 99, 99], [token_media()])
    h.precompute_prefill(fresh)
    assert model.calls == 1
    assert h.vision_cache.hits == 1
    feature.clear_pixel_values()
    feature.placeholder_range = PlaceholderRange(2, 4)
    h.vision_cache.clear()
    second = request([9, 10, 99, 99, 99, 99], [feature])
    h.precompute_prefill(second)
    np.testing.assert_array_equal(second.token_types, [-1, -1, 0, 1, 2, 3])
    np.testing.assert_array_equal(second.prefill_inputs_embeds[:, 0], [9, 10, 99, 5, 99, 99])
    assert second.vision_processed
    assert model.calls == 1
    h.reset_vision_cache()
    with pytest.raises(RuntimeError, match="earlier model"):
        h.precompute_prefill(request([9, 10, 99, 99, 99, 99], [feature]))


def test_token_type_provider_assembles_mixed_images_and_validates_cached_grid(monkeypatch):
    model = TokenTypeModel()
    h = helper(monkeypatch, model)
    monkeypatch.setattr(jax, "numpy", np)
    cached = token_media(5)
    h.precompute_prefill(request([7, 99, 99, 99, 99], [cached]))
    cached.clear_pixel_values()
    raw = token_media(6, offset=5)
    mixed = request([7] + [99] * 8, [cached, raw])
    h.precompute_prefill(mixed)
    np.testing.assert_array_equal(mixed.token_types, [-1, 0, 1, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(mixed.prefill_inputs_embeds[:, 0], [7, 99, 5, 99, 99, 99, 6, 99, 99])
    assert model.calls == 2
    cached.grid_thw = np.array([[1, 3, 2]])
    bad = request([7, 99, 99, 99, 99], [cached])
    with pytest.raises(RuntimeError, match="different grid"):
        h.precompute_prefill(bad)
    assert not bad.vision_processed and bad.prefill_inputs_embeds is None
    assert model.calls == 2


def test_token_type_missing_encoder_rows_fails_instead_of_marking_processed(monkeypatch):
    model = TokenTypeModel()
    h = helper(monkeypatch, model)
    monkeypatch.setattr(jax, "numpy", np)
    feature = token_media()
    feature.pixel_values = None
    req = request([7, 99, 99, 99, 99], [feature])
    with pytest.raises(RuntimeError, match="neither cached encoder rows nor raw pixels"):
        h.precompute_prefill(req)
    assert not req.vision_processed
    assert req.prefill_inputs_embeds is None
