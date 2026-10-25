"""Token-type prefill cache orchestration; NumPy doubles perform no JAX computation."""

from contextlib import nullcontext
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange
from easydel.modules.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForConditionalGeneration
from easydel.modules.deepseek_v41.processing_deepseek_v41 import compute_prefill_embeddings

pytestmark = pytest.mark.cpu_ok


@pytest.fixture
def model(monkeypatch):
    """Replace only array transport and model computation, keeping real hook orchestration."""
    monkeypatch.setattr(
        jax, "numpy", SimpleNamespace(asarray=np.asarray, concatenate=np.concatenate, float32=np.float32)
    )
    monkeypatch.setattr(jax, "device_get", lambda value: value)

    def encode(patches, image_grid):
        # Distinct image content produces distinct aligned rows.
        return np.full((1, (image_grid[0] // 3) * (image_grid[1] // 3), 2), patches.flat[0], dtype=np.float32)

    def merge(embeds, types, rows):
        result = embeds.copy()
        result[types == 1] = rows.reshape(-1, 2)
        for token_type, value in ((0, -10), (3, -20), (2, -30)):
            result[types == token_type] = value
        return result

    return SimpleNamespace(
        config=SimpleNamespace(vision_params=lambda: {"downsample_ratio": 3}),
        mesh=nullcontext(),
        dtype=np.float32,
        compute_embedding=lambda ids: np.repeat(ids[..., None], 2, axis=-1).astype(np.float32),
        base_model=SimpleNamespace(get_image_features=encode, merge_image_embeddings=merge),
    )


def image(value=5, offset=1):
    return MultiModalFeature.from_image(
        np.full((9, 3, 1, 1), value, dtype=np.float32),
        np.array([[1, 3, 3]], dtype=np.int32),
        PlaceholderRange(offset=offset, length=4, modality="image"),
    )


def call(model, ids, features, provider=None):
    return DeepseekV41ForConditionalGeneration.esurge_compute_prefill_embeddings(
        model, ids, features, encoder_cache_provider=provider
    )


def test_feature_cache_survives_pixel_and_grid_clearing_and_new_prompt(model):
    feature = image()
    embeds, types = call(model, [7, 99, 99, 99, 99, 8], [feature])
    np.testing.assert_array_equal(types, [-1, 0, 1, 2, 3, -1])
    np.testing.assert_array_equal(embeds[:, 0], [7, -10, 5, -30, -20, 8])
    assert feature.has_cached_embeddings
    feature.clear_pixel_values()
    feature.placeholder_range = PlaceholderRange(offset=2, length=4, modality="image")
    # A new prompt must get fresh text embeddings and its own image-slot positions.
    embeds, types = call(model, [10, 11, 99, 99, 99, 99, 12], [feature])
    np.testing.assert_array_equal(types, [-1, -1, 0, 1, 2, 3, -1])
    np.testing.assert_array_equal(embeds[:, 0], [10, 11, -10, 5, -30, -20, 12])
    assert embeds.dtype == np.float32


def test_shared_provider_reuses_encoder_rows_across_distinct_features(model):
    cache = {}

    def provider(feature, encode):
        if feature.mm_hash not in cache:
            cache[feature.mm_hash] = encode()
        return cache[feature.mm_hash]

    first, _ = call(model, [1, 99, 99, 99, 99], [image()], provider)
    second_feature = image(offset=2)
    # Cache hit must not even construct patches from missing pixels.
    second_feature.pixel_values = None
    second, types = call(model, [2, 3, 99, 99, 99, 99], [second_feature], provider)
    np.testing.assert_array_equal(first[:, 0], [1, -10, 5, -30, -20])
    np.testing.assert_array_equal(second[:, 0], [2, 3, -10, 5, -30, -20])
    np.testing.assert_array_equal(types, [-1, -1, 0, 1, 2, 3])


def test_cached_features_are_merged_in_placeholder_order(model):
    first, second = image(5), image(9, offset=5)
    first.set_cached_embeddings(np.array([[5, 5]], dtype=np.float32))
    second.set_cached_embeddings(np.array([[9, 9]], dtype=np.float32))
    first.pixel_values = second.pixel_values = None
    embeds, _ = call(model, [7] + [99] * 8, [second, first])
    np.testing.assert_array_equal(embeds[:, 0], [7, -10, 5, -30, -20, -10, 9, -30, -20])


def test_provider_can_reject_stale_feature_cache(model):
    feature = image()
    feature.set_cached_embeddings(np.array([[5, 5]], dtype=np.float32))

    def stale_provider(feature, encode):
        raise ValueError("encoder weight version changed")

    with pytest.raises(ValueError, match="weight version changed"):
        call(model, [7] + [99] * 4, [feature], stale_provider)


@pytest.mark.parametrize("use_provider", [False, True])
def test_missing_pixels_without_encoder_result_is_an_error(model, use_provider):
    feature = image()
    feature.pixel_values = None
    provider = (lambda feature, encode: encode()) if use_provider else None
    with pytest.raises(ValueError, match="neither cached encoder rows nor raw pixels"):
        call(model, [7] + [99] * 4, [feature], provider)


def test_cached_row_count_must_match_image_grid(model):
    feature = image()
    feature.set_cached_embeddings(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="must provide 1 rows"):
        call(model, [7] + [99] * 4, [feature])


def test_legacy_three_argument_helper_and_empty_features(model):
    embeds, types = compute_prefill_embeddings(model, [7, 8], [])
    np.testing.assert_array_equal(embeds, [[7, 7], [8, 8]])
    np.testing.assert_array_equal(types, [-1, -1])
