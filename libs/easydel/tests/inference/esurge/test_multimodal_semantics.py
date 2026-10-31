"""Public host-only multimodal input contracts and independent layout references."""

from collections import UserDict
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.core import utils as cache_utils
from easydel.inference.esurge.core.utils import hash_request_tokens
from easydel.inference.esurge.multimodal.manager import MultiModalManager
from easydel.inference.esurge.multimodal.types import BatchedMultiModalInputs, MultiModalFeature
from easydel.inference.esurge.request import EngineRequest

from .test_mm_manager import _image

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("modality", ["image", "video"])
def test_equal_pixels_with_different_grids_cannot_share_prefix_pages(modality, monkeypatch):
    factory = getattr(MultiModalFeature, f"from_{modality}")
    pixels = np.zeros((16, 3), dtype=np.float32)
    grids = [np.array([[1, 2, 8]]), np.array([[1, 4, 4]])]
    features = [factory(pixels, grid) for grid in grids]
    assert features[0].mm_hash != features[1].mm_hash
    assert factory(pixels).mm_hash not in {feature.mm_hash for feature in features}
    assert factory(pixels, grids[0].reshape(3).astype(np.int32)).mm_hash == features[0].mm_hash
    monkeypatch.setattr(cache_utils, "none_hash", 0, raising=False)
    requests = [
        EngineRequest(str(i), [1, 2, 3, 4], None, 2, mm_features=[feature]) for i, feature in enumerate(features)
    ]
    first, second = [hash_request_tokens(hash, 2, request) for request in requests]
    assert len(first) == len(second) == 2
    assert all(a != b for a, b in zip(first, second, strict=True))
    features[0].clear_pixel_values()
    assert hash_request_tokens(hash, 2, requests[0]) == first


@pytest.mark.parametrize("modality,shape", [("image", (3, 4, 4)), ("video", (2, 3, 4, 4))])
@pytest.mark.parametrize("batched", [False, True])
def test_standard_batching_preserves_media_channels_frames_and_values(modality, shape, batched):
    arrays = [np.arange(np.prod(shape)).reshape(shape) + i * 1000 for i in range(2)]
    factory = getattr(MultiModalFeature, f"from_{modality}")
    features = [factory(array[None] if batched else array, request_idx=i) for i, array in enumerate(arrays)]
    batch = BatchedMultiModalInputs.from_features(features)
    actual = batch.pixel_values if modality == "image" else batch.pixel_values_videos
    np.testing.assert_array_equal(actual, np.stack(arrays))
    assert getattr(batch, f"num_{modality}s") == 2
    assert getattr(batch, f"get_request_{modality}_count")(1) == 1
    assert all(feature.pixel_values.ndim == len(shape) + int(batched) for feature in features)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_flat_patch_batching_concatenates_patches_and_normalizes_each_grid(modality):
    factory = getattr(MultiModalFeature, f"from_{modality}")
    arrays = [np.zeros((8, 3)), np.ones((16, 3))]
    features = [factory(arrays[0], np.array([1, 2, 4])), factory(arrays[1], np.array([[2, 2, 4]]))]
    batch = BatchedMultiModalInputs.from_features(features)
    pixels = batch.pixel_values if modality == "image" else batch.pixel_values_videos
    grids = batch.image_grid_thw if modality == "image" else batch.video_grid_thw
    np.testing.assert_array_equal(pixels, np.concatenate(arrays))
    np.testing.assert_array_equal(grids, [[1, 2, 4], [2, 2, 4]])


def _manager(*, patch=1, temporal=1, resolution=(4, 4), processor=None):
    vision = SimpleNamespace(patch_size=patch, temporal_patch_size=temporal, spatial_merge_size=2)
    model = SimpleNamespace(supports_flat_patch_inputs=True, config=SimpleNamespace(vision_config=vision))
    return MultiModalManager(processor=processor, model=model, resolution_buckets=[resolution])


def _reference_patches(frames, patch, temporal, merge):
    """Direct scalar-coordinate reference, deliberately not reshape/transposition based."""
    t, h, w, channels = frames.shape
    target_t = ((t + temporal - 1) // temporal) * temporal
    target_h = ((h + patch * merge - 1) // (patch * merge)) * patch * merge
    target_w = ((w + patch * merge - 1) // (patch * merge)) * patch * merge
    rows = []
    for time in range(0, target_t, temporal):
        for block_h in range(0, target_h, patch * merge):
            for block_w in range(0, target_w, patch * merge):
                for local_h in range(merge):
                    for local_w in range(merge):
                        rows.append(
                            [
                                frames[
                                    min(time + dt, t - 1),
                                    min(block_h + local_h * patch + dh, h - 1),
                                    min(block_w + local_w * patch + dw, w - 1),
                                    c,
                                ]
                                for c in range(channels)
                                for dt in range(temporal)
                                for dh in range(patch)
                                for dw in range(patch)
                            ]
                        )
    return np.asarray(rows), [[target_t // temporal, target_h // patch, target_w // patch]]


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("patch,temporal,side", [(1, 1, 4), (2, 2, 8), (1, 2, 4)])
def test_fallback_media_uses_merge_blocks_and_channel_temporal_patch_order(modality, patch, temporal, side):
    manager = _manager(patch=patch, temporal=temporal, resolution=(side, side))
    if modality == "image":
        image = _image(size=(side, side))
        image.pixels[:] = np.arange(side * side * 3, dtype=np.uint8).reshape(side, side, 3)
        frames = image.pixels[None]
        actual, grids = manager.process_images([image])
    else:
        frames = np.arange(3 * side * side * 3, dtype=np.uint8).reshape(3, side, side, 3)
        actual, grids = manager.process_videos([frames])
    # CLIP normalization, independent constants and no production helper calls.
    normalized = (
        frames.astype(np.float32) / 255 - np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    ) / np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    expected, expected_grids = _reference_patches(normalized, patch, temporal, 2)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(grids, expected_grids)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_processor_odd_grid_padding_emits_merge_order_without_reordering_aligned_outputs(modality):
    class Processor:
        def __call__(self, **kwargs):
            pixels = "pixel_values" if modality == "image" else "pixel_values_videos"
            grid = "image_grid_thw" if modality == "image" else "video_grid_thw"
            # Raster rows for an incomplete grid, followed by an already merged grid.
            return UserDict({pixels: np.arange(25, dtype=np.float32)[:, None], grid: np.array([[1, 3, 3], [1, 4, 4]])})

    manager = _manager(processor=Processor())
    media = [_image(size=(4, 4))] if modality == "image" else [np.zeros((1, 4, 4, 3), dtype=np.uint8)]
    actual, grids = getattr(manager, f"process_{modality}s")(media)
    expected, _ = _reference_patches(np.arange(9).reshape(1, 3, 3, 1), 1, 1, 2)
    np.testing.assert_array_equal(actual, np.concatenate([expected, np.arange(9, 25)[:, None]]))
    np.testing.assert_array_equal(grids, [[1, 4, 4], [1, 4, 4]])
