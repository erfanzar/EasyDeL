"""MultiModalManager preprocessing: BatchFeature outputs, error surfacing, caching, declared flat-patch inputs."""

from collections import UserDict
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.multimodal.manager import MultiModalManager
from easydel.inference.esurge.multimodal.types import BatchedMultiModalInputs, MultiModalFeature

pytestmark = pytest.mark.cpu_ok


class _Processor:
    """Stub HF processor returning a ``UserDict`` (like ``BatchFeature``) or raising."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = 0

    def __call__(self, images=None, videos=None, text=None, return_tensors=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if images is not None:
            return UserDict({"pixel_values": np.ones((len(images), 3, 4, 4), dtype=np.float32)})
        return UserDict({"pixel_values_videos": np.ones((len(videos), 2, 3, 4, 4), dtype=np.float32)})


class _Image:
    """RGB image protocol for processor-only tests, which do not need Pillow."""

    def __init__(self, color, size):
        self.size = size
        self.pixels = np.empty((size[1], size[0], 3), dtype=np.uint8)
        self.pixels[:] = color

    def convert(self, mode):
        assert mode == "RGB"
        return self

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.pixels, dtype=dtype)


def _image(color=(10, 20, 30), size=(32, 32)):
    return _Image(color, size)


def test_batchfeature_like_processor_output_is_used():
    processor = _Processor()
    manager = MultiModalManager(processor=processor, model=None)
    pixel_values, grid = manager.process_images([_image()])
    assert grid is None
    assert pixel_values.shape == (1, 3, 4, 4)
    video = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    pixel_values_videos, _ = manager.process_videos([video])
    assert pixel_values_videos.shape == (1, 2, 3, 4, 4)


def test_processor_failure_is_reported_not_hidden():
    manager = MultiModalManager(processor=_Processor(error=RuntimeError("boom-in-processor")), model=None)
    with pytest.raises(ValueError, match="boom-in-processor"):
        manager.process_images([_image()])


def test_repeated_media_is_served_from_the_preprocessing_cache():
    processor = _Processor()
    manager = MultiModalManager(processor=processor, model=None)
    first = manager.process_images([_image()])
    second = manager.process_images([_image()])
    assert processor.calls == 1
    assert np.array_equal(first[0], second[0])
    assert manager.get_cache_stats()["hits"] == 1
    manager.process_images([_image(color=(200, 1, 1))])
    assert processor.calls == 2


def test_features_can_reuse_already_processed_images():
    processor = _Processor()
    manager = MultiModalManager(processor=processor, model=None, enable_cache=False)
    images = [_image()]
    processed = manager.process_images(images)
    features = manager.process_images_to_features(images, processed=processed)
    assert processor.calls == 1
    assert len(features) == 1 and features[0].pixel_values.shape == (1, 3, 4, 4)
    batched = BatchedMultiModalInputs.from_features(features)
    assert batched.num_images == 1 and batched.pixel_values.shape == (1, 3, 4, 4)


def test_resize_to_bucket_uses_configured_buckets():
    image = pytest.importorskip("PIL.Image")
    manager = MultiModalManager(processor=None, model=None, resolution_buckets=[(32, 32), (64, 64)])
    assert manager.resize_to_bucket(image.new("RGB", (40, 40))).size == (32, 32)
    assert manager.resize_to_bucket(image.new("RGB", (60, 60))).size == (64, 64)


def _flat_patch_model(**declared):
    vision = SimpleNamespace(patch_size=14, temporal_patch_size=2, spatial_merge_size=1)
    return SimpleNamespace(config=SimpleNamespace(model_type="my_new_vl", vision_config=vision), **declared)


def test_model_declared_flat_patch_inputs_enable_fallback_preprocessing():
    broken = _Processor(error=RuntimeError("no processor support"))
    manager = MultiModalManager(
        processor=broken,
        model=_flat_patch_model(esurge_flat_patch_inputs=True),
        resolution_buckets=[(28, 28)],
    )
    pixel_values, grid = manager.process_images([_image(size=(28, 28))])
    assert pixel_values.ndim == 2 and pixel_values.shape[1] == 3 * 2 * 14 * 14
    assert grid.shape == (1, 3)
    assert pixel_values.shape[0] == int(grid[0].prod())


def test_undeclared_unknown_model_does_not_take_flat_patch_fallback():
    manager = MultiModalManager(
        processor=_Processor(error=RuntimeError("x")), model=_flat_patch_model(), resolution_buckets=[(28, 28)]
    )
    with pytest.raises(ValueError):
        manager.process_images([_image(size=(28, 28))])


def test_video_processor_failure_preserves_the_cause():
    failure = RuntimeError("invalid-video")
    manager = MultiModalManager(processor=_Processor(error=failure))
    with pytest.raises(ValueError, match="invalid-video") as caught:
        manager.process_videos([np.zeros((2, 32, 32, 3), dtype=np.uint8)])
    assert caught.value.__cause__ is failure


def test_features_can_reuse_already_processed_videos():
    processor = _Processor()
    manager = MultiModalManager(processor=processor, enable_cache=False)
    videos = [np.zeros((2, 32, 32, 3), dtype=np.uint8)]
    processed = manager.process_videos(videos)
    features = manager.process_videos_to_features(videos, processed=processed, request_idx=2)
    assert processor.calls == 1
    assert len(features) == 1 and features[0].request_idx == 2
    np.testing.assert_array_equal(features[0].pixel_values, processed[0])


def test_preprocessing_cache_can_be_cleared_and_disabled():
    processor = _Processor()
    manager = MultiModalManager(processor=processor)
    manager.process_images([_image()])
    manager.clear_cache()
    assert manager.get_cache_stats()["num_entries"] == 0
    manager.process_images([_image()])
    assert processor.calls == 2
    uncached = MultiModalManager(processor=processor, enable_cache=False)
    uncached.process_images([_image()])
    uncached.process_images([_image()])
    assert processor.calls == 4 and uncached.get_cache_stats() is None


def test_documented_batch_api_routes_images_and_videos_to_requests():
    image = MultiModalFeature.from_image(np.ones((2, 4)), np.array([[1, 1, 2]]), request_idx=3)
    video = MultiModalFeature.from_video(np.ones((6, 4)), np.array([[3, 1, 2]]), request_idx=7)
    batch = BatchedMultiModalInputs.from_features([image, video])
    assert batch.has_images and batch.has_videos and batch.has_vision
    assert batch.num_images == batch.num_videos == 1
    assert batch.get_request_image_count(3) == batch.get_request_video_count(7) == 1
    assert batch.get_request_image_count(7) == batch.get_request_video_count(3) == 0
    np.testing.assert_array_equal(batch.pixel_values, image.pixel_values)
    np.testing.assert_array_equal(batch.pixel_values_videos, video.pixel_values)
    assert not BatchedMultiModalInputs.empty().has_vision
