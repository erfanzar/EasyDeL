"""Multimodal content hashes must cover the whole buffer (they are prefix-cache correctness keys)."""

import numpy as np
import pytest
from easydel.inference.esurge.multimodal.cache import VisionEncoderCache
from easydel.inference.esurge.multimodal.types import MultiModalFeature, compute_content_hash

pytestmark = pytest.mark.cpu_ok

_LARGE = 1_100_000  # > 1M elements, where a strided-sample hash would skip data


def _pair() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((_LARGE,), dtype=np.float32)
    b = a.copy()
    b[1] = 1.0  # index 1 is never visited by a ``flat[::100]`` sample
    return a, b


def test_large_arrays_differing_off_the_old_sample_get_distinct_feature_hashes():
    a, b = _pair()
    assert MultiModalFeature.from_image(a).mm_hash != MultiModalFeature.from_image(b).mm_hash
    assert MultiModalFeature.from_video(a).mm_hash != MultiModalFeature.from_video(b).mm_hash


def test_encoder_cache_key_matches_feature_hash_and_is_exhaustive():
    a, b = _pair()
    cache = VisionEncoderCache(capacity_mb=1)
    assert cache.compute_hash(a) != cache.compute_hash(b)
    assert cache.compute_hash(a) == MultiModalFeature.from_image(a).mm_hash


def test_hash_is_deterministic_and_covers_shape_and_dtype():
    x = np.arange(12, dtype=np.float32)
    assert compute_content_hash(x) == compute_content_hash(x.copy())
    assert compute_content_hash(x) != compute_content_hash(x.reshape(3, 4))
    assert compute_content_hash(x) != compute_content_hash(x.astype(np.float64))
    assert compute_content_hash(x[::2]) == compute_content_hash(np.ascontiguousarray(x[::2]))
    assert compute_content_hash(np.array(1)) != compute_content_hash(np.array([1]))
    assert compute_content_hash(np.empty((0, 2))) != compute_content_hash(np.empty((0, 3)))
    matrix = x.reshape(3, 4)
    assert compute_content_hash(matrix) == compute_content_hash(np.asfortranarray(matrix))
