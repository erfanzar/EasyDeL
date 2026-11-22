"""Numerical shared Qwen-VL regressions (accelerator only)."""

from types import SimpleNamespace

import jax
import numpy as np
from easydel.modules._base import DeepStackVisualMixin, QwenVLMultimodalMixin
from jax import numpy as jnp


def test_deepstack_interleaving_is_batch_ordered_and_jittable():
    image_mask = jnp.array([[False, True, False, False], [True, False, False, False]])
    video_mask = jnp.array([[True, False, False, False], [False, False, True, False]])
    images = [jnp.array([[11.0, 12.0], [21.0, 22.0]])]
    videos = [jnp.array([[31.0, 32.0], [41.0, 42.0]])]
    merge = jax.jit(DeepStackVisualMixin.merge_deepstack_features)
    mask, tables = merge(image_mask, video_mask, images, videos)
    np.testing.assert_array_equal(mask, [[True, True, False, False], [True, False, True, False]])
    np.testing.assert_array_equal(tables[0], [[31, 32], [11, 12], [21, 22], [41, 42]])


def test_deepstack_injection_preserves_text_dtype_and_empty_table():
    mixin = DeepStackVisualMixin()
    hidden = jnp.ones((2, 3, 2), dtype=jnp.bfloat16)
    mask = jnp.array([[True, False, False], [False, True, False]])
    residuals = jnp.array([[2.0, 3.0], [4.0, 5.0]], dtype=jnp.float32)
    result = jax.jit(mixin._deepstack_process)(hidden, mask, residuals)
    expected = np.ones((2, 3, 2), dtype=np.float32)
    expected[0, 0] = [3, 4]
    expected[1, 1] = [5, 6]
    assert result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(result.astype(jnp.float32), expected)
    empty = mixin._deepstack_process(hidden, jnp.zeros((2, 3), dtype=bool), jnp.empty((0, 2)))
    np.testing.assert_array_equal(empty, hidden)


class _TextBase:
    def compute_embedding(self, input_ids):
        return jnp.broadcast_to(jnp.array([1.0, 2.0]), (*input_ids.shape, 2))


class _Model(QwenVLMultimodalMixin, _TextBase):
    config = SimpleNamespace(image_token_id=10, video_token_id=20)

    def get_rope_index(self, input_ids, **kwargs):
        return jnp.broadcast_to(jnp.arange(input_ids.shape[-1]), input_ids.shape), jnp.zeros((input_ids.shape[0], 1))


def test_precomputed_features_take_priority_and_preserve_text_positions():
    # There is no vision tower on this harness: honoring precomputed features is
    # required even when both pixel arguments are also supplied.
    model = _Model()
    ids = jnp.array([[20, 0, 10], [10, 20, 0]])
    embeddings, info = model.compute_embedding_with_info(
        ids,
        pixel_values=object(),
        pixel_values_videos=object(),
        image_embeds=jnp.array([[3.0, 4.0], [5.0, 6.0]]),
        video_embeds=jnp.array([[7.0, 8.0], [9.0, 10.0]]),
    )
    np.testing.assert_array_equal(embeddings, [[[7, 8], [1, 2], [3, 4]], [[5, 6], [9, 10], [1, 2]]])
    np.testing.assert_array_equal(info.position_ids, [[0, 1, 2], [0, 1, 2]])
    assert info.deepstack_visual_embeds is None


def test_raw_images_and_videos_share_forward_preparation_and_embedding_info():
    class Model(_Model):
        def get_image_features(self, pixel_values, image_grid_thw=None, image_max_grid_size=None):
            return (jnp.array([[3.0, 4.0]]),), [jnp.array([[30.0, 40.0]])]

        def get_video_features(self, pixel_values_videos, video_grid_thw=None, video_max_grid_size=None):
            return (jnp.array([[5.0, 6.0]]),), [jnp.array([[50.0, 60.0]])]

    model = Model()
    ids = jnp.array([[20, 0, 10]])
    embeddings, info = model.prepare_multimodal_embeddings(ids, pixel_values=object(), pixel_values_videos=object())
    np.testing.assert_array_equal(embeddings, [[[5, 6], [1, 2], [3, 4]]])
    np.testing.assert_array_equal(info.visual_pos_masks, [[True, False, True]])
    np.testing.assert_array_equal(info.deepstack_visual_embeds[0], [[50, 60], [30, 40]])
    np.testing.assert_array_equal(info.deepstack_image_embeds[0], [[30, 40]])
    np.testing.assert_array_equal(info.deepstack_video_embeds[0], [[50, 60]])
