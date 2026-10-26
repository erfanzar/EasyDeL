"""Real JAX embedding-merge regression (accelerator only)."""

from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import numpy as np
from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange
from easydel.modules.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForConditionalGeneration, DeepseekV41VLModel
from jax import numpy as jnp


def test_cached_encoder_rows_preserve_real_merge_delimiters_and_new_prompt():
    def encode(patches, image_grid):
        assert image_grid == (3, 3)
        return jnp.full((1, 1, 2), patches[0, 0, 0, 0, 0], dtype=jnp.float32)

    base = SimpleNamespace(
        get_image_features=encode,
        image_start=SimpleNamespace(value=jnp.array([-10.0, -11.0])),
        image_end=SimpleNamespace(value=jnp.array([-20.0, -21.0])),
        image_newline=SimpleNamespace(value=jnp.array([-30.0, -31.0])),
    )
    base.merge_image_embeddings = partial(DeepseekV41VLModel.merge_image_embeddings, base)
    model = SimpleNamespace(
        config=SimpleNamespace(vision_params=lambda: {"downsample_ratio": 3}),
        mesh=nullcontext(),
        dtype=jnp.float32,
        compute_embedding=lambda ids: jnp.repeat(ids[..., None], 2, axis=-1).astype(jnp.float32),
        base_model=base,
    )
    feature = MultiModalFeature.from_image(
        np.full((9, 3, 1, 1), 5, dtype=np.float32),
        np.array([[1, 3, 3]], dtype=np.int32),
        PlaceholderRange(offset=1, length=4, modality="image"),
    )
    hook = DeepseekV41ForConditionalGeneration.esurge_compute_prefill_embeddings
    first, _ = hook(model, [7, 99, 99, 99, 99], [feature])
    feature.clear_pixel_values()
    feature.placeholder_range = PlaceholderRange(offset=2, length=4, modality="image")
    second, types = hook(model, [8, 9, 99, 99, 99, 99], [feature])
    np.testing.assert_array_equal(first, [[7, 7], [-10, -11], [5, 5], [-30, -31], [-20, -21]])
    np.testing.assert_array_equal(second, [[8, 8], [9, 9], [-10, -11], [5, 5], [-30, -31], [-20, -21]])
    np.testing.assert_array_equal(types, [-1, -1, 0, 1, 2, 3])
