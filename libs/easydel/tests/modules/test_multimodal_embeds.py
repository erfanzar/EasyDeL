"""Public multimodal-forward regressions with a transparent decoder (accelerator-only)."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.modules._base import BaseVisionLanguageModule
from easydel.modules.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel, get_rope_index
from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from easydel.modules.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel
from jax import numpy as jnp


class _TransparentMultimodalModel:
    """A language decoder that exposes merged embeddings plus 3D positions."""

    def __init__(self):
        self.config = SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
            text_config=SimpleNamespace(use_cache=True, rope_scaling={"mrope_section": [1, 1, 1]}),
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )

    def compute_embedding(self, input_ids, inputs_embeds=None, image_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = jnp.zeros((*input_ids.shape, 3), dtype=jnp.float32)
        if image_embeds is not None:
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=image_embeds,
                placeholder_token_id=9,
            )
        return inputs_embeds

    def get_rope_index(self, **kwargs):
        return get_rope_index(
            **kwargs, spatial_merge_size=2, image_token_id=9, video_token_id=10, vision_start_token_id=8
        )

    def language_model(self, *, inputs_embeds, position_ids, **kwargs):
        return BaseModelOutput(last_hidden_state=inputs_embeds + jnp.transpose(position_ids, (1, 2, 0)))


@pytest.mark.parametrize("model_cls", [Qwen2VLModel, Qwen3_5Model, Qwen3_5MoeModel])
def test_precomputed_image_embeddings_retain_image_grid_for_mrope(model_cls):
    model = _TransparentMultimodalModel()
    ids = np.array([[8, 9, 9, 9, 9, 7]], dtype=np.int32)
    image_embeds = jnp.array([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]], dtype=jnp.float32)
    output = model_cls.forward(
        model,
        input_ids=ids,
        image_embeds=image_embeds,
        image_grid_thw=np.array([[1, 4, 4]], dtype=np.int32),
    )
    # Four image tokens occupy a 2x2 spatial grid, not four text positions.
    positions = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2], [3, 3, 3]])
    merged = np.zeros((6, 3), dtype=np.float32)
    merged[1:5] = np.asarray(image_embeds)
    np.testing.assert_allclose(output.last_hidden_state, (merged + positions)[None], atol=0)
    if model_cls is not Qwen2VLModel:
        np.testing.assert_array_equal(output.rope_deltas, [[-2]])
