"""Forward-input preparation keeps vision validation and embedding overrides."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor
from jax import numpy as jnp


def _metadata(**overrides):
    fields = dict(
        model_side_inputs=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
        model_input_ids=None,
        prefill_embeds=None,
        prefill_embeds_mask=None,
        mrope_position_ids=None,
        position_ids_buf=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.cpu_ok
def test_deepstack_without_mask_is_rejected_before_model_execution():
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    with pytest.raises(ValueError):
        executor._prepare_model_call(
            None,
            _metadata(deepstack_visual_embeds=(object(),)),
            scope="test",
        )


def test_prefill_overrides_preserve_base_embeddings_on_unmasked_rows():
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.metadata = SimpleNamespace(data_parallel_size=1)
    base_embeds = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    model = SimpleNamespace(compute_embedding=lambda ids: base_embeds)
    metadata = _metadata(
        model_input_ids=jnp.asarray([5, 6]),
        position_ids_buf=jnp.asarray([0, 1]),
        prefill_embeds=jnp.asarray([[10.0, 20.0], [30.0, 40.0]]),
        prefill_embeds_mask=jnp.asarray([True, False]),
        pixel_values=jnp.ones((1, 2)),
        image_grid_thw=jnp.asarray([[1, 1, 1]]),
    )
    inputs, positions, external = executor._prepare_model_call(model, metadata, scope="test")
    assert inputs["input_ids"] is None
    np.testing.assert_array_equal(np.asarray(inputs["inputs_embeds"]), [[[10.0, 20.0], [3.0, 4.0]]])
    np.testing.assert_array_equal(np.asarray(positions), [[0, 1]])
    assert external["pixel_values"] is metadata.pixel_values
    assert external["image_grid_thw"] is metadata.image_grid_thw
