"""Accelerator-only runner-to-compiled-model DeepStack residual placement parity."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor
from easydel.modules._base.qwen_vl import DeepStackVisualMixin
from jax import numpy as jnp

from .test_sampling_probabilities import run_host_runner_with_vision_payload


@pytest.mark.parametrize("dp_size", [2, 4])
@pytest.mark.parametrize("extra_vision", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_runner_rank_padding_preserves_compiled_deepstack_injection(monkeypatch, dp_size, extra_vision, deferred):
    # The runner and scheduler are host control flow. Restore their host-only
    # adapters before compiling the actual executor input preparation and the
    # model family's shared residual-injection implementation on the accelerator.
    with monkeypatch.context() as host_patch:
        payload = run_host_runner_with_vision_payload(host_patch, deferred, dp_size, extra_vision)

    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.metadata = SimpleNamespace(data_parallel_size=dp_size)

    class Embedding:
        def compute_embedding(self, ids):
            return jnp.zeros((*ids.shape, 2), dtype=jnp.float32)

    model = Embedding()
    mixin = DeepStackVisualMixin()

    @jax.jit
    def forward(embeddings, mask, positions, visual_mask, residuals):
        batch = SimpleNamespace(
            model_side_inputs={},
            pixel_values=None,
            pixel_values_videos=None,
            deepstack_visual_embeds=(residuals,),
            visual_pos_masks=visual_mask,
            model_input_ids=jnp.zeros(4, dtype=jnp.int32),
            prefill_embeds=embeddings,
            prefill_embeds_mask=mask,
            mrope_position_ids=positions,
        )
        inputs, model_positions, external = executor._prepare_model_call(model, batch, scope="dp_vision_parity")
        injected = mixin._deepstack_process(
            inputs["inputs_embeds"], external["visual_pos_masks"], external["deepstack_visual_embeds"][0]
        )
        return injected, model_positions

    actual, positions = forward(
        jnp.asarray(payload["prefill_embeds_cpu"], dtype=jnp.float32),
        jnp.asarray(payload["prefill_embeds_mask_cpu"]),
        jnp.asarray(payload["mrope_position_ids_cpu"]),
        jnp.asarray(payload["visual_pos_masks_cpu"]),
        jnp.asarray(payload["deepstack_visual_embeds_cpu"][0], dtype=jnp.float32),
    )
    # Independent arithmetic: image embedding [11,12] + residual [21,22]
    # belongs to flat token 2, never token 0 or a padded rank. A second video
    # at token 0 consumes [41,42], before the image despite scheduler order.
    expected = [[72, 74] if extra_vision else [0, 0], [0, 0], [32, 34], [0, 0]]
    np.testing.assert_array_equal(actual, np.asarray(expected).reshape(dp_size, 4 // dp_size, 2))
    np.testing.assert_array_equal(positions[:, 2 // (4 // dp_size), 2 % (4 // dp_size)], [5, 6, 7])
