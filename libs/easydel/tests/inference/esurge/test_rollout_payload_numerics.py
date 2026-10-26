"""Accelerator-only pipeline-to-trainer numerical payload regression.

Deliberately not cpu_ok. The host-only pipeline tests live beside it.
"""

import numpy as np
import pytest
from easydel.inference.esurge.engine_types import EngineCoreOutput, FinishReason
from easydel.trainers.base_trainer import BaseTrainer
from jax import numpy as jnp

from .test_output_logprobs import pipeline_factory as pipeline_factory
from .test_output_logprobs import probabilities, submit


@pytest.mark.parametrize("include_logprobs", [True, False])
def test_pipeline_payload_reconstructs_rollout_sampled_logps_on_accelerator(pipeline_factory, include_logprobs):
    pipeline, _, output, _ = pipeline_factory(n=2)
    output.prompt_token_ids = [[1]]
    for sample_index, tokens, logps in ((0, [65, 0], [-0.2, -0.4]), (1, [66], [-0.6])):
        submit(
            pipeline,
            EngineCoreOutput(
                f"parent-{sample_index}",
                tokens,
                new_logprobs=probabilities(tokens, logps) if include_logprobs else None,
                finish_reason=FinishReason.STOP,
            ),
        )
    assert output.finished
    arrays = BaseTrainer._esurge_outputs_to_arrays(
        None,
        [output],
        input_ids=None,
        attention_mask=None,
        return_prompts=["prompt"],
        pad_token_id=99,
        max_seq_len=2,
        max_new_tokens=3,
    )
    np.testing.assert_array_equal(arrays.completion_ids, [[65, 0, 99], [66, 99, 99]])
    np.testing.assert_array_equal(arrays.completion_mask, [[1, 1, 0], [1, 0, 0]])
    if include_logprobs:
        assert arrays.sampling_per_token_logps.dtype == jnp.float32
        np.testing.assert_allclose(arrays.sampling_per_token_logps, [[-0.2, -0.4, 0], [-0.6, 0, 0]])
    else:
        assert arrays.sampling_per_token_logps is None
