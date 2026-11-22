"""Accelerator-only TPO scoring of chosen, rejected and reference log-probabilities."""

import math
from types import SimpleNamespace

import pytest
from easydel.trainers.tpo_trainer._fn import tpo_concatenated_forward
from jax import numpy as jnp


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_tpo_scores_chosen_rejected_and_reference_with_shared_logprobs(chunk_size):
    class FixedLogitsModel:
        def __call__(self, input_ids, attention_mask, labels):
            return SimpleNamespace(logits=jnp.broadcast_to(jnp.asarray([0.0, 1.0, 2.0]), (*labels.shape, 3)))

    batch = {
        "prompt_input_ids": jnp.asarray([[0]]),
        "prompt_attention_mask": jnp.asarray([[1]]),
        "chosen_input_ids": jnp.asarray([[1, 2]]),
        "chosen_attention_mask": jnp.asarray([[1, 0]]),
        "rejected_input_ids": jnp.asarray([[2, 2]]),
        "rejected_attention_mask": jnp.asarray([[1, 1]]),
        "reference_input_ids": jnp.asarray([[2, 1]]),
        "reference_attention_mask": jnp.asarray([[1, 1]]),
    }
    output = tpo_concatenated_forward(
        FixedLogitsModel(),
        batch,
        is_encoder_decoder=True,
        label_pad_token_id=-100,
        padding_value=0,
        logprob_vocab_chunk_size=chunk_size,
    )
    log_z = math.log(1.0 + math.exp(1.0) + math.exp(2.0))
    assert float(output["chosen_logps"][0]) == pytest.approx(1.0 - log_z, abs=1e-5)
    assert float(output["rejected_logps"][0]) == pytest.approx(2.0 * (2.0 - log_z), abs=1e-5)
    assert float(output["reference_nll"]) == pytest.approx(log_z - 1.5, abs=1e-5)
    assert float(output["chosen_lengths"][0]) == 1.0
    assert float(output["rejected_lengths"][0]) == 2.0
