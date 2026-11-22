"""Accelerator-only contracts for valid generation history and minimum length."""

import jax
import numpy as np
import pytest
from easydel.inference.logits_process import (
    FrequencyPenaltyLogitsProcessor,
    MinLengthLogitsProcessor,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
from jax import numpy as jnp


@pytest.mark.parametrize("length", [2, 3, 4])
def test_eos_is_allowed_only_above_minimum_length(length):
    scores = jnp.asarray([[1.0, 4.0, 2.0]])
    result = jax.jit(MinLengthLogitsProcessor(min_length=3, eos_token_id=1))(
        jnp.zeros((1, 5), dtype=jnp.int32), scores, jnp.asarray(length)
    )
    expected = np.asarray([[1.0, -np.inf if length <= 3 else 4.0, 2.0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("kind", ["presence", "frequency", "repetition"])
def test_penalties_ignore_unfilled_capacity_and_match_independent_reference(kind):
    scores = jnp.asarray([[4.0, -2.0, 6.0], [3.0, 5.0, -4.0]])
    ids = jnp.asarray([[1, 1, 0, 0], [2, 1, 0, 0]], dtype=jnp.int32)
    processors = {
        "presence": PresencePenaltyLogitsProcessor(presence_penalty=0.5),
        "frequency": FrequencyPenaltyLogitsProcessor(frequency_penalty=0.5),
        "repetition": RepetitionPenaltyLogitsProcessor(repetition_penalty=2.0),
    }
    processor = processors[kind]
    counts = np.asarray([[0, 2, 0], [0, 1, 1]])
    reference = np.asarray(scores).copy()
    if kind == "presence":
        reference -= (counts > 0) * 0.5
    elif kind == "frequency":
        reference -= counts * 0.5
    else:
        reference = np.where(counts > 0, np.where(reference > 0, reference / 2, reference * 2), reference)
    result = jax.jit(processor)(ids, scores, jnp.asarray(2))
    np.testing.assert_array_equal(result, reference)
    np.testing.assert_array_equal(result, processor(ids[:, :2], scores, 2))
