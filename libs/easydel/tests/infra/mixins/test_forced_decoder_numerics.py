"""Hardware-only regression of the real forced-token processor built by generation."""

import numpy as np
import pytest
from easydel.infra.mixins.generation import EasyGenerationMixin
from jax import numpy as jnp
from transformers import GenerationConfig


@pytest.mark.parametrize("prompt_length", [1, 4])
def test_forced_decoder_pairs_select_tokens_and_preserve_none_steps(prompt_length):
    config = GenerationConfig(forced_decoder_ids=[(1, 7), (2, None), (3, 11)])
    processors = EasyGenerationMixin()._get_logits_processor(config, prompt_length, None)
    scores = jnp.arange(16, dtype=jnp.float32)[None, :]
    input_ids = jnp.zeros((1, prompt_length + 3), dtype=jnp.int32)
    for offset, token in enumerate((7, None, 11)):
        output = processors(input_ids, scores, prompt_length + offset)
        if token is None:
            np.testing.assert_array_equal(np.asarray(output), np.asarray(scores))
        else:
            expected = np.full((1, 16), -np.inf, dtype=np.float32)
            expected[0, token] = 0
            np.testing.assert_array_equal(np.asarray(output), expected)
