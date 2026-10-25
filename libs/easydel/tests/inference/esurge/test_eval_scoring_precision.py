"""Accelerator-only likelihood parity for low-precision model-head outputs.

These tests execute JAX numerics and intentionally have no cpu_ok marker.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.evaluations.esurge_eval import eSurgeLMEvalAdapter
from jax import numpy as jnp


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("continuation_length", [1, 1024])
def test_low_precision_scoring_matches_independent_fp32_normalization(dtype, continuation_length):
    adapter = object.__new__(eSurgeLMEvalAdapter)
    adapter.model = SimpleNamespace(mesh=None)
    adapter.surge = SimpleNamespace(runner=SimpleNamespace(mesh=None))
    adapter.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0, bos_token_id=0)
    adapter.max_length = 2048
    adapter._get_scoring_model = lambda: adapter.model

    # Large equal logits expose cancellation: bf16 normalization without promotion
    # rounds 128 + log(2) to 129, giving -1 instead of -log(2).
    logits = jnp.full((1, continuation_length, 2), 128, dtype=dtype)
    adapter._get_scoring_logits_fn = lambda: lambda *_: logits
    result = adapter._loglikelihood_token_ids([[0]], [[1] * continuation_length])
    reference_logits = np.asarray(logits, dtype=np.float64)
    centered = reference_logits - reference_logits.max(axis=-1, keepdims=True)
    reference = centered - np.log(np.exp(centered).sum(axis=-1, keepdims=True))
    expected = float(reference[0, :, 1].sum())
    assert result[0][0] == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert result[0][1] is False
