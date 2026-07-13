# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for the eSurge sampler on MIXED (greedy + sampling) batches.

Two default-path correctness bugs, both silent and model-agnostic:

* Greedy rows carry the reset sentinel temperature ``-1.0`` (``SequenceBuffer``).
  The all-greedy short-circuit only fires when the *whole* batch is greedy, so a
  mixed batch ran ``_regular_sample`` for the greedy row too, where ``apply_temp``
  divided its logits by ``-1.0`` (negating them) and sampled the *lowest*-logit
  tokens. Fix: per-row greedy override to argmax + safe temperature.
* min-p was thresholded in LOGIT space and ``need_min_p_sampling`` is a
  batch-global flag, so a request with the default ``min_p=0`` had its
  negative-logit tokens masked whenever any *other* row in the batch used min-p.
  Fix: probability-space min-p, where ``min_p=0`` is a true no-op.
"""

import jax
import jax.numpy as jnp
from easydel.inference.esurge.core.binary_search import apply_min_p_mask
from easydel.inference.esurge.core.sampler import sample_tokens
from easydel.inference.esurge.core.sampling_metadata import SamplingMetadata


def _meta(*, temperatures, min_ps, need_min_p_sampling):
    batch = temperatures.shape[0]
    return SamplingMetadata(
        temperatures=temperatures,
        top_ps=jnp.ones((batch,), dtype=jnp.float32),
        top_ks=jnp.zeros((batch,), dtype=jnp.int32),
        min_ps=min_ps,
        sampling_seeds=None,
        is_all_greedy=False,
        need_min_p_sampling=need_min_p_sampling,
        do_penalties=False,
        linear_penalty=None,
    )


def test_greedy_row_takes_argmax_when_co_batched_with_sampling():
    """A greedy row batched with a sampling row must emit its argmax, not argmin."""
    vocab = 32
    peak_token = 5
    # Row 0: greedy (sentinel temp -1.0), sharply peaked at `peak_token`.
    # Row 1: ordinary sampling row (temp 0.8) so the batch is NOT all-greedy.
    row0 = jnp.full((vocab,), 0.0, dtype=jnp.float32).at[peak_token].set(20.0)
    row1 = jax.random.normal(jax.random.PRNGKey(1), (vocab,), dtype=jnp.float32)
    logits = jnp.stack([row0, row1], axis=0)

    meta = _meta(
        temperatures=jnp.array([[-1.0], [0.8]], dtype=jnp.float32),
        min_ps=jnp.zeros((2,), dtype=jnp.float32),
        need_min_p_sampling=False,
    )
    tokens = sample_tokens(logits, meta, jax.random.PRNGKey(0))

    # Greedy override is deterministic: the greedy row is exactly its argmax.
    # On the buggy code the greedy row sampled from -logits (peak_token had the
    # lowest probability) and would essentially never be `peak_token`.
    assert int(tokens[0]) == peak_token, (
        f"greedy row emitted {int(tokens[0])}, expected argmax {peak_token} "
        "(mixed-batch greedy negation regression)"
    )


def test_min_p_zero_is_a_noop_even_when_another_row_uses_min_p():
    """min_p=0 rows must be untouched even though min-p is a batch-global flag."""
    # Row 0: min_p=0 (default) with negative logits that a logit-space threshold
    # would wrongly mask. Row 1: an actual min-p user (triggers the batch flag).
    logits = jnp.array(
        [
            [3.0, -0.5, -1.0, -4.0],
            [2.0, 1.5, -0.2, -3.0],
        ],
        dtype=jnp.float32,
    )
    meta = _meta(
        temperatures=jnp.array([[1.0], [1.0]], dtype=jnp.float32),
        min_ps=jnp.array([0.0, 0.1], dtype=jnp.float32),
        need_min_p_sampling=True,
    )
    masked = apply_min_p_mask(logits, meta)

    # min_p == 0 -> threshold 0 in probability space -> keep every token.
    assert bool(jnp.allclose(masked[0], logits[0])), (
        "min_p=0 row was altered by min-p filtering (logit-space / batch-global "
        f"contamination regression); got {masked[0]} vs {logits[0]}"
    )
    # The genuine min-p row (row 1) still filters something (min_p=0.1 drops the
    # deep-negative token), proving the filter is actually active for real users.
    assert bool(jnp.any(masked[1] <= -1e9)), "min_p=0.1 row should have masked its low-probability tail"


def test_min_p_does_not_mask_entire_vocab_when_all_logits_negative():
    """All-negative logits with min-p must not collapse to a uniform (fully masked) row."""
    logits = jnp.array([[-1.0, -2.0, -5.0, -8.0]], dtype=jnp.float32)
    meta = _meta(
        temperatures=jnp.array([[1.0]], dtype=jnp.float32),
        min_ps=jnp.array([0.1], dtype=jnp.float32),
        need_min_p_sampling=True,
    )
    masked = apply_min_p_mask(logits, meta)
    # The top token (prob-max) must survive; the row must not be all -1e10.
    assert not bool(jnp.all(masked[0] <= -1e9)), "min-p masked the entire vocab (logit-space max<=0 bug)"
    assert int(jnp.argmax(masked[0])) == 0, "the highest-probability token must be kept"


if __name__ == "__main__":
    test_greedy_row_takes_argmax_when_co_batched_with_sampling()
    test_min_p_zero_is_a_noop_even_when_another_row_uses_min_p()
    test_min_p_does_not_mask_entire_vocab_when_all_logits_negative()
    print("ok")
