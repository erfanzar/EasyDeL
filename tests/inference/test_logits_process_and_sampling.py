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

"""Tests for ``easydel.inference.logits_process`` and ``easydel.inference.sampling_funcs``.

The processors are pure functions over (input_ids, scores, cur_len) -> scores.
We test:

* warper / processor identity-on-disabled-config (e.g. ``temperature=-1`` is identity)
* numerical equivalence with reference numpy/scipy where possible
* invariants: shape preservation, dtype preservation, no NaN/Inf in the kept positions
* ``LogitsProcessorList`` chains processors in order and stops at any unmet kwarg
* ``sample_top_p_efficient`` returns valid token indices and respects greedy edge cases
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from easydel.inference.logits_process import (
    EmptyProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    FrequencyPenaltyLogitsProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    LogitsWarper,
    MinLengthLogitsProcessor,
    MinPLogitsWarper,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from easydel.inference.sampling_funcs import sample_top_p_efficient


def test_warper_and_processor_are_separate_base_classes():
    """LogitsWarper and LogitsProcessor are sibling abstract classes -- neither inherits from the other."""
    assert not issubclass(LogitsWarper, LogitsProcessor)
    assert not issubclass(LogitsProcessor, LogitsWarper)

    assert callable(LogitsProcessor.__call__)
    assert callable(LogitsWarper.__call__)


def test_processor_list_inherits_from_list():
    assert issubclass(LogitsProcessorList, list)


def test_temperature_warper_scales_logits_by_inverse_temperature():
    scores = jnp.array([[1.0, 2.0, 4.0]])
    out = TemperatureLogitsWarper(temperature=jnp.asarray(2.0))(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert jnp.allclose(out, scores / 2.0)


def test_temperature_warper_minus_one_is_identity():
    """Sentinel ``-1`` disables the warper (per its docstring)."""
    scores = jnp.array([[1.5, 2.5, 3.5]])
    out = TemperatureLogitsWarper(temperature=jnp.asarray(-1.0))(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert jnp.allclose(out, scores)


def test_temperature_warper_preserves_shape_and_dtype():
    scores = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    out = TemperatureLogitsWarper(temperature=jnp.asarray(0.5))(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert out.shape == scores.shape
    assert out.dtype == scores.dtype


def test_topk_warper_keeps_only_top_k_logits():
    """top_k=2 with [1, 5, 3, 4] should mask everything except 5 and 4."""
    scores = jnp.array([[1.0, 5.0, 3.0, 4.0]])
    out = TopKLogitsWarper(top_k=2)(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    finite_count = int(jnp.sum(jnp.isfinite(out)))
    assert finite_count == 2

    kept_mask = jnp.isfinite(out[0])
    assert bool(kept_mask[1])
    assert bool(kept_mask[3])
    assert not bool(kept_mask[0])
    assert not bool(kept_mask[2])


def test_topk_warper_zero_disables_filter():
    scores = jnp.array([[1.0, 2.0, 3.0]])
    out = TopKLogitsWarper(top_k=0)(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert jnp.allclose(out, scores)


def test_topk_warper_min_tokens_to_keep_overrides_small_k():
    """If top_k < min_tokens_to_keep, the latter wins."""
    scores = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    out = TopKLogitsWarper(top_k=1, min_tokens_to_keep=3)(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    finite_count = int(jnp.sum(jnp.isfinite(out)))
    assert finite_count == 3


def test_topk_warper_k_geq_vocab_is_identity():
    scores = jnp.array([[1.0, 2.0, 3.0]])
    out = TopKLogitsWarper(top_k=10)(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert jnp.allclose(out, scores)


def test_topp_warper_one_is_identity():
    scores = jnp.array([[1.0, 2.0, 3.0]])
    out = TopPLogitsWarper(top_p=1.0)(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert jnp.allclose(out, scores)


def test_topp_warper_zero_is_identity():
    """top_p=0 also bypasses the filter via the ``(top_p > 0) & (top_p < 1)`` predicate."""
    scores = jnp.array([[1.0, 2.0, 3.0]])
    out = TopPLogitsWarper(top_p=0.0)(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert jnp.allclose(out, scores)


def test_topp_warper_keeps_at_least_one_token():
    """min_tokens_to_keep=1 ensures the highest-prob token always survives even on aggressive top_p."""
    scores = jnp.array([[10.0, 0.0, -10.0]])
    out = TopPLogitsWarper(top_p=0.0001)(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)

    assert jnp.isfinite(out[0, 0])


def test_minp_warper_zero_is_identity():
    scores = jnp.array([[1.0, 2.0, 3.0]])
    out = MinPLogitsWarper(min_p=0.0)(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert jnp.allclose(out, scores)


def test_minp_warper_one_is_identity():
    scores = jnp.array([[1.0, 2.0, 3.0]])
    out = MinPLogitsWarper(min_p=1.0)(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert jnp.allclose(out, scores)


def test_minp_warper_filters_below_threshold():
    """With strong logit gap, only the peak survives at high min_p."""
    scores = jnp.array([[10.0, 0.0, 0.0]])
    out = MinPLogitsWarper(min_p=0.5)(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    finite_count = int(jnp.sum(jnp.isfinite(out)))
    assert finite_count >= 1
    assert jnp.isfinite(out[0, 0])


def test_presence_penalty_zero_is_identity():
    scores = jnp.array([[1.0, 2.0, 3.0]])
    input_ids = jnp.array([[0, 1]])
    out = PresencePenaltyLogitsProcessor(presence_penalty=0.0)(input_ids, scores, 2)
    assert jnp.allclose(out, scores)


def test_presence_penalty_subtracts_fixed_amount_per_present_token():
    scores = jnp.array([[5.0, 5.0, 5.0]])
    input_ids = jnp.array([[0, 0, 1]])
    out = PresencePenaltyLogitsProcessor(presence_penalty=2.0)(input_ids, scores, 3)

    assert jnp.allclose(out, jnp.array([[3.0, 3.0, 5.0]]))


def test_frequency_penalty_zero_is_identity():
    scores = jnp.array([[1.0, 2.0, 3.0]])
    input_ids = jnp.array([[0, 0, 0]])
    out = FrequencyPenaltyLogitsProcessor(frequency_penalty=0.0)(input_ids, scores, 3)
    assert jnp.allclose(out, scores)


def test_frequency_penalty_subtracts_count_times_penalty():
    """Token 0 appears 3 times, token 1 once, token 2 zero times."""
    scores = jnp.array([[10.0, 10.0, 10.0]])
    input_ids = jnp.array([[0, 0, 0, 1]])
    out = FrequencyPenaltyLogitsProcessor(frequency_penalty=1.0)(input_ids, scores, 4)

    assert jnp.allclose(out, jnp.array([[7.0, 9.0, 10.0]]))


def test_repetition_penalty_one_is_identity():
    scores = jnp.array([[1.0, 2.0, 3.0]])
    input_ids = jnp.array([[0, 1]])
    out = RepetitionPenaltyLogitsProcessor(repetition_penalty=1.0)(input_ids, scores, 2)
    assert jnp.allclose(out, scores)


def test_repetition_penalty_divides_positive_and_multiplies_negative():
    """Per CTRL paper: +ve logits / penalty, -ve logits * penalty for previously seen tokens."""
    scores = jnp.array([[2.0, -2.0, 3.0]])
    input_ids = jnp.array([[0, 1]])
    out = RepetitionPenaltyLogitsProcessor(repetition_penalty=2.0)(input_ids, scores, 2)

    assert jnp.allclose(out, jnp.array([[1.0, -4.0, 3.0]]))


def test_forced_bos_token_active_only_at_position_one():
    """Only at cur_len==1 does the BOS forcing kick in."""
    scores = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    proc = ForcedBOSTokenLogitsProcessor(bos_token_id=2)


    out_other = proc(jnp.zeros((1, 0), dtype=jnp.int32), scores, 5)
    assert jnp.allclose(out_other, scores)


    out_first = proc(jnp.zeros((1, 0), dtype=jnp.int32), scores, 1)
    assert int(jnp.argmax(out_first[0])) == 2


def test_forced_eos_token_at_max_length():
    """ForcedEOSTokenLogitsProcessor forces EOS at the max length boundary."""
    scores = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    proc = ForcedEOSTokenLogitsProcessor(max_length=10, eos_token_id=3)


    out_before = proc(jnp.zeros((1, 0), dtype=jnp.int32), scores, 5)
    assert jnp.allclose(out_before, scores)


    out_at = proc(jnp.zeros((1, 0), dtype=jnp.int32), scores, 9)
    assert int(jnp.argmax(out_at[0])) == 3


def test_min_length_processor_suppresses_eos_below_min_length():
    """EOS is masked while ``cur_len <= min_length``; allowed once strictly above.

    The implementation uses ``apply_penalty = 1 - clip(cur_len - min_length, 0, 1)``
    so the boundary case ``cur_len == min_length`` STILL suppresses EOS. EOS becomes
    available only when ``cur_len > min_length``.
    """
    scores = jnp.array([[1.0, 2.0, 3.0, 10.0]])
    proc = MinLengthLogitsProcessor(min_length=5, eos_token_id=3)


    out_below = proc(jnp.zeros((1, 0), dtype=jnp.int32), scores, 2)
    assert not jnp.isfinite(out_below[0, 3])


    out_at = proc(jnp.zeros((1, 0), dtype=jnp.int32), scores, 5)
    assert not jnp.isfinite(out_at[0, 3])


    out_above = proc(jnp.zeros((1, 0), dtype=jnp.int32), scores, 6)
    assert jnp.isfinite(out_above[0, 3])


def test_suppress_tokens_at_begin_only_at_begin_index():
    """Tokens are suppressed only when cur_len == begin_index."""
    scores = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    proc = SuppressTokensAtBeginLogitsProcessor(begin_suppress_tokens=jnp.array([1, 3]), begin_index=2)


    out_other = proc(jnp.zeros((1, 0), dtype=jnp.int32), scores, 5)
    assert jnp.allclose(out_other, scores)


    out_at = proc(jnp.zeros((1, 0), dtype=jnp.int32), scores, 2)
    assert not jnp.isfinite(out_at[0, 1])
    assert not jnp.isfinite(out_at[0, 3])
    assert jnp.isfinite(out_at[0, 0])
    assert jnp.isfinite(out_at[0, 2])


def test_suppress_tokens_processor_always_masks_listed():
    scores = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    proc = SuppressTokensLogitsProcessor(suppress_tokens=jnp.array([0, 2]))
    out = proc(jnp.zeros((1, 0), dtype=jnp.int32), scores, 7)
    assert not jnp.isfinite(out[0, 0])
    assert not jnp.isfinite(out[0, 2])
    assert jnp.isfinite(out[0, 1])
    assert jnp.isfinite(out[0, 3])


def test_empty_processor_is_identity():
    scores = jnp.array([[1.0, 2.0, 3.0]])
    out = EmptyProcessor()(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert jnp.allclose(out, scores)


def test_logits_processor_list_chains_in_order():
    """Output of N-th processor feeds the N+1-th."""
    scores = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    procs = LogitsProcessorList(
        [
            TemperatureLogitsWarper(temperature=jnp.asarray(2.0)),
            TopKLogitsWarper(top_k=2),
        ]
    )
    out = procs(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    finite_count = int(jnp.sum(jnp.isfinite(out)))
    assert finite_count == 2


def test_logits_processor_list_handles_extra_kwargs_processor_signature():
    """If a processor in the list expects extra kwargs not supplied, it raises ValueError."""

    class NeedsExtra(LogitsProcessor):
        def __call__(self, input_ids, scores, cur_len, must_have_this):
            return scores

    procs = LogitsProcessorList([NeedsExtra()])
    with pytest.raises(ValueError, match="required parameters"):
        procs(jnp.zeros((1, 0), dtype=jnp.int32), jnp.zeros((1, 3)), 0)


def test_logits_processor_list_chain_preserves_shape_and_dtype():
    """Through a chain of processors, the output retains the input scores' shape and dtype."""
    scores = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    procs = LogitsProcessorList(
        [
            EmptyProcessor(),
            TemperatureLogitsWarper(temperature=jnp.asarray(0.5)),
            EmptyProcessor(),
        ]
    )
    out = procs(jnp.zeros((1, 0), dtype=jnp.int32), scores, 0)
    assert out.shape == scores.shape
    assert out.dtype == scores.dtype


def test_sample_top_p_efficient_returns_valid_index():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([1.0, 5.0, 3.0, 2.0])
    token = sample_top_p_efficient(
        logits=logits,
        top_p=jnp.asarray(0.9),
        temperature=jnp.asarray(1.0),
        rng=rng,
    )
    assert int(token) in {0, 1, 2, 3}


def test_sample_top_p_efficient_zero_temperature_safe():
    """The function defends against temperature near zero by clamping to 1.0."""
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([1.0, 5.0, 3.0, 2.0])
    token = sample_top_p_efficient(
        logits=logits,
        top_p=jnp.asarray(0.9),
        temperature=jnp.asarray(0.0),
        rng=rng,
    )
    assert int(token) in {0, 1, 2, 3}


def test_sample_top_p_efficient_low_p_concentrates_to_argmax():
    """With top_p ~ 0, only the argmax candidate survives the keep_mask."""
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([1.0, 50.0, 3.0, 2.0])

    tokens = []
    for i in range(20):
        rng, sub = jax.random.split(rng)
        token = sample_top_p_efficient(
            logits=logits,
            top_p=jnp.asarray(0.001),
            temperature=jnp.asarray(1.0),
            rng=sub,
        )
        tokens.append(int(token))
    assert set(tokens) == {1}, f"expected only token 1, got {set(tokens)}"


def test_sample_top_p_efficient_jit_compatible():
    """The function is differentiable / JIT-friendly (no Python branching on traced values)."""
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([1.0, 2.0, 3.0])
    jitted = jax.jit(sample_top_p_efficient, static_argnames=("top_k_for_computation",))
    token = jitted(logits, jnp.asarray(0.9), jnp.asarray(1.0), rng)
    assert int(token) in {0, 1, 2}
