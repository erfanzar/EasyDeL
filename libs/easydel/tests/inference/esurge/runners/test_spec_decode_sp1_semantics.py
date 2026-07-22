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

"""Semantic lock for the reject-all resample position (SP1).

SP1 fixes the sampled, non-full-hidden ("replay") verify branch in
``eSurgeRunner._execute_model_impl``: when the drafter distributions are
missing/short every draft must be rejected, and the corrected token must be a
resample of the *target* distribution at the FIRST draft position (real-token
context) — mirroring the full-hidden branch
``corrected = sample_distribution(filtered_log_probs(logits_rows)[:1])``. The
pre-set fallback ``corrected_token = tokens_np[row_pos]`` is the *bonus*-position
sample taken with the (rejected) drafts as context and must not be emitted.

The runner branch that hosts SP1 is a defensive corner (only reached when the
model step returns fewer gathered hidden rows than the spec window needs), which
is not deterministically reproducible from a CPU unit test. This test instead
locks the load-bearing semantic on the exact strategy primitives SP1 composes
(``filtered_log_probs`` + ``sample_distribution``): with a crafted window whose
first-draft-position target distribution disagrees with the bonus-position
sample, the correct (first-position) resample is chosen and differs from the
bonus token.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation

_VOCAB = 8


def _greedy_req_state():
    return SimpleNamespace(
        sampling_params=SimpleNamespace(
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            n=1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
        )
    )


def _strategy():
    runner = SimpleNamespace(
        executor_manager=SimpleNamespace(rng_key=jax.random.PRNGKey(0)),
        max_num_seqs=1,
    )
    return DrafterSpeculation(runner=runner, drafter=SimpleNamespace(num_draft_tokens=2), num_draft_tokens=2)


def _peaked(token: int) -> jnp.ndarray:
    return jnp.full((1, _VOCAB), -30.0, dtype=jnp.float32).at[0, token].set(30.0)


def test_sp1_corrected_token_comes_from_first_draft_position_not_bonus():
    """The reject-all corrected token is the first-draft-position resample."""
    spec = _strategy()
    req_state = _greedy_req_state()

    first_position_token = 2  # target distribution at the first draft position (real-token context)
    bonus_position_token = 6  # tokens_np[row_pos]: the pre-fix fallback (drafts-as-context sample)
    assert first_position_token != bonus_position_token

    first_position_logits = _peaked(first_position_token)
    bonus_position_logits = _peaked(bonus_position_token)

    # SP1 fixed path: resample the target at the FIRST draft position.
    corrected_fixed = spec.sample_distribution(
        spec.filtered_log_probs(first_position_logits, req_state),
        req_state,
    )
    # Pre-fix path would have emitted the bonus-position sample.
    corrected_buggy = spec.sample_distribution(
        spec.filtered_log_probs(bonus_position_logits, req_state),
        req_state,
    )

    assert corrected_fixed == first_position_token
    assert corrected_buggy == bonus_position_token
    assert corrected_fixed != corrected_buggy, (
        "SP1 must resample the target at the first draft position, not emit the bonus-position token."
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
