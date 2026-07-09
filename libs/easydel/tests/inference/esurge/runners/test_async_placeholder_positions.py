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

"""Async-scheduling placeholder bookkeeping regression tests.

The async placeholder slot for a step's sampled token is
``num_computed_tokens + scheduled`` recorded at postprocess time.
``num_tokens_no_spec`` is only maintained by the async path itself, so it goes
stale (one slot behind) whenever the previous step for a request was finalized
synchronously (e.g. the prefill step). These tests pin the invariant that the
placeholder machinery trusts the recorded position, not the stale counter:
re-reading ``num_tokens_no_spec`` made every async finalize overwrite the
previously sampled token and left the true append slot as a ``0`` placeholder
that was then fed back into the model (garbage decode from step 2 onward).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from easydel.inference.esurge.runners.model_runner import eSurgeRunner


def _runner_with_buffer(num_tokens_no_spec: list[int], max_model_len: int = 64) -> eSurgeRunner:
    runner = object.__new__(eSurgeRunner)
    runner.max_model_len = max_model_len
    runner.sequence_buffer = SimpleNamespace(
        num_tokens_no_spec=np.asarray(num_tokens_no_spec, dtype=np.int32),
        num_tokens=np.asarray(num_tokens_no_spec, dtype=np.int32),
        req_id_to_index={},
        token_ids=np.zeros((len(num_tokens_no_spec), max_model_len), dtype=np.int32),
    )
    runner.requests = {}
    return runner


def test_update_placeholder_uses_recorded_position_over_stale_counter():
    """After a sync-finalized prefill, ``num_tokens_no_spec`` lags one behind.

    Prompt length 14, prefill token already appended synchronously at slot 14
    (counter left at 14). The first async decode step's sampled token lands at
    slot 15 = ``num_computed(14) + scheduled(1)``; the placeholder bump must
    land there, not at the stale counter value.
    """
    runner = _runner_with_buffer([14])
    req_state = SimpleNamespace(req_id="r0", output_token_ids=[105261])

    placeholder_idx = 15  # num_computed_tokens (14) + scheduled (1)
    runner._update_placeholder(
        discard_sampled_tokens_req_indices=[],
        request_seq_lens=[(0, 0, req_state, placeholder_idx)],
    )

    assert int(runner.sequence_buffer.num_tokens_no_spec[0]) == 16
    assert int(runner.sequence_buffer.num_tokens[0]) == 16
    # One placeholder appended for the in-flight token; the previously
    # finalized token is untouched.
    assert req_state.output_token_ids == [105261, 0]


def test_update_placeholder_matches_counter_in_pure_async_chain():
    """In an uninterrupted async chain the recorded slot equals the counter."""
    runner = _runner_with_buffer([16])
    req_state = SimpleNamespace(req_id="r0", output_token_ids=[7, 8])

    runner._update_placeholder(
        discard_sampled_tokens_req_indices=[],
        request_seq_lens=[(0, 0, req_state, 16)],
    )

    assert int(runner.sequence_buffer.num_tokens_no_spec[0]) == 17
    assert req_state.output_token_ids == [7, 8, 0]


def test_update_placeholder_discard_and_length_guard():
    """Discarded rows are skipped; overflow of ``max_model_len`` raises."""
    runner = _runner_with_buffer([10, 10], max_model_len=16)
    kept = SimpleNamespace(req_id="keep", output_token_ids=[])
    discarded = SimpleNamespace(req_id="drop", output_token_ids=[])

    runner._update_placeholder(
        discard_sampled_tokens_req_indices=[1],
        request_seq_lens=[(0, 0, kept, 11), (1, 1, discarded, 11)],
    )
    assert int(runner.sequence_buffer.num_tokens_no_spec[0]) == 12
    assert int(runner.sequence_buffer.num_tokens_no_spec[1]) == 10
    assert discarded.output_token_ids == []

    overflowing = SimpleNamespace(req_id="over", output_token_ids=[])
    with pytest.raises(ValueError, match="max model length"):
        runner._update_placeholder(
            discard_sampled_tokens_req_indices=[],
            request_seq_lens=[(0, 0, overflowing, 16)],
        )


def test_finalize_async_state_writes_token_at_recorded_placeholder():
    """The drained async token replaces exactly the recorded placeholder slot."""
    runner = _runner_with_buffer([14])
    # Sync-finalized prefill token at slot 14; async placeholder at slot 15.
    runner.sequence_buffer.token_ids[0, 14] = 105261
    req_state = SimpleNamespace(req_id="r0", output_token_ids=[105261, 0], num_prompt_tokens=14)
    runner.sequence_buffer.req_id_to_index = {"r0": 0}
    runner.requests = {"r0": req_state}
    runner._pre_async_results = None

    runner._finalize_async_scheduler_runner_state(
        [[110807]],
        request_seq_lens=[(0, 0, req_state, 15)],
    )

    assert int(runner.sequence_buffer.token_ids[0, 14]) == 105261, "previous token must not be overwritten"
    assert int(runner.sequence_buffer.token_ids[0, 15]) == 110807
    assert req_state.output_token_ids == [105261, 110807]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
