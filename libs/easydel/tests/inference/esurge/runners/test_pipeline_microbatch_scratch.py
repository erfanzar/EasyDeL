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

import numpy as np

from easydel.inference.esurge.runners.execution_manager import ExecutionManager


def _scratch_slot(max_reqs: int = 5, max_len: int = 4, max_pages: int = 3) -> dict:
    return {
        "scheduled": np.zeros((max_reqs,), dtype=np.int32),
        "active": np.zeros((max_reqs,), dtype=np.bool_),
        "token_ids": np.zeros((max_reqs, max_len), dtype=np.int32),
        "num_computed": np.zeros((max_reqs,), dtype=np.int32),
        "temperature": np.ones((max_reqs,), dtype=np.float32),
        "top_p": np.ones((max_reqs,), dtype=np.float32),
        "top_k": np.zeros((max_reqs,), dtype=np.int32),
        "min_p": np.zeros((max_reqs,), dtype=np.float32),
        "frequency": np.zeros((max_reqs,), dtype=np.float32),
        "presence": np.zeros((max_reqs,), dtype=np.float32),
        "repetition": np.ones((max_reqs,), dtype=np.float32),
        "page_table": np.zeros((max_reqs, max_pages), dtype=np.int32),
        "prev_count": 0,
    }


def test_pipeline_microbatch_scratch_clears_stale_rows_when_batch_shrinks():
    scheduled = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    token_ids = np.arange(20, dtype=np.int32).reshape(5, 4)
    num_computed = np.array([4, 5, 6, 7, 8], dtype=np.int32)
    temperature = np.array([0.7, 0.8, 0.9, 1.0, 1.1], dtype=np.float32)
    top_p = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    top_k = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    min_p = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    frequency = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    presence = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    repetition = np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32)
    page_table = np.arange(15, dtype=np.int32).reshape(5, 3)
    slot = _scratch_slot()

    ExecutionManager._populate_pipeline_microbatch_scratch(
        slot,
        chunk=np.array([0, 2, 4], dtype=np.int32),
        scheduled_full_cpu=scheduled,
        token_ids_cpu=token_ids,
        num_computed_tokens_cpu=num_computed,
        temperature_cpu=temperature,
        top_p_cpu=top_p,
        top_k_cpu=top_k,
        min_p_cpu=min_p,
        frequency_penalties_cpu=frequency,
        presence_penalties_cpu=presence,
        repetition_penalties_cpu=repetition,
        page_table_cpu=page_table,
    )
    ExecutionManager._populate_pipeline_microbatch_scratch(
        slot,
        chunk=np.array([3], dtype=np.int32),
        scheduled_full_cpu=scheduled,
        token_ids_cpu=token_ids,
        num_computed_tokens_cpu=num_computed,
        temperature_cpu=temperature,
        top_p_cpu=top_p,
        top_k_cpu=top_k,
        min_p_cpu=min_p,
        frequency_penalties_cpu=frequency,
        presence_penalties_cpu=presence,
        repetition_penalties_cpu=repetition,
        page_table_cpu=page_table,
    )

    np.testing.assert_array_equal(slot["active"], np.array([True, False, False, False, False]))
    np.testing.assert_array_equal(slot["scheduled"], np.array([1, 0, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(slot["token_ids"][0], token_ids[3])
    np.testing.assert_array_equal(slot["token_ids"][1:3], np.zeros((2, 4), dtype=np.int32))
    np.testing.assert_array_equal(slot["page_table"][0], page_table[3])
    np.testing.assert_array_equal(slot["page_table"][1:3], np.zeros((2, 3), dtype=np.int32))
    np.testing.assert_allclose(slot["temperature"], np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(slot["top_p"], np.array([0.6, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(slot["repetition"], np.array([1.3, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))


def test_pipeline_microbatching_rejects_underfilled_token_buckets():
    assert ExecutionManager._use_pipeline_microbatching(
        active_count=16,
        num_stages=4,
        microbatch_count=4,
        microbatch_req_count=4,
        min_token_bucket=4,
        has_compiled_handoff=False,
    )
    assert not ExecutionManager._use_pipeline_microbatching(
        active_count=4,
        num_stages=4,
        microbatch_count=4,
        microbatch_req_count=1,
        min_token_bucket=4,
        has_compiled_handoff=False,
    )
    assert ExecutionManager._use_pipeline_microbatching(
        active_count=4,
        num_stages=4,
        microbatch_count=4,
        microbatch_req_count=1,
        min_token_bucket=4,
        has_compiled_handoff=True,
    )
    assert not ExecutionManager._use_pipeline_microbatching(
        active_count=3,
        num_stages=4,
        microbatch_count=3,
        microbatch_req_count=1,
        min_token_bucket=4,
        has_compiled_handoff=True,
    )


def test_pipeline_microbatch_shape_supports_auto_count_size_and_disable():
    assert ExecutionManager._resolve_pipeline_microbatch_shape(
        active_count=16,
        num_stages=4,
        pp_microbatch_count="auto",
        pp_microbatch_size="auto",
    ) == (4, 4)
    assert ExecutionManager._resolve_pipeline_microbatch_shape(
        active_count=16,
        num_stages=4,
        pp_microbatch_count=8,
        pp_microbatch_size="auto",
    ) == (8, 2)
    assert ExecutionManager._resolve_pipeline_microbatch_shape(
        active_count=16,
        num_stages=4,
        pp_microbatch_count="auto",
        pp_microbatch_size=5,
    ) == (4, 5)
    assert (
        ExecutionManager._resolve_pipeline_microbatch_shape(
            active_count=16,
            num_stages=4,
            pp_microbatch_count=0,
            pp_microbatch_size="auto",
        )
        is None
    )
