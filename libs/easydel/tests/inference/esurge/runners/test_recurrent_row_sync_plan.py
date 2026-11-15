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

"""Host-only contract of the identity-layout recurrent row sync: one permute both moves survivors and zeroes freed rows.

The device-side permute (``perm[t] == -1`` zeroes row ``t``) is covered on TPU by ``test_recurrent_row_sync.py``; this
file pins the plan the runner hands it, and that the state sync no longer issues its own clear on that path.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.model_runner import eSurgeRunner
from easydel.inference.esurge.runners.slot_pool import RecurrentSlotPool
from easydel.inference.esurge.runners.state_sync import SequenceStateSync

pytestmark = pytest.mark.cpu_ok


def _runner(rows: list[str | None]):
    perms: list[object] = []
    runner = object.__new__(eSurgeRunner)
    runner.max_num_reqs = len(rows)
    runner.sequence_buffer = SimpleNamespace(req_ids=list(rows))
    runner.executor_manager = SimpleNamespace(
        permute_recurrent_slots=lambda perm: perms.append(("permute", np.asarray(perm).tolist())),
        clear_recurrent_slots=lambda slots: perms.append(("clear", list(slots))),
    )
    return runner, perms


def test_freed_row_without_moves_is_zeroed_by_one_targeted_clear():
    # "b" finished at row 1; nobody moved. The deferred clear must still zero row 1, and only row 1.
    runner, perms = _runner(["a", None, "c", None])
    runner._sync_recurrent_rows(["a", "b", "c", None], removed_req_ids={"b"})
    assert perms == [("clear", [1])]


def test_condense_moves_survivor_and_zeroes_the_freed_row_in_the_same_call():
    # "b" finished at row 1; condense moved "d" from row 3 to row 1. One permute, no separate clear.
    runner, perms = _runner(["a", "d", "c", None])
    runner._sync_recurrent_rows(["a", "b", "c", "d"], removed_req_ids={"b"})
    assert perms == [("permute", [0, 3, 2, -1])]


def test_preempted_and_readmitted_in_place_is_not_carried_over():
    # "b" was preempted and re-admitted into its old row in the same step: its stale state must be zeroed.
    runner, perms = _runner(["a", "b", "c", None])
    runner._sync_recurrent_rows(["a", "b", "c", None], removed_req_ids={"b"})
    assert perms == [("clear", [1])]


def test_no_removal_and_no_move_dispatches_nothing():
    runner, perms = _runner(["a", "b", None, None])
    runner._sync_recurrent_rows(["a", "b", None, None], removed_req_ids=())
    assert perms == []


def test_disabled_row_sync_dispatches_nothing():
    runner, perms = _runner(["a", None])
    runner._sync_recurrent_rows(None, removed_req_ids={"b"})
    assert perms == []


@pytest.mark.parametrize(
    ("defer", "pooled", "expect_clear"), [(True, False, False), (False, False, True), (True, True, True)]
)
def test_state_sync_leaves_identity_layout_clears_to_the_row_sync(defer, pooled, expect_clear):
    from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
    from easydel.inference.esurge.scheduler.output import CachedRequestData, NewRequestData, SchedulerOutput
    from easydel.inference.sampling_params import SamplingParams

    metadata = SimpleNamespace(data_parallel_size=1, num_pages=16)
    buffer = SequenceBuffer(4, 16, 16, 32, [2])
    pool = RecurrentSlotPool(metadata=metadata, max_num_reqs=4, slot_indexed_state=pooled)
    clears: list[list[int]] = []
    sync = SequenceStateSync(
        sequence_buffer=buffer,
        requests={},
        slot_pool=pool,
        metadata=metadata,
        executor_manager=SimpleNamespace(clear_recurrent_slots=lambda slots: clears.append(list(slots))),
        max_model_len=16,
        spec_decode_recurrent_candidates=False,
        spec_recurrent_commit_cpu=np.zeros((1, 4), dtype=np.int32),
        pending_spec_recurrent_commit_by_req={},
        defer_row_clears=defer,
    )
    admit = SchedulerOutput(
        [
            NewRequestData("a", [1, 2], SamplingParams(), ([1],), 0),
            NewRequestData("b", [1, 2], SamplingParams(), ([2],), 0),
        ],
        CachedRequestData.make_empty(),
        {"a": 2, "b": 2},
        4,
        {},
        [0],
        set(),
    )
    sync.update_states(admit)
    if pooled:
        pool.assign_slot("b", None)  # the runner pins pooled requests to a slot at admission
    finish = SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], {"b"})
    sync.update_states(finish)
    assert bool(clears) == expect_clear
