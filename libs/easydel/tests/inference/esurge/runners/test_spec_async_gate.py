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

"""Which speculative steps may be dispatched before the previous step is drained (host-only gate)."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.model_runner import eSurgeRunner

pytestmark = pytest.mark.cpu_ok

K = 3


class _ProgramDrafter:
    def spec_step(self, *args, **kwargs):
        raise AssertionError("not called by the gate")


def _runner(drafter=None, version="v3", dp=1, penalty=0.0):
    runner = object.__new__(eSurgeRunner)
    runner.async_scheduling = True
    runner.drafter = drafter if drafter is not None else _ProgramDrafter()
    runner.num_draft_tokens = K
    runner.metadata = SimpleNamespace(version=version, data_parallel_size=dp)
    runner.executor_manager = SimpleNamespace(cache_group_configs=())
    runner.model = SimpleNamespace()
    runner._model_uses_vlm_inputs = lambda: False
    runner.sequence_buffer = SimpleNamespace(
        frequency_penalties=np.full((4,), penalty, np.float32),
        presence_penalties=np.zeros((4,), np.float32),
        repetition_penalties=np.ones((4,), np.float32),
    )
    return runner


def _plan(*, async_safe=True, windows=None, drafts=None):
    windows = {"a": 1 + K, "b": 1 + K, "c": 1} if windows is None else windows
    drafts = {"a": [-1] * K, "b": [-1] * K} if drafts is None else drafts
    return SimpleNamespace(
        async_scheduling=True,
        async_safe=async_safe,
        total_num_scheduled_tokens=sum(windows.values()),
        num_scheduled_tokens=windows,
        scheduled_spec_decode_tokens=drafts,
        pending_structured_output_tokens=False,
    )


def test_verify_windows_of_one_plus_k_are_deferred():
    assert _runner().can_dispatch_next_before_async_drain(_plan())


@pytest.mark.parametrize(
    ("runner_kwargs", "plan_kwargs"),
    [
        ({}, {"async_safe": False}),  # the core did not mark the plan async-safe
        ({}, {"windows": {"a": 5}, "drafts": {}}),  # a chunk, not a verify window
        ({}, {"windows": {"a": 3}, "drafts": {"a": [-1, -1]}}),  # a trimmed window (budgeted, sync only)
        ({"drafter": SimpleNamespace()}, {}),  # a drafter that drafts on the host
        ({"version": "v2"}, {}),  # host-built position-dependent KV write map
        ({"dp": 2}, {}),  # rank-major DP rows
        ({"penalty": 0.5}, {}),  # penalties read host token history
    ],
)
def test_steps_the_device_rewrite_cannot_serve_stay_synchronous(runner_kwargs, plan_kwargs):
    assert not _runner(**runner_kwargs).can_dispatch_next_before_async_drain(_plan(**plan_kwargs))
