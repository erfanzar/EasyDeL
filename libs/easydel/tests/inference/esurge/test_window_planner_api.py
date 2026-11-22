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

"""WindowPlanner honors configured and live runtime token bucket ladders."""

import pytest
from easydel.inference.esurge.runners.window_planner import WindowPlanner

pytestmark = pytest.mark.cpu_ok


def test_window_planner_selects_bucket_with_or_without_legacy_token_ladder() -> None:
    buckets = [2, 4, 8]
    legacy = WindowPlanner(num_tokens_paddings=[16], max_num_seq_buckets=buckets)
    current = WindowPlanner(max_num_seq_buckets=buckets)
    for planner in (legacy, current):
        assert planner.get_current_bucket(3) == 4
        assert planner.get_current_bucket(100) == 8
        assert planner.get_current_bucket(3, active_buckets=[2, 3]) == 3


def test_token_bucket_selection_uses_live_runtime_ladder_when_supplied() -> None:
    planner = WindowPlanner(num_tokens_paddings=[16, 32], max_num_seq_buckets=[1])
    assert planner.get_token_bucket(17) == 32
    assert planner.get_token_bucket(17, active_buckets=[24, 48]) == 24
    assert planner.get_token_bucket(100) == 32
    planner.num_tokens_paddings = [64, 128]
    assert planner.get_token_bucket(17) == 64


def test_token_bucket_selection_requires_a_ladder() -> None:
    planner = WindowPlanner(max_num_seq_buckets=[1])
    with pytest.raises(ValueError, match="token compile bucket"):
        planner.get_token_bucket(1)
