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

"""Static window size of the inference-mode deferred MoE dispatch (pure integer arithmetic, no array computation).

The dispatch's numerics with windows shorter than ``tokens * top_k`` (several windows, skewed routing, padded last
window) are covered on TPU by ``test_moe_defer_tp_reduce_replicated.py``.
"""

import pytest
from easydel.layers.moe._moe_module import _deferred_window_rows

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    ("total_rows", "ep", "expected"),
    [
        (1024, 8, 256),  # 128 decode tokens x top-8 at ep8: 160 expected -> 256, not all 1,024 rows
        (256, 8, 256),  # 32 decode tokens: the window covers every row
        (128, 8, 128),  # 16 decode tokens: capped at total_rows
        (1024, 4, 512),  # ep4: 320 -> 512
        (32768 * 8, 8, 40960),  # prefill window keeps the 1024 granularity: 5/4 x 262,144 / 8 = 40,960
        (4096, 2, 3072),  # unchanged from before: 2,560 -> 3,072
        (1024, 1, 1024),  # ep1: no windows
    ],
)
def test_window_rows(total_rows, ep, expected):
    assert _deferred_window_rows(total_rows, ep, None) == expected


def test_override_wins_and_is_capped():
    assert _deferred_window_rows(1024, 8, 48) == 48
    assert _deferred_window_rows(100, 8, 4096) == 100


@pytest.mark.parametrize("total_rows", [8, 128, 1000, 1024, 4096, 81920, 262144])
@pytest.mark.parametrize("ep", [2, 4, 8, 16])
def test_window_covers_five_quarters_of_a_balanced_share(total_rows, ep):
    rows = _deferred_window_rows(total_rows, ep, None)
    assert rows == total_rows or rows >= -(-5 * total_rows // (4 * ep))
    assert rows == total_rows or rows % 256 == 0
