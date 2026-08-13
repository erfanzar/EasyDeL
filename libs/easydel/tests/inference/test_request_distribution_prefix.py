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

"""Regression: v3 request_distribution must honor the POSITIONAL contract.

The ragged-page-attention v3 kernel runs rows [0, decode_end) on a static
q_len=1 path. Rows are persistent slots, so when a finished request's slot is
reused by a new prefill while later slots still decode, count-based
classification put the prefill inside the decode range — desyncing the
kernel's offsets and hanging the TPU (reproduced: qwen3_5-4B and
qwen3-30B-A3B, any run with more requests than seats; scheduler heartbeat
stale within ~2 min of the first admission-after-completion).
"""

import numpy as np
from easydel.inference.esurge.runners.executors.batch_preparer import compute_request_distribution_prefix


def _dist(scheduled, computed):
    scheduled = np.asarray(scheduled, dtype=np.int32)
    computed = np.asarray(computed, dtype=np.int32)
    return compute_request_distribution_prefix(scheduled, computed, len(scheduled))


def test_pure_decode_batch_keeps_fast_path():
    d, p, t = _dist([1] * 8, [100] * 8)
    assert (d, p, t) == (8, 8, 8)


def test_initial_prefill_wave_keeps_fast_path():
    d, p, t = _dist([512] * 8, [0] * 8)
    assert (d, p, t) == (0, 8, 8)


def test_ordered_mixed_batch_keeps_both_fast_paths():
    d, p, t = _dist([1, 1, 1, 512, 512], [50, 60, 70, 0, 0])
    assert (d, p, t) == (3, 5, 5)


def test_reused_slot_prefill_amid_decodes_goes_to_mixed():
    """THE hang case: slot 2 freed and reused by a new 512-token prefill
    while slots 0-1 and 3-4 still decode. The prefill and everything after
    it must leave the static-decode range."""
    d, p, t = _dist([1, 1, 512, 1, 1], [50, 60, 0, 70, 80])
    assert d == 2, "decode prefix must stop at the interleaved prefill"
    assert p == 3, "prefill run is exactly the one interleaved prefill"
    # rows 3-4 (decodes out of order) are in the mixed range [p, t)
    assert t == 5


def test_budget_skipped_row_terminates_prefix():
    """An inactive (scheduled=0) row mid-array must not sit inside the
    static decode range either."""
    d, p, t = _dist([1, 0, 1, 1], [50, 60, 70, 80])
    assert d == 1
    assert p == 1
    assert t == 4


def test_chunked_prefill_resume_is_prefill_not_decode():
    """A resumed chunked prefill has computed>0 but scheduled>1."""
    d, p, _t = _dist([1, 256, 1], [50, 512, 60])
    assert (d, p) == (1, 2)


def test_single_token_prompt_is_prefill():
    """scheduled==1 with computed==0 is a 1-token prompt, not a decode."""
    d, p, _t = _dist([1, 1], [0, 100])
    assert d == 0 and p == 1


def test_empty_batch():
    assert _dist([], []) == (0, 0, 0)
