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
"""Decode buckets shorter than the row count run one entry per token, with the row-major results bit for bit.

The same decode step (12 rows, 6 live tokens, explicit slots: an out-of-range slot, empty rows naming slot 0, a fresh
row) runs once in an 8-token bucket (token-major: fewer tokens than rows) and once in a 12-token bucket (row-major).
Live outputs, padding outputs and every pool leaf must be bitwise equal, and the recurrence must really run over 8
entries in the short bucket and over 12 rows in the long one.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from easydel.operations.kernels import kda_rows_forward
from easydel.operations.kernels.ragged_kda_decode import RaggedKDADecodeOp

from .test_kda_decode_rows import LEAVES, POOL, _case, _f64, _run

LENS = (1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0)
CONTEXT = (9, 0, 1, 40, 0, 0, 3, 17, 0, 2, 0, 0)
SLOTS = (4, 0, 1, 7, 0, 0, 9, 2, 0, POOL + 3, 0, 0)
SHORT = 8


@pytest.mark.parametrize(("dtype", "tp"), [(jnp.float32, 1), (jnp.bfloat16, 4)], ids=["f32-tp1", "bf16-tp4"])
@pytest.mark.parametrize("implicit_slots", [False, True], ids=["explicit-slots", "row-owns-slot"])
def test_token_major_decode_matches_row_major_bitwise(dtype, tp, implicit_slots, monkeypatch):
    entries = []
    original = RaggedKDADecodeOp._run

    def recording(self, query, *args, **kwargs):
        entries.append(int(query.shape[0]))
        return original(self, query, *args, **kwargs)

    monkeypatch.setattr(RaggedKDADecodeOp, "_run", recording)
    # Without explicit slots row r owns slot r, so the rows must fit the POOL-slot pool (eSurge: max_num_reqs slots).
    lens, context, slots = (LENS[:POOL], CONTEXT[:POOL], None) if implicit_slots else (LENS, CONTEXT, SLOTS)
    long_len = len(lens)
    long_case = _case((lens, context, long_len), seed=3, dtype=dtype)
    short_case = dict(long_case)
    for name in ("q", "k", "v", "beta", "decay"):
        short_case[name] = long_case[name][:SHORT]
    short_out, short = _run(kda_rows_forward, short_case, (lens, context, SHORT), dtype, tp=tp, slots=slots)
    long_out, long = _run(kda_rows_forward, long_case, (lens, context, long_len), dtype, tp=tp, slots=slots)
    assert entries == [SHORT, long_len], entries
    np.testing.assert_array_equal(_f64(short_out)[0], _f64(long_out)[0][:SHORT])
    for name in LEAVES:
        a, b = np.asarray(short[name]), np.asarray(long[name])
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name
