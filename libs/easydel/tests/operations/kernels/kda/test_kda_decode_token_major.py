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
    # Every traced branch runs SHORT entries in the short bucket and long_len entries in the long one.
    assert set(entries) == {SHORT, long_len}, entries
    np.testing.assert_array_equal(_f64(short_out)[0], _f64(long_out)[0][:SHORT])
    for name in LEAVES:
        a, b = np.asarray(short[name]), np.asarray(long[name])
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name


@pytest.mark.parametrize(("dtype", "tp"), [(jnp.float32, 1), (jnp.bfloat16, 4)], ids=["f32-tp1", "bf16-tp4"])
@pytest.mark.parametrize("live", [5, 8, 0], ids=["five-leading-rows", "bucket-full", "no-live-row"])
def test_prefix_rows_decode_matches_row_major_bitwise(dtype, tp, live):
    # Leading live rows (row r's token at position r) take the prefix branch in the short bucket; it must equal the
    # row-major decode of the long bucket bit for bit, outputs and every pool leaf, slots past the prefix untouched.
    lens = tuple([1] * live + [0] * (POOL - live))
    context = tuple(([9, 1, 40, 2, 3, 17, 4, 6] * 2)[:live] + [0] * (POOL - live))
    long_case = _case((lens, context, POOL), seed=5, dtype=dtype)
    short_case = dict(long_case)
    for name in ("q", "k", "v", "beta", "decay"):
        short_case[name] = long_case[name][:SHORT]
    short_out, short = _run(kda_rows_forward, short_case, (lens, context, SHORT), dtype, tp=tp, slots=None)
    long_out, long = _run(kda_rows_forward, long_case, (lens, context, POOL), dtype, tp=tp, slots=None)
    np.testing.assert_array_equal(_f64(short_out)[0], _f64(long_out)[0][:SHORT])
    for name in LEAVES:
        a, b = np.asarray(short[name]), np.asarray(long[name])
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name


def test_rows_are_tokens_rejects_a_stream_longer_than_the_rows():
    # The flag is a static promise of one stream position per row; a mismatched call must fail at trace time.
    from easydel.operations.kernels.ragged_kda_decode import kda_decode_rows_forward

    def driver(*, chunk_size, decay_lower_bound, precision, **kwargs):
        del chunk_size, decay_lower_bound, precision
        return kda_decode_rows_forward(**kwargs, rows_are_tokens=True)

    lens, context = (1, 1, 0, 0), (3, 4, 0, 0)
    case = _case((lens, context, 8), seed=2, dtype=jnp.float32)
    with pytest.raises(ValueError, match="rows_are_tokens"):
        _run(driver, case, (lens, context, 8), jnp.float32, tp=1)


@pytest.mark.parametrize(("dtype", "tp"), [(jnp.float32, 1), (jnp.bfloat16, 4)], ids=["f32-tp1", "bf16-tp4"])
@pytest.mark.parametrize("live", [POOL, 7], ids=["all-rows", "seven-leading-rows"])
def test_full_bucket_prefix_matches_row_major_bitwise(dtype, tp, live):
    # A bucket of exactly one position per row whose live rows lead takes the prefix branch; it must equal the
    # row-major decode driver called directly (per-row gathers and output scatter) bit for bit.
    from easydel.operations.kernels.ragged_kda_decode import kda_decode_rows_forward

    def row_major(*, chunk_size, decay_lower_bound, precision, **kwargs):
        del chunk_size, decay_lower_bound, precision
        return kda_decode_rows_forward(**kwargs)

    lens = tuple([1] * live + [0] * (POOL - live))
    context = tuple(([9, 1, 40, 2, 3, 17, 4, 6, 12, 5] * 2)[:live] + [0] * (POOL - live))
    case = _case((lens, context, POOL), seed=6, dtype=dtype)
    routed_out, routed = _run(kda_rows_forward, case, (lens, context, POOL), dtype, tp=tp, slots=None)
    direct_out, direct = _run(row_major, case, (lens, context, POOL), dtype, tp=tp, slots=None)
    assert np.asarray(routed_out).tobytes() == np.asarray(direct_out).tobytes()
    for name in LEAVES:
        assert np.asarray(routed[name]).tobytes() == np.asarray(direct[name]).tobytes(), name


@pytest.mark.parametrize(("dtype", "tp"), [(jnp.float32, 1), (jnp.bfloat16, 4)], ids=["f32-tp1", "bf16-tp4"])
def test_conv_window_select_and_concat_forms_match_bitwise(dtype, tp, monkeypatch):
    # Small steps build the shifted conv window with a select, large ones with a concatenate; forcing the
    # concatenate on a small step must not change any output or pool leaf.
    from easydel.operations.kernels import ragged_kda_decode

    lens = tuple([1] * 7 + [0] * (POOL - 7))
    context = tuple([9, 1, 40, 2, 3, 17, 4] + [0] * (POOL - 7))
    case = _case((lens, context, POOL), seed=7, dtype=dtype)
    assert POOL <= ragged_kda_decode._CONV_SELECT_MAX_ENTRIES
    select_out, select = _run(kda_rows_forward, case, (lens, context, POOL), dtype, tp=tp, slots=None)
    monkeypatch.setattr(ragged_kda_decode, "_CONV_SELECT_MAX_ENTRIES", 0)
    concat_out, concat = _run(kda_rows_forward, case, (lens, context, POOL), dtype, tp=tp, slots=None)
    assert np.asarray(select_out).tobytes() == np.asarray(concat_out).tobytes()
    for name in LEAVES:
        assert np.asarray(select[name]).tobytes() == np.asarray(concat[name]).tobytes(), name


@pytest.mark.parametrize(("dtype", "tp"), [(jnp.float32, 1), (jnp.bfloat16, 4)], ids=["f32-tp1", "bf16-tp4"])
@pytest.mark.parametrize(
    ("live", "gapped"), [(POOL, False), (7, False), (6, True), (0, False)], ids=["all", "seven", "gapped", "none"]
)
def test_single_token_program_keeps_the_leading_row_path_bitwise(dtype, tp, live, gapped):
    # eSurge's pure-decode program (single_token_rows) with implicit slots must still take the in-place leading-row
    # path when the live rows lead, and the plain decode driver otherwise; both must equal the general program bit
    # for bit (outputs and every pool leaf).
    from easydel.infra.sharding import single_token_rows

    if gapped:
        lens = (1, 0, 1, 1, 0, 0, 1, 1, 0, 1)
        context = (9, 0, 1, 40, 0, 0, 3, 17, 0, 2)
    else:
        lens = tuple([1] * live + [0] * (POOL - live))
        context = tuple(([9, 1, 40, 2, 3, 17, 4, 6, 12, 5] * 2)[:live] + [0] * (POOL - live))
    layout = (lens, context, POOL)
    case = _case(layout, seed=8, dtype=dtype)
    general_out, general = _run(kda_rows_forward, case, layout, dtype, tp=tp)
    with single_token_rows():
        single_out, single = _run(kda_rows_forward, case, layout, dtype, tp=tp)
    assert np.asarray(single_out).tobytes() == np.asarray(general_out).tobytes()
    for name in LEAVES:
        assert np.asarray(single[name]).tobytes() == np.asarray(general[name]).tobytes(), name
