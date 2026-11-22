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

"""Parity and invariants of surgecore's HostPayloadBuilder (eSurge's step-input builder).

``data/host_payload_goldens.npz`` holds inputs and outputs captured from eSurge's former Python builder
(``BatchMetadataPreparer._build_host_payload``; the fixture header names the commit it came from). The parity test
replays every case through the native builder over staging buffers that start from the same fill pattern and compares
every staging buffer bitwise, so the regions the builder must leave untouched are compared too. The invariant tests
need no reference: they check the packed step inputs against their definitions.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest
from surgecore import HostPayloadBuilder, request_distribution_prefix

pytestmark = pytest.mark.cpu_ok

_FIXTURE = pathlib.Path(__file__).parent / "data" / "host_payload_goldens.npz"
with np.load(_FIXTURE) as _npz:
    _ARRAYS = {name: _npz[name] for name in _npz.files}
_HEADER = json.loads(str(_ARRAYS.pop("header")))
_CASES = {case["name"]: case for case in _HEADER["cases"]}


def _unpack(prefix: str, key: str, ref: list) -> np.ndarray:
    offset, shape = ref
    flat = _ARRAYS[f"{prefix}_{key}"]
    return flat[offset : offset + math.prod(shape)].reshape(shape).copy()


def _golden_staging(case: dict) -> dict[str, np.ndarray]:
    fill = _HEADER["fill"]
    return {
        name: np.full(shape, fill["f32"], np.float32)
        if name == "packed_f32_padded"
        else np.full(shape, fill["i32"], np.int32)
        for name, shape in case["buffers"].items()
    }


def _positional(scheduled: list[int], computed: list[int]) -> tuple[int, int]:
    """(decode_end, prefill_end) of the v3 positional request-distribution contract, from its definition."""
    kinds = [0 if s <= 0 else (1 if s == 1 and c > 0 else 2) for s, c in zip(scheduled, computed, strict=True)]
    decode_end = 0
    while decode_end < len(kinds) and kinds[decode_end] == 1:
        decode_end += 1
    prefill_end = decode_end
    while prefill_end < len(kinds) and kinds[prefill_end] == 2:
        prefill_end += 1
    return decode_end, prefill_end


def test_fixture_names_its_source():
    assert _HEADER["source"].endswith("BatchMetadataPreparer._build_host_payload")
    assert len(_HEADER["source_commit"]) == 12
    assert len(_CASES) >= 100
    assert sum(case["error"] is not None for case in _CASES.values()) >= 5


@pytest.mark.parametrize("name", sorted(_CASES))
def test_matches_python_goldens_bitwise(name):
    case = _CASES[name]
    buffers = _golden_staging(case)
    inputs = {key: _unpack("in", key, ref) for key, ref in case["inputs"].items()}
    builder = HostPayloadBuilder(buffers=buffers, **case["config"])
    if case["error"] is not None:
        with pytest.raises(ValueError) as excinfo:
            builder.build(**inputs, **case["scalars"])
        assert str(excinfo.value) == case["error"]
        return
    result = builder.build(**inputs, **case["scalars"])
    assert list(result) == case["result"]
    for buffer_name, ref in case["outputs"].items():
        expected = _unpack("out", buffer_name, ref)
        got = buffers[buffer_name]
        assert got.dtype == expected.dtype and got.shape == expected.shape, buffer_name
        np.testing.assert_array_equal(got.view(np.int32), expected.view(np.int32), err_msg=buffer_name)


_ROWS, _TOKENS, _MODEL_LEN, _PAGES, _PAGE_SIZE, _SLICES = 16, 96, 64, 8, 8, 64


def _layout(*, version: str, dp: int, W: int) -> tuple[dict[str, np.ndarray], dict]:
    rows_per_rank = _ROWS // dp
    buffers = {
        "scheduled": np.zeros(_ROWS, np.int32),
        "query_start_loc": np.zeros(_ROWS + 1, np.int32),
        "input_ids": np.zeros(_TOKENS, np.int32),
        "positions": np.zeros(_TOKENS, np.int32),
        "seq_lens": np.zeros(_ROWS, np.int32),
        "logits_indices": np.zeros(_ROWS, np.int32),
        "pages_tables": np.zeros((_ROWS, _PAGES), np.int32),
        "packed_qsl_seqlens": np.zeros((2, _ROWS + 1), np.int32),
        "packed_i32_padded": np.zeros((3, _ROWS), np.int32),
        "packed_f32_padded": np.zeros((6, _ROWS), np.float32),
        "packed_misc_i32": np.zeros(5, np.int32),
        "spec_recurrent_commit": np.zeros((2, _ROWS), np.int32),
        "request_distribution": np.zeros(3, np.int32),
        "dp_query_start_loc": np.zeros((dp, rows_per_rank + 1), np.int32),
        "dp_request_distribution": np.zeros((dp, 3), np.int32),
        "dp_context_lens": np.zeros((dp, rows_per_rank), np.int32),
        "dp_recurrent_state_indices": np.zeros((dp, rows_per_rank), np.int32),
        "dp_model_row_for_request": np.full(_ROWS, -1, np.int32),
        "num_kv_update": np.zeros(1, np.int32),
    }
    if version == "v2":
        buffers["slot_mapping"] = np.zeros((3, _SLICES), np.int32)
    config = {
        "max_num_reqs": _ROWS,
        "num_reqs_max_model_len": _ROWS,
        "data_parallel_size": dp,
        "use_slot_mapping": version == "v2",
        "use_request_distribution": version != "v2",
        "plain_recurrent_slots": False,
        "recurrent_slot_group_size": W,
        "enable_spec_recurrent_commit": False,
        "enable_dp_local_page_path": True,
        "page_size": _PAGE_SIZE,
        "max_num_pages_per_req": _PAGES,
        "slices_per_page": 2,
        "max_padded_slices": _SLICES,
    }
    return buffers, config


def _random_window(rng: np.random.Generator, *, dp: int, W: int) -> dict:
    rows_per_rank = _ROWS // dp
    window = rng.permutation(_ROWS).astype(np.int32) if dp > 1 else np.arange(_ROWS, dtype=np.int32)
    rank = window // rows_per_rank
    budget = np.full(dp, _TOKENS // dp)
    scheduled = np.zeros(_ROWS, np.int32)
    computed = np.zeros(_ROWS, np.int32)
    active = np.zeros(_ROWS, np.bool_)
    for i in range(_ROWS):
        kind = int(rng.integers(0, 5))  # decode, prefill chunk, prefix-cache hit, budget-skipped, inactive
        if kind == 0:
            c, s = int(rng.integers(1, _MODEL_LEN - 1)), 1
        elif kind == 1:
            c, s = 0, int(rng.integers(1, 24))
        elif kind == 2:
            c = int(rng.choice([_PAGE_SIZE, 2 * _PAGE_SIZE, int(rng.integers(1, _MODEL_LEN - 2))]))
            s = int(rng.integers(2, _MODEL_LEN - c + 1))
        else:
            c, s = int(rng.integers(0, _MODEL_LEN - 1)), 0
        s = int(min(s, budget[rank[i]], _MODEL_LEN - c))
        budget[rank[i]] -= s
        scheduled[i], computed[i], active[i] = s, c, kind != 4
    pages_per_shard = 64
    table = np.zeros((_ROWS, _PAGES), np.int32)
    for i in range(_ROWS):
        lo = 1 if rank[i] == 0 else int(rank[i]) * pages_per_shard
        table[i] = rng.permutation(np.arange(lo, (int(rank[i]) + 1) * pages_per_shard))[:_PAGES]
    inputs = {
        "scheduled_full": scheduled,
        "active_mask_full": active,
        "token_ids": (np.arange(_ROWS)[:, None] * 1000 + np.arange(_MODEL_LEN)[None, :]).astype(np.int32),
        "num_computed_tokens": computed,
        "temperature": rng.random(_ROWS, dtype=np.float32),
        "top_p": rng.random(_ROWS, dtype=np.float32),
        "top_k": rng.integers(0, 64, _ROWS).astype(np.int32),
        "min_p": rng.random(_ROWS, dtype=np.float32),
        "frequency_penalties": rng.random(_ROWS, dtype=np.float32),
        "presence_penalties": rng.random(_ROWS, dtype=np.float32),
        "repetition_penalties": rng.random(_ROWS, dtype=np.float32),
        "page_table": table,
        "num_tokens_static": _TOKENS,
        "padded_num_reqs_in": int(rng.integers(1, _ROWS + 1)),
        "num_pages": pages_per_shard * dp,
        "metadata_data_parallel_size": dp,
    }
    if dp > 1:
        slots_per_rank = rows_per_rank * W
        inputs["window_row_indices"] = window
        inputs["recurrent_slot_indices"] = (rank * slots_per_rank + rng.integers(0, slots_per_rank, _ROWS)).astype(
            np.int32
        )
    return inputs


@pytest.mark.parametrize("version", ["v3", "v2"])
@pytest.mark.parametrize(("dp", "W"), [(1, 1), (2, 1), (2, 2), (4, 3)])
@pytest.mark.parametrize("seed", range(8))
def test_invariants_without_reference(version, dp, W, seed):
    rng = np.random.default_rng([seed, dp, W, version == "v2"])
    buffers, config = _layout(version=version, dp=dp, W=W)
    inp = _random_window(rng, dp=dp, W=W)
    num_requests, padded, rows_to_copy, _reuse, total_pages = HostPayloadBuilder(buffers=buffers, **config).build(**inp)

    live = np.flatnonzero(inp["active_mask_full"])
    assert num_requests == (int(live[-1]) + 1 if live.size else 0)
    n = num_requests
    sched = inp["scheduled_full"][:n].astype(np.int64)
    computed = inp["num_computed_tokens"][:n].astype(np.int64)
    tokens, table = inp["token_ids"], inp["page_table"]
    qsl, ids, pos = buffers["query_start_loc"], buffers["input_ids"], buffers["positions"]
    # qsl is the prefix sum of the scheduled counts; seq_lens are computed + scheduled (zero when nothing is scheduled).
    assert qsl[0] == 0
    np.testing.assert_array_equal(qsl[1 : n + 1], np.cumsum(sched))
    assert (qsl[n + 1 :] == qsl[n]).all()
    np.testing.assert_array_equal(buffers["seq_lens"][:n], np.where(sched > 0, computed + sched, 0))
    np.testing.assert_array_equal(buffers["packed_qsl_seqlens"][0], qsl)
    np.testing.assert_array_equal(buffers["packed_qsl_seqlens"][1, :_ROWS], buffers["seq_lens"])
    np.testing.assert_array_equal(buffers["packed_i32_padded"][2, :padded], inp["top_k"][:padded])
    np.testing.assert_array_equal(buffers["packed_f32_padded"][5, :padded], inp["repetition_penalties"][:padded])

    # Positions are the computed counts plus the row offsets, and each run carries its row's tokens.
    rows_per_rank, tokens_per_rank = _ROWS // dp, _TOKENS // dp
    used = np.zeros(_TOKENS, np.bool_)
    for i in range(n):
        s, c = int(sched[i]), int(computed[i])
        if s == 0:
            continue
        if dp == 1:
            start = int(qsl[i])
        else:
            rank = int(inp["window_row_indices"][i]) // rows_per_rank
            model_row = int(buffers["dp_model_row_for_request"][i])
            assert model_row // rows_per_rank == rank
            local = model_row - rank * rows_per_rank
            rank_qsl = buffers["dp_query_start_loc"][rank]
            assert int(rank_qsl[local + 1]) - int(rank_qsl[local]) == s
            assert int(rank_qsl[local + 1]) <= tokens_per_rank
            start = rank * tokens_per_rank + int(rank_qsl[local])
            assert buffers["dp_context_lens"][rank, local] == c + s
            slot = int(inp["recurrent_slot_indices"][i])
            assert buffers["dp_recurrent_state_indices"][rank, local] == slot - rank * rows_per_rank * W
            np.testing.assert_array_equal(buffers["pages_tables"][model_row], table[i])
        np.testing.assert_array_equal(pos[start : start + s], c + np.arange(s))
        np.testing.assert_array_equal(ids[start : start + s], tokens[i, c : c + s])
        assert buffers["logits_indices"][i] == start + s - 1
        assert not used[start : start + s].any()
        used[start : start + s] = True
    assert not ids[~used].any() and not pos[~used].any()

    if dp == 1:
        np.testing.assert_array_equal(buffers["logits_indices"][:n], qsl[1 : n + 1] - 1)
        np.testing.assert_array_equal(buffers["pages_tables"][:rows_to_copy], table[:rows_to_copy])
        assert (buffers["pages_tables"][rows_to_copy:] == 0).all()
    else:
        # DP blocks are rank-major: a rank's requests take its local rows in request order.
        for rank in range(dp):
            members = [
                i for i in range(n) if sched[i] > 0 and int(inp["window_row_indices"][i]) // rows_per_rank == rank
            ]
            locals_ = [int(buffers["dp_model_row_for_request"][i]) - rank * rows_per_rank for i in members]
            assert locals_ == list(range(len(members)))
            lens = [int(sched[i]) for i in members]
            rank_qsl = buffers["dp_query_start_loc"][rank]
            np.testing.assert_array_equal(rank_qsl[1 : len(lens) + 1], np.cumsum(lens, dtype=np.int64))
            assert (rank_qsl[len(lens) + 1 :] == sum(lens)).all()
            decode_end, prefill_end = _positional(lens, [int(computed[i]) for i in members])
            np.testing.assert_array_equal(buffers["dp_request_distribution"][rank], [decode_end, prefill_end, len(lens)])

    if version == "v3":
        decode_end, prefill_end = _positional(sched.tolist(), computed.tolist())
        np.testing.assert_array_equal(buffers["request_distribution"], [decode_end, prefill_end, n])
        np.testing.assert_array_equal(buffers["packed_misc_i32"], [n, padded, decode_end, prefill_end, n])
        assert total_pages == 0
    else:
        # Every v2 slot lies in a page of its request's page-table row and moves exactly that request's tokens.
        slots = buffers["slot_mapping"]
        column = 0
        for i in range(n):
            s, c = int(sched[i]), int(computed[i])
            if s == 0:
                continue
            covered = 0
            for page_index in range(c // _PAGE_SIZE, (c + s - 1) // _PAGE_SIZE + 1):
                kv_start, new_start, length = (int(v) for v in slots[:, column])
                page, offset = divmod(kv_start, _PAGE_SIZE)
                assert page == table[i, page_index]
                assert length > 0 and offset + length <= _PAGE_SIZE
                first = page_index * _PAGE_SIZE + offset
                np.testing.assert_array_equal(pos[new_start : new_start + length], first + np.arange(length))
                np.testing.assert_array_equal(ids[new_start : new_start + length], tokens[i, first : first + length])
                covered += length
                column += 1
            assert covered == s
        assert total_pages == column == int(buffers["num_kv_update"][0])
        assert not slots[:, column:].any()


@pytest.mark.parametrize("seed", range(24))
def test_request_distribution_prefix_follows_the_positional_contract(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 12))
    scheduled = rng.choice([0, 1, 1, 1, 3], size=n).astype(np.int32)
    computed = rng.choice([0, 0, 5], size=n).astype(np.int32)
    assert request_distribution_prefix(scheduled, computed, n) == (
        *_positional(scheduled.tolist(), computed.tolist()),
        n,
    )


def test_missing_staging_buffer_is_rejected():
    buffers, config = _layout(version="v2", dp=1, W=1)
    del buffers["slot_mapping"]
    with pytest.raises(ValueError, match="slot_mapping"):
        HostPayloadBuilder(buffers=buffers, **config)
