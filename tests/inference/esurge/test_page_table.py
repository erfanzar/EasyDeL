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

"""Tests for ``easydel.inference.esurge.page_table``.

The page table is a CPU/GPU dual-mirror data structure backing paged-attention
KV cache management. CPU writes go to ``page_table_cpu`` and only become
visible on device after ``commit(num_reqs)``. We test:

* ``cdiv`` ceiling division
* ``PageTable.append_row`` / ``clear_row`` / ``add_row`` / ``move_row`` /
  ``swap_row`` / ``clear`` semantics on the CPU mirror
* CPU-version bumping (every mutation increments)
* ``commit(N)`` propagates the first N rows to the device array
* ``MultiGroupPageTable`` fans out operations across groups and indexes via
  ``[]``
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from easydel.inference.esurge.page_table import (
    PAGE_TABLE_PADDING_VAL,
    MultiGroupPageTable,
    PageTable,
    cdiv,
)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (7, 3, 3),
        (6, 3, 2),
        (5, 3, 2),
        (0, 5, 0),
        (1, 1, 1),
        (10, 1, 10),
        (100, 7, 15),
    ],
)
def test_cdiv(a: int, b: int, expected: int):
    assert cdiv(a, b) == expected


def _make_table(*, max_num_reqs=4, max_pages=8, max_batch=64) -> PageTable:
    return PageTable(
        max_num_reqs=max_num_reqs,
        max_num_pages_per_req=max_pages,
        max_num_batched_tokens=max_batch,
    )


def test_page_table_initial_state_is_zero():
    pt = _make_table()
    assert pt.page_table_cpu.shape == (4, 8)
    assert np.all(pt.page_table_cpu == 0)
    assert np.all(pt.num_pages_per_row == 0)
    assert pt.cpu_version == 0


def test_append_row_writes_pages_at_offset():
    pt = _make_table()
    pt.append_row([10, 11, 12], row_idx=0)
    assert list(pt.page_table_cpu[0, :3]) == [10, 11, 12]
    assert pt.num_pages_per_row[0] == 3
    assert pt.cpu_version == 1


def test_append_row_appends_to_existing_pages():
    pt = _make_table()
    pt.append_row([10, 11], row_idx=0)
    pt.append_row([20, 21], row_idx=0)
    assert list(pt.page_table_cpu[0, :4]) == [10, 11, 20, 21]
    assert pt.num_pages_per_row[0] == 4
    assert pt.cpu_version == 2


def test_append_row_empty_is_noop():
    pt = _make_table()
    before = pt.cpu_version
    pt.append_row([], row_idx=0)
    assert pt.cpu_version == before
    assert pt.num_pages_per_row[0] == 0


def test_clear_row_resets_pages_and_count():
    pt = _make_table()
    pt.append_row([10, 11, 12], row_idx=0)
    pt.clear_row(0)
    assert list(pt.page_table_cpu[0, :3]) == [PAGE_TABLE_PADDING_VAL] * 3
    assert pt.num_pages_per_row[0] == 0


def test_clear_row_independent_of_other_rows():
    pt = _make_table()
    pt.append_row([10, 11], row_idx=0)
    pt.append_row([20, 21, 22], row_idx=1)
    pt.clear_row(0)

    assert list(pt.page_table_cpu[1, :3]) == [20, 21, 22]
    assert pt.num_pages_per_row[1] == 3


def test_add_row_replaces_existing_content():
    pt = _make_table()
    pt.append_row([10, 11, 12], row_idx=0)
    pt.add_row([99, 100], row_idx=0)
    assert list(pt.page_table_cpu[0, :2]) == [99, 100]
    assert pt.num_pages_per_row[0] == 2

    assert pt.page_table_cpu[0, 2] == 0


def test_move_row_copies_source_to_target():
    pt = _make_table()
    pt.append_row([10, 11, 12], row_idx=0)
    pt.move_row(src=0, tgt=2)
    assert list(pt.page_table_cpu[2, :3]) == [10, 11, 12]
    assert pt.num_pages_per_row[2] == 3

    assert list(pt.page_table_cpu[0, :3]) == [10, 11, 12]
    assert pt.num_pages_per_row[0] == 3


def test_move_row_clears_target_beyond_new_length():
    """Per the docstring: stale pages beyond the moved range must be wiped."""
    pt = _make_table()
    pt.append_row([10, 11, 12, 13, 14], row_idx=2)
    pt.append_row([99], row_idx=0)
    pt.move_row(src=0, tgt=2)

    assert pt.page_table_cpu[2, 0] == 99
    assert all(pt.page_table_cpu[2, 1:] == PAGE_TABLE_PADDING_VAL)
    assert pt.num_pages_per_row[2] == 1


def test_swap_row_exchanges_pages_and_counts():
    pt = _make_table()
    pt.append_row([10, 11], row_idx=0)
    pt.append_row([20, 21, 22], row_idx=1)
    pt.swap_row(src=0, tgt=1)
    assert pt.num_pages_per_row[0] == 3
    assert pt.num_pages_per_row[1] == 2
    assert list(pt.page_table_cpu[0, :3]) == [20, 21, 22]
    assert list(pt.page_table_cpu[1, :2]) == [10, 11]


def test_clear_resets_everything():
    pt = _make_table()
    pt.append_row([10, 11], row_idx=0)
    pt.append_row([20], row_idx=2)
    pt.commit(num_reqs=4)
    pt.clear()
    assert np.all(pt.page_table_cpu == 0)
    assert np.all(pt.num_pages_per_row == 0)
    assert np.all(pt.page_table == 0)


def test_commit_propagates_cpu_to_device():
    pt = _make_table()
    pt.append_row([10, 11, 12], row_idx=0)
    pt.append_row([20, 21], row_idx=1)

    assert int(jnp.sum(pt.page_table)) == 0
    pt.commit(num_reqs=2)

    device_rows = np.asarray(pt.page_table)[:2]
    assert list(device_rows[0, :3]) == [10, 11, 12]
    assert list(device_rows[1, :2]) == [20, 21]


def test_commit_partial_only_first_n_rows():
    """``commit(N)`` only mirrors the first N rows; later rows stay stale on device."""
    pt = _make_table()
    pt.append_row([10, 11], row_idx=0)
    pt.append_row([20, 21], row_idx=1)
    pt.append_row([30, 31], row_idx=2)
    pt.commit(num_reqs=2)
    device = np.asarray(pt.page_table)

    assert int(np.sum(device[2])) == 0


def test_get_device_and_cpu_tensor_handles():
    pt = _make_table()
    cpu = pt.get_cpu_tensor()
    dev = pt.get_device_tensor()
    assert cpu is pt.page_table_cpu

    assert dev.shape == cpu.shape


def test_cpu_version_increments_on_each_mutation():
    pt = _make_table()
    pt.append_row([1], row_idx=0)
    v1 = pt.cpu_version
    pt.clear_row(0)
    v2 = pt.cpu_version
    pt.add_row([2], row_idx=0)
    v3 = pt.cpu_version
    pt.move_row(0, 1)
    v4 = pt.cpu_version
    pt.swap_row(0, 1)
    v5 = pt.cpu_version
    assert v1 < v2 < v3 < v4 < v5


def _make_multi_group(*, page_sizes=(16, 32)) -> MultiGroupPageTable:
    return MultiGroupPageTable(
        max_num_reqs=4,
        max_model_len=128,
        max_num_batched_tokens=64,
        page_sizes=list(page_sizes),
    )


def test_multi_group_creates_one_table_per_page_size():
    mg = _make_multi_group(page_sizes=(16, 32, 64))
    assert len(mg.page_tables) == 3

    assert mg.page_tables[0].max_num_pages_per_req == 8
    assert mg.page_tables[1].max_num_pages_per_req == 4
    assert mg.page_tables[2].max_num_pages_per_req == 2


def test_multi_group_getitem_returns_specific_table():
    mg = _make_multi_group()
    assert isinstance(mg[0], PageTable)
    assert mg[0] is mg.page_tables[0]
    assert mg[1] is mg.page_tables[1]


def test_multi_group_append_row_fans_out():
    mg = _make_multi_group()
    mg.append_row([[10, 11], [20]], row_idx=0)
    assert list(mg[0].page_table_cpu[0, :2]) == [10, 11]
    assert mg[0].num_pages_per_row[0] == 2
    assert list(mg[1].page_table_cpu[0, :1]) == [20]
    assert mg[1].num_pages_per_row[0] == 1


def test_multi_group_add_row_replaces_in_each_group():
    mg = _make_multi_group()
    mg.append_row([[10, 11, 12], [20, 21]], row_idx=0)
    mg.add_row([[99], [100, 101]], row_idx=0)
    assert mg[0].num_pages_per_row[0] == 1
    assert mg[1].num_pages_per_row[0] == 2
    assert mg[0].page_table_cpu[0, 0] == 99
    assert list(mg[1].page_table_cpu[0, :2]) == [100, 101]


def test_multi_group_clear_row_clears_all_groups():
    mg = _make_multi_group()
    mg.append_row([[10, 11], [20, 21]], row_idx=0)
    mg.clear_row(row_idx=0)
    assert mg[0].num_pages_per_row[0] == 0
    assert mg[1].num_pages_per_row[0] == 0


def test_multi_group_move_row_fans_out_across_groups():
    mg = _make_multi_group()
    mg.append_row([[10, 11], [20]], row_idx=0)
    mg.move_row(src=0, tgt=2)
    assert list(mg[0].page_table_cpu[2, :2]) == [10, 11]
    assert list(mg[1].page_table_cpu[2, :1]) == [20]


def test_multi_group_swap_row_fans_out():
    mg = _make_multi_group()
    mg.append_row([[10, 11], [20]], row_idx=0)
    mg.append_row([[30], [40, 41]], row_idx=1)
    mg.swap_row(src=0, tgt=1)
    assert list(mg[0].page_table_cpu[0, :1]) == [30]
    assert list(mg[1].page_table_cpu[0, :2]) == [40, 41]
    assert list(mg[0].page_table_cpu[1, :2]) == [10, 11]
    assert list(mg[1].page_table_cpu[1, :1]) == [20]


def test_multi_group_commit_propagates_to_each_group():
    mg = _make_multi_group()
    mg.append_row([[10, 11], [20]], row_idx=0)
    mg.commit(num_reqs=1)
    assert int(np.asarray(mg[0].page_table)[0, 0]) == 10
    assert int(np.asarray(mg[1].page_table)[0, 0]) == 20


def test_multi_group_clear_clears_all_groups():
    mg = _make_multi_group()
    mg.append_row([[10, 11], [20]], row_idx=0)
    mg.clear()
    for pt in mg.page_tables:
        assert np.all(pt.page_table_cpu == 0)
        assert np.all(pt.num_pages_per_row == 0)


def test_multi_group_append_rows_batch_handles_multiple_requests():
    mg = _make_multi_group()
    mg.append_rows_batch(
        page_ids_per_req=[
            [[10, 11], [20, 21]],
            [[30], [40, 41, 42]],
        ],
        req_indices=[0, 1],
    )
    assert list(mg[0].page_table_cpu[0, :2]) == [10, 11]
    assert list(mg[1].page_table_cpu[0, :2]) == [20, 21]
    assert list(mg[0].page_table_cpu[1, :1]) == [30]
    assert list(mg[1].page_table_cpu[1, :3]) == [40, 41, 42]


def test_multi_group_append_rows_batch_empty_input_is_noop():
    mg = _make_multi_group()
    mg.append_rows_batch([], [])

    for pt in mg.page_tables:
        assert int(np.sum(pt.page_table_cpu)) == 0


def test_multi_group_append_rows_batch_strict_zip_mismatch_raises():
    """The ``zip(..., strict=True)`` enforces matching lengths."""
    mg = _make_multi_group()
    with pytest.raises(ValueError):
        mg.append_rows_batch(
            page_ids_per_req=[[[1], [2]], [[3], [4]]],
            req_indices=[0],
        )
