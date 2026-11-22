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

"""Recurrent slot groups and the slot-contract decision: host bookkeeping only, no array computation."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.runners.slot_contract import (
    check_identity_layout_row_windows,
    declared_slot_contract,
    lowest_slot_contract,
    uses_pooled_recurrent_slots,
)
from easydel.inference.esurge.runners.slot_pool import RecurrentSlotPool

pytestmark = pytest.mark.cpu_ok


def _pool(dp: int, max_num_reqs: int, group_size: int, pooled: bool = True) -> RecurrentSlotPool:
    return RecurrentSlotPool(
        metadata=SimpleNamespace(data_parallel_size=dp),
        max_num_reqs=max_num_reqs,
        slot_indexed_state=pooled,
        group_size=group_size,
    )


def test_one_slot_per_request_is_todays_layout():
    pool = _pool(1, 4, 1)
    assert [pool.assign_slot(f"r{i}", None) for i in range(4)] == [0, 1, 2, 3]
    assert pool.num_slots() == 4
    assert pool.group_slots(2) == [2]
    with pytest.raises(RuntimeError, match="No free recurrent state slots"):
        pool.assign_slot("r4", None)


def test_groups_are_contiguous_stable_and_reused():
    pool = _pool(1, 3, 4)
    assert pool.num_slots() == 12
    assert pool.assign_slot("a", None) == 0
    assert pool.assign_slot("b", None) == 4
    assert pool.assign_slot("a", None) == 0, "a request keeps its group"
    assert pool.group_slots(4) == [4, 5, 6, 7]
    assert pool.release_slot("a", forget_rank=True) == 0
    assert pool.assign_slot("c", None) == 0, "the lowest free group is handed out first"
    assert pool.assign_slot("d", None) == 8


def test_rank_major_groups_stay_inside_their_rank_block():
    pool = _pool(2, 4, 3)  # 2 requests per rank, 3 slots per request: 6 slots per rank
    assert pool.rows_per_dp_rank() == 2
    assert pool.slots_per_dp_rank() == 6
    assert pool.assign_slot("a", 1) == 6
    assert pool.assign_slot("b", 1) == 9
    assert all(6 <= slot < 12 for slot in pool.group_slots(9))
    with pytest.raises(RuntimeError, match="DP rank 1"):
        pool.assign_slot("c", 1)
    assert pool.assign_slot("d", 0) == 0
    with pytest.raises(RuntimeError, match="does not match"):
        pool.assign_slot("a", 0)
    assert pool.release_slot("b", forget_rank=False) == 9
    assert pool.assign_slot("e", 1) == 9


def test_pool_disabled_without_pooled_state_or_dp():
    pool = _pool(1, 4, 1, pooled=False)
    assert not pool.is_enabled()
    assert pool.assign_slot("a", None) is None


def test_contract_level_is_the_weakest_declaration_covering_every_recurrent_layer():
    assert lowest_slot_contract(["slots", "slots", "slots"], num_recurrent_layers=3) == "slots"
    assert lowest_slot_contract(["verify", "slots", "verify"], num_recurrent_layers=3) == "slots"
    assert lowest_slot_contract(["verify"] * 4, num_recurrent_layers=3) == "verify"
    assert lowest_slot_contract(["slots", "slots"], num_recurrent_layers=3) == "none"
    assert lowest_slot_contract([], num_recurrent_layers=0) == "none"


def test_declarations_are_read_from_the_class():
    class Declares:
        esurge_recurrent_slot_contract = "slots"

    class Silent:
        pass

    class Wrong:
        esurge_recurrent_slot_contract = "always"

    assert declared_slot_contract(Declares()) == "slots"
    assert declared_slot_contract(Silent()) is None
    with pytest.raises(ValueError, match="always"):
        declared_slot_contract(Wrong())


def test_identity_layout_refuses_several_row_windows():
    check_identity_layout_row_windows(
        family="tiny", rows_per_forward=4, max_num_reqs=4, speculative_candidate_rows=False
    )
    with pytest.raises(ValueError, match=r"rows per forward=2 .*recurrent_state_indices"):
        check_identity_layout_row_windows(
            family="tiny", rows_per_forward=2, max_num_reqs=4, speculative_candidate_rows=False
        )
    with pytest.raises(ValueError, match=r"rows per forward=2 .*speculative decoding"):
        check_identity_layout_row_windows(
            family="tiny", rows_per_forward=2, max_num_reqs=4, speculative_candidate_rows=True
        )


def _decide(**overrides) -> bool:
    options = dict(
        family="tiny",
        contract="slots",
        has_recurrent_layers=True,
        slot_indexed_cache=False,
        rank_major_dp=False,
        speculative_candidate_rows=False,
        rows_per_forward=4,
        max_num_reqs=4,
        sparse_row_packing=False,
        group_size=1,
    )
    options.update(overrides)
    return uses_pooled_recurrent_slots(**options)


def test_slots_are_pooled_exactly_where_the_row_layout_needs_them():
    # One window without sparse packing keeps the identity layout (in-place recurrent updates).
    assert _decide() is False
    assert _decide(sparse_row_packing=True) is True
    assert _decide(rows_per_forward=2) is True
    assert _decide(contract="verify", group_size=2) is True
    # Compressed-window caches are always pooled; rank-major DP and models without recurrent state never are.
    assert _decide(slot_indexed_cache=True, has_recurrent_layers=False) is True
    assert _decide(rows_per_forward=2, rank_major_dp=True) is False
    assert _decide(rows_per_forward=2, has_recurrent_layers=False) is False
    # Layers that do not declare the contract, and speculative candidate rows, keep the identity layout.
    assert _decide(sparse_row_packing=True, contract="none") is False
    assert _decide(sparse_row_packing=True, speculative_candidate_rows=True) is False
    with pytest.raises(ValueError, match="speculative decoding"):
        _decide(rows_per_forward=2, speculative_candidate_rows=True)
    with pytest.raises(ValueError, match="recurrent_state_indices"):
        _decide(rows_per_forward=2, contract="none")
    with pytest.raises(ValueError, match="verify"):
        _decide(group_size=2)
