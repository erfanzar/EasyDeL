# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for eray.resources.topology — host chip-split planning."""

import pytest

from eray.resources.topology import (
    DEFAULT_HOST_CHIP_GRIDS,
    DEFAULT_SPLIT_BASE_PORT,
    plan_host_split,
)


class TestIsolatedMode:
    def test_one_chip_per_split_on_v4_v5p_host(self):
        plan = plan_host_split(4, 4, mode="isolated")
        assert plan.chips_per_split == 1
        assert plan.chip_assignments is None  # delegated to Ray
        assert len(plan.env_per_split) == 4
        for i, env in enumerate(plan.env_per_split):
            assert env["ERAY_SPLIT_ID"] == str(i)
            assert env["ERAY_NUM_SPLITS"] == "4"
            assert env["ERAY_SPLIT_MODE"] == "isolated"
            # Chip assignment is Ray's job in isolated mode: the plan must
            # not pin chips or disable Ray's accelerator manager.
            assert "TPU_VISIBLE_CHIPS" not in env
            assert "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS" not in env

    def test_two_chips_per_split(self):
        plan = plan_host_split(4, 2, mode="isolated")
        assert plan.chips_per_split == 2
        assert plan.resources_per_split() == {"TPU": 2.0}

    def test_eight_chip_host_single_chip_splits(self):
        plan = plan_host_split(8, 8, mode="isolated")
        assert plan.chips_per_split == 1
        assert [e["ERAY_SPLIT_ID"] for e in plan.env_per_split] == [str(i) for i in range(8)]

    def test_whole_host_is_trivial(self):
        plan = plan_host_split(4, 1, mode="isolated")
        assert plan.chips_per_split == 4
        assert plan.chip_assignments == (tuple(range(4)),)
        assert plan.env_per_split[0]["ERAY_SPLIT_ID"] == "0"

    def test_four_chip_splits_rejected(self):
        # Ray's TPU accelerator manager only writes bounds env for 1- and
        # 2-chip subsets, so a 4-of-8 isolated split must be refused.
        with pytest.raises(ValueError, match="1 or 2 chips"):
            plan_host_split(8, 2, mode="isolated")


class TestCooperativeMode:
    def test_v4_v5p_host_process_topology(self):
        plan = plan_host_split(4, 4, mode="cooperative")
        # Chip assignment is Ray's; chip-dependent vars are bound in-task.
        assert plan.chip_assignments is None
        assert plan.base_port == DEFAULT_SPLIT_BASE_PORT
        addresses = ",".join(f"localhost:{DEFAULT_SPLIT_BASE_PORT + c}" for c in range(4))
        for i, env in enumerate(plan.env_per_split):
            assert env["ERAY_SPLIT_ID"] == str(i)
            assert env["TPU_PROCESS_BOUNDS"] == "2,2,1"
            assert env["TPU_CHIPS_PER_PROCESS_BOUNDS"] == "1,1,1"
            assert env["TPU_PROCESS_ADDRESSES"] == addresses
            assert env["RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS"] == "1"
            # Bound at runtime from Ray's assignment, never planned statically:
            assert "TPU_VISIBLE_CHIPS" not in env
            assert "CLOUD_TPU_TASK_ID" not in env
            assert "TPU_PROCESS_PORT" not in env

    def test_eight_chip_host_process_bounds(self):
        plan = plan_host_split(8, 8, mode="cooperative")
        assert plan.env_per_split[0]["TPU_PROCESS_BOUNDS"] == "2,4,1"

    def test_addresses_are_chip_ordered_from_base_port(self):
        plan = plan_host_split(4, 4, mode="cooperative", base_port=9000)
        assert plan.base_port == 9000
        assert plan.env_per_split[0]["TPU_PROCESS_ADDRESSES"] == (
            "localhost:9000,localhost:9001,localhost:9002,localhost:9003"
        )

    def test_explicit_chip_grid_override(self):
        plan = plan_host_split(4, 4, mode="cooperative", chip_grid=(4, 1, 1))
        assert plan.env_per_split[0]["TPU_PROCESS_BOUNDS"] == "4,1,1"

    def test_multi_chip_per_process_rejected(self):
        with pytest.raises(NotImplementedError, match="one chip per split"):
            plan_host_split(4, 2, mode="cooperative")

    def test_unknown_grid_requires_explicit(self):
        with pytest.raises(ValueError, match="chip_grid"):
            plan_host_split(16, 16, mode="cooperative")

    def test_grid_must_cover_chips(self):
        with pytest.raises(ValueError, match="covers"):
            plan_host_split(4, 4, mode="cooperative", chip_grid=(2, 4, 1))


class TestValidation:
    def test_non_divisible_split_rejected(self):
        with pytest.raises(ValueError, match="divide"):
            plan_host_split(4, 3)

    def test_non_positive_counts_rejected(self):
        with pytest.raises(ValueError, match="num_chips"):
            plan_host_split(0, 1)
        with pytest.raises(ValueError, match="num_splits"):
            plan_host_split(4, 0)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown split mode"):
            plan_host_split(4, 4, mode="magic")

    def test_plan_is_frozen(self):
        plan = plan_host_split(4, 4)
        with pytest.raises(AttributeError):
            plan.num_splits = 2

    def test_default_grids_are_consistent(self):
        for chips, grid in DEFAULT_HOST_CHIP_GRIDS.items():
            assert grid[0] * grid[1] * grid[2] == chips
