# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Per-role loader/sharding overrides for auxiliary models (teacher/target/reference/reward).

Auxiliary models run inside the primary model's compiled step, so the
override contract is: ``<role>_loader`` / ``<role>_sharding`` replace the
shared sections for that role only, must keep the primary mesh, and fall
back to the shared sections when absent.
"""

import easydel as ed
import easydel.infra.elarge.model as elarge_model_mod
import pytest
from easydel.infra.elarge.processing import coerce_partition_axis
from spectrax import PartitionAxis

_BASE_CFG = {
    "model": {"name_or_path": "stub/student"},
    "teacher_model": {"name_or_path": "stub/teacher"},
    "reference_model": {"name_or_path": "stub/reference"},
    "loader": {"dtype": "bf16", "verbose": True},
    "sharding": {"axis_dims": (1, 2, 4, 1, 1, 1), "axis_names": ("pp", "dp", "fsdp", "ep", "tp", "sp")},
}


@pytest.fixture
def captured_build(monkeypatch):
    """Intercept build_model and capture the config each aux build receives."""
    calls = []

    def fake_build_model(cfg):
        calls.append(cfg)
        return object()

    monkeypatch.setattr(elarge_model_mod, "build_model", fake_build_model)
    return calls


def test_teacher_falls_back_to_shared_sections(captured_build):
    elm = ed.eLargeModel(dict(_BASE_CFG))
    elm.build_teacher_model()
    cfg = captured_build[-1]
    assert cfg["model"]["name_or_path"] == "stub/teacher"
    assert cfg["loader"]["dtype"] == "bf16"
    assert tuple(cfg["sharding"]["axis_dims"]) == (1, 2, 4, 1, 1, 1)


def test_teacher_overrides_win_for_teacher_only(captured_build):
    cfg_in = dict(_BASE_CFG)
    cfg_in["teacher_loader"] = {"dtype": "fp32"}
    cfg_in["teacher_sharding"] = {
        "axis_dims": (1, 2, 4, 1, 1, 1),
        "partition_axis": PartitionAxis(fully_sharded_data_parallel_axis=None, batch_axis=("fsdp", "dp")),
    }
    elm = ed.eLargeModel(cfg_in)

    elm.build_teacher_model()
    teacher_cfg = captured_build[-1]
    assert teacher_cfg["loader"]["dtype"] == "fp32"
    assert teacher_cfg["sharding"]["partition_axis"].fully_sharded_data_parallel_axis is None

    elm.build_reference_model()
    reference_cfg = captured_build[-1]
    assert reference_cfg["model"]["name_or_path"] == "stub/reference"
    assert reference_cfg["loader"]["dtype"] == "bf16"
    assert "partition_axis" not in reference_cfg["sharding"]


def test_target_falls_back_to_teacher_section_and_overrides(captured_build):
    cfg_in = dict(_BASE_CFG)
    cfg_in["teacher_sharding"] = {"axis_dims": (1, 2, 4, 1, 1, 1), "auto_shard_model": True}
    elm = ed.eLargeModel(cfg_in)
    elm.build_target_model()
    cfg = captured_build[-1]
    assert cfg["model"]["name_or_path"] == "stub/teacher"
    assert cfg["sharding"]["auto_shard_model"] is True


def test_mesh_divergence_is_rejected_with_partition_axis_guidance(captured_build):
    cfg_in = dict(_BASE_CFG)
    cfg_in["teacher_sharding"] = {"axis_dims": (1, 8, 1, 1, 1, 1)}
    elm = ed.eLargeModel(cfg_in)
    with pytest.raises(ValueError, match="partition_axis"):
        elm.build_teacher_model()
    assert not captured_build  # rejected before any build


def test_missing_role_section_returns_none(captured_build):
    elm = ed.eLargeModel({"model": {"name_or_path": "stub/student"}})
    assert elm.build_teacher_model() is None
    assert elm.build_reward_model() is None
    assert not captured_build


def test_coerce_partition_axis_accepts_dict_lists_and_instances():
    inst = PartitionAxis(fully_sharded_data_parallel_axis=None)
    assert coerce_partition_axis(inst) is inst
    assert coerce_partition_axis(None) is None
    coerced = coerce_partition_axis({"fully_sharded_data_parallel_axis": None, "batch_axis": ["fsdp", "dp"]})
    assert isinstance(coerced, PartitionAxis)
    assert coerced.fully_sharded_data_parallel_axis is None
    assert coerced.batch_axis == ("fsdp", "dp")
