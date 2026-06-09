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

from __future__ import annotations

from types import SimpleNamespace

from ejkernel.types import MaskInfo
from jax.sharding import PartitionSpec

from easydel.operations import _operation_meta as op_meta
from easydel.operations.kernels._mask_info import align_mask_info_to_qkv_specs


def test_operation_metadata_mesh_uses_spectrax_stage_mesh(monkeypatch):
    calls = []

    def resolve_stage_mesh(mesh):
        calls.append(mesh)
        return "stage-mesh"

    monkeypatch.setattr(op_meta, "resolve_stage_mesh", resolve_stage_mesh)
    metadata = SimpleNamespace(base_config=None, _stored_mesh="config-mesh")

    assert op_meta.OperationMetadata.mesh.fget(metadata) == "stage-mesh"
    assert calls == ["config-mesh"]


def test_operation_metadata_mesh_prefers_base_config_mesh(monkeypatch):
    calls = []

    def resolve_stage_mesh(mesh):
        calls.append(mesh)
        return "stage-mesh"

    monkeypatch.setattr(op_meta, "resolve_stage_mesh", resolve_stage_mesh)
    metadata = SimpleNamespace(base_config=SimpleNamespace(mesh="base-config-mesh"), _stored_mesh="stored-mesh")

    assert op_meta.OperationMetadata.mesh.fget(metadata) == "stage-mesh"
    assert calls == ["base-config-mesh"]


def test_operation_metadata_mesh_keeps_empty_mesh_none(monkeypatch):
    calls = []

    def resolve_stage_mesh(mesh):
        calls.append(mesh)
        return "stage-mesh"

    monkeypatch.setattr(op_meta, "resolve_stage_mesh", resolve_stage_mesh)
    metadata = SimpleNamespace(base_config=None, _stored_mesh=None)

    assert op_meta.OperationMetadata.mesh.fget(metadata) is None
    assert calls == []


def test_mask_info_axes_follow_attention_specs():
    mask_info = MaskInfo(
        batch_axis_name=("dp", "fsdp"),
        qheads_axis_name="tp",
        kvheads_axis_name="tp",
        sequence_axis_name="sp",
    )

    aligned = align_mask_info_to_qkv_specs(
        mask_info,
        query_spec=PartitionSpec(None, None, "tp", None),
        key_spec=PartitionSpec(None, None, None, None),
        layout="bthd",
    )

    assert aligned.batch_axis_name is None
    assert aligned.qheads_axis_name == "tp"
    assert aligned.kvheads_axis_name is None
    assert aligned.sequence_axis_name is None
