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

"""Regression: ``EasyDeLBaseConfig._stage_spmd_devices`` is stage-aware under MPMD.

When the parent mesh carries an MPMD pipeline axis (``pp>1``), the expert
mesh must be built from the *current pipeline stage's* SPMD sub-mesh -- not
the global device grid. Each MoE layer caches ``auto_expert_mesh`` at
``BaseMoeModule.__init__`` while a ``spx.assign_stage(total=N, current=i)``
context is active; the per-layer cache must capture that stage's devices,
not always stage 0's.

Without this, MoE shard_map collectives on stages != 0 would use stage 0's
devices -- catastrophic on DCN topologies (cross-host all-to-all per layer).

Stage selection is in priority order:
  1. Active ``spx.assign_stage`` context, resolved via ``resolve_stage_rank``.
  2. Stage 0 fallback when no hint is available.

This test covers both branches plus the non-MPMD path.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
import spectrax as spx

import easydel as ed

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _make_config(*, pp: int, dp: int, fsdp: int, ep: int, tp: int, sp: int) -> ed.LlamaConfig:
    """Build a minimal config with an explicit mpmd-tagged mesh."""
    mesh = spx.create_mesh(
        axis_dims=(pp, dp, fsdp, ep, tp, sp),
        axis_names=AXIS_NAMES,
        mpmd_axis="pp" if pp > 1 else None,
    )
    config = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=8,
        sharding_axis_dims=(pp, dp, fsdp, ep, tp, sp),
        sharding_axis_names=AXIS_NAMES,
    )
    config.set_model_mesh(mesh)
    return config


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires >= 4 devices for pp>1")
def test_stage_spmd_devices_returns_full_mesh_when_not_mpmd():
    """pp=1 -> SpxMesh.is_mpmd is False -> full mesh devices, no stage logic."""
    config = _make_config(pp=1, dp=1, fsdp=4, ep=1, tp=1, sp=1)
    devices = config._stage_spmd_devices()
    expected = config.mesh.devices.flatten()
    assert devices.shape == expected.shape
    assert set(map(id, devices.tolist())) == set(map(id, expected.tolist()))


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires >= 4 devices for pp=2")
def test_stage_spmd_devices_falls_back_to_stage_zero_outside_assign_stage():
    """No ``assign_stage`` context active -> stage 0 fallback."""
    config = _make_config(pp=2, dp=1, fsdp=2, ep=1, tp=1, sp=1)
    mpmd = config.mesh.mpmd_mesh
    expected_stage0 = mpmd.submesh(0).devices.flatten()
    devices = config._stage_spmd_devices()
    assert devices.shape == expected_stage0.shape

    assert set(devices.tolist()) == set(expected_stage0.tolist())


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires >= 4 devices for pp=2")
def test_stage_spmd_devices_disjoint_per_stage_under_assign_stage():
    """Each ``spx.assign_stage(current=i)`` scope must yield its own stage's device set."""
    config = _make_config(pp=2, dp=1, fsdp=2, ep=1, tp=1, sp=1)
    mpmd = config.mesh.mpmd_mesh
    n_layers = 4

    devices_by_logical_layer: dict[int, set] = {}
    for layer_idx in range(n_layers):
        with spx.assign_stage(total=n_layers, current=layer_idx):
            stage_devices = config._stage_spmd_devices()
        devices_by_logical_layer[layer_idx] = set(stage_devices.tolist())


    expected_stage0 = set(mpmd.submesh(0).devices.flatten().tolist())
    expected_stage1 = set(mpmd.submesh(1).devices.flatten().tolist())

    assert devices_by_logical_layer[0] == expected_stage0
    assert devices_by_logical_layer[1] == expected_stage0
    assert devices_by_logical_layer[2] == expected_stage1
    assert devices_by_logical_layer[3] == expected_stage1


    assert devices_by_logical_layer[0].isdisjoint(devices_by_logical_layer[2])
    assert devices_by_logical_layer[1].isdisjoint(devices_by_logical_layer[3])


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires >= 4 devices for pp>1")
def test_expert_mesh_reshape_uses_stage_local_devices():
    """``auto_expert_mesh`` reshapes (dp, ep, tp) over the current stage's devices."""
    config = _make_config(pp=2, dp=1, fsdp=2, ep=1, tp=1, sp=1)
    mpmd = config.mesh.mpmd_mesh
    expected_stage1 = set(mpmd.submesh(1).devices.flatten().tolist())

    n_layers = 4

    with spx.assign_stage(total=n_layers, current=3):
        expert_mesh = config.auto_expert_mesh

    expert_devices = set(expert_mesh.jax_mesh.devices.flatten().tolist())
    assert expert_devices == expected_stage1, (
        "auto_expert_mesh built under assign_stage(current=3) must reshape over stage 1's "
        f"devices; got {expert_devices} vs expected {expected_stage1}"
    )


    assert "pp" not in expert_mesh.jax_mesh.axis_names
    assert expert_mesh.is_mpmd is False


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires >= 4 devices for pp>1")
def test_stage_spmd_devices_total_count_matches_per_stage():
    """Stage submesh device count == global / pp_size."""
    config = _make_config(pp=2, dp=1, fsdp=2, ep=1, tp=1, sp=1)
    global_count = config.mesh.devices.size
    n_layers = 2

    with spx.assign_stage(total=n_layers, current=0):
        stage_devices = config._stage_spmd_devices()

    pp_size = int(config.mesh.shape["pp"])
    assert stage_devices.size == global_count // pp_size
