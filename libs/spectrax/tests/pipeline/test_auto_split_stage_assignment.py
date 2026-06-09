# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pipeline auto-split placement and nested-state sharding regressions."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import spectrax as spx
from spectrax import nn
from spectrax.runtime.mpmd.runtime import _place_state_on_rank
from spectrax.runtime.primitives.split import auto_split
from spectrax.sharding import create_mesh, logical_axis_rules


class _TaggedBlock(spx.Module):
    """Tiny block with a visible index and nested parameter path."""

    def __init__(self, idx: int, *, rngs: spx.Rngs):
        """Initialize with idx, fc."""
        super().__init__()
        self.idx = idx
        self.fc = nn.Linear(4, 4, sharding=("fsdp", "tp"), rngs=rngs)

    def forward(self, x):
        """Run the forward pass."""
        return self.fc(x)


class _TaggedStack(spx.Module):
    """``embed -> blocks[*] -> head`` stack used by split tests."""

    def __init__(self, n_blocks: int, *, rngs: spx.Rngs):
        """Initialize with embed, blocks, head."""
        super().__init__()
        self.embed = nn.Linear(4, 4, sharding=("fsdp", "tp"), rngs=rngs)
        self.blocks = nn.ModuleList([_TaggedBlock(i, rngs=rngs) for i in range(n_blocks)])
        self.head = nn.Linear(4, 4, sharding=("fsdp", "tp"), rngs=rngs)

    def forward(self, x):
        """Run the forward pass."""
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _block_ids(stage: spx.Module) -> list[int]:
    """Extract block IDs from a module."""
    return [block.idx for block in stage.blocks]


def test_auto_split_respects_explicit_uneven_block_pp_stage_layout():
    """Block-level ``pp_stage`` supports uneven dynamic pipeline layouts."""
    model = _TaggedStack(28, rngs=spx.Rngs(0))
    model.embed.pp_stage = 0
    model.head.pp_stage = 3
    for i, block in enumerate(model.blocks):
        if i <= 4:
            block.pp_stage = 0
        elif i <= 9:
            block.pp_stage = 1
        elif i <= 14:
            block.pp_stage = 2
        else:
            block.pp_stage = 3

    stages = auto_split(model, 4)

    assert stages[0].pre_names == ("embed",)
    assert _block_ids(stages[0]) == list(range(0, 5))
    assert _block_ids(stages[1]) == list(range(5, 10))
    assert _block_ids(stages[2]) == list(range(10, 15))
    assert _block_ids(stages[3]) == list(range(15, 28))
    assert stages[3].post_names == ("head",)


def test_auto_split_rejects_non_monotonic_block_pp_stage_annotations():
    """A later block cannot move to an earlier stage without changing semantics."""
    model = _TaggedStack(4, rngs=spx.Rngs(0))
    model.blocks[0].pp_stage = 1
    model.blocks[1].pp_stage = 0

    with pytest.raises(ValueError, match="non-decreasing"):
        auto_split(model, 2)


def test_nested_stage_state_is_placed_on_owner_submesh_with_tp_sharding():
    """Nested paths such as ``blocks.0.fc.weight`` keep per-leaf sharding."""
    if len(jax.devices()) < 4:
        pytest.skip("need 4 devices for pp=2,tp=2 placement test")

    mesh = create_mesh(axis_dims=(2, 1, 1, 1, 2, 1), mpmd_axis="pp")
    rules = [("fsdp", "fsdp"), ("tp", "tp")]
    with logical_axis_rules(rules), mesh:
        model = _TaggedStack(4, rngs=spx.Rngs(0))
        stages = auto_split(model, 2)

    stage_shardings = [mesh.mpmd_mesh.sub_sharding(i) for i in range(2)]
    rank_submeshes = [mesh.mpmd_mesh.submesh(i) for i in range(2)]
    with logical_axis_rules(rules):
        for rank, stage in enumerate(stages):
            _gdef, state = spx.export(stage)
            placed = _place_state_on_rank(state, rank, stage, rank_submeshes, stage_shardings)
            for _collection, path, leaf in placed.items():
                assert isinstance(leaf, jax.Array), path
                assert np.array_equal(leaf.sharding.mesh.devices, rank_submeshes[rank].devices), path
            weight = placed.get("parameters", "blocks.0.fc.weight")
            assert tuple(weight.sharding.spec) == ("fsdp", "tp")


def test_spmd_place_state_uses_dotted_nested_paths_for_leaf_sharding():
    """The public SPMD placement helper must not treat nested dict keys as leaves."""
    if len(jax.devices()) < 2:
        pytest.skip("need at least 2 devices for tp sharding test")

    from spectrax.api import _place_state

    mesh = create_mesh(axis_dims=(1, 1, 1, 1, -1, 1))
    rules = [("fsdp", None), ("tp", "tp")]
    with logical_axis_rules(rules), mesh:
        model = _TaggedStack(2, rngs=spx.Rngs(0))
        _gdef, state = spx.export(model)
        placed = _place_state(state, model, mesh.jax_mesh)

    weight = placed.get("parameters", "blocks.0.fc.weight")
    assert isinstance(weight, jax.Array)
    assert tuple(weight.sharding.spec) == (None, "tp")
