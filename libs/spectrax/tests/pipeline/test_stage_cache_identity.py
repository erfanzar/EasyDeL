# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage cache identity must include concrete meshes inside nested JAXPRs."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec
from spectrax.runtime.mpmd.pscan_compiler import _stage_jit_name_suffix


@pytest.mark.parametrize("nested_jit", [False, True])
def test_stage_cache_identity_tracks_nested_mesh_placement(nested_jit):
    """Equivalent traces share a key, but nested device permutations must not."""
    devices = jax.devices()[:2]
    if len(devices) < 2:
        pytest.skip("need two devices to distinguish concrete mesh placements")
    stage_mesh = Mesh(np.asarray(devices), ("tp",))
    captured = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    def trace(device_order):
        mesh = Mesh(np.asarray(device_order), ("tp",))
        mapped = jax.shard_map(
            lambda x: x + captured,
            mesh=mesh,
            in_specs=PartitionSpec(),
            out_specs=PartitionSpec(),
        )
        fn = jax.jit(mapped) if nested_jit else mapped
        return jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((4,), np.float32))

    original = trace(devices)
    equivalent = trace(devices)
    permuted = trace(devices[::-1])
    suffix = _stage_jit_name_suffix(original, stage_mesh)

    # JAX 0.10 returns a distinct ClosedJaxpr wrapper; in JAX 0.11 .jaxpr
    # points back to the same object. Both representations must work.
    assert suffix == _stage_jit_name_suffix(original.jaxpr, stage_mesh)
    assert suffix == _stage_jit_name_suffix(equivalent, stage_mesh)
    # Keep the outer stage mesh fixed: only a nested mesh changes, and its
    # shape, axis names, and minimum device id all remain identical.
    assert suffix != _stage_jit_name_suffix(permuted, stage_mesh)
