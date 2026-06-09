# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inspect MPMD stage-local meshes and boundary shardings.

SpectraX treats the pipeline axis as a program-selection axis, not as
an intra-stage sharding axis. A full mesh may be ``(pp, dp, tp)``, but
each compiled stage sees only the local SPMD sub-mesh ``(dp, tp)``.
That is the key layout rule behind true MPMD:

* ``pp`` chooses which program/rank owns the stage;
* non-``pp`` axes remain available for FSDP, TP, SP, EP, and DP.

Run::

    python -m examples.07_mpmd.13_stage_local_mesh_layout
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec

from spectrax.runtime.types import MpMdMesh


def main() -> None:
    """Build a tiny MPMD mesh and print the stage-local view."""
    devices = jax.devices()
    if len(devices) < 2:
        print("need at least two visible devices for a non-trivial pp mesh")
        print(f"visible devices: {len(devices)}")
        return

    pp = 2
    dp = max(1, len(devices) // pp)
    used = pp * dp
    mesh = Mesh(np.asarray(devices[:used], dtype=object).reshape(pp, dp), ("pp", "dp"))
    mpmd_mesh = MpMdMesh(mesh, "pp")

    print("full mesh shape:", dict(mpmd_mesh.jax_mesh.shape))
    print("MPMD axis      :", mpmd_mesh.mpmd_axis_name)
    print("SPMD axes      :", mpmd_mesh.spmd_axis_names)
    print()

    for rank, submesh in enumerate(mpmd_mesh.unstack()):
        print(f"rank {rank} stage mesh:", dict(submesh.shape))
        print(f"  axis names: {submesh.axis_names}")
        print(f"  devices   : {[d.id for d in submesh.devices.flat]}")

    print()
    print("replicated boundary sharding:", mpmd_mesh.sub_sharding(0).spec)
    print("dp-sharded boundary sharding:", mpmd_mesh.sub_sharding(0, PartitionSpec("dp")).spec)
    try:
        mpmd_mesh.sub_sharding(0, PartitionSpec("pp"))
    except ValueError as exc:
        print("pp in a stage-local spec is rejected:", str(exc).split(".")[0])


if __name__ == "__main__":
    main()
