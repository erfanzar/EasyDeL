# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regressions for removing legacy SPMD/MPMD hybrid entry points."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

import spectrax.runtime.spmd as spmd
import spectrax.runtime.spmd.api as spmd_api
from spectrax.runtime.spmd.hybrid import hybrid_linear_run
from spectrax.runtime.types import MpMdMesh
from spectrax.sharding.mesh import SpxMesh


def _one_axis_mesh() -> Mesh:
    """Build the smallest mesh needed to exercise API-boundary checks."""
    return Mesh(np.asarray(jax.devices()[:1], dtype=object), axis_names=("pp",))


def test_pipeline_call_removed_from_spmd_public_api():
    """``pipeline_call`` must not remain as a public or direct API symbol."""
    assert not hasattr(spmd, "pipeline_call")
    assert not hasattr(spmd_api, "pipeline_call")
    assert "hybrid_linear_run" not in spmd.__all__


def test_hybrid_linear_run_rejects_direct_calls():
    """The old hybrid helper should not remain as an executable fallback."""
    with pytest.raises(NotImplementedError, match="not a true MPMD"):
        hybrid_linear_run()


def test_pipeline_step_rejects_explicit_mpmd_mesh():
    """The SPMD step wrapper must not accept an explicit MPMD mesh."""
    mesh = MpMdMesh(_one_axis_mesh(), "pp")
    with pytest.raises(ValueError, match="SPMD-only"):
        spmd_api.pipeline_step(
            object(),  # Rejection happens before model inspection.
            (jnp.ones((1, 1), dtype=jnp.float32),),
            mesh=mesh,
            axis="pp",
            schedule=None,
            loss_fn=lambda x: x.mean(),
        )


def test_pipeline_step_rejects_spxmesh_with_mpmd_axis():
    """Even degenerate MPMD metadata should not enter the SPMD runtime."""
    mesh = SpxMesh(_one_axis_mesh(), mpmd_axis="pp")
    with pytest.raises(ValueError, match="mpmd_axis"):
        spmd_api.pipeline_step(
            object(),  # Rejection happens before model inspection.
            (jnp.ones((1, 1), dtype=jnp.float32),),
            mesh=mesh,
            axis="pp",
            schedule=None,
            loss_fn=lambda x: x.mean(),
        )
