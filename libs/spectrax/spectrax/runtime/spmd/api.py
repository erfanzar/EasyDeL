# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public SPMD pipeline API.

This module intentionally contains only SPMD entry points.  True MPMD
execution lives under :mod:`spectrax.runtime.mpmd` and is reached through
``sxjit``, ``sxcall``, ``sxgrad``, ``sxvalue_and_grad``, or
``spx.run(..., mesh=<SpxMesh with mpmd_axis>)``.

The old ``pipeline_call`` helper has been removed.  It accepted MPMD-shaped
meshes while mixing ``shard_map`` SPMD and host-dispatched multi-jit paths,
which made it too easy to believe a call was running through the true MPMD
scheduler when it was not.
"""

from __future__ import annotations

from collections.abc import Callable

from jax.sharding import Mesh

from spectrax.nn import PipelineSequential

from ...core.state import State
from ...sharding.mesh import SpxMesh
from ..types.mesh import MpMdMesh
from ..types.stage import PipelineStage
from .runtime import spmd_run

__all__ = ["PipelineStage", "pipeline_step"]


def _unwrap_spmd_mesh(mesh: SpxMesh | MpMdMesh | Mesh, *, axis: str, api_name: str) -> Mesh:
    """Return a raw JAX mesh for SPMD-only APIs.

    Args:
        mesh: A raw :class:`jax.sharding.Mesh`, a pure-SPMD
            :class:`~spectrax.sharding.mesh.SpxMesh`, or an MPMD mesh.
        axis: Pipeline axis name requested by the caller.  Only used in
            the error text so the failure points at the wrong boundary.
        api_name: Name of the public API performing the unwrap.

    Returns:
        The raw JAX mesh that :func:`spmd_run` expects.

    Raises:
        ValueError: If ``mesh`` carries MPMD metadata.  SPMD APIs must not
            silently consume an MPMD-tagged mesh; true MPMD callers should
            use the :mod:`spectrax.runtime.mpmd` entry points instead.
    """
    if isinstance(mesh, SpxMesh):
        if mesh.mpmd_axis is not None:
            raise ValueError(
                f"{api_name} is SPMD-only and does not accept an SpxMesh with "
                f"mpmd_axis={mesh.mpmd_axis!r} (requested axis={axis!r}). "
                "Use sxcall/sxjit or spx.run with the MPMD mesh for true MPMD execution."
            )
        return mesh.jax_mesh
    if isinstance(mesh, MpMdMesh):
        raise ValueError(
            f"{api_name} is SPMD-only and does not accept MpMdMesh(axis={mesh.mpmd_axis_name!r}). "
            "Use sxcall/sxjit or spx.run with the MPMD mesh for true MPMD execution."
        )
    return mesh


def pipeline_step(
    model: PipelineSequential,
    batch: tuple[object, ...],
    *,
    mesh: SpxMesh | MpMdMesh | Mesh,
    axis: str = "pp",
    schedule: object,
    loss_fn: Callable[..., object],
) -> tuple[object, tuple[State, ...]]:
    """Execute one SPMD pipeline-parallel forward + backward step.

    Thin wrapper over :func:`~spectrax.runtime.spmd.spmd_run`.  The
    caller supplies a raw JAX mesh, or an :class:`SpxMesh` without
    ``mpmd_axis`` metadata, and ``spmd_run`` compiles a single SPMD HLO
    whose params are sharded along ``axis``.

    This function is deliberately not an MPMD compatibility layer.  If
    ``mesh`` is an :class:`MpMdMesh` or an MPMD-tagged :class:`SpxMesh`,
    it raises before dispatch so MPMD-marked calls cannot accidentally
    run through the SPMD runtime.

    Args:
        model: :class:`PipelineSequential` whose ``num_stages`` must
            equal the mesh's ``axis`` dimension. All stages must share a
            structurally identical ``GraphDef``.
        batch: Tuple of positional tensors. The first element is the
            pipeline input; remaining elements are targets or auxiliary
            arguments passed to ``loss_fn`` on the final stage.
        mesh: Raw :class:`jax.sharding.Mesh` or a pure-SPMD
            :class:`SpxMesh`. MPMD-tagged meshes are rejected.
        axis: Named axis of ``mesh`` reserved for SPMD pipeline stages.
        schedule: One of :class:`GPipe`, :class:`Std1F1B`,
            :class:`ZeroBubbleH1`, :class:`InterleavedH1`, or another
            schedule supported by :func:`spmd_run`.
        loss_fn: Scalar loss. Called as ``loss_fn(final_stage_output,
            *batch[1:])`` on each microbatch and mean-reduced over
            microbatches.

    Returns:
        ``(loss, per_stage_grads)`` where ``per_stage_grads`` is a tuple
        of :class:`State` s, one per stage.
    """
    jax_mesh = _unwrap_spmd_mesh(mesh, axis=axis, api_name="pipeline_step")
    return spmd_run(
        model,
        batch,
        mesh=jax_mesh,
        axis=axis,
        schedule=schedule,
        loss_fn=loss_fn,
    )
