# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared pipeline-parallel types.

The types in this package are consumed by every spectrax pipeline
runtime — both the SPMD path (:mod:`spectrax.runtime.spmd`) and the
MPMD path (:mod:`spectrax.runtime.mpmd`).

* :class:`MpMdMesh` — a :class:`~jax.sharding.Mesh` with one named axis
  designated as the pipeline (multi-program) axis.
* :class:`PipelineStage` — the ``(fn, parameters, init_state)`` triple
  that the runtimes invoke for one logical stage.
* :class:`StagesArray` — a logical array that lives on a subset of
  pipeline stages, used to represent values (activations, tied
  embeddings) that aren't replicated across every rank.

The :func:`resolve_mpmd_mesh` and :func:`abstract_stages_array` helpers
are convenience constructors. :func:`_is_empty_state` is a private
sentinel-detection helper exposed here so the runtimes can share its
behaviour for the ``()``/``None`` empty-state convention.
"""

from __future__ import annotations

from .array import StagesArray, abstract_stages_array
from .mesh import MpMdMesh, resolve_mpmd_mesh
from .stage import PipelineStage, _is_empty_state

__all__ = [
    "MpMdMesh",
    "PipelineStage",
    "StagesArray",
    "_is_empty_state",
    "abstract_stages_array",
    "resolve_mpmd_mesh",
]
