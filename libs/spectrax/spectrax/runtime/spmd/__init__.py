# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SPMD pipeline runtime.

Compiles each pipeline step into a single HLO program. This package is
SPMD-only: MPMD-tagged meshes are rejected at the public API boundary so
callers do not accidentally bypass the true MPMD scheduler.

Public surface
--------------
* :func:`pipeline_step` — thin convenience wrapper over
  :func:`spmd_run` for use with :class:`PipelineSequential` modules.
* :func:`spmd_run` — scan-free runtime for :class:`PipelineSequential`:
  shards stacked params along the pipeline axis and lets XLA route
  forward/backward across stages based on the placement.
* :func:`make_scheduled_body` — turn any
  :class:`~spectrax.runtime.schedules.Schedule` into a ``shard_map``
  body for lower-level SPMD schedule construction.
"""

from __future__ import annotations

from .api import pipeline_step
from .runtime import spmd_run
from .shard_map import make_scheduled_body

__all__ = [
    "make_scheduled_body",
    "pipeline_step",
    "spmd_run",
]
