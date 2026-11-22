# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pipeline-parallel runtimes and schedules.

This package gathers spectrax's two pipeline runtimes plus the
schedule library and the types they share. The two runtimes are
complementary: the SPMD path emits a single HLO program that XLA
partitions across pipeline ranks via ``shard_map`` with the pipeline
axis manual; the MPMD path emits one program per rank through the
scheduled MPMD dispatcher, supporting genuinely heterogeneous stage
shapes.

Subpackages
-----------
* :mod:`spectrax.runtime.mpmd` — True MPMD runtime (per-rank separate
  HLOs). Use for heterogeneous pipelines or when you need explicit
  per-rank dispatch control.
* :mod:`spectrax.runtime.spmd` — SPMD runtime (single HLO via
  :func:`jax.shard_map`). Requires homogeneous stages and rejects
  MPMD-tagged meshes at the public API boundary.
* :mod:`spectrax.runtime.schedules` — Pipeline schedules
  (:class:`GPipe`, :class:`Std1F1B`, :class:`ZeroBubbleH1`,
  :class:`InterleavedH1`, :class:`KimiK2`, :class:`DualPipeV`, ...).
* :mod:`spectrax.runtime.types` — Shared types: :class:`MpMdMesh`,
  :class:`PipelineStage`, :class:`StagesArray`.
* :mod:`spectrax.runtime.primitives` — Shared primitives:
  :func:`boundary` (inline split marker), :func:`auto_split`
  (per-block stage assignment helper).

Top-level re-exports
--------------------
This module re-exports the MPMD entry points (:func:`sxcall`,
:func:`sxjit`, :func:`sxgrad`, :func:`sxvalue_and_grad`,
:func:`sxloop`, :func:`sxenter_loop`, :func:`sxexit_loop`,
:func:`sxstage_region`, :func:`sxstage_iter`), the forward-only
:class:`MpmdPipelineExecutor` used by inference runtimes, and every schedule
class plus the fusion helpers, so users typically only need
``from spectrax.runtime import ...``.
"""

from __future__ import annotations

from .mpmd import (
    DispatchWatchdog,
    MpmdPipelineDispatchStats,
    MpmdPipelineExecutor,
    ScheduleTracer,
    StallReport,
    dispatch_watchdog,
    format_schedule_grid,
    sxcall,
    sxenter_loop,
    sxexit_loop,
    sxgrad,
    sxjit,
    sxloop,
    sxstage_iter,
    sxstage_region,
    sxvalue_and_grad,
    sxvalue_and_grad_and_apply,
    trace_schedule,
)
from .schedules import (
    Action,
    DualPipeV,
    Eager1F1B,
    FusedTask,
    GPipe,
    Interleaved1F1BPlusOne,
    InterleavedGPipe,
    InterleavedH1,
    KimiK2,
    Phase,
    Schedule,
    Std1F1B,
    ZeroBubbleH1,
    dualpipev_tasks,
    fuse_1f1b_steady_state,
    fuse_zerobubble_bwd_pair,
)

__all__ = (
    "Action",
    "DispatchWatchdog",
    "DualPipeV",
    "Eager1F1B",
    "FusedTask",
    "GPipe",
    "Interleaved1F1BPlusOne",
    "InterleavedGPipe",
    "InterleavedH1",
    "KimiK2",
    "MpmdPipelineDispatchStats",
    "MpmdPipelineExecutor",
    "Phase",
    "Schedule",
    "ScheduleTracer",
    "StallReport",
    "Std1F1B",
    "ZeroBubbleH1",
    "dispatch_watchdog",
    "dualpipev_tasks",
    "format_schedule_grid",
    "fuse_1f1b_steady_state",
    "fuse_zerobubble_bwd_pair",
    "sxcall",
    "sxenter_loop",
    "sxexit_loop",
    "sxgrad",
    "sxjit",
    "sxloop",
    "sxstage_iter",
    "sxstage_region",
    "sxvalue_and_grad",
    "sxvalue_and_grad_and_apply",
    "trace_schedule",
)
