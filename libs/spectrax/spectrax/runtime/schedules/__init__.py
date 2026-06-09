# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pipeline schedules — when each stage runs which microbatch.

A pipeline schedule is a small, pure-Python object that describes
**which microbatch each stage works on at each time step**. The
schedule does not execute anything itself; the runtimes in
:mod:`spectrax.runtime.spmd` and :mod:`spectrax.runtime.mpmd` consume
:meth:`Schedule.build` and turn the resulting grid into either a
``shard_map`` body (SPMD) or a per-rank action queue (MPMD).

Schedules ship in this package fall into three families:

* **Flat schedules** (one logical stage per physical rank):
  :class:`GPipe`, :class:`Std1F1B`, :class:`Eager1F1B`,
  :class:`ZeroBubbleH1`. They differ in whether all forwards run
  before any backward (GPipe), whether forward and backward
  alternate after a warmup (1F1B), and whether the backward is split
  into BWD_I + BWD_W to fill bubbles (zero-bubble H1).
* **Virtual-stage schedules** (multiple logical stages per physical
  rank): :class:`InterleavedH1`, :class:`InterleavedGPipe`,
  :class:`Interleaved1F1BPlusOne`, :class:`KimiK2`. Each device hosts
  several non-contiguous logical stages, dramatically shrinking the
  bubble at the cost of extra per-microbatch transport hops.
* **Bidirectional schedules**: :class:`DualPipeV` mirrors the
  pipeline so each rank hosts one forward virtual stage and one
  reverse virtual stage; the matching :func:`dualpipev_tasks` helper
  emits the per-rank task list used by DeepSeek's reference
  implementation.

The :class:`Action` / :class:`FusedTask` / :class:`Phase` types live
in :mod:`spectrax.runtime.schedules.base` and are the primitives
every schedule emits. The :func:`fuse_1f1b_steady_state` and
:func:`fuse_zerobubble_bwd_pair` post-processors collapse adjacent
schedule cells into single fused dispatches when the runtime can
benefit from them.
"""

from .base import Action, FusedTask, Phase, Schedule
from .dualpipe import DualPipeV, dualpipev_tasks
from .fusion import fuse_1f1b_steady_state, fuse_zerobubble_bwd_pair
from .gpipe import GPipe
from .interleaved import (
    Interleaved1F1BPlusOne,
    InterleavedGPipe,
    InterleavedH1,
    KimiK2,
)
from .one_f_one_b import Eager1F1B, Std1F1B
from .zero_bubble import ZeroBubbleH1

__all__ = [
    "Action",
    "DualPipeV",
    "Eager1F1B",
    "FusedTask",
    "GPipe",
    "Interleaved1F1BPlusOne",
    "InterleavedGPipe",
    "InterleavedH1",
    "KimiK2",
    "Phase",
    "Schedule",
    "Std1F1B",
    "ZeroBubbleH1",
    "dualpipev_tasks",
    "fuse_1f1b_steady_state",
    "fuse_zerobubble_bwd_pair",
]
