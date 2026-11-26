# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Eyvan run tracking: log metrics from inside training and evaluation jobs.

Example::

    import eyvan.track as track

    with track.start("ppo-atari", group="sweep-3", config=cfg) as run:
        for update in range(updates):
            run.log({"loss/policy": loss}, step=update, axes={"env_steps": frames})
        run.summary(best_return=best)

Standard library only, and never fails the job on storage errors. The tracker
service lives in ``eyvan.track.tracker`` and is deliberately not imported here.
"""

from eyvan.track._format import run_id_for_workload
from eyvan.track._run import (
    DEFAULT_LOCAL_STORE,
    DEFAULT_SPILL_DIR,
    ReplayResult,
    Run,
    TrackLimits,
    replay_spill,
    start,
)

__all__ = (
    "DEFAULT_LOCAL_STORE",
    "DEFAULT_SPILL_DIR",
    "ReplayResult",
    "Run",
    "TrackLimits",
    "replay_spill",
    "run_id_for_workload",
    "start",
)
