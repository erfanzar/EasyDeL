# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read the training loss each finished research job printed.

Kept beside the research subsystem rather than inside it: the plan runner calls
this once per step, outside any transaction, with a log reader the API layer
supplies (the service itself has no transport).
"""

from __future__ import annotations

import logging
import typing as tp
from datetime import UTC, datetime

from eyvan._internal.training_metrics import extract_loss
from eyvan.records import (
    MAX_RESEARCH_NOTE_CHARS,
    ResearchLoss,
    ResearchNote,
    ResearchPlanRecord,
    WorkloadRecord,
)
from eyvan.server.service.state import TERMINAL_WORKLOAD_STATES

logger = logging.getLogger(__name__)

#: How long after a job ends the plan keeps retrying an unreadable output
#: (an unreachable cell, an archive still uploading) before finishing without
#: that job's loss. Bounded: a plan must never hang on a log it cannot read.
LOSS_READ_GRACE_S = 600.0

#: ``workload_id -> its printed output``; supplied by the runner.
ResearchLogReader: tp.TypeAlias = tp.Callable[[str], tp.Awaitable[str]]


class _Seen(tp.Protocol):
    @property
    def workload(self) -> WorkloadRecord | None: ...


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _note(now: datetime, text: str) -> ResearchNote:
    return ResearchNote(t=now, kind="loss", text=text[:MAX_RESEARCH_NOTE_CHARS])


async def read_research_losses(
    plan: ResearchPlanRecord,
    observed: tp.Mapping[str, _Seen],
    read_logs: ResearchLogReader | None,
    now: datetime,
    notes: list[ResearchNote],
    waiting: set[int] | None = None,
) -> dict[int, ResearchLoss]:
    """Read each newly final job's output once and keep the training loss it printed.

    An unreadable output is retried on the next step and, for up to
    :data:`LOSS_READ_GRACE_S` after the job ended, its index is added to
    ``waiting`` so the plan does not finish without it. Output read with no
    loss in it is recorded as such (``final=None``) and never re-read.
    """

    found: dict[int, ResearchLoss] = {}
    if read_logs is None:
        return found
    for item in plan.submitted:
        seen = observed.get(item.workload_id or "")
        workload = seen.workload if seen is not None else None
        if item.loss is not None or workload is None or workload.state not in TERMINAL_WORKLOAD_STATES:
            continue
        label = f"job {item.job_index} {plan.jobs[item.job_index].name!r}"
        try:
            curve = extract_loss(await read_logs(workload.id))
        except Exception as exc:
            logger.warning("research plan %s: output of %s unreadable: %s", plan.id, workload.id, exc)
            if (now - _aware(workload.updated_at)).total_seconds() < LOSS_READ_GRACE_S:
                if waiting is not None:
                    waiting.add(item.job_index)
            else:
                waited = f"{LOSS_READ_GRACE_S:g}s"
                notes.append(_note(now, f"{label}: output still unreadable {waited} after it ended; no loss recorded"))
            continue
        if curve is None:
            found[item.job_index] = ResearchLoss()
            notes.append(_note(now, f"{label} printed no training loss"))
            continue
        found[item.job_index] = ResearchLoss(
            final=curve["final"],
            last_step=curve["last_step"],
            total_steps=curve["total_steps"],
            points=tuple(curve["points"]),
        )
        where = "" if curve["last_step"] is None else f" at step {curve['last_step']}"
        where += f"/{curve['total_steps']}" if where and curve["total_steps"] else ""
        notes.append(_note(now, f"{label} final loss {curve['final']:g}{where}"))
    return found


__all__ = ("LOSS_READ_GRACE_S", "ResearchLogReader", "read_research_losses")
