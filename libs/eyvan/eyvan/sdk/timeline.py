# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One ordered narrative of what happened to a workload.

Every fact is already recorded. What was missing is the sentence: to learn why
a job took forty minutes to start you read the workload record, then its
leases, then the audit ledger, then correlated three timestamps by hand. The
answer is usually obvious once the events are in one column -- eight minutes
queued, thirty provisioning, two preemptions -- and invisible until then.

Derived entirely from the audit ledger, which is ordered by a monotonic
sequence and immutable. No new store, and nothing here can disagree with what
the control plane recorded.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from datetime import datetime, timedelta

from eyvan.records import AuditEvent

#: How each recorded action reads in a narrative. An action with no entry is
#: still shown, spelled as it was recorded: an unknown event is a fact, and
#: hiding it would make the timeline quietly incomplete the moment the control
#: plane learns to record something new.
_PHRASES: dict[str, str] = {
    "workload.submitted": "submitted",
    "workload.admitted": "admitted",
    "workload.wave_admitted": "admitted as part of a wave",
    "workload.dag_released": "released by its dependencies",
    "workload.group_released": "released with its group",
    "workload.leased": "leased capacity",
    "workload.dispatch_reserved": "dispatch reserved",
    "workload.dispatch_reservation_failed": "dispatch reservation failed",
    "workload.dispatched": "dispatched to its cells",
    "workload.running": "running",
    "workload.preemption_started": "preempted; checkpointing",
    "workload.checkpoint_completed": "checkpoint written",
    "workload.kicked": "kicked off its host",
    "workload.recovery_queued": "queued to run again",
    "workload.capacity_recovery_queued": "queued again after losing capacity",
    "workload.cancelled_after_loss": "cancelled after losing capacity",
    "workload.lost": "lost",
    "workload.orphaned": "orphaned; its lease was gone",
    "workload.force_finalized": "force finalized by an operator",
    "workload.agent_fence_conflict": "refused by its agent as out of date",
    "workload.batch_settled": "settled with its batch",
    "workload.cancelled": "cancelled",
    "workload.succeeded": "succeeded",
    "workload.failed": "failed",
    "lease.created": "capacity leased",
    "lease.settled": "capacity released",
    "lease.released_unstarted": "capacity released before it started",
}

#: Payload keys worth showing beside an event, in the order they read best.
#: A timeline that printed whole payloads would be a log; these are the fields
#: that answer "why" rather than restating the event.
_DETAIL_KEYS: tuple[str, ...] = (
    "reason",
    "dispatch_elapsed_s",
    "failure_reason",
    "scheduling_reason",
    "diagnostic",
    "detail",
    "cell_id",
    "zone",
    "pool_id",
    "members",
    "attempt",
)


@dataclass(frozen=True, slots=True)
class Moment:
    """One thing that happened, and how long after the first thing."""

    at: datetime
    since_start: timedelta
    headline: str
    detail: str
    state: str | None

    @property
    def elapsed(self) -> str:
        """Return the offset from the first event as ``h:mm:ss``."""

        total = int(self.since_start.total_seconds())
        hours, remainder = divmod(max(total, 0), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"


#: Actions whose only content is the state they moved to. Phrasing them as
#: "state changed (running)" says the same thing twice and buries the word that
#: matters.
_STATE_IS_THE_STORY = frozenset({"workload.state_changed"})


def _phrase(event: AuditEvent) -> str:
    """Return how one recorded action reads, or empty to let its state speak."""

    if event.action in _STATE_IS_THE_STORY:
        return ""
    known = _PHRASES.get(event.action)
    if known is not None:
        return known
    # Recorded but not yet phrased. Shown as the action itself rather than
    # dropped, because an event nobody thought to name is exactly the one worth
    # seeing when something unusual happened.
    return event.action.split(".", 1)[-1].replace("_", " ")


def _detail(event: AuditEvent) -> str:
    """Return the short why beside one event, or nothing when it adds none."""

    payload = event.payload if isinstance(event.payload, dict) else {}
    parts = []
    for key in _DETAIL_KEYS:
        value = payload.get(key)
        if value is None or value == "" or value == []:
            continue
        parts.append(f"{key}={value}" if not isinstance(value, str) else value)
        if len(parts) == 2:
            break
    return " | ".join(parts)


def timeline(events: tp.Sequence[AuditEvent]) -> tuple[Moment, ...]:
    """Order one workload's recorded events into a narrative.

    Ordered by the ledger's sequence rather than by timestamp. The sequence is
    the order things were committed; two events written in the same transaction
    share a timestamp, and sorting those by clock would reorder cause and
    effect.

    Args:
        events: Every event about this workload and its leases.

    Returns:
        One moment per event, earliest first, each carrying its offset from the
        first. Empty when nothing was recorded.
    """

    ordered = sorted(events, key=lambda event: (event.sequence, event.timestamp))
    if not ordered:
        return ()
    start = ordered[0].timestamp
    return tuple(
        Moment(
            at=event.timestamp,
            since_start=event.timestamp - start,
            headline=_phrase(event),
            detail=_detail(event),
            state=event.new_state,
        )
        for event in ordered
    )


def render(moments: tp.Sequence[Moment]) -> tuple[str, ...]:
    """Render a narrative as aligned lines, newest last."""

    if not moments:
        return ("[eyvan] nothing has been recorded about this workload yet.",)
    width = max(len(moment.elapsed) for moment in moments)
    lines = []
    for moment in moments:
        headline = moment.headline or moment.state or "state changed"
        if moment.state and moment.state != headline:
            headline = f"{headline} ({moment.state})"
        line = f"  {moment.elapsed:>{width}}  {headline}"
        if moment.detail:
            line = f"{line}  --  {moment.detail}"
        lines.append(line)
    return tuple(lines)


__all__ = ("Moment", "render", "timeline")
