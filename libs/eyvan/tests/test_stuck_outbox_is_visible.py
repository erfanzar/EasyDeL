# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""I3 must see a command that is retrying, not only one that has never been tried.

Found on hardware. A cancel was dispatched to a native worker that answered 500;
the agent retried it twenty-four times; the workload never left `cancelling`, its
lease was never released, `eyvan down` could not drain the cell, and a v5p-16 sat
pinned and billing until it was force-deleted through gcloud.

`eyvan capacity invariants` reported no violations the entire time, and I3 is the
invariant whose whole job is "no outbox command sits undelivered indefinitely".

The reason is where a failing command lives. Between attempts it is settled to
FAILED with `available_at` in the future; it is PENDING or PROCESSING only for
the instant it is actually being tried. The check listed those two states, so a
command retrying forever was invisible to it -- the check could only ever catch
a command nobody had picked up, which is the easy case.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import OutboxRecord, OutboxState, utcnow
from eyvan.server.invariants import STUCK_OUTBOX_AFTER_S, Invariant, check_fleet
from eyvan.server.persistence import InMemoryRepository

NOW = utcnow()


def _command(state: OutboxState, *, age_s: float, attempts: int = 0) -> OutboxRecord:
    created = NOW - timedelta(seconds=age_s)
    return OutboxRecord(
        id=f"cmd-{state.value}-{int(age_s)}",
        topic="workload.cancel",
        aggregate_type="workload",
        aggregate_id="wl-1",
        idempotency_key=f"key-{state.value}-{int(age_s)}",
        generation=1,
        state=state,
        attempts=attempts,
        created_at=created,
        updated_at=created,
        available_at=NOW + timedelta(seconds=60),
    )


async def _violations(*commands: OutboxRecord):
    repository = InMemoryRepository()
    await repository.open()
    async with repository.transaction() as tx:
        for command in commands:
            await tx.put_outbox(command)
    return [v for v in await check_fleet(repository, now=NOW) if v.invariant is Invariant.NO_STUCK_OUTBOX]


@pytest.mark.asyncio
async def test_a_command_stuck_retrying_is_reported() -> None:
    """The hardware case: FAILED, old, and being retried forever."""

    found = await _violations(_command(OutboxState.FAILED, age_s=STUCK_OUTBOX_AFTER_S + 60, attempts=24))

    assert len(found) == 1
    assert "workload.cancel" in found[0].detail


@pytest.mark.asyncio
async def test_a_command_never_picked_up_is_still_reported() -> None:
    """The case the check already caught, asserted so the fix cannot have lost it."""

    found = await _violations(_command(OutboxState.PENDING, age_s=STUCK_OUTBOX_AFTER_S + 60))

    assert len(found) == 1


@pytest.mark.asyncio
async def test_a_command_in_flight_is_reported_when_it_is_old_enough() -> None:
    found = await _violations(_command(OutboxState.PROCESSING, age_s=STUCK_OUTBOX_AFTER_S + 60))

    assert len(found) == 1


@pytest.mark.asyncio
async def test_a_recent_failure_is_not_a_violation() -> None:
    """A command that failed once and is backing off is ordinary.

    Firing here would report every transient provider hiccup, and an invariant
    that cries wolf is one an operator learns to skip.
    """

    found = await _violations(_command(OutboxState.FAILED, age_s=30, attempts=1))

    assert found == []


@pytest.mark.asyncio
async def test_a_settled_command_is_never_a_violation() -> None:
    """SUCCEEDED and DISCARDED are finished; age says nothing about them."""

    found = await _violations(
        _command(OutboxState.SUCCEEDED, age_s=STUCK_OUTBOX_AFTER_S * 10),
        _command(OutboxState.DISCARDED, age_s=STUCK_OUTBOX_AFTER_S * 10),
    )

    assert found == []


@pytest.mark.asyncio
async def test_the_age_is_measured_from_creation_not_from_the_last_attempt() -> None:
    """Otherwise each retry resets the clock and nothing is ever stuck.

    This is the subtle way the check could be reintroduced broken: a command
    retried every 300s would look 300s old forever.
    """

    command = _command(OutboxState.FAILED, age_s=STUCK_OUTBOX_AFTER_S + 600, attempts=12)
    command = command.model_copy(update={"updated_at": NOW})

    found = await _violations(command)

    assert len(found) == 1


@pytest.mark.asyncio
async def test_the_message_states_what_it_measured() -> None:
    """ "Created Ns ago and still X" and "has been X for Ns" are different claims.

    Only the first is measured: the age comes from `created_at`, so a command
    that has been retrying for hours may have entered PROCESSING a second ago.
    The original wording asserted the second, and an operator reading it would go
    looking for a four-hour hang in whichever state the command happened to be in
    when the check ran.
    """

    found = await _violations(_command(OutboxState.PROCESSING, age_s=STUCK_OUTBOX_AFTER_S + 100, attempts=9))

    assert len(found) == 1
    assert "was created" in found[0].detail
    assert "is still processing" in found[0].detail
    assert "has been processing for" not in found[0].detail
