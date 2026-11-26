# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capacity observation must not depend on a full outbox scan.

``_observe_cells_once``, ``_current_observable_cell``, and
``requeue_unprovisioned_cell`` each used ``list_outbox(limit=10_000)``. That is
O(cells x outbox) per reconcile cycle, and past the limit the scan can miss the
very command the check exists to find -- which would let the controller observe
a cell in the middle of provisioning, or requeue a live provision.
"""

from __future__ import annotations

from eyvan.records import OutboxRecord, OutboxState
from eyvan.server.persistence import InMemoryRepository


async def test_targeted_outbox_query_returns_only_the_requested_generation() -> None:
    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        for generation in (1, 2):
            await transaction.put_outbox(
                OutboxRecord(
                    id=f"cmd-{generation}",
                    topic="capacity.apply",
                    aggregate_type="cell",
                    aggregate_id="cell-a",
                    idempotency_key=f"capacity:cell-a:{generation}",
                    generation=generation,
                    state=OutboxState.PENDING,
                    payload={},
                )
            )
        # A command for a different cell must never be returned.
        await transaction.put_outbox(
            OutboxRecord(
                id="cmd-other",
                topic="capacity.apply",
                aggregate_type="cell",
                aggregate_id="cell-b",
                idempotency_key="capacity:cell-b:1",
                generation=1,
                state=OutboxState.PENDING,
                payload={},
            )
        )

    async with repository.transaction() as transaction:
        matched = await transaction.list_outbox_for_aggregates("cell", {"cell-a": 2}, topic="capacity.apply")

    assert [command.id for command in matched] == ["cmd-2"]


async def test_a_different_topic_is_not_returned() -> None:
    """The capacity check must not be satisfied by an unrelated command."""

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_outbox(
            OutboxRecord(
                id="cmd-dispatch",
                topic="workload.dispatch",
                aggregate_type="cell",
                aggregate_id="cell-a",
                idempotency_key="dispatch:cell-a:1",
                generation=1,
                state=OutboxState.PENDING,
                payload={},
            )
        )

    async with repository.transaction() as transaction:
        matched = await transaction.list_outbox_for_aggregates("cell", {"cell-a": 1}, topic="capacity.apply")

    assert matched == []


async def test_empty_aggregate_set_short_circuits_without_scanning() -> None:
    """No REGISTERING cells means no outbox work at all."""

    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_outbox(
            OutboxRecord(
                id="cmd-1",
                topic="capacity.apply",
                aggregate_type="cell",
                aggregate_id="cell-a",
                idempotency_key="capacity:cell-a:1",
                generation=1,
                state=OutboxState.PENDING,
                payload={},
            )
        )

    async with repository.transaction() as transaction:
        assert await transaction.list_outbox_for_aggregates("cell", {}, topic="capacity.apply") == []
