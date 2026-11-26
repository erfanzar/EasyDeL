# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The fleet health check must not scale with everything the fleet ever ran."""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import (
    CommandExecution,
    LeaseRecord,
    LeaseState,
    ResourceGang,
    ResourceVector,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.invariants import LEASE_HOLDING_STATES, Invariant, check_fleet
from eyvan.server.persistence.memory import InMemoryRepository

_SPEC = WorkloadDefinition(name="t", execution=CommandExecution(command="echo hi"))


class _CountingRepository(InMemoryRepository):
    """Counts the workload rows the health check actually reads.

    Rows read is the cost that mattered: the check ran inside one transaction on
    a 60-second poll, so a fleet with a long history slowed its own health check
    until the poll timed out -- and a timing-out health check reports
    "unreachable", which an operator cannot tell apart from a plane that is down.
    """

    def __init__(self) -> None:
        super().__init__()
        self.workload_rows_read = 0

    def transaction(self):
        session = super().transaction()
        repository = self

        class _Counting:
            async def __aenter__(self):
                self._inner = await session.__aenter__()
                return self

            async def __aexit__(self, *exc):
                return await session.__aexit__(*exc)

            def __getattr__(self, name):
                return getattr(self._inner, name)

            async def list_workloads(self, **kwargs):
                rows = await self._inner.list_workloads(**kwargs)
                repository.workload_rows_read += len(rows)
                return rows

        return _Counting()


async def _seed(repository, *, terminal: int, live: int) -> None:
    async with repository.transaction() as tx:
        for index in range(terminal):
            await tx.put_workload(
                WorkloadRecord(
                    id=f"done-{index}",
                    org_id="acme",
                    project_id="research",
                    spec=_SPEC,
                    state=WorkloadState.SUCCEEDED,
                    created_by="alice",
                )
            )
        for index in range(live):
            await tx.put_workload(
                WorkloadRecord(
                    id=f"live-{index}",
                    org_id="acme",
                    project_id="research",
                    spec=_SPEC,
                    state=WorkloadState.RUNNING,
                    created_by="alice",
                )
            )


async def test_the_health_check_reads_live_workloads_not_the_whole_history() -> None:
    """Rows read must track the live fleet, not everything it has ever run."""

    repository = _CountingRepository()
    await _seed(repository, terminal=500, live=3)

    await check_fleet(repository)

    assert repository.workload_rows_read == 3


async def test_a_longer_history_does_not_cost_the_health_check_more() -> None:
    """The property, stated directly: history is not an input to the cost."""

    small = _CountingRepository()
    await _seed(small, terminal=50, live=2)
    large = _CountingRepository()
    await _seed(large, terminal=5_000, live=2)

    await check_fleet(small)
    await check_fleet(large)

    assert small.workload_rows_read == large.workload_rows_read == 2


async def test_a_lease_outliving_a_settled_workload_is_still_caught() -> None:
    """The narrowed read must not lose the violation it exists to find.

    I2 is precisely about a lease whose workload has already settled, and a
    settled workload is not in the live set. Fetching those by id keeps the check
    bounded by active leases instead of by history.
    """

    repository = _CountingRepository()
    async with repository.transaction() as tx:
        await tx.put_workload(
            WorkloadRecord(
                id="wl-settled",
                org_id="acme",
                project_id="research",
                spec=_SPEC,
                state=WorkloadState.SUCCEEDED,
                created_by="alice",
            )
        )
        await tx.put_lease(
            LeaseRecord(
                id="lease-1",
                workload_id="wl-settled",
                cell_id="cell-a",
                cell_generation=1,
                org_id="acme",
                project_id="research",
                state=LeaseState.ACTIVE,
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                generation=1,
                fencing_token="token-1",
                expires_at=utcnow() + timedelta(hours=24),
            )
        )

    violations = await check_fleet(repository)

    assert Invariant.NO_LEASES_OUTLIVING_WORKLOADS in {item.invariant for item in violations}
    assert any("wl-settled" in item.detail for item in violations)


@pytest.mark.parametrize("state", sorted(LEASE_HOLDING_STATES, key=lambda item: item.value))
async def test_every_lease_holding_state_is_still_read(state: WorkloadState) -> None:
    """A state dropped from the narrowed read would silently stop being checked."""

    repository = _CountingRepository()
    async with repository.transaction() as tx:
        await tx.put_workload(
            WorkloadRecord(
                id="wl-1",
                org_id="acme",
                project_id="research",
                spec=_SPEC,
                state=state,
                created_by="alice",
            )
        )

    violations = await check_fleet(repository)

    assert repository.workload_rows_read == 1
    assert Invariant.NO_RUNNING_WORKLOAD_WITHOUT_A_LEASE in {item.invariant for item in violations}
