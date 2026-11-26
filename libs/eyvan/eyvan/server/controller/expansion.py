# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Releasing bounded fan-out waves, and placing heterogeneous groups whole."""

from __future__ import annotations

import uuid

from eyvan.records import (
    AuditEvent,
    CoordinationKind,
    ExpansionAdmissionRecord,
    ExpansionAdmissionState,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.controller._support import logger
from eyvan.server.persistence.base import RepositorySession
from eyvan.server.service.batch import task_index
from eyvan.server.service.errors import ConflictError, InputError
from eyvan.server.service.state import TERMINAL_WORKLOAD_STATES


class _ControllerExpansion:
    """Releasing bounded fan-out waves, and placing heterogeneous groups whole.

    A concern of :class:`~eyvan.server.controller.EyvanController`; not
    instantiated on its own.
    """

    async def release_expansion_waves_once(self) -> int:
        """Release the next wave of children for every expansion still owing one.

        An expansion admitted under a wave plan created all of its children and
        made only the first wave schedulable. This is what makes the rest
        schedulable, as earlier children stop competing -- so a ten-thousand-task
        array occupies a bounded slice of the queue for its whole life instead of
        all of it for the first minute.

        Everything this decides is read from durable rows. The plan comes from the
        expansion ledger, the occupancy from the children themselves, and *which*
        child is next from the children's own stable order -- never from a counter
        that could disagree with them. A leader that dies between flipping a child
        and committing the ledger therefore resumes with the same answer: that
        child is already schedulable, so it is not released again, and the next
        one is still gated, so it is not lost.

        Returns:
            How many children were made schedulable in this pass.
        """

        async with self.repository.transaction() as transaction:
            ledgers = await transaction.list_expansion_admissions(state=ExpansionAdmissionState.ADMITTING)
        released = 0
        for ledger in ledgers:
            released += await self._release_one_expansion_wave(ledger)
        return released

    async def _release_one_expansion_wave(self, ledger: ExpansionAdmissionRecord) -> int:
        """Release one expansion's next wave inside a single transaction."""

        async with self.repository.transaction() as transaction:
            current = await transaction.get_expansion_admission(ledger.workload_id)
            if current is None or current.state is not ExpansionAdmissionState.ADMITTING:
                return 0
            coordinator = await transaction.get_workload(current.workload_id)
            if coordinator is None or coordinator.state in TERMINAL_WORKLOAD_STATES:
                # The array was cancelled or has finished; its remaining children
                # went terminal with it. Closing the ledger stops this phase from
                # scanning a plan that can never release anything again.
                await self._close_expansion_ledger(transaction, current, reason="coordinator is no longer running")
                return 0
            children = await transaction.list_child_workloads(current.workload_id)
            candidates = self._wave_release_candidates(children)
            waiting = [child for child in candidates if not child.schedulable and child.state is WorkloadState.PENDING]
            # Dependents are excluded from ``candidates`` -- their gate is their
            # predecessors, not this plan -- but they are still children this
            # ledger owes a bound. Closing while one is still gated would hand
            # the DAG phase a ``wave_size=None`` and let its next release flood
            # the queue, so the ledger stays admitting until every child has
            # been released or finished.
            if not waiting and not any(
                not child.schedulable and child.state is WorkloadState.PENDING for child in children
            ):
                await self._close_expansion_ledger(transaction, current, reason="every child has been released")
                return 0
            # Occupancy counts *every* child of the expansion that is competing,
            # not only the ones this ledger opened. A DAG node released because
            # its predecessor succeeded holds a place in the queue exactly like a
            # root does, and ignoring it would let an expansion exceed its own
            # bound the moment one stage finished.
            occupying = sum(1 for child in children if child.schedulable and child.state not in TERMINAL_WORKLOAD_STATES)
            openings = current.wave_size - occupying
            if openings < 1:
                return 0
            now = utcnow()
            releasing = waiting[:openings]
            if not releasing:
                # Nothing this plan can release -- the remaining gated children
                # are dependents the DAG phase owns, bounded by this ledger.
                return 0
            for child in releasing:
                await transaction.put_workload(
                    child.model_copy(
                        update={
                            "schedulable": True,
                            "scheduling_reason": "expansion wave admitted; waiting for scheduling",
                            "updated_at": now,
                        }
                    )
                )
                await transaction.append_event(
                    AuditEvent(
                        id=f"evt_{uuid.uuid4().hex}",
                        actor=self._principal.subject,
                        org_id=child.org_id,
                        project_id=child.project_id,
                        entity_type="workload",
                        entity_id=child.id,
                        action="workload.wave_admitted",
                        payload={
                            "expansion_id": current.id,
                            "coordinator": current.workload_id,
                            "wave_size": current.wave_size,
                        },
                    )
                )
            released_total = min(current.released_children + len(releasing), current.total_children)
            await transaction.put_expansion_admission(
                current.model_copy(update={"released_children": released_total, "updated_at": now})
            )
            return len(releasing)

    @staticmethod
    def _wave_release_candidates(children: list[WorkloadRecord]) -> list[WorkloadRecord]:
        """Return the children one ledger may release, in their stable order.

        A DAG's dependent nodes are excluded. Their gate is their predecessors,
        not this plan, and releasing one early would run a stage before the stage
        it waits for -- so waves bound a DAG's entry fan-out and nothing else.
        Ordering is by task index for an array and by name otherwise, so a replay
        or a restart releases the same child next.
        """

        candidates = [child for child in children if not child.depends_on]
        return sorted(candidates, key=lambda child: (task_index(child.spec.environment), child.spec.name, child.id))

    async def _close_expansion_ledger(
        self,
        transaction: RepositorySession,
        ledger: ExpansionAdmissionRecord,
        *,
        reason: str,
    ) -> None:
        """Mark one admission plan absorbing, so the wave phase stops scanning it."""

        logger.debug("closing expansion admission %s: %s", ledger.id, reason)
        await transaction.put_expansion_admission(
            ledger.model_copy(update={"state": ExpansionAdmissionState.COMPLETE, "updated_at": utcnow()})
        )

    async def place_admitted_groups_once(self) -> int:
        """Place every member of each feasible heterogeneous group, or none of it.

        Off unless ``atomic_group_placement`` is configured, because it changes
        which component places a group's members: with it, the group is leased as
        a unit inside one transaction; without it, the members are released to the
        ordinary scheduler and placed one at a time, which is what shipped.

        A group that cannot be placed whole in this pass is left exactly as it
        was -- gated, holding nothing -- and reconsidered next cycle.

        Returns:
            How many members were leased in this pass.
        """

        if not self.settings.atomic_group_placement or self.service.scheduler is None:
            return 0
        async with self.repository.transaction() as transaction:
            coordinators = [
                workload
                for workload in await transaction.list_workloads(state=WorkloadState.PENDING)
                if workload.coordination_kind is CoordinationKind.GROUP
            ]
        placed = 0
        for coordinator in coordinators:
            try:
                members = await self.service.place_group_atomically(
                    self._principal,
                    coordinator_id=coordinator.id,
                    lease_ttl_s=self.settings.lease_ttl_s,
                )
            except (ConflictError, InputError) as exc:
                # State moved under the decision, or the group is not placeable on
                # the terms it was admitted under. Neither is fatal and neither
                # left a lease behind: the transaction that would have written one
                # is the one that raised.
                logger.warning("could not place group %s atomically: %s", coordinator.id, exc)
                continue
            placed += len(members)
        return placed


__all__ = ("_ControllerExpansion",)
