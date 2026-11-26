# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    AuditEvent,
    BudgetRecord,
    Permission,
    Principal,
)


class EventsMixin:
    """Transactional events operations."""

    async def list_budgets(self, principal: Principal, *, org_id: str) -> list[BudgetRecord]:
        self.authorizer.require(principal, Permission.READ, org_id=org_id)
        async with self.repository.transaction() as tx:
            budgets = await tx.list_budgets(org_id=org_id)
            return [
                budget
                for budget in budgets
                if self.authorizer.allows(principal, Permission.READ, org_id=budget.org_id, project_id=budget.project_id)
            ]

    async def list_events(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        after_sequence: int = 0,
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            return await tx.list_events(
                after_sequence=after_sequence,
                org_id=org_id,
                project_id=project_id,
                entity_type=entity_type,
                entity_id=entity_id,
                limit=limit,
            )

    async def list_fleet_events(
        self,
        principal: Principal,
        *,
        after_sequence: int = 0,
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Return fleet-wide audit history to fleet administrators."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            return await tx.list_events(
                after_sequence=after_sequence,
                entity_type=entity_type,
                entity_id=entity_id,
                limit=limit,
            )

    async def wait_for_events(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        after_sequence: int,
        timeout: float | None = None,
    ) -> list[AuditEvent]:
        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        return await self.repository.wait_for_events(
            after_sequence=after_sequence,
            org_id=org_id,
            project_id=project_id,
            timeout=timeout,
        )
