# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from math import isfinite

from eyvan.server.service._control_plane.shared import (
    AuditEvent,
    LeaseRecord,
    NotFoundError,
    Permission,
    Principal,
    WebhookRecord,
    WorkloadSpendRecord,
    _new_id,
    _on_demand_price,
    spend_of,
    validate_webhook_target,
)


class WebhooksMixin:
    """Transactional webhooks operations."""

    async def create_webhook(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        url: str,
        secret: str,
        entity_types: tuple[str, ...] = (),
        actions: tuple[str, ...] = (),
        allow_private_targets: bool = False,
    ) -> WebhookRecord:
        """Register where this project wants its lifecycle events pushed.

        Authorized as managing the project, not as reading it: a subscription
        sends this project's audit trail somewhere, and anybody who can create
        one decides where its history goes.

        The subscription is created starting from *now*, not from the beginning
        of the ledger. A new integration replaying a month of history into
        somebody's incident channel is the first impression this feature would
        otherwise make.
        """

        self.authorizer.require(principal, Permission.MANAGE_PROJECT, org_id=org_id, project_id=project_id)
        validate_webhook_target(url, allow_private=allow_private_targets)
        async with self.repository.transaction() as tx:
            latest = await tx.list_events(org_id=org_id, project_id=project_id, newest_first=True, limit=1)
            webhook = WebhookRecord(
                id=_new_id("wh"),
                org_id=org_id,
                project_id=project_id,
                url=url,
                secret=secret,
                entity_types=entity_types,
                actions=actions,
                delivered_sequence=latest[0].sequence if latest else 0,
                created_by=principal.subject,
            )
            await tx.put_webhook(webhook)
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="webhook",
                    entity_id=webhook.id,
                    action="webhook.created",
                    org_id=org_id,
                    project_id=project_id,
                )
            )
        return webhook

    async def list_webhooks(self, principal: Principal, *, org_id: str, project_id: str) -> list[WebhookRecord]:
        """Return this project's subscriptions, with their signing keys removed."""

        self.authorizer.require(principal, Permission.MANAGE_PROJECT, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            webhooks = await tx.list_webhooks(org_id=org_id, project_id=project_id)
        # Redacted on the way out rather than on the way in: the plane needs the
        # key to sign, and a signature anybody can forge proves nothing.
        return [webhook.redacted() for webhook in webhooks]

    async def delete_webhook(self, principal: Principal, webhook_id: str) -> WebhookRecord:
        """Stop delivering to one subscription and forget it."""

        async with self.repository.transaction() as tx:
            webhook = await tx.get_webhook(webhook_id)
            if webhook is None:
                raise NotFoundError(f"webhook {webhook_id!r} was not found")
            self.authorizer.require(
                principal,
                Permission.MANAGE_PROJECT,
                org_id=webhook.org_id,
                project_id=webhook.project_id,
            )
            await tx.delete_webhook(webhook_id)
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    entity_type="webhook",
                    entity_id=webhook_id,
                    action="webhook.deleted",
                    org_id=webhook.org_id,
                    project_id=webhook.project_id,
                )
            )
        return webhook.redacted()

    async def workload_cost(
        self,
        principal: Principal,
        workload_id: str,
        *,
        on_demand_hourly: float | None = None,
    ) -> WorkloadSpendRecord:
        """Total one run's capacity spend, and the on-demand alternative.

        Computed here rather than by the caller because the honest answer needs
        pool unit prices, and pool records are a fleet read. Assembled in the
        client, `eyvan workload cost` had to list capacity pools -- so the one
        command that answers "what did this run cost me" was refused to
        everybody who could not administer the fleet, while its help said
        nothing about needing that. Authorized exactly like reading the workload
        it is about, which is the permission the question actually implies.

        No price is invented. The comparison uses a price an operator declared
        on an on-demand pool of the same shape, or one the caller passed, and
        says which. An invented saving is the figure a team puts in a budget.
        """

        workload = await self.get_workload(principal, workload_id)
        async with self.repository.transaction() as tx:
            leases = await tx.list_leases(workload_id=workload_id)
            pools = await tx.list_capacity_pools()
            cells = {cell_id: await tx.get_cell(cell_id) for cell_id in {lease.cell_id for lease in leases}}

        # The request's primary pool is only a first choice. Describe observed
        # cells instead, and never let a replacement incarnation describe an
        # older lease. Missing or mixed observations cannot prove one shape/class.
        observed_cells = [
            cell
            for lease in leases
            if (cell := cells[lease.cell_id]) is not None and cell.generation == lease.cell_generation
        ]
        shapes = {cell.accelerator_type for cell in observed_cells}
        classes = {cell.capacity_class.value for cell in observed_cells}
        complete = bool(leases) and len(observed_cells) == len(leases)
        shape = next(iter(shapes)) if complete and len(shapes) == 1 else None
        capacity_class = next(iter(classes)) if complete and len(classes) == 1 else None

        # Admission freezes the actual billing rate on each lease. Today's pool
        # or cell declarations cannot establish historical pricing. A zero or
        # legacy-missing rate has no provenance proving "free", so stays unknown.
        # Check every lease before constructing the cell set: retries can reuse
        # the same cell with different frozen rates.
        priced_cells = (
            {lease.cell_id for lease in leases}
            if all(isfinite(lease.hourly_cost) and lease.hourly_cost > 0 for lease in leases)
            else None
        )
        spend = spend_of(
            [lease.model_dump(mode="json") for lease in leases],
            priced_cells=priced_cells,
        )
        hourly, source = _on_demand_price(shape, pools, on_demand_hourly)
        return WorkloadSpendRecord(
            workload_id=workload_id,
            state=workload.state.value,
            accelerator_hours=spend.accelerator_hours,
            settled=spend.settled,
            priced=spend.priced,
            reserved=spend.reserved,
            leases=spend.leases,
            shape=shape,
            capacity_class=capacity_class,
            on_demand_hourly=hourly,
            on_demand_source=source,
        )

    async def list_leases(
        self, principal: Principal, *, org_id: str, project_id: str, workload_id: str | None = None
    ) -> list[LeaseRecord]:
        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            return await tx.list_leases(org_id=org_id, project_id=project_id, workload_id=workload_id)

    async def get_lease(self, principal: Principal, lease_id: str) -> LeaseRecord:
        async with self.repository.transaction() as tx:
            lease = await tx.get_lease(lease_id)
            if lease is None:
                raise NotFoundError(f"lease {lease_id!r} was not found")
            self.authorizer.require(principal, Permission.READ, org_id=lease.org_id, project_id=lease.project_id)
            return lease
