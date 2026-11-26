# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.server.service._control_plane.shared import (
    _LIVE_OUTBOX_STATES,
    CandidateZone,
    CellState,
    InputError,
    OutboxRecord,
    OutboxState,
    Permission,
    Principal,
    WorkloadDefinition,
    WorkloadExplanation,
    WorkloadRecord,
    WorkloadState,
    _explain_base,
    _explain_from_scheduling_result,
    candidate_zone_report,
    datetime,
    locality_index,
    preferred_cells_for,
    tp,
    utcnow,
)


class SchedulingMixin:
    """Transactional scheduling operations."""

    async def schedule_once(self, principal: Principal):
        """Return one pure scheduling plan without committing its proposed leases."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        if self.scheduler is None:
            raise InputError("a scheduling service is not configured")
        async with self.repository.transaction() as tx:
            workloads = [
                workload
                for workload in await tx.list_workloads()
                if workload.state is not WorkloadState.PENDING or workload.preemption_id is None
            ]
            cells = await tx.list_cells()
            leases = await tx.list_leases()
            budgets = await tx.list_budgets()
            quotas = await tx.list_quotas()
            policies = await tx.list_policies()
            # The read that made grants real. They were issued, reported on and
            # expired, and never consulted by the thing that decides whether a
            # workload may run -- so a grant was recorded and licensed nothing.
            grants = await tx.list_budget_grants() if self.plane_id else []
            quota_grants = await tx.list_quota_grants() if self.plane_id else []
        return self.scheduler.schedule(
            workloads,
            cells,
            leases=leases,
            budgets=budgets,
            quotas=quotas,
            policies=policies,
            grants=grants,
            quota_grants=quota_grants,
            plane_id=self.plane_id,
        )

    @staticmethod
    async def _workload_delivery(tx: tp.Any, workload: WorkloadRecord) -> OutboxRecord | None:
        """Return the newest still-moving command carrying this workload.

        Looked up by aggregate rather than remembered on the workload, so it
        cannot drift from the record that actually decides whether the workload
        reaches its cell. Settled commands -- delivered, failed for good, or
        discarded -- are excluded: explaining a succeeded run with "its dispatch
        command is being retried" described a corpse as if it were moving.
        """

        try:
            commands = await tx.list_outbox_for_aggregates("workload", {workload.id: workload.generation})
        except Exception:
            # Explaining a workload must never fail because the delivery lookup
            # did; the scheduling answer above is still worth returning.
            return None
        # A still-scheduled workload's failed command is alive -- the outbox is
        # retrying it. The same state on a terminal workload is a corpse: settle
        # stops the clock, so nothing can move it again.
        terminal = workload.state in {
            WorkloadState.SUCCEEDED,
            WorkloadState.FAILED,
            WorkloadState.CANCELLED,
        }
        live = _LIVE_OUTBOX_STATES if terminal else _LIVE_OUTBOX_STATES | {OutboxState.FAILED}
        commands = [command for command in commands if command.state in live]
        if not commands:
            return None
        return max(commands, key=lambda command: command.updated_at)

    async def explain_workload(self, principal: Principal, workload_id: str) -> WorkloadExplanation:
        """Return why one workload is or is not placeable right now.

        The scheduler is pure, so this re-runs it against current state instead
        of reporting a verdict cached when the workload was last considered.
        The answer therefore reflects the fleet as it is now, and no second
        durable field has to be kept in sync at every ``scheduling_reason``
        write site.
        """

        if self.scheduler is None:
            raise InputError("a scheduling service is not configured")
        async with self.repository.transaction() as tx:
            workload = await self._require_workload(tx, workload_id)
            self.authorizer.require(principal, Permission.READ, org_id=workload.org_id, project_id=workload.project_id)
            if workload.state is not WorkloadState.PENDING:
                # Only pending workloads are scheduled, so skip the full snapshot
                # load -- but do not stop at "not my department". Past PENDING the
                # question "why is this not running" is answered by delivery, and
                # reporting only the state left an operator watching a spinner
                # while the command behind it failed and retried in silence.
                delivery = await self._workload_delivery(tx, workload)
                message = f"workload is {workload.state.value}; only pending workloads are scheduled"
                if delivery is not None and delivery.attempts > 1:
                    message = (
                        f"workload is {workload.state.value}; its {delivery.topic} command has failed "
                        f"{delivery.attempts} time(s) and is being retried"
                    )
                return WorkloadExplanation(
                    **_explain_base(workload),
                    placeable=False,
                    message=message,
                    delivery_state=None if delivery is None else delivery.state.value,
                    delivery_attempts=0 if delivery is None else delivery.attempts,
                    delivery_error=None if delivery is None else delivery.last_error,
                )
            workloads = [
                candidate
                for candidate in await tx.list_workloads()
                if candidate.state is not WorkloadState.PENDING or candidate.preemption_id is None
            ]
            cells = await tx.list_cells()
            leases = await tx.list_leases()
            budgets = await tx.list_budgets()
            quotas = await tx.list_quotas()
            policies = await tx.list_policies()
            # The read that made grants real. They were issued, reported on and
            # expired, and never consulted by the thing that decides whether a
            # workload may run -- so a grant was recorded and licensed nothing.
            grants = await tx.list_budget_grants() if self.plane_id else []
            quota_grants = await tx.list_quota_grants() if self.plane_id else []
            pools_for_explain = await tx.list_capacity_pools()
        result = self.scheduler.schedule(
            workloads,
            cells,
            leases=leases,
            budgets=budgets,
            quotas=quotas,
            policies=policies,
            grants=grants,
            quota_grants=quota_grants,
            plane_id=self.plane_id,
        )
        explanation = _explain_from_scheduling_result(workload, result)
        zones = candidate_zone_report(workload, pools_for_explain, cells, leases, now=utcnow())
        if zones:
            # Which zone this workload will ask next, and on what evidence. The
            # belief state has been deciding it since alternates existed and
            # saying so nowhere, so "why is it waiting on a zone that keeps
            # reclaiming while another sits idle" had no answer short of reading
            # cell records by hand.
            explanation = explanation.model_copy(
                update={"candidate_zones": tuple(CandidateZone.model_validate(zone) for zone in zones)}
            )
        if explanation.placeable:
            return explanation
        provisioning = tuple(
            sorted(
                cell.id
                for cell in cells
                if cell.autoscale_request_workload_id == workload.id
                and cell.state in {CellState.REGISTERING, CellState.READY}
            )
        )
        if not provisioning:
            return explanation
        # The scheduler only sees cells that exist, so on its own it reports a
        # workload waiting for capacity exactly like one that can never run.
        return explanation.model_copy(
            update={
                "provisioning_cell_ids": provisioning,
                "message": (f"capacity is being provisioned for this workload: {', '.join(provisioning)}"),
            }
        )

    async def resolve_locality_preferences(
        self,
        spec: WorkloadDefinition,
        *,
        now: datetime | None = None,
    ) -> WorkloadDefinition:
        """Fill ``preferred_cells`` from where this spec's named assets recently ran.

        An explicit ``preferred_cells`` is never overwritten. A pin is a statement
        the submitter made about placement, and a cache guess has no standing to
        argue with it -- silently replacing it would make an explicit pin behave
        differently depending on unrelated fleet history.

        Confirmed managed stages outrank both weaker signals, and only confirmed
        ones are read at all: a stage that is being fetched, could not be
        observed, or was proven against a generation the cell no longer runs
        contributes nothing, because preferring a cell on the strength of bytes
        nobody confirmed is exactly the failure managed staging exists to remove.
        Stages are fetched for this spec's assets only, so the submission path
        stays proportional to what the workload named rather than to the fleet's
        whole cache.
        """

        if not spec.named_assets or spec.preferred_cells:
            return spec
        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            index = locality_index(
                await tx.list_workloads(),
                await tx.list_leases(),
                now=moment,
                cells=await tx.list_cells(),
                stages=await tx.list_cell_stages(assets=spec.named_assets),
            )
        preferred = preferred_cells_for(spec.named_assets, index)
        if not preferred:
            return spec
        return spec.model_copy(update={"preferred_cells": preferred})

    async def group_is_placeable(self, principal: Principal, *, member_ids: tp.Sequence[str]) -> bool:
        """Return whether every named member could be placed in one scheduling pass.

        Members are presented to the scheduler as schedulable, because the flag
        this answer decides is the very one the scheduler filters on -- asking
        without flipping it would report every gated member as unplaceable and the
        group would never open.

        One pass over all of them together, not one call each: the members compete
        for the same cells, and evaluating them separately would happily report
        that two members each fit a single free cell.
        """

        if self.scheduler is None:
            raise InputError("a scheduling service is not configured")
        wanted = set(member_ids)
        if not wanted:
            return False
        async with self.repository.transaction() as tx:
            candidates: list[WorkloadRecord] = []
            for workload in await tx.list_workloads():
                if workload.id in wanted:
                    self.authorizer.require(
                        principal, Permission.READ, org_id=workload.org_id, project_id=workload.project_id
                    )
                    candidates.append(workload.model_copy(update={"schedulable": True}))
                elif workload.state is not WorkloadState.PENDING or workload.preemption_id is None:
                    candidates.append(workload)
            cells = await tx.list_cells()
            leases = await tx.list_leases()
            budgets = await tx.list_budgets()
            quotas = await tx.list_quotas()
            policies = await tx.list_policies()
            # The read that made grants real. They were issued, reported on and
            # expired, and never consulted by the thing that decides whether a
            # workload may run -- so a grant was recorded and licensed nothing.
            grants = await tx.list_budget_grants() if self.plane_id else []
            quota_grants = await tx.list_quota_grants() if self.plane_id else []
        result = self.scheduler.schedule(
            candidates,
            cells,
            leases=leases,
            budgets=budgets,
            quotas=quotas,
            policies=policies,
            grants=grants,
            quota_grants=quota_grants,
            plane_id=self.plane_id,
        )
        placed = {placement.workload_id for placement in result.placements}
        # Subset, not intersection: a member that no longer exists cannot appear in
        # the placements, so this one comparison also answers "is a member missing"
        # without a separate existence check that could drift from it.
        return wanted <= placed
