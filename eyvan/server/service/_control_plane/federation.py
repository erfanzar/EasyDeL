# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import math
from decimal import Decimal, localcontext

from eyvan.server.service._control_plane.shared import (
    BudgetGrantRecord,
    BudgetRecord,
    FederatedBudgetStatus,
    GrantState,
    InputError,
    NotFoundError,
    Permission,
    PlanePeerRecord,
    Principal,
    QuotaGrantRecord,
    ResourceLimits,
    _dump_limits,
    _fingerprint,
    _new_id,
    _scope,
    _validate_idempotency_key,
    datetime,
    federated_spend,
    outstanding_total,
    resource_quantities,
    utcnow,
)
from eyvan.server.service.federation import _QUOTA_PRECISION, _issuable_amount_exact, _quota_headroom_exact, _sum_money
from eyvan.server.service.scheduling import to_core_lease
from eyvan.server.service.state import ACTIVE_LEASE_STATES


class FederationMixin:
    """Transactional federation operations."""

    async def put_plane_peer(
        self,
        principal: Principal,
        *,
        plane_id: str,
        endpoint: str,
        description: str = "",
        enabled: bool = True,
    ) -> PlanePeerRecord:
        """Register or update another control plane this one may grant budget to.

        Peers are configured, never discovered. Discovery would let a plane claim a
        tenant's budget by asserting it is the home plane, and nothing in this
        design has the authority to refuse that claim -- so the list of who counts
        as a peer stays something an operator wrote down.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        if plane_id == self.plane_id:
            raise InputError("a plane does not register itself as its own peer")
        async with self.repository.transaction() as tx:
            existing = await tx.get_plane_peer(plane_id)
            peer = PlanePeerRecord(
                plane_id=plane_id,
                endpoint=endpoint,
                description=description,
                enabled=enabled,
                created_at=existing.created_at if existing is not None else utcnow(),
            )
            await tx.put_plane_peer(peer)
            await self._audit_scoped(
                tx,
                principal,
                org_id=None,
                project_id=None,
                entity_type="plane_peer",
                entity_id=plane_id,
                action="plane.peer_registered",
                payload={"endpoint": endpoint, "enabled": enabled},
            )
            return peer

    async def list_plane_peers(self, principal: Principal) -> list[PlanePeerRecord]:
        """List the control planes this one knows about."""

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            return await tx.list_plane_peers()

    async def record_grant_spend(
        self,
        *,
        grant_id: str,
        reported_spend: float,
        now: datetime | None = None,
    ) -> BudgetGrantRecord:
        """Record what a grantee reports having spent against its grant.

        Monotone: a report lower than one already recorded is ignored rather than
        applied. A peer restarting mid-period and reporting its fresh counter would
        otherwise walk the fleet-wide total *backwards*, which reads as money
        returned and is the one direction this number must never move on its own.
        """

        if not math.isfinite(reported_spend):
            raise InputError("reported grant spend must be finite")
        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            grant = await tx.get_budget_grant(grant_id)
            if grant is None:
                raise NotFoundError(f"budget grant {grant_id!r} was not found")
            if reported_spend <= grant.reported_spend:
                return grant
            updated = grant.model_copy(update={"reported_spend": reported_spend, "reported_at": moment})
            await tx.put_budget_grant(updated)
            return updated

    async def issue_budget_grant(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        grantee_plane_id: str,
        amount: float,
        expires_at: datetime,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> BudgetGrantRecord:
        """Grant a peer plane a slice of this tenant's budget.

        The home plane refuses to issue past its own limit, counting both what it
        has already granted away and what it has itself spent. That single check is
        the entire cross-plane guarantee: no plane consults another at admission
        time, and the fleet-wide total is bounded because the slices were bounded
        when they were cut.

        Replacement retires the old new-admission authority, but keeps its full
        monetary reservation and all reports. Neither retirement nor a report is
        reliable proof of final consumption. Local active reservations and all
        retained issued rights partition the same current-period limit under the
        admission lock; no final-consumption/refund protocol is invented here.

        Raises:
            InputError: If the grant would exceed the tenant's remaining limit, or
                if this plane has no identity and therefore cannot be a home plane.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        if not self.plane_id:
            raise InputError("this control plane has no plane_id, so it cannot issue budget grants")
        if grantee_plane_id == self.plane_id:
            raise InputError("a plane does not grant budget to itself; it holds the limit directly")
        if not math.isfinite(amount):
            raise InputError("budget grant amount must be finite")
        moment = now or utcnow()
        key = _validate_idempotency_key(idempotency_key)
        scope = _scope(org_id, project_id)
        fingerprint = _fingerprint({"grantee": grantee_plane_id, "amount": amount, "expires_at": expires_at.isoformat()})
        async with self.repository.transaction() as tx:
            await self._lock_admission_scope(tx, org_id, project_id)
            existing_key = await self._idempotent_resource(tx, scope, "budget.grant", key, fingerprint)
            if existing_key is not None:
                stored = await tx.get_budget_grant(existing_key.resource_id)
                if stored is not None:
                    return stored
            budget = await tx.get_budget(org_id, project_id)
            if budget is None:
                raise NotFoundError(f"budget for {org_id}/{project_id} was not found")
            peer = await tx.get_plane_peer(grantee_plane_id)
            if peer is None or not peer.enabled:
                raise NotFoundError(f"plane peer {grantee_plane_id!r} is not registered or is disabled")
            rolled = self._rolled_budget(budget, moment)
            if rolled != budget:
                await tx.put_budget(rolled)
            budget = rolled
            grants = await tx.list_budget_grants(org_id=org_id, project_id=project_id)
            if any(grant.grantee_plane_id == self.plane_id for grant in grants):
                # The local record is the peer's consumption ledger, not a new
                # home limit. No multi-hop delegation/finality protocol exists.
                raise InputError("an inbound budget scope cannot issue independent home authority")
            grants = [grant for grant in grants if grant.home_plane_id == self.plane_id]
            local_reserved = _sum_money(
                lease.reserved_cost
                for lease in await tx.list_leases()
                if lease.org_id == org_id and lease.project_id == project_id and lease.state in ACTIVE_LEASE_STATES
            )
            replacing = next(
                (grant for grant in grants if grant.grantee_plane_id == grantee_plane_id and grant.active_at(moment)),
                None,
            )
            headroom = _issuable_amount_exact(
                budget.limit,
                grants,
                now=moment,
                local_spent=budget.spent,
                local_reserved=local_reserved,
            )
            if Decimal(str(amount)) > headroom:
                raise InputError(
                    f"grant of {amount} exceeds the {headroom} remaining on {org_id}/{project_id}; "
                    f"the limit is {budget.limit}, {budget.spent} is spent here and the rest is granted away"
                )
            if replacing is not None:
                await tx.put_budget_grant(replacing.model_copy(update={"state": GrantState.REVOKED}))
            grant = BudgetGrantRecord(
                id=_new_id("bgr"),
                org_id=org_id,
                project_id=project_id,
                home_plane_id=self.plane_id,
                grantee_plane_id=grantee_plane_id,
                amount=amount,
                currency=budget.currency,
                issued_at=moment,
                expires_at=expires_at,
            )
            await tx.put_budget_grant(grant)
            await self._bind_idempotency(tx, scope, "budget.grant", key, fingerprint, grant)
            await self._audit_scoped(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                entity_type="budget_grant",
                entity_id=grant.id,
                action="budget.granted",
                payload={
                    "grantee_plane_id": grantee_plane_id,
                    "amount": amount,
                    "replaced": None if replacing is None else replacing.id,
                    "remaining_headroom": float(headroom - Decimal(str(amount))),
                },
            )
            return grant

    async def list_budget_grants(
        self,
        principal: Principal,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        include_inactive: bool = False,
        now: datetime | None = None,
    ) -> list[BudgetGrantRecord]:
        """List the budget slices this plane has issued or been issued.

        Fleet administrators see all scopes, organization administrators see
        their organizations, and project administrators see only their projects.
        A grant names a tenant and a sum of money, so an unfiltered listing would
        be every tenant's budget posture readable by anyone with a token.

        Expired grants are excluded by default and not deleted: the sweep marks
        them EXPIRED and the record stays, because "this peer used to hold
        headroom and stopped" is the first question asked when a grantee suddenly
        refuses work. ``include_inactive`` is how that question gets answered.
        """

        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            grants = await tx.list_budget_grants(org_id=org_id, project_id=project_id)
        if not include_inactive:
            grants = [grant for grant in grants if grant.active_at(moment)]
        if self.authorizer.allows(principal, Permission.MANAGE_FLEET):
            return list(grants)
        return [
            grant
            for grant in grants
            if self.authorizer.allows(
                principal, Permission.MANAGE_PROJECT, org_id=grant.org_id, project_id=grant.project_id
            )
        ]

    async def revoke_budget_grant(self, principal: Principal, *, grant_id: str) -> BudgetGrantRecord:
        """Withdraw a peer's authority to spend.

        Stops *new* admission on the grantee and says nothing about what already
        ran: that money is spent, and pretending otherwise by clawing back would
        make the home plane's ledger disagree with the invoice.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            grant = await tx.get_budget_grant(grant_id)
            if grant is None:
                raise NotFoundError(f"budget grant {grant_id!r} was not found")
            if grant.state is not GrantState.ACTIVE:
                return grant
            revoked = grant.model_copy(update={"state": GrantState.REVOKED})
            await tx.put_budget_grant(revoked)
            await self._audit_scoped(
                tx,
                principal,
                org_id=grant.org_id,
                project_id=grant.project_id,
                entity_type="budget_grant",
                entity_id=grant.id,
                action="budget.grant_revoked",
                payload={"grantee_plane_id": grant.grantee_plane_id, "amount": grant.amount},
            )
            return revoked

    async def federated_budget_status(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        now: datetime | None = None,
    ) -> FederatedBudgetStatus:
        """Report a tenant's spend across every plane that holds a grant.

        The peer figures are as fresh as each peer's last report and no fresher,
        which is why this informs and alerts but never admits. Admission stays
        local, against a grant that was bounded when it was issued -- so a stale
        report can make this number wrong without ever making spend unbounded.
        """

        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            budget = await tx.get_budget(org_id, project_id)
            if budget is None:
                raise NotFoundError(f"budget for {org_id}/{project_id} was not found")
            grants = await tx.list_budget_grants(org_id=org_id, project_id=project_id)
        budget = self._rolled_budget(budget, moment)
        grants = [grant for grant in grants if not self.plane_id or grant.home_plane_id == self.plane_id]
        active = [grant for grant in grants if grant.active_at(moment)]
        return FederatedBudgetStatus(
            org_id=org_id,
            project_id=project_id,
            home_plane_id=self.plane_id or None,
            limit=budget.limit,
            local_spent=budget.spent,
            granted_away=outstanding_total(grants, now=moment),
            reported_peer_spend=federated_spend(0.0, grants),
            total_spent=federated_spend(budget.spent, grants),
            active_grants=len(active),
            stale_reports=sum(1 for grant in active if grant.reported_at is None),
        )

    async def issue_quota_grant(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        grantee_plane_id: str,
        limits: ResourceLimits,
        expires_at: datetime,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> QuotaGrantRecord:
        """Grant a peer plane a slice of this tenant's quota.

        The budget analogue with one difference that matters: headroom has to hold
        in *every* dimension at once. A grant of 8 TPU and 64 CPU against a quota
        with TPU to spare and no CPU left is refused, and the refusal names every
        overrun dimension rather than the first -- an operator correcting one
        number at a time against a four-dimensional ceiling is four round trips,
        and the fourth is where they grant themselves fleet admin instead.

        Raises:
            InputError: If the grant overruns the tenant's remaining ceiling in any
                dimension, or if this plane has no identity.
            NotFoundError: If the tenant has no quota, or the peer is unregistered.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        if not self.plane_id:
            raise InputError("this control plane has no plane_id, so it cannot issue quota grants")
        if grantee_plane_id == self.plane_id:
            raise InputError("a plane does not grant quota to itself; it holds the limit directly")
        moment = now or utcnow()
        key = _validate_idempotency_key(idempotency_key)
        scope = _scope(org_id, project_id)
        fingerprint = _fingerprint(
            {"grantee": grantee_plane_id, "limits": _dump_limits(limits), "expires_at": expires_at.isoformat()}
        )
        async with self.repository.transaction() as tx:
            await self._lock_admission_scope(tx, org_id, project_id)
            existing_key = await self._idempotent_resource(tx, scope, "quota.grant", key, fingerprint)
            if existing_key is not None:
                stored = await tx.get_quota_grant(existing_key.resource_id)
                if stored is not None:
                    return stored
            quota = await tx.get_quota(org_id, project_id, None)
            if quota is None:
                raise NotFoundError(f"quota for {org_id}/{project_id} was not found")
            peer = await tx.get_plane_peer(grantee_plane_id)
            if peer is None or not peer.enabled:
                raise NotFoundError(f"plane peer {grantee_plane_id!r} is not registered or is disabled")
            grants = await tx.list_quota_grants(org_id=org_id, project_id=project_id)
            if any(grant.grantee_plane_id == self.plane_id for grant in grants):
                raise InputError("an inbound quota scope cannot issue independent home authority")
            grants = [grant for grant in grants if grant.home_plane_id == self.plane_id]
            replacing = next(
                (grant for grant in grants if grant.grantee_plane_id == grantee_plane_id and grant.active_at(moment)),
                None,
            )
            local_usage: dict[str, Decimal] = {}
            # Use the scheduler's canonical worker/elastic scaling, rather than
            # counting raw per-worker gangs as if they occupied only one worker.
            # Keep both scaling and accumulation exact through the admission
            # decision: float subtraction can erase a tiny active reservation.
            with localcontext(prec=_QUOTA_PRECISION):
                for lease in await tx.list_leases():
                    if (
                        lease.org_id != org_id
                        or lease.project_id != project_id
                        or lease.state not in ACTIVE_LEASE_STATES
                    ):
                        continue
                    workload = await tx.get_workload(lease.workload_id)
                    cell = await tx.get_cell(lease.cell_id)
                    if workload is None or cell is None:
                        raise InputError("active quota reservation lacks its workload or cell accounting")
                    usage = to_core_lease(lease, workload, landed_worker_count=cell.worker_count).resources
                    for name, quantity in usage.items():
                        local_usage[name] = local_usage.get(name, Decimal(0)) + quantity
            headroom = _quota_headroom_exact(quota.limits, grants, now=moment, local_usage=local_usage)
            overrun = tuple(
                sorted(
                    name
                    for name, quantity in resource_quantities(limits).items()
                    if Decimal(str(quantity)) > headroom.get(name, Decimal(0))
                )
            )
            if overrun:
                detail = ", ".join(
                    f"{name} {resource_quantities(limits)[name]} > {headroom.get(name, 0.0)}" for name in overrun
                )
                raise InputError(f"quota grant exceeds the headroom on {org_id}/{project_id}: {detail}")
            if replacing is not None:
                await tx.put_quota_grant(replacing.model_copy(update={"state": GrantState.REVOKED}))
            grant = QuotaGrantRecord(
                id=_new_id("qgr"),
                org_id=org_id,
                project_id=project_id,
                home_plane_id=self.plane_id,
                grantee_plane_id=grantee_plane_id,
                limits=limits,
                issued_at=moment,
                expires_at=expires_at,
            )
            await tx.put_quota_grant(grant)
            await self._bind_idempotency(tx, scope, "quota.grant", key, fingerprint, grant)
            await self._audit_scoped(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                entity_type="quota_grant",
                entity_id=grant.id,
                action="quota.granted",
                payload={
                    "grantee_plane_id": grantee_plane_id,
                    "limits": _dump_limits(limits),
                    "replaced": None if replacing is None else replacing.id,
                },
            )
            return grant

    async def list_quota_grants(
        self,
        principal: Principal,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
        include_inactive: bool = False,
        now: datetime | None = None,
    ) -> list[QuotaGrantRecord]:
        """List the quota slices this plane has issued or been issued."""

        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            grants = await tx.list_quota_grants(org_id=org_id, project_id=project_id)
        if not include_inactive:
            grants = [grant for grant in grants if grant.active_at(moment)]
        if self.authorizer.allows(principal, Permission.MANAGE_FLEET):
            return list(grants)
        return [
            grant
            for grant in grants
            if self.authorizer.allows(
                principal, Permission.MANAGE_PROJECT, org_id=grant.org_id, project_id=grant.project_id
            )
        ]

    async def revoke_quota_grant(self, principal: Principal, *, grant_id: str) -> QuotaGrantRecord:
        """Withdraw a peer's authority to place work against this tenant's quota.

        Stops *new* admission on the grantee and does not disturb what is already
        running: a quota is a concurrent ceiling, and evicting live work to enforce
        a ceiling change would make revocation a fleet-wide outage rather than an
        administrative act.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            grant = await tx.get_quota_grant(grant_id)
            if grant is None:
                raise NotFoundError(f"quota grant {grant_id!r} was not found")
            if grant.state is not GrantState.ACTIVE:
                return grant
            revoked = grant.model_copy(update={"state": GrantState.REVOKED})
            await tx.put_quota_grant(revoked)
            await self._audit_scoped(
                tx,
                principal,
                org_id=grant.org_id,
                project_id=grant.project_id,
                entity_type="quota_grant",
                entity_id=grant.id,
                action="quota.grant_revoked",
                payload={"grantee_plane_id": grant.grantee_plane_id},
            )
            return revoked

    async def expire_budget_grants_once(self, *, now: datetime | None = None) -> int:
        """Mark elapsed grants EXPIRED so an unreachable home plane fails closed.

        The grantee already refuses to admit past ``expires_at`` -- the check is on
        the timestamp, not on this sweep -- so this exists to make the *state* match
        what enforcement already does, which is what an operator reads when asking
        why a peer stopped admitting.
        """

        moment = now or utcnow()
        expired = 0
        async with self.repository.transaction() as tx:
            for grant in await tx.list_budget_grants():
                if grant.state is GrantState.ACTIVE and moment >= grant.expires_at:
                    await tx.put_budget_grant(grant.model_copy(update={"state": GrantState.EXPIRED}))
                    expired += 1
            # Quota slices lapse in the same pass rather than in a second leader
            # phase. They expire on the same clock for the same reason, and a
            # separate phase would be one more thing to forget to register --
            # which is exactly how the budget sweep spent its first life.
            for quota_grant in await tx.list_quota_grants():
                if quota_grant.state is GrantState.ACTIVE and moment >= quota_grant.expires_at:
                    await tx.put_quota_grant(quota_grant.model_copy(update={"state": GrantState.EXPIRED}))
                    expired += 1
        return expired

    async def roll_budget_period(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str,
        now: datetime | None = None,
    ) -> BudgetRecord:
        """Close the current budget period if it has elapsed, and persist the result.

        Settlement already rolls a stale period before charging, so this exists for
        the fleet that stops running work: a project that spends nothing for two
        months keeps reporting last quarter's ``spent`` until something settles,
        and an operator looking at that number cannot tell a stale window from a
        genuinely expensive one.

        Deliberately *not* a force-reset. Zeroing a period that has not elapsed
        would erase spend the limit is still supposed to bind, which is a way to
        silently disable a budget rather than to administer one.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        moment = now or utcnow()
        async with self.repository.transaction() as tx:
            await self._lock_admission_scope(tx, org_id, project_id)
            budget = await tx.get_budget(org_id, project_id)
            if budget is None:
                raise NotFoundError(f"budget for {org_id}/{project_id} was not found")
            rolled = self._rolled_budget(budget, moment)
            if rolled.period_started_at == budget.period_started_at:
                return budget
            rolled = rolled.model_copy(
                update={"revision": budget.revision + 1, "updated_by": principal.subject, "updated_at": moment}
            )
            await tx.put_budget(rolled)
            await self._audit_scoped(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                entity_type="budget",
                entity_id=_scope(org_id, project_id),
                action="budget.period_rolled",
                payload={
                    "period": rolled.period,
                    "closed_spend": budget.spent,
                    "period_started_at": rolled.period_started_at.isoformat()
                    if rolled.period_started_at is not None
                    else None,
                },
            )
            return rolled
