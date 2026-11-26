# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from uuid import uuid4

from eyvan.server.service._control_plane.shared import (
    _QUERY_AUDIT_STATEMENT_LIMIT,
    BudgetRecord,
    ConflictError,
    EffectiveQuota,
    IdempotencyRecord,
    NotFoundError,
    Permission,
    PolicyRecord,
    Principal,
    QuotaRecord,
    ReadOnlyQueryResult,
    ScopeQuota,
    _fingerprint,
    _scope,
    _validate_idempotency_key,
    datetime,
    utcnow,
)


class PoliciesMixin:
    """Transactional policies operations."""

    async def put_policy(
        self, principal: Principal, policy: PolicyRecord, *, idempotency_key: str
    ) -> tuple[PolicyRecord, bool]:
        self.authorizer.require(principal, Permission.MANAGE_PROJECT, org_id=policy.org_id, project_id=policy.project_id)
        key = _validate_idempotency_key(idempotency_key)
        scope = _scope(policy.org_id, policy.project_id)
        fingerprint = _fingerprint(
            policy.model_copy(update={"revision": 0, "updated_by": "", "updated_at": datetime.min})
        )
        async with self.repository.transaction() as tx:
            await self._lock_admission_scope(tx, policy.org_id, policy.project_id)
            await self._require_registered_scope(tx, policy.org_id, policy.project_id)
            existing_key = await self._idempotent_resource(tx, scope, "policy.put", key, fingerprint)
            if existing_key is not None:
                existing = await tx.get_policy(policy.org_id, policy.project_id)
                if existing is None:
                    raise ConflictError("idempotency record points to a missing policy")
                return existing, True
            previous = await tx.get_policy(policy.org_id, policy.project_id)
            stored = policy.model_copy(
                update={
                    "revision": 1 if previous is None else previous.revision + 1,
                    "updated_by": principal.subject,
                    "updated_at": utcnow(),
                }
            )
            await tx.put_policy(stored)
            await self._bind_idempotency(tx, scope, "policy.put", key, fingerprint, stored)
            await self._audit_scoped(
                tx,
                principal,
                org_id=stored.org_id,
                project_id=stored.project_id,
                entity_type="policy",
                entity_id=_scope(stored.org_id, stored.project_id),
                action="policy.updated",
                payload={"revision": stored.revision},
            )
            return stored, False

    async def get_policy(self, principal: Principal, *, org_id: str, project_id: str) -> PolicyRecord:
        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            policy = await tx.get_policy(org_id, project_id)
            if policy is None:
                raise NotFoundError(f"policy for {org_id}/{project_id} was not found")
            return policy

    async def list_policies(self, principal: Principal, *, org_id: str) -> list[PolicyRecord]:
        self.authorizer.require(principal, Permission.READ, org_id=org_id)
        async with self.repository.transaction() as tx:
            policies = await tx.list_policies(org_id=org_id)
            return [
                policy
                for policy in policies
                if self.authorizer.allows(principal, Permission.READ, org_id=policy.org_id, project_id=policy.project_id)
            ]

    async def put_budget(
        self, principal: Principal, budget: BudgetRecord, *, idempotency_key: str
    ) -> tuple[BudgetRecord, bool]:
        self.authorizer.require(principal, Permission.MANAGE_PROJECT, org_id=budget.org_id, project_id=budget.project_id)
        key = _validate_idempotency_key(idempotency_key)
        scope = _scope(budget.org_id, budget.project_id)
        fingerprint = _fingerprint(
            budget.model_copy(
                update={
                    "spent": 0,
                    "settlement_period_id": None,
                    "period_opening_spent": None,
                    "accelerator_hours_spent": {},
                    "revision": 0,
                    "updated_by": "",
                    "updated_at": datetime.min,
                }
            )
        )
        async with self.repository.transaction() as tx:
            await self._lock_admission_scope(tx, budget.org_id, budget.project_id)
            await self._require_registered_scope(tx, budget.org_id, budget.project_id)
            existing_key = await self._idempotent_resource(tx, scope, "budget.put", key, fingerprint)
            if existing_key is not None:
                existing = await tx.get_budget(budget.org_id, budget.project_id)
                if existing is None:
                    raise ConflictError("idempotency record points to a missing budget")
                return existing, True
            await tx.get_idempotency(scope, "budget.settle", scope)
            previous = await tx.get_budget(budget.org_id, budget.project_id)
            if previous is not None and (previous.spent or previous.accelerator_hours_spent):
                if budget.currency != previous.currency or budget.period != previous.period:
                    raise ConflictError("budget currency and period cannot change after usage has been settled")
            stored = budget.model_copy(
                update={
                    "spent": budget.spent if previous is None else previous.spent,
                    # Attribution is server-owned. Configuration requests cannot
                    # rewrite the opening ledger or forge a settlement epoch.
                    "settlement_period_id": uuid4().hex if previous is None else previous.settlement_period_id,
                    "period_opening_spent": budget.spent if previous is None else previous.period_opening_spent,
                    "accelerator_hours_spent": (
                        budget.accelerator_hours_spent if previous is None else previous.accelerator_hours_spent
                    ),
                    # Configuration edits do not restart the charged window or
                    # re-announce thresholds that already fired within it.
                    "period_started_at": budget.period_started_at if previous is None else previous.period_started_at,
                    "alerted_thresholds": (
                        budget.alerted_thresholds
                        if previous is None
                        else tuple(
                            threshold
                            for threshold in previous.alerted_thresholds
                            if threshold in budget.alert_thresholds
                        )
                    ),
                    "revision": 1 if previous is None else previous.revision + 1,
                    "updated_by": principal.subject,
                    "updated_at": utcnow(),
                }
            )
            await tx.put_budget(stored)
            await self._bind_idempotency(tx, scope, "budget.put", key, fingerprint, stored)
            await self._audit_scoped(
                tx,
                principal,
                org_id=stored.org_id,
                project_id=stored.project_id,
                entity_type="budget",
                entity_id=_scope(stored.org_id, stored.project_id),
                action="budget.updated",
                payload={
                    "revision": stored.revision,
                    "limit": stored.limit,
                    "spent": stored.spent,
                    "accelerator_hour_limits": stored.accelerator_hour_limits,
                    "accelerator_hours_spent": stored.accelerator_hours_spent,
                },
            )
            return stored, False

    async def put_quota(
        self, principal: Principal, quota: QuotaRecord, *, idempotency_key: str
    ) -> tuple[QuotaRecord, bool]:
        self._require_quota_admin(principal, quota.org_id, quota.project_id)
        key = _validate_idempotency_key(idempotency_key)
        scope = self._quota_id(quota.org_id, quota.project_id, quota.principal_id)
        fingerprint = _fingerprint(
            quota.model_copy(update={"revision": 0, "updated_by": "", "updated_at": datetime.min})
        )
        async with self.repository.transaction() as tx:
            await self._lock_admission_scope(tx, quota.org_id, quota.project_id)
            await self._require_registered_scope(tx, quota.org_id, quota.project_id)
            existing_key = await self._idempotent_resource(tx, scope, "quota.put", key, fingerprint)
            if existing_key is not None:
                existing = await tx.get_quota(quota.org_id, quota.project_id, quota.principal_id)
                if existing is None:
                    raise ConflictError("idempotency record points to a missing quota")
                return existing, True
            previous = await tx.get_quota(quota.org_id, quota.project_id, quota.principal_id)
            stored = quota.model_copy(
                update={
                    "revision": 1 if previous is None else previous.revision + 1,
                    "updated_by": principal.subject,
                    "updated_at": utcnow(),
                }
            )
            await tx.put_quota(stored)
            await self._bind_idempotency(tx, scope, "quota.put", key, fingerprint, stored)
            await self._audit_scoped(
                tx,
                principal,
                org_id=stored.org_id,
                project_id=stored.project_id,
                entity_type="quota",
                entity_id=self._quota_id(stored.org_id, stored.project_id, stored.principal_id),
                action="quota.updated",
                payload={"revision": stored.revision, "weight": stored.weight, "limits": stored.limits.model_dump()},
            )
            return stored, False

    async def get_quota(
        self, principal: Principal, *, org_id: str, project_id: str | None, principal_id: str | None
    ) -> EffectiveQuota:
        """Return every quota that applies to one scope, and whether any binds.

        Admission walks org, project and principal and enforces *every* quota it
        finds, treating a missing one as "no limit from that scope". This
        answers the same question the same way.

        It used to be an exact-row lookup that raised ``NotFoundError`` when the
        row was absent -- so asking a fresh fleet what its quota was returned a
        404, calling the ordinary unconfigured state an error, while the command
        that called it advertised hierarchy resolution it never did.
        """

        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            scopes: list[ScopeQuota] = [
                ScopeQuota(scope="org", org_id=org_id, quota=await tx.get_quota(org_id, None, None))
            ]
            if project_id is not None:
                scopes.append(
                    ScopeQuota(
                        scope="project",
                        org_id=org_id,
                        project_id=project_id,
                        quota=await tx.get_quota(org_id, project_id, None),
                    )
                )
            if project_id is not None and principal_id is not None:
                scopes.append(
                    ScopeQuota(
                        scope="principal",
                        org_id=org_id,
                        project_id=project_id,
                        principal_id=principal_id,
                        quota=await tx.get_quota(org_id, project_id, principal_id),
                    )
                )
        return EffectiveQuota(
            org_id=org_id,
            project_id=project_id,
            principal_id=principal_id,
            scopes=tuple(scopes),
        )

    async def run_readonly_query(
        self,
        principal: Principal,
        statement: str,
        *,
        limit: int,
        timeout_ms: int,
    ) -> ReadOnlyQueryResult:
        """Answer one read-only statement against this plane's own database.

        Gated on ``MANAGE_FLEET`` rather than ``OPERATE``. A statement reads
        whatever the schema holds, across every organization and project, and no
        row filter applied afterwards could make that safe for a per-project
        operator -- the scope of a raw read is the database, not the caller's
        grants, so only the role whose scope is also the whole fleet may have it.

        Audited before it runs and whether or not it succeeds. The interesting
        query during an incident is often the one that timed out or was refused,
        and a ledger that records only successes hides exactly the attempts a
        reviewer would want to see.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        async with self.repository.transaction() as tx:
            await self._audit_scoped(
                tx,
                principal,
                org_id=None,
                project_id=None,
                entity_type="control-plane",
                entity_id="readonly-query",
                action="query.executed",
                payload={"statement": statement[:_QUERY_AUDIT_STATEMENT_LIMIT], "limit": limit},
            )
        return await self.repository.readonly_query(statement, limit=limit, timeout_ms=timeout_ms)

    async def list_quotas(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str | None = None,
        principal_id: str | None = None,
    ) -> list[QuotaRecord]:
        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            quotas = await tx.list_quotas(org_id=org_id, project_id=project_id, principal_id=principal_id)
            return [
                quota
                for quota in quotas
                if self.authorizer.allows(principal, Permission.READ, org_id=quota.org_id, project_id=quota.project_id)
            ]

    async def delete_quota(
        self,
        principal: Principal,
        *,
        org_id: str,
        project_id: str | None,
        principal_id: str | None,
        idempotency_key: str,
    ) -> bool:
        self._require_quota_admin(principal, org_id, project_id)
        key = _validate_idempotency_key(idempotency_key)
        scope = self._quota_id(org_id, project_id, principal_id)
        fingerprint = _fingerprint({"org_id": org_id, "project_id": project_id, "principal_id": principal_id})
        async with self.repository.transaction() as tx:
            await self._lock_admission_scope(tx, org_id, project_id)
            if await self._idempotent_resource(tx, scope, "quota.delete", key, fingerprint) is not None:
                return True
            if not await tx.delete_quota(org_id, project_id, principal_id):
                raise NotFoundError("quota was not found")
            quota_id = self._quota_id(org_id, project_id, principal_id)
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope=scope,
                    operation="quota.delete",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="quota",
                    resource_id=quota_id,
                )
            )
            await self._audit_scoped(
                tx,
                principal,
                org_id=org_id,
                project_id=project_id,
                entity_type="quota",
                entity_id=quota_id,
                action="quota.deleted",
            )
            return False

    async def get_budget(self, principal: Principal, *, org_id: str, project_id: str) -> BudgetRecord:
        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        async with self.repository.transaction() as tx:
            budget = await tx.get_budget(org_id, project_id)
            if budget is None:
                raise NotFoundError(f"budget for {org_id}/{project_id} was not found")
            return budget
