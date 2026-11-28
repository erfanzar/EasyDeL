# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from eyvan.domain.governance import enabled_fleet_admins
from eyvan.server.service._control_plane.shared import (
    AuditEvent,
    AuthorizationError,
    ConflictError,
    IdempotencyRecord,
    InputError,
    NotFoundError,
    OrganizationRecord,
    Permission,
    Principal,
    ProjectRecord,
    RepositorySession,
    Role,
    RoleGrant,
    RoleGrantRecord,
    _fingerprint,
    _new_id,
    _scope,
    _validate_idempotency_key,
    datetime,
    utcnow,
)


class TenancyMixin:
    """Transactional tenancy operations."""

    # ----------------------------------------------------------------- tenancy

    def _require_org_administration(self, principal: Principal, org_id: str) -> None:
        """Require authority over one whole organization, not one project in it.

        ``MANAGE_PROJECT`` is the wrong gate here: a project admin holds it, and
        registering or disabling a *project* changes what its siblings can be
        named and whether the organization accepts work at all. The same shape
        as ``_require_quota_admin``'s unscoped branch, for the same reason.
        """

        if not any(
            grant.role == Role.FLEET_ADMIN or (grant.role == Role.ORG_ADMIN and grant.org_id == org_id)
            for grant in principal.grants
        ):
            raise AuthorizationError(f"{principal.subject!r} lacks administration of organization {org_id!r}")

    async def register_organization(
        self,
        principal: Principal,
        organization: OrganizationRecord,
        *,
        idempotency_key: str,
    ) -> tuple[OrganizationRecord, bool]:
        """Record a tenant this control plane serves.

        Fleet administration only. An organization admin can administer the
        organization it was granted, and creating a *new* one is how a tenant
        would otherwise mint itself scopes nobody allocated -- including scopes
        that then collect quota and budget rows.
        """

        self.authorizer.require(principal, Permission.MANAGE_FLEET)
        key = _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint(
            organization.model_copy(update={"created_at": datetime.min, "updated_at": datetime.min, "created_by": ""})
        )
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(
                tx, organization.id, "organization.register", key, fingerprint
            )
            if existing_key is not None:
                existing = await tx.get_organization(organization.id)
                if existing is None:
                    raise ConflictError("idempotency record points to a missing organization")
                return existing, True
            previous = await tx.get_organization(organization.id)
            stored = organization.model_copy(
                update={
                    "created_at": organization.created_at if previous is None else previous.created_at,
                    "created_by": principal.subject if previous is None else previous.created_by,
                    "updated_at": utcnow(),
                }
            )
            await tx.put_organization(stored)
            await self._bind_idempotency(tx, organization.id, "organization.register", key, fingerprint, stored)
            await self._audit_scoped(
                tx,
                principal,
                org_id=stored.id,
                project_id="",
                entity_type="organization",
                entity_id=stored.id,
                action="organization.registered" if previous is None else "organization.updated",
                payload={"owner": stored.owner, "enabled": stored.enabled},
            )
            return stored, False

    async def list_organizations(self, principal: Principal) -> list[OrganizationRecord]:
        """List tenants, narrowed to those the caller has any grant for.

        A viewer in one organization asking "which tenants exist" must not learn
        the customer list. Fleet administrators see everything, which is what
        makes the route usable for operating the plane.
        """

        async with self.repository.transaction() as tx:
            organizations = await tx.list_organizations()
        if self.authorizer.allows(principal, Permission.MANAGE_FLEET):
            return organizations
        visible = {grant.org_id for grant in principal.grants if grant.org_id is not None}
        return [organization for organization in organizations if organization.id in visible]

    async def get_organization(self, principal: Principal, org_id: str) -> OrganizationRecord:
        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=None)
        async with self.repository.transaction() as tx:
            organization = await tx.get_organization(org_id)
        if organization is None:
            raise NotFoundError(f"organization {org_id!r} is not registered")
        return organization

    async def register_project(
        self,
        principal: Principal,
        project: ProjectRecord,
        *,
        idempotency_key: str,
    ) -> tuple[ProjectRecord, bool]:
        """Record a project inside a registered organization.

        Refuses an unregistered organization rather than creating one
        implicitly: the whole point of the registry is that a scope exists on
        purpose, and auto-creating the parent from a typo in the child would
        reproduce exactly the problem it was built to remove.
        """

        self._require_org_administration(principal, project.org_id)
        key = _validate_idempotency_key(idempotency_key)
        scope = _scope(project.org_id, project.id)
        fingerprint = _fingerprint(
            project.model_copy(update={"created_at": datetime.min, "updated_at": datetime.min, "created_by": ""})
        )
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, scope, "project.register", key, fingerprint)
            if existing_key is not None:
                existing = await tx.get_project(project.org_id, project.id)
                if existing is None:
                    raise ConflictError("idempotency record points to a missing project")
                return existing, True
            organization = await tx.get_organization(project.org_id)
            if organization is None:
                raise ConflictError(
                    f"organization {project.org_id!r} is not registered; register it before its projects"
                )
            previous = await tx.get_project(project.org_id, project.id)
            stored = project.model_copy(
                update={
                    "created_at": project.created_at if previous is None else previous.created_at,
                    "created_by": principal.subject if previous is None else previous.created_by,
                    "updated_at": utcnow(),
                }
            )
            await tx.put_project(stored)
            await self._bind_idempotency(tx, scope, "project.register", key, fingerprint, stored)
            await self._audit_scoped(
                tx,
                principal,
                org_id=stored.org_id,
                project_id=stored.id,
                entity_type="project",
                entity_id=scope,
                action="project.registered" if previous is None else "project.updated",
                payload={"owner": stored.owner, "enabled": stored.enabled},
            )
            return stored, False

    async def list_projects(self, principal: Principal, *, org_id: str) -> list[ProjectRecord]:
        self.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=None)
        async with self.repository.transaction() as tx:
            return await tx.list_projects(org_id=org_id)

    async def _require_registered_scope(
        self,
        tx: RepositorySession,
        org_id: str,
        project_id: str | None,
    ) -> None:
        """Refuse a scope the registry contradicts.

        Scoped enforcement, not a global switch: an organization nobody
        registered is the pre-registry world and is left alone, so upgrading
        breaks no existing deployment. Registering one is the operator saying
        they are managing tenancy here -- from then on its projects must be real,
        which is what turns a typo in ``PUT /quotas`` from a silent success into
        an error naming the scope.
        """

        organization = await tx.get_organization(org_id)
        if organization is None:
            return
        if not organization.enabled:
            raise ConflictError(f"organization {org_id!r} is disabled and accepts no new work")
        if project_id is None or not project_id:
            return
        project = await tx.get_project(org_id, project_id)
        if project is None:
            raise ConflictError(
                f"project {project_id!r} is not registered in organization {org_id!r}; register it or correct the scope"
            )
        if not project.enabled:
            raise ConflictError(f"project {org_id}/{project_id} is disabled and accepts no new work")

    # ------------------------------------------------------------------ grants

    def _require_grant_administration(self, principal: Principal, grant: RoleGrant) -> None:
        """Require authority to issue the specific role being granted.

        Escalation is the whole hazard: whoever can issue grants can otherwise
        write themselves a better one. So the gate scales with what is being
        handed out rather than being a single "may administer grants"
        permission.

        * ``fleet_admin`` -- fleet administration, because it is unscoped and
          reaches every tenant.
        * ``org_admin`` -- administration of that organization. An org admin may
          appoint a peer in their own organization and nowhere else.
        * project roles -- ``MANAGE_PROJECT`` in that exact project, which is
          ordinary delegation: a project admin adding an operator to their team.
        """

        if grant.role is Role.FLEET_ADMIN:
            self.authorizer.require(principal, Permission.MANAGE_FLEET)
            return
        if grant.role is Role.ORG_ADMIN:
            self._require_org_administration(principal, grant.org_id or "")
            return
        self.authorizer.require(principal, Permission.MANAGE_PROJECT, org_id=grant.org_id, project_id=grant.project_id)

    async def issue_role_grant(
        self,
        principal: Principal,
        record: RoleGrantRecord,
        *,
        idempotency_key: str,
    ) -> tuple[RoleGrantRecord, bool]:
        """Grant a role durably, so authority does not depend on an IdP claim."""

        self._require_grant_administration(principal, record.grant)
        key = _validate_idempotency_key(idempotency_key)
        scope = ":".join(record.key)
        fingerprint = _fingerprint(record.model_copy(update={"granted_at": datetime.min, "granted_by": ""}))
        async with self.repository.transaction() as tx:
            existing_key = await self._idempotent_resource(tx, scope, "grant.issue", key, fingerprint)
            if existing_key is not None:
                existing = next(
                    (item for item in await tx.list_role_grants(subject=record.subject) if item.key == record.key),
                    None,
                )
                if existing is None:
                    raise ConflictError("idempotency record points to a missing role grant")
                return existing, True
            stored = record.model_copy(update={"granted_by": principal.subject, "granted_at": utcnow()})
            await tx.put_role_grant(stored)
            await tx.put_idempotency(
                IdempotencyRecord(
                    scope=scope,
                    operation="grant.issue",
                    key=key,
                    fingerprint=fingerprint,
                    resource_type="role_grant",
                    resource_id=scope,
                )
            )
            await tx.append_event(
                AuditEvent(
                    id=_new_id("evt"),
                    actor=principal.subject,
                    org_id=stored.grant.org_id,
                    project_id=stored.grant.project_id,
                    entity_type="role_grant",
                    entity_id=scope,
                    action="grant.issued",
                    payload={
                        "subject": stored.subject,
                        "role": stored.grant.role.value,
                        "expires_at": None if stored.expires_at is None else stored.expires_at.isoformat(),
                        "reason": stored.reason,
                    },
                )
            )
            return stored, False

    async def revoke_role_grant(self, principal: Principal, record_key: tuple[str, str, str, str]) -> bool:
        """Withdraw a durable grant.

        Revocation takes the same authority as issuance: someone who could grant
        a role should be able to take it back, and nobody else should be able to
        strip access they could not have given.
        """

        subject, role, org_id, project_id = record_key
        try:
            grant = RoleGrant(role=Role(role), org_id=org_id or None, project_id=project_id or None)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        self._require_grant_administration(principal, grant)
        async with self.repository.transaction() as tx:
            guard = getattr(self, "governance_enabled", False) and grant.role is Role.FLEET_ADMIN
            before = await self._enabled_fleet_admin_subjects(tx) if guard else set()
            removed = await tx.delete_role_grant(record_key)
            if removed and subject in before and not await self._enabled_fleet_admin_subjects(tx):
                # Rolled back with the transaction: the grant is not deleted.
                raise ConflictError(
                    f"{subject!r} is the last enabled fleet admin; make someone else an admin before revoking this"
                )
            if removed:
                await tx.append_event(
                    AuditEvent(
                        id=_new_id("evt"),
                        actor=principal.subject,
                        org_id=grant.org_id,
                        project_id=grant.project_id,
                        entity_type="role_grant",
                        entity_id=":".join(record_key),
                        action="grant.revoked",
                        payload={"subject": subject, "role": role},
                    )
                )
            return removed

    async def list_role_grants(
        self,
        principal: Principal,
        *,
        subject: str | None = None,
        org_id: str | None = None,
    ) -> list[RoleGrantRecord]:
        """List durable grants the caller is entitled to see.

        Narrowed the same way the organization list is: a grant names a person
        and what they can reach, so an unfiltered listing would be a directory
        of every tenant's staff.
        """

        async with self.repository.transaction() as tx:
            grants = await tx.list_role_grants(subject=subject)
        if self.authorizer.allows(principal, Permission.MANAGE_FLEET):
            return [item for item in grants if org_id is None or item.grant.org_id == org_id]
        visible = {
            grant.org_id
            for grant in principal.grants
            if grant.org_id is not None and grant.role in {Role.ORG_ADMIN, Role.PROJECT_ADMIN}
        }
        # PROJECT_ADMIN authority is project-granular: within an organization
        # where the caller administers only some projects, sibling projects'
        # grants are another tenant's staff directory. Org-level rows and rows
        # in the caller's own administered projects stay visible; an ORG_ADMIN
        # keeps org-wide visibility.
        administered_orgs = {
            grant.org_id for grant in principal.grants if grant.org_id is not None and grant.role is Role.ORG_ADMIN
        }
        administered_projects = {
            (grant.org_id, grant.project_id)
            for grant in principal.grants
            if grant.org_id is not None and grant.role is Role.PROJECT_ADMIN and grant.project_id is not None
        }
        return [
            item
            for item in grants
            if item.grant.org_id in visible
            and (org_id is None or item.grant.org_id == org_id)
            and (
                item.grant.org_id in administered_orgs
                or item.grant.project_id is None
                or (item.grant.org_id, item.grant.project_id) in administered_projects
            )
        ]

    @staticmethod
    async def _enabled_fleet_admin_subjects(tx: RepositorySession) -> set[str]:
        """Enabled team accounts holding an active ``fleet_admin`` grant, as this transaction sees them."""

        return enabled_fleet_admins(await tx.list_users(), await tx.list_role_grants(), utcnow())

    async def active_grants_for(self, subject: str, *, now: datetime | None = None) -> tuple[RoleGrant, ...]:
        """Return the unexpired durable grants held by one subject.

        Read on every authenticated request rather than cached: a revocation
        that takes effect on the next request is the behaviour an operator
        assumes, and a cache would make "I removed their access" true only
        eventually. It is one indexed read on an indexed column.
        """

        moment = now or utcnow()
        # The committed snapshot where the store offers one: a revocation still
        # counts from the request after it commits, but authentication no longer
        # queues behind every in-flight commit (over a second each on a large
        # local store, which stalled every request including assistant streams).
        snapshot = getattr(self.repository, "snapshot", None)
        view = snapshot() if snapshot is not None else self.repository.transaction()
        async with view as tx:
            grants = await tx.list_role_grants(subject=subject)
        return tuple(item.grant for item in grants if item.active_at(moment))
