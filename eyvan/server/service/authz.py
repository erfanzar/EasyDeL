# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Role-based authorization independent of HTTP authentication mechanics."""

from __future__ import annotations

from eyvan.records import Permission, Principal, Role
from eyvan.server.service.errors import AuthorizationError

_ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.FLEET_ADMIN: frozenset(Permission),
    Role.ORG_ADMIN: frozenset(
        {
            Permission.READ,
            Permission.SUBMIT,
            Permission.OPERATE,
            Permission.MANAGE_PROJECT,
            # Its own capacity, not the fleet's. Before this, every one of the
            # cell and pool mutations was MANAGE_FLEET-only, so a tenant could
            # not resize a pool it paid for without an operator holding rights
            # over every other tenant's capacity as well.
            Permission.MANAGE_OWN_CAPACITY,
        }
    ),
    # Deliberately not project roles. Capacity is an organization-level asset:
    # a pool serves every project in its org, so letting one project's admin
    # resize it would let them take capacity from a sibling project.
    Role.PROJECT_ADMIN: frozenset({Permission.READ, Permission.SUBMIT, Permission.OPERATE, Permission.MANAGE_PROJECT}),
    Role.PROJECT_OPERATOR: frozenset({Permission.READ, Permission.SUBMIT, Permission.OPERATE}),
    # Submission without read or lifecycle authority. This is intentionally a
    # separate kernel role rather than a dashboard-only label: automation can be
    # allowed to enqueue work without learning what anyone else ran or being able
    # to cancel it afterwards.
    Role.PROJECT_SUBMITTER: frozenset({Permission.SUBMIT}),
    Role.PROJECT_VIEWER: frozenset({Permission.READ}),
}


class Authorizer:
    """Evaluate explicit organization/project grants without implicit inheritance."""

    def allows(
        self,
        principal: Principal,
        permission: Permission,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> bool:
        for grant in principal.grants:
            if permission not in _ROLE_PERMISSIONS[grant.role]:
                continue
            if grant.role == Role.FLEET_ADMIN:
                return True
            if org_id is None or grant.org_id != org_id:
                continue
            if grant.role == Role.ORG_ADMIN:
                return permission != Permission.MANAGE_FLEET
            if grant.project_id is not None and grant.project_id == project_id:
                return True
        return False

    def permissions(
        self,
        principal: Principal,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> tuple[Permission, ...]:
        """Return the effective permissions at one scope in stable enum order.

        This is the same evaluator used by every enforcement point, exposed for
        authenticated authority introspection. Keeping introspection on this
        path prevents a dashboard role table from drifting away from the policy
        the API actually enforces.
        """

        return tuple(
            permission
            for permission in Permission
            if self.allows(principal, permission, org_id=org_id, project_id=project_id)
        )

    def require(
        self,
        principal: Principal,
        permission: Permission,
        *,
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """Raise when the principal lacks a scoped permission."""

        if not self.allows(principal, permission, org_id=org_id, project_id=project_id):
            raise AuthorizationError(
                f"{principal.subject!r} lacks {permission.value} permission"
                + (f" for {org_id}/{project_id}" if org_id is not None else "")
            )
