# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The single source of tenant isolation, exercised on purpose.

``Authorizer`` decides every cross-tenant question in this control plane, and no
test imported it. It was exercised incidentally -- a route test that happens to
pass a principal with the right grant proves the route calls *an* authorizer,
not that the matrix behind it denies what it should. The two failures look
identical from a passing suite and completely different in production.

So this is table-driven over every (role, permission, scope) triple rather than
a handful of cases. A matrix is exactly the shape where hand-picked examples
miss the one combination that matters, and the combination that matters here is
"a principal scoped to org A acting on org B".
"""

from __future__ import annotations

import itertools

import pytest

from eyvan.records import Permission, Principal, Role, RoleGrant
from eyvan.server.service.authz import Authorizer
from eyvan.server.service.errors import AuthorizationError

#: Every permission each role is expected to hold *within its own scope*. Written
#: out rather than imported from the implementation: a table that reads its
#: answers from the thing it is testing proves only that the code equals itself.
EXPECTED_WITHIN_SCOPE: dict[Role, frozenset[Permission]] = {
    Role.FLEET_ADMIN: frozenset(Permission),
    Role.ORG_ADMIN: frozenset(
        {
            Permission.READ,
            Permission.SUBMIT,
            Permission.OPERATE,
            Permission.MANAGE_PROJECT,
            Permission.MANAGE_OWN_CAPACITY,
        }
    ),
    Role.PROJECT_ADMIN: frozenset({Permission.READ, Permission.SUBMIT, Permission.OPERATE, Permission.MANAGE_PROJECT}),
    Role.PROJECT_OPERATOR: frozenset({Permission.READ, Permission.SUBMIT, Permission.OPERATE}),
    Role.PROJECT_SUBMITTER: frozenset({Permission.SUBMIT}),
    Role.PROJECT_VIEWER: frozenset({Permission.READ}),
}

_ORG_SCOPED = frozenset({Role.ORG_ADMIN})
_PROJECT_SCOPED = frozenset({Role.PROJECT_ADMIN, Role.PROJECT_OPERATOR, Role.PROJECT_SUBMITTER, Role.PROJECT_VIEWER})


def _principal(role: Role, *, org: str | None = "acme", project: str | None = "research") -> Principal:
    if role is Role.FLEET_ADMIN:
        return Principal(subject="p", grants=(RoleGrant(role=role),))
    if role in _ORG_SCOPED:
        return Principal(subject="p", grants=(RoleGrant(role=role, org_id=org),))
    return Principal(subject="p", grants=(RoleGrant(role=role, org_id=org, project_id=project),))


@pytest.mark.parametrize(
    ("role", "permission"),
    list(itertools.product(list(Role), list(Permission))),
    ids=lambda value: value.value if hasattr(value, "value") else str(value),
)
def test_a_role_holds_exactly_its_declared_permissions_in_its_own_scope(
    role: Role,
    permission: Permission,
) -> None:
    """Every cell of the matrix, in the scope the grant names."""

    authorizer = Authorizer()
    granted = authorizer.allows(
        _principal(role),
        permission,
        org_id="acme",
        project_id="research",
    )

    assert granted is (permission in EXPECTED_WITHIN_SCOPE[role]), (
        f"{role.value} should {'hold' if permission in EXPECTED_WITHIN_SCOPE[role] else 'not hold'} "
        f"{permission.value} within its own scope"
    )


@pytest.mark.parametrize(
    ("role", "permission"),
    list(itertools.product([role for role in Role if role is not Role.FLEET_ADMIN], list(Permission))),
    ids=lambda value: value.value if hasattr(value, "value") else str(value),
)
def test_no_role_reaches_another_organization(role: Role, permission: Permission) -> None:
    """The one property the whole design rests on.

    Every route in this package resolves ``org_id``/``project_id`` from the
    request and asks the authorizer. If a grant scoped to one organization ever
    answered yes for another, every one of those routes would leak at once --
    reads, cancels, quota writes, capacity placement.
    """

    assert (
        Authorizer().allows(
            _principal(role, org="acme"),
            permission,
            org_id="rival",
            project_id="research",
        )
        is False
    ), f"{role.value} scoped to acme was allowed {permission.value} on rival"


@pytest.mark.parametrize(
    ("role", "permission"),
    list(itertools.product(sorted(_PROJECT_SCOPED, key=lambda r: r.value), list(Permission))),
    ids=lambda value: value.value if hasattr(value, "value") else str(value),
)
def test_a_project_grant_does_not_reach_a_sibling_project(role: Role, permission: Permission) -> None:
    """Two teams in one organization is the ordinary case, not an exotic one."""

    assert (
        Authorizer().allows(
            _principal(role, org="acme", project="research"),
            permission,
            org_id="acme",
            project_id="platform",
        )
        is False
    ), f"{role.value} scoped to research was allowed {permission.value} on platform"


def test_an_org_admin_covers_every_project_in_its_own_organization() -> None:
    """Otherwise an org admin could administer nothing without a grant per project."""

    admin = _principal(Role.ORG_ADMIN, org="acme")

    for project in ("research", "platform", "a-project-created-tomorrow"):
        assert Authorizer().allows(admin, Permission.MANAGE_PROJECT, org_id="acme", project_id=project) is True


def test_an_org_admin_never_reaches_fleet_administration() -> None:
    """Fleet administration is the boundary between a tenant and the operator.

    ``MANAGE_FLEET`` gates cell deletion, force-finalize, and provider
    configuration -- things that affect every other tenant on the plane.
    """

    admin = _principal(Role.ORG_ADMIN, org="acme")

    assert Authorizer().allows(admin, Permission.MANAGE_FLEET, org_id="acme", project_id="research") is False
    assert Authorizer().allows(admin, Permission.MANAGE_FLEET) is False


def test_the_fleet_administration_boundary_survives_a_widened_role_table() -> None:
    """``ORG_ADMIN`` is denied ``MANAGE_FLEET`` twice, and the second one matters.

    The role table excludes it, and the evaluator refuses it again explicitly.
    That second check is unreachable while the table is correct -- which is
    exactly why it needs a test: without one it looks like dead code and gets
    deleted, and the boundary between a tenant and the fleet operator then rests
    on a single line in a dictionary.

    So this asserts the property survives the table being wrong. It patches the
    private table on purpose: the point is a single-layer failure, which no
    public call can produce.
    """

    from unittest import mock

    from eyvan.server.service import authz

    admin = _principal(Role.ORG_ADMIN, org="acme")
    widened = dict(authz._ROLE_PERMISSIONS)
    widened[Role.ORG_ADMIN] = frozenset(Permission)

    with mock.patch.object(authz, "_ROLE_PERMISSIONS", widened):
        assert authz.Authorizer().allows(admin, Permission.MANAGE_FLEET, org_id="acme") is False
        # The widening is real -- otherwise the assertion above would pass for
        # the wrong reason, which is how a defence-in-depth test becomes a lie.
        assert authz.Authorizer().allows(admin, Permission.OPERATE, org_id="acme") is True


def test_a_fleet_admin_needs_no_scope_and_is_not_confined_by_one() -> None:
    """A global role answering only for a scope it happens to be asked about would
    fail exactly where an operator needs it: an unscoped fleet-wide question."""

    admin = _principal(Role.FLEET_ADMIN)

    for permission in Permission:
        assert Authorizer().allows(admin, permission) is True
        assert Authorizer().allows(admin, permission, org_id="anyone", project_id="anything") is True


def test_an_unscoped_question_is_denied_to_every_scoped_role() -> None:
    """A route that forgot to pass a scope must fail closed.

    This is the failure mode that turns one missing keyword argument into a
    cross-tenant hole: with ``org_id=None`` there is nothing to compare against,
    so the only safe answer is no.
    """

    for role in Role:
        if role is Role.FLEET_ADMIN:
            continue
        for permission in Permission:
            assert Authorizer().allows(_principal(role), permission) is False, (
                f"{role.value} was allowed unscoped {permission.value}"
            )


def test_a_principal_with_no_grants_is_allowed_nothing() -> None:
    """An authenticated identity is not an authorized one."""

    anonymous = Principal(subject="nobody", grants=())

    for permission in Permission:
        assert Authorizer().allows(anonymous, permission, org_id="acme", project_id="research") is False


def test_grants_combine_rather_than_the_first_one_deciding() -> None:
    """A real principal holds several grants, and evaluation order is not a policy.

    An implementation that returned the first grant's verdict would make access
    depend on claim ordering in a token -- the same user allowed or denied
    depending on how an identity provider serialized a list.
    """

    principal = Principal(
        subject="p",
        grants=(
            RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),
            RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="platform"),
        ),
    )

    assert Authorizer().allows(principal, Permission.READ, org_id="acme", project_id="research") is True
    assert Authorizer().allows(principal, Permission.SUBMIT, org_id="acme", project_id="research") is False
    assert Authorizer().allows(principal, Permission.MANAGE_PROJECT, org_id="acme", project_id="platform") is True


def test_require_names_the_permission_and_the_scope_it_was_denied_for() -> None:
    """A denial an operator cannot act on produces a support ticket, not a fix."""

    with pytest.raises(AuthorizationError) as raised:
        Authorizer().require(
            _principal(Role.PROJECT_VIEWER),
            Permission.SUBMIT,
            org_id="acme",
            project_id="research",
        )

    message = str(raised.value)
    assert "submit" in message
    assert "acme/research" in message


def test_require_stays_silent_when_the_permission_is_held() -> None:
    """The allow path must not raise, or every authorized request would 403."""

    Authorizer().require(
        _principal(Role.PROJECT_OPERATOR),
        Permission.SUBMIT,
        org_id="acme",
        project_id="research",
    )


def test_an_org_admin_grant_cannot_be_narrowed_into_looking_project_scoped() -> None:
    """The same hazard as fleet admin, one level down, and it was live.

    Organization administration covers every project in the organization -- the
    evaluator never reads ``project_id`` for this role. A grant reading
    ``org_admin @ acme/research`` therefore conferred authority over
    ``acme/platform`` while naming only ``research``: a boundary that does not
    exist, shown to whoever reads the audit log or the grant list.
    """

    grant = RoleGrant(role=Role.ORG_ADMIN, org_id="acme", project_id="research")

    assert (grant.org_id, grant.project_id) == ("acme", None)
    admin = Principal(subject="p", grants=(grant,))
    assert Authorizer().allows(admin, Permission.MANAGE_PROJECT, org_id="acme", project_id="platform") is True


def test_a_fleet_admin_grant_cannot_be_narrowed_into_looking_scoped() -> None:
    """A stored grant must never appear narrower than the access it confers.

    ``RoleGrant`` drops the scope on a fleet-admin grant for this reason: a row
    reading ``fleet_admin @ acme`` in an audit log or an admin UI would suggest a
    boundary that does not exist.
    """

    grant = RoleGrant(role=Role.FLEET_ADMIN, org_id="acme", project_id="research")

    assert (grant.org_id, grant.project_id) == (None, None)
    assert Authorizer().allows(Principal(subject="p", grants=(grant,)), Permission.MANAGE_FLEET, org_id="rival") is True
