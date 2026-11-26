# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`eyvan quota get` answers what applies, and says so when nothing does.

It used to 404. A fresh fleet has no quota row, so asking what quota applied
called the ordinary unconfigured state an error -- and the command's own help
promised the *effective* quota "not a stored row: the hierarchy resolves org,
project and principal scopes", which the exact-row lookup behind it never did.

Admission is the reference: it walks the whole chain and enforces every quota
it finds, treating a missing one as no limit from that scope. These tests hold
the read to the same rule, because a report that disagrees with the scheduler
teaches an operator a model of it that is wrong.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    EffectiveQuota,
    Principal,
    QuotaRecord,
    ResourceLimits,
    Role,
    RoleGrant,
    ScopeQuota,
)
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService

ORG = "acme"
PROJECT = "research"


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def _service(*quotas: QuotaRecord) -> ControlPlaneService:
    service = ControlPlaneService(InMemoryRepository())
    for index, quota in enumerate(quotas):
        await service.put_quota(_admin(), quota, idempotency_key=f"quota-{index}")
    return service


def _quota(*, project_id: str | None = None, cpu: float = 0.0, tpu: int = 0) -> QuotaRecord:
    return QuotaRecord(
        org_id=ORG,
        project_id=project_id,
        limits=ResourceLimits(cpu=cpu, tpu=tpu),
        updated_by="root",
    )


# --------------------------------------------------------------------------- nothing set


@pytest.mark.asyncio
async def test_an_unconfigured_scope_is_answered_rather_than_refused() -> None:
    """The state every fleet starts in. A 404 here calls "not configured" an error."""

    service = await _service()

    effective = await service.get_quota(_admin(), org_id=ORG, project_id=PROJECT, principal_id=None)

    assert effective.unlimited
    assert [entry.scope for entry in effective.scopes] == ["org", "project"]
    assert all(entry.quota is None for entry in effective.scopes)


@pytest.mark.asyncio
async def test_unlimited_is_stated_rather_than_left_to_be_inferred() -> None:
    """An empty list is also what a silently failed read looks like."""

    service = await _service()

    document = (await service.get_quota(_admin(), org_id=ORG, project_id=PROJECT, principal_id=None)).model_dump(
        mode="json"
    )

    assert EffectiveQuota.model_validate(document).unlimited is True


# --------------------------------------------------------------------------- the hierarchy


@pytest.mark.asyncio
async def test_an_org_quota_is_reported_when_the_project_declares_none() -> None:
    """The org limit applies to the project; an exact-row lookup would miss it.

    This is the whole reason the command promised hierarchy resolution: a
    project with no row of its own is not unlimited if its org has one.
    """

    service = await _service(_quota(cpu=64, tpu=16))

    effective = await service.get_quota(_admin(), org_id=ORG, project_id=PROJECT, principal_id=None)

    assert not effective.unlimited
    org = next(entry for entry in effective.scopes if entry.scope == "org")
    project = next(entry for entry in effective.scopes if entry.scope == "project")
    assert org.quota is not None and org.quota.limits.cpu == 64
    assert project.quota is None, "the project declares nothing; saying so is the point"


@pytest.mark.asyncio
async def test_both_scopes_are_reported_when_both_declare_a_limit() -> None:
    """Admission enforces every quota it finds, so every one of them is shown."""

    service = await _service(_quota(cpu=64), _quota(project_id=PROJECT, cpu=8))

    effective = await service.get_quota(_admin(), org_id=ORG, project_id=PROJECT, principal_id=None)

    limits = {entry.scope: (entry.quota.limits.cpu if entry.quota else None) for entry in effective.scopes}
    assert limits == {"org": 64, "project": 8}


@pytest.mark.asyncio
async def test_the_chain_stops_where_the_question_does() -> None:
    """Asking about an org must not invent a project scope to report on."""

    service = await _service(_quota(cpu=64))

    effective = await service.get_quota(_admin(), org_id=ORG, project_id=None, principal_id=None)

    assert [entry.scope for entry in effective.scopes] == ["org"]


@pytest.mark.asyncio
async def test_a_principal_scope_is_reported_when_one_is_asked_about() -> None:
    service = await _service()

    effective = await service.get_quota(_admin(), org_id=ORG, project_id=PROJECT, principal_id="alice")

    assert [entry.scope for entry in effective.scopes] == ["org", "project", "principal"]


def test_the_reported_chain_keeps_the_order_admission_walks() -> None:
    """Broadest first. A chain reported inside-out reads as the wrong precedence."""

    with pytest.raises(ValueError, match="ordered org, project, principal"):
        EffectiveQuota(
            org_id=ORG,
            scopes=(
                ScopeQuota(scope="project", org_id=ORG, project_id=PROJECT),
                ScopeQuota(scope="org", org_id=ORG),
            ),
        )


# --------------------------------------------------------------------------- writing one


def test_setting_a_quota_says_which_scope_it_lands_on(monkeypatch, capsys) -> None:
    """Omitting --project writes an org-wide quota, whatever the session says.

    That asymmetry is deliberate -- it is the only way to write an org quota
    from a session that has a project -- and it is invisible, which is what
    makes it dangerous: a quota written broader than intended lifts a limit for
    everybody rather than tightening it for one project.
    """

    from eyvan.sdk.commands import governance

    # Patched through monkeypatch so it is undone even if this fails, and only
    # the name governance resolves rather than click's own attribute -- writing
    # to that would leak a stub into every later test in the session.
    monkeypatch.setattr(governance, "stream_is_terminal", lambda _stream: True)

    governance._announce_quota_scope(ORG, None, None)
    governance._announce_quota_scope(ORG, PROJECT, None)
    governance._announce_quota_scope(ORG, PROJECT, "alice")

    written = capsys.readouterr().err.splitlines()
    assert "whole acme organization" in written[0]
    assert "--project to narrow it" in written[0]
    assert "project acme/research" in written[1]
    assert "principal alice" in written[2]
