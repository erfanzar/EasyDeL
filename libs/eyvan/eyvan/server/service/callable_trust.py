# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Explicit project trust policy for serialized Python callables."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from eyvan.server.service.errors import AuthorizationError


@dataclass(frozen=True, slots=True)
class CallableProjectTrustPolicy:
    """Authorize callable deserialization for explicitly selected projects."""

    trusted_projects: frozenset[str] = frozenset()

    @classmethod
    def from_projects(cls, projects: tp.Collection[str]) -> CallableProjectTrustPolicy:
        """Build a policy from fully scoped ``org/project`` IDs or bare project IDs.

        Bare project IDs match that project in every organization, so they must
        be globally unique; prefer fully scoped IDs whenever they are not.
        """

        return cls(trusted_projects=frozenset(projects))

    def allows(self, org_id: str, project_id: str) -> bool:
        """Return whether a project may submit serialized callables.

        A bare (unscoped) trusted project ID matches in any organization; a
        scoped ``org/project`` entry matches only within that organization.
        """

        return project_id in self.trusted_projects or f"{org_id}/{project_id}" in self.trusted_projects

    def require(self, org_id: str, project_id: str) -> None:
        """Reject callable submission for a project outside this policy.

        The refusal names what *is* trusted. The identity in a submission comes
        from the caller's active context, not from the fleet they named, so the
        common way to arrive here is to pass ``init(fleet=...)`` while a context
        selecting some other org and project is active -- the fleet is
        overridden and the identity is inherited. Reporting only the rejected
        project leaves that invisible: the submitter sees a project they never
        typed and no way to tell which of the two halves is wrong.
        """

        if self.allows(org_id, project_id):
            return
        if self.trusted_projects:
            trusted = ", ".join(sorted(self.trusted_projects))
            remedy = f"this plane trusts {trusted}"
        else:
            remedy = (
                "this plane trusts no project for callables; set EYVAN_TRUSTED_CALLABLE_PROJECTS "
                "or run it as a loopback development plane"
            )
        raise AuthorizationError(
            f"callable workloads are not trusted for project {org_id}/{project_id}; {remedy}. "
            "Submissions take their org and project from the active Eyvan context, which "
            "'eyvan context' will show and 'eyvan use' will change"
        )


__all__ = ("CallableProjectTrustPolicy",)
