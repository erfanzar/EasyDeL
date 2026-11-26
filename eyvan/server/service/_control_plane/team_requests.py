# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capacity and approval requests users file for the fleet admins.

A request is a message with a lifecycle, never an authority: approving one
records an admin's decision, and any capacity, quota or workload change still
goes through the ordinary audited routes. Anyone authenticated may file one for
themselves; only fleet admins (``MANAGE_FLEET``) see everyone's and resolve them.
"""

from __future__ import annotations

import secrets
import typing as tp

from eyvan.domain.governance import HELD_WORKLOAD_KEY
from eyvan.records import TeamRequestRecord
from eyvan.server.service._control_plane.shared import (
    AuditEvent,
    AuthorizationError,
    ConflictError,
    InputError,
    NotFoundError,
    Permission,
    Principal,
    _new_id,
    utcnow,
)

#: Upper bound on open requests one person may hold, so a looping client (or a
#: confused assistant) cannot flood the admins' queue.
MAX_OPEN_REQUESTS_PER_SUBJECT = 20
_RESOLUTIONS = ("approved", "declined")


def _event(actor: str, request: TeamRequestRecord, action: str, payload: dict[str, tp.Any]) -> AuditEvent:
    return AuditEvent(
        id=_new_id("evt"),
        actor=actor,
        entity_type="team_request",
        entity_id=request.id,
        action=action,
        org_id=request.org_id,
        project_id=request.project_id,
        payload=payload,
    )


class TeamRequestsMixin:
    """Transactional request operations."""

    def _is_fleet_admin(self, principal: Principal) -> bool:
        return bool(self.authorizer.allows(principal, Permission.MANAGE_FLEET))  # type: ignore[attr-defined]

    async def file_team_request(
        self,
        principal: Principal,
        *,
        kind: str,
        summary: str,
        details: dict[str, tp.Any] | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> TeamRequestRecord:
        """File a capacity or approval request as ``principal``."""

        if details and HELD_WORKLOAD_KEY in details:
            # Only admission links a request to a held workload; a user-set link
            # would let anyone's approval release a workload governance never held.
            raise InputError(f"{HELD_WORKLOAD_KEY!r} is set by the plane, not by a request")
        try:
            request = TeamRequestRecord(
                id=f"req_{secrets.token_hex(8)}",
                kind=kind,  # type: ignore[arg-type]
                subject=principal.subject,
                org_id=org_id,
                project_id=project_id,
                summary=summary.strip(),
                details=dict(details or {}),
            )
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            open_requests = await tx.list_team_requests(subject=principal.subject, state="open")
            if len(open_requests) >= MAX_OPEN_REQUESTS_PER_SUBJECT:
                raise ConflictError(
                    f"{principal.subject} already has {len(open_requests)} open requests; "
                    "withdraw some or wait for an admin"
                )
            await tx.put_team_request(request)
            await tx.append_event(_event(principal.subject, request, "team_request.filed", {"kind": request.kind}))
        return request

    async def list_team_requests(self, principal: Principal, *, state: str | None = None) -> list[TeamRequestRecord]:
        """Fleet admins see every request; everyone else sees only their own."""

        subject = None if self._is_fleet_admin(principal) else principal.subject
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            return await tx.list_team_requests(subject=subject, state=state)

    async def resolve_team_request(
        self, principal: Principal, request_id: str, *, resolution: str, note: str = ""
    ) -> TeamRequestRecord:
        """Approve or decline (fleet admins), or withdraw (the filer) an open request."""

        if resolution not in (*_RESOLUTIONS, "withdrawn"):
            raise InputError("resolution must be approved, declined or withdrawn")
        async with self.repository.transaction() as tx:  # type: ignore[attr-defined]
            request = await tx.get_team_request(request_id)
            if request is None:
                raise NotFoundError(f"request {request_id!r} not found")
            if resolution == "withdrawn":
                if request.subject != principal.subject:
                    raise AuthorizationError("only the person who filed a request may withdraw it")
            elif not self._is_fleet_admin(principal):
                if request.subject != principal.subject:
                    raise NotFoundError(f"request {request_id!r} not found")
                raise AuthorizationError(f"{principal.subject!r} may not resolve requests")
            if request.state != "open":
                raise ConflictError(f"request {request_id} is already {request.state}")
            resolved = request.model_copy(
                update={
                    "state": resolution,
                    "resolved_at": utcnow(),
                    "resolved_by": principal.subject,
                    "resolution_note": note.strip()[:2000],
                }
            )
            await tx.put_team_request(resolved)
            await tx.append_event(
                _event(principal.subject, resolved, f"team_request.{resolution}", {"note": resolved.resolution_note})
            )
            # Same transaction: an approval that releases its held workload, or a
            # decline/withdrawal that cancels it, commits together with the decision.
            await self._apply_approval_resolution(tx, principal, resolved, resolution)  # type: ignore[attr-defined]
            return resolved


__all__ = ("MAX_OPEN_REQUESTS_PER_SUBJECT", "TeamRequestsMixin")
