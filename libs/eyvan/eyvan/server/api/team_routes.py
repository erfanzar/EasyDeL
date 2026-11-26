# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Team-mode HTTP routes: sign-in, invites, personal API tokens, users.

Every route is registered on every plane so the route inventory and the API
reference stay stable, but each answers 404 unless the plane runs team mode
(``eyvan-server --team`` or ``team: true`` in the platform config file),
except ``/api/v1/team/status`` which tells the dashboard which mode it is in.

Secrets never travel in a URL path or query: uvicorn's access log records
paths, so invite tokens are POSTed and session tokens live in an HttpOnly cookie.
"""

from __future__ import annotations

import typing as tp

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from eyvan.config import EyvanSettings
from eyvan.identity.team_secrets import TEAM_SESSION_COOKIE, SignInThrottle
from eyvan.records import ApiTokenRecord, InviteRecord, Permission, Principal, Role, RoleGrant, TeamTier, UserRecord
from eyvan.server.api.auth import authenticated_principal, require_dashboard_origin, team_session_cookie_name

if tp.TYPE_CHECKING:
    from eyvan.server.service import ControlPlaneService

_Principal = tp.Annotated[Principal, Depends(authenticated_principal)]


def require_team_mode(request: Request) -> None:
    """Answer 404 for every team route unless this plane runs team mode."""

    if not getattr(request.app.state, "team_mode", False):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="team mode is not enabled")


#: Module level, not a closure: with postponed annotations FastAPI resolves
#: dependency annotations against module globals, and a local alias is invisible.
_On = tp.Annotated[None, Depends(require_team_mode)]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginBody(_Body):
    #: The invite's email address or the account's user name; the field keeps
    #: its original wire name so existing clients still work.
    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=1024)
    #: Set by `eyvan login`: return a named personal API token in the body
    #: instead of setting a browser session cookie.
    token_name: str | None = Field(default=None, min_length=1, max_length=64)


class InviteBody(_Body):
    email: str = Field(min_length=3, max_length=320)
    subject: str = Field(min_length=1, max_length=64)
    role: Role | None = None
    #: The simple alternative to ``role``: admin, trusted or user (not both).
    tier: TeamTier | None = None
    org_id: str | None = Field(default=None, max_length=128)
    project_id: str | None = Field(default=None, max_length=128)


class TierBody(_Body):
    tier: TeamTier
    org_id: str | None = Field(default=None, max_length=128)
    project_id: str | None = Field(default=None, max_length=128)


class InviteTokenBody(_Body):
    token: str = Field(min_length=8, max_length=128)


class AcceptBody(InviteTokenBody):
    password: str = Field(min_length=1, max_length=1024)
    display_name: str = Field(default="", max_length=128)


class TokenBody(_Body):
    name: str = Field(min_length=1, max_length=64)
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)


class RulesBody(_Body):
    """One account's governance rules: ``null`` clears a rule, an omitted key keeps the stored value.

    Omission is preserved so a client written before a rule existed (which never
    sends its key) cannot clear it by accident. The dashboard sends only edits;
    ``eyvan users rules`` sends every key, so its omitted options still clear.
    """

    allowed_fleets: list[str] | None = Field(default=None, max_length=64)
    approval_tpu_threshold: int | None = Field(default=None, ge=0, le=1_000_000)
    guaranteed_tpu_units: int | None = Field(default=None, ge=0, le=1_000_000)
    max_tpu_units: int | None = Field(default=None, ge=0, le=1_000_000)
    max_cpu_units: float | None = Field(default=None, ge=0, le=1_000_000, strict=True)
    max_gpu_units: int | None = Field(default=None, ge=0, le=1_000_000)


class ReclaimBody(_Body):
    tpu_units: int = Field(ge=1, le=1_000_000)
    reason: str = Field(default="", max_length=400)


class WorkloadReclaimBody(_Body):
    reason: str = Field(min_length=1, max_length=500)


def user_document(user: UserRecord, tier: TeamTier | None = None) -> dict[str, tp.Any]:
    """Public view of an account; never the password hash.

    ``tier`` is derived from grants, so a caller that has it passes it in; without
    it the flag alone decides between ``trusted`` and ``user``.
    """

    derived = tier if tier is not None else (TeamTier.TRUSTED if user.trusted else TeamTier.USER)
    return {
        "subject": user.subject,
        "email": user.email,
        "display_name": user.display_name,
        "disabled": user.disabled,
        "created_at": user.created_at.isoformat(),
        "created_by": user.created_by,
        "tier": derived.value,
        "trusted": user.trusted,
        "rules": {
            "allowed_fleets": None if user.allowed_fleets is None else list(user.allowed_fleets),
            "approval_tpu_threshold": user.approval_tpu_threshold,
            "guaranteed_tpu_units": user.guaranteed_tpu_units,
            "max_tpu_units": user.max_tpu_units,
            "max_cpu_units": user.max_cpu_units,
            "max_gpu_units": user.max_gpu_units,
        },
    }


def token_document(record: ApiTokenRecord) -> dict[str, tp.Any]:
    """Public view of a token; never its hash."""

    return {
        "id": record.id,
        "name": record.name,
        "prefix": record.prefix,
        "created_at": record.created_at.isoformat(),
        "expires_at": None if record.expires_at is None else record.expires_at.isoformat(),
        "last_used_at": None if record.last_used_at is None else record.last_used_at.isoformat(),
        "revoked_at": None if record.revoked_at is None else record.revoked_at.isoformat(),
    }


def invite_document(invite: InviteRecord) -> dict[str, tp.Any]:
    return {
        "id": invite.id,
        "email": invite.email,
        "subject": invite.subject,
        "roles": [grant.model_dump(mode="json") for grant in invite.grants],
        "created_by": invite.created_by,
        "created_at": invite.created_at.isoformat(),
        "expires_at": invite.expires_at.isoformat(),
        "accepted_at": None if invite.accepted_at is None else invite.accepted_at.isoformat(),
        "revoked": invite.revoked,
    }


def register_team_routes(app: FastAPI, *, settings: EyvanSettings, service: ControlPlaneService) -> None:
    """Register the team account routes; all but ``status`` answer 404 when team mode is off."""

    throttle = SignInThrottle()
    app.state.team_sign_in_throttle = throttle
    app.state.team_mode = settings.team_mode
    ttl = settings.dashboard_session_ttl_s

    def set_session(request: Request, response: Response, raw: str) -> None:
        name = team_session_cookie_name(settings, request)
        response.set_cookie(
            key=name,
            value=raw,
            max_age=ttl,
            httponly=True,
            secure=name == TEAM_SESSION_COOKIE,
            samesite="strict",
            path="/",
        )

    def bad_input(error: ValueError) -> HTTPException:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))

    async def document(user: UserRecord) -> dict[str, tp.Any]:
        tiers = await service.team_tiers((user.subject,))
        return user_document(user, tiers.get(user.subject))

    @app.get("/api/v1/team/status", tags=["team"])
    async def team_status() -> dict[str, tp.Any]:
        """Report whether this plane runs team accounts (unauthenticated)."""

        return {"team_mode": settings.team_mode}

    @app.post("/api/v1/team/login", tags=["team"])
    async def team_login(body: LoginBody, request: Request, response: Response, _on: _On) -> dict[str, tp.Any]:
        """Sign in with user name (or email) and password; sets an HttpOnly session cookie."""

        key = body.email.strip().lower()
        if throttle.blocked(key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="too many failed sign-ins; wait a minute and try again",
                headers={"Retry-After": "60"},
            )
        result = await service.password_login(key, body.password, session_ttl_s=ttl, token_name=body.token_name)
        if result is None:
            throttle.failed(key)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="wrong user name or password")
        throttle.succeeded(key)
        user, record, raw = result
        expires_at = record.expires_at.isoformat() if record.expires_at else None
        if body.token_name is not None:
            # A CLI credential is returned once, in the body; no cookie is set.
            return {
                "user": await document(user),
                "expires_at": expires_at,
                "token": raw,
                "record": token_document(record),
            }
        set_session(request, response, raw)
        return {"user": await document(user), "expires_at": expires_at}

    @app.post("/api/v1/team/logout", tags=["team"])
    async def team_logout(request: Request, response: Response, _on: _On) -> dict[str, tp.Any]:
        """End the dashboard's cookie session."""

        require_dashboard_origin(settings, request)
        name = team_session_cookie_name(settings, request)
        cookie = request.cookies.get(name)
        if cookie:
            await service.end_session(cookie)
        response.delete_cookie(key=name, path="/", httponly=True, secure=name == TEAM_SESSION_COOKIE, samesite="strict")
        return {"signed_out": True}

    @app.get("/api/v1/team/me", tags=["team"])
    async def team_me(principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        """Return who the caller is and the grants this plane resolved for them."""

        user = await service.get_team_user(principal.subject)
        fleet_admin = any(grant.role is Role.FLEET_ADMIN for grant in principal.grants)
        user_view = None if user is None else await document(user)
        if user_view is not None:
            tier = user_view["tier"]
        else:
            tier = TeamTier.ADMIN.value if fleet_admin else None
        return {
            "subject": principal.subject,
            "display_name": principal.display_name,
            "user": user_view,
            "tier": tier,
            "grants": [grant.model_dump(mode="json") for grant in principal.grants],
            "fleet_admin": fleet_admin,
        }

    @app.post("/api/v1/team/tokens", status_code=201, tags=["team"])
    async def create_team_token(body: TokenBody, principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        """Mint a personal API token; the raw value is returned only here."""

        record, raw = await service.create_api_token(principal, name=body.name, expires_in_days=body.expires_in_days)
        return {"token": raw, "record": token_document(record)}

    @app.get("/api/v1/team/tokens", tags=["team"])
    async def list_team_tokens(principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        return {"tokens": [token_document(item) for item in await service.list_api_tokens(principal)]}

    @app.delete("/api/v1/team/tokens/{token_id}", tags=["team"])
    async def revoke_team_token(token_id: str, principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        return token_document(await service.revoke_api_token(principal, token_id))

    @app.post("/api/v1/team/invites", status_code=201, tags=["team"])
    async def create_team_invite(body: InviteBody, principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        """Invite someone; the link is returned once and should be sent to them privately."""

        if body.role is not None and body.tier is not None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="pass tier or role, not both")
        grants: tuple[RoleGrant, ...] = ()
        trusted = False
        try:
            if body.role is not None:
                grants = (RoleGrant(role=body.role, org_id=body.org_id, project_id=body.project_id),)
            elif body.tier is TeamTier.ADMIN:
                grants = (RoleGrant(role=Role.FLEET_ADMIN),)
            elif body.tier is not None:
                if not body.org_id or not body.project_id:
                    raise ValueError(f"the {body.tier.value} tier needs an org_id and a project_id")
                role = Role.PROJECT_OPERATOR if body.tier is TeamTier.TRUSTED else Role.PROJECT_SUBMITTER
                grants = (RoleGrant(role=role, org_id=body.org_id, project_id=body.project_id),)
                trusted = body.tier is TeamTier.TRUSTED
        except ValueError as error:
            raise bad_input(error) from None
        invite, raw = await service.invite_user(
            principal, email=body.email, subject=body.subject, grants=grants, trusted=trusted
        )
        return {
            "invite": invite_document(invite),
            "url": f"{settings.public_url.rstrip('/')}/#/accept-invite/{raw}",
            "token": raw,
        }

    @app.get("/api/v1/team/invites", tags=["team"])
    async def list_team_invites(principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        return {"invites": [invite_document(item) for item in await service.list_invites(principal)]}

    @app.delete("/api/v1/team/invites/{invite_id}", tags=["team"])
    async def revoke_team_invite(invite_id: str, principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        return invite_document(await service.revoke_invite(principal, invite_id))

    @app.post("/api/v1/team/invites/preview", tags=["team"])
    async def preview_team_invite(body: InviteTokenBody, _on: _On) -> dict[str, tp.Any]:
        """Show who an invite is for (unauthenticated: the token is the authority)."""

        invite = await service.preview_invite(body.token)
        if invite is None:
            raise HTTPException(status_code=404, detail="this invite is invalid, expired or already used")
        return {"email": invite.email, "subject": invite.subject, "expires_at": invite.expires_at.isoformat()}

    @app.post("/api/v1/team/invites/accept", tags=["team"])
    async def accept_team_invite(body: AcceptBody, request: Request, response: Response, _on: _On) -> dict[str, tp.Any]:
        """Create the account from an invite, set its password, and sign in."""

        try:
            user, record, raw = await service.accept_invite(
                body.token, password=body.password, display_name=body.display_name, session_ttl_s=ttl
            )
        except ValueError as error:
            raise bad_input(error) from None
        set_session(request, response, raw)
        return {"user": await document(user), "expires_at": record.expires_at.isoformat() if record.expires_at else None}

    @app.get("/api/v1/team/users", tags=["team"])
    async def list_team_users(principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        users = await service.list_team_users(principal)
        tiers = await service.team_tiers()
        return {"users": [user_document(item, tiers.get(item.subject)) for item in users]}

    @app.post("/api/v1/team/users/{subject}/disable", tags=["team"])
    async def disable_team_user(subject: str, principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        return await document(await service.set_user_disabled(principal, subject, disabled=True))

    @app.post("/api/v1/team/users/{subject}/enable", tags=["team"])
    async def enable_team_user(subject: str, principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        return await document(await service.set_user_disabled(principal, subject, disabled=False))

    @app.put("/api/v1/team/users/{subject}/tier", tags=["team"])
    async def set_team_user_tier(subject: str, body: TierBody, principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        """Make an account admin, trusted or user: its grants and trusted flag change together (fleet admins)."""

        return await document(
            await service.set_user_tier(principal, subject, body.tier, org_id=body.org_id, project_id=body.project_id)
        )

    @app.put("/api/v1/team/users/{subject}/rules", tags=["team"])
    async def set_team_user_rules(subject: str, body: RulesBody, principal: _Principal, _on: _On) -> dict[str, tp.Any]:
        """Update the supplied rules, including hard CPU, GPU and TPU caps (fleet admins)."""

        sent = {name: getattr(body, name) for name in body.model_fields_set}
        return await document(await service.set_user_rules(principal, subject, **sent))

    @app.get("/api/v1/team/reclaim/plan", tags=["team"])
    async def plan_team_reclaim(
        principal: _Principal, _on: _On, tpu_units: tp.Annotated[int, Query(ge=1, le=1_000_000)]
    ) -> dict[str, tp.Any]:
        """What reclaiming ``tpu_units`` for the caller would take, by the deterministic borrow rule."""

        plan = await service.plan_reclaim_for(principal, tpu_units=tpu_units)
        document: dict[str, tp.Any] = {
            "allowed": plan.allowed,
            "reason": plan.reason,
            "freed_tpu_units": plan.freed_tpu_units,
            "requester_usage": plan.requester_usage,
            "requester_guarantee": plan.requester_guarantee,
        }
        # Which jobs would be taken, and who borrows how much, names other
        # people's work: only a fleet admin sees it. Everyone else gets the verdict.
        if service.authorizer.allows(principal, Permission.MANAGE_FLEET):
            document["victims"] = list(plan.victims)
            document["borrowed_by"] = dict(plan.borrowed_by or {})
        return document

    @app.post("/api/v1/team/reclaim", tags=["team"])
    async def reclaim_team_borrowed(
        body: ReclaimBody,
        principal: _Principal,
        _on: _On,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=160)],
    ) -> dict[str, tp.Any]:
        """Take back borrowed capacity for the caller; the rule picks victims, each is checkpointed first."""

        plan, issued = await service.reclaim_borrowed(
            principal, tpu_units=body.tpu_units, reason=body.reason, idempotency_key=idempotency_key
        )
        document: dict[str, tp.Any] = {
            "allowed": plan.allowed,
            "freed_tpu_units": plan.freed_tpu_units,
            "reclaimed": len(issued),
        }
        if service.authorizer.allows(principal, Permission.MANAGE_FLEET):
            document["victims"] = issued
        return document

    @app.post("/api/v1/team/workloads/{workload_id}/reclaim", tags=["team"])
    async def reclaim_team_workload(
        workload_id: str,
        body: WorkloadReclaimBody,
        principal: _Principal,
        _on: _On,
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)],
    ) -> dict[str, tp.Any]:
        """Checkpoint, stop and requeue one running workload (fleet admins)."""

        workload, replayed = await service.reclaim_workload(
            principal, workload_id, idempotency_key=idempotency_key, reason=body.reason
        )
        return {"id": workload.id, "state": workload.state.value, "replayed": replayed}


__all__ = ("TEAM_SESSION_COOKIE", "register_team_routes")
