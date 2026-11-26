# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan-server bootstrap-admin``: the first fleet-admin invite, created on the plane host.

A network-bound team plane refuses development headers, so nobody can sign in
to create the first account over HTTP. Authority here is the ability to open
the plane's own store with the plane's own configuration (database credentials
or the owner-only local state file), which only the operator of the host has.

Postgres planes (what a network bind requires) are bootstrapped while running.
A local-document plane holds an exclusive lease, so it must be stopped first.
The command prints exactly one secret: the one-time invite link.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import typing as tp
from dataclasses import replace
from pathlib import Path

from eyvan._internal.platform import load_platform_config, refuse_removed_team_variable, resolve_team_mode
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.server.persistence.base import Repository
from eyvan.server.persistence.local import LocalControlPlaneInUseError, LocalJsonRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import ConflictError, InputError


class BootstrapError(RuntimeError):
    """The first admin could not be bootstrapped; the message says why and what to do."""


def _repository(settings: EyvanSettings) -> tuple[Repository, tp.Any]:
    """Open the same durable store the plane uses, or refuse with the reason."""

    if settings.repository_backend is RepositoryBackend.MEMORY:
        raise BootstrapError(
            "this plane keeps its state in memory (EYVAN_REPOSITORY_BACKEND=memory), so an invite written "
            "from here would never reach it; run bootstrap-admin with the same EYVAN_* settings as the server"
        )
    if settings.repository_backend is RepositoryBackend.LOCAL:
        return LocalJsonRepository(settings.local_state_path, lease_timeout_s=0), None
    from sqlalchemy.ext.asyncio import create_async_engine

    from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository

    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return SqlAlchemyRepository(engine), engine


async def bootstrap_admin(
    settings: EyvanSettings, *, email: str, username: str, public_url: str | None = None
) -> dict[str, str]:
    """Create the first fleet-admin invite and return ``{url, email, username, expires_at}``."""

    if not settings.team_mode:
        raise BootstrapError(
            "team mode is off for this plane; pass --team (as the server is started with `eyvan-server --team`) "
            "or set `team: true` in the platform file, and use the same EYVAN_* settings as the server, or the "
            "link would point at a plane that does not serve invites"
        )
    repository, engine = _repository(settings)
    try:
        try:
            await repository.open()
        except LocalControlPlaneInUseError as exc:
            raise BootstrapError(
                f"the local plane is running and owns {settings.local_state_path}; stop it, run bootstrap-admin, "
                "then start it again (a Postgres-backed plane can be bootstrapped while running)"
            ) from exc
        try:
            service = ControlPlaneService(repository)
            try:
                invite, raw = await service.bootstrap_admin_invite(email=email, subject=username)
            except (ConflictError, InputError) as exc:
                raise BootstrapError(str(exc)) from exc
        finally:
            await repository.close()
    finally:
        if engine is not None:
            await engine.dispose()
    base = (public_url or settings.public_url).rstrip("/")
    return {
        "url": f"{base}/#/accept-invite/{raw}",
        "email": invite.email,
        "username": invite.subject,
        "expires_at": invite.expires_at.isoformat(),
    }


def main(argv: tp.Sequence[str] | None = None) -> int:
    """Run ``eyvan-server bootstrap-admin``; returns a process exit status."""

    parser = argparse.ArgumentParser(
        prog="eyvan-server bootstrap-admin",
        description="Create the first fleet-admin invite on a team-mode plane that has no accounts yet.",
    )
    parser.add_argument("--username", required=True, help="User name for the admin account (e.g. erfan).")
    parser.add_argument(
        "--email",
        help="Label stored on the account; nobody is emailed. Defaults to USERNAME@localhost.invalid.",
    )
    parser.add_argument("--public-url", help="Base URL people open (defaults to EYVAN_PUBLIC_URL).")
    parser.add_argument(
        "--local-state-path",
        type=Path,
        help="State document of a local-backend plane (the server's own --local-state-path; "
        "default ~/.eyvan/control-plane.json).",
    )
    parser.add_argument(
        "--team",
        action="store_true",
        help="The plane runs `eyvan-server --team` (not needed when the platform file says `team: true`).",
    )
    parser.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    args = parser.parse_args(argv)
    username = args.username.strip().lower()
    email = args.email or f"{username}@localhost.invalid"
    try:
        refuse_removed_team_variable()
        settings = EyvanSettings.from_env()
        settings = resolve_team_mode(
            settings, flag=args.team, platform=load_platform_config(settings.platform_config_path)
        )
    except ValueError as exc:
        print(f"bootstrap-admin: {exc}", file=sys.stderr)
        return 1
    if args.local_state_path is not None:
        settings = replace(settings, local_state_path=args.local_state_path.expanduser())
    try:
        result = asyncio.run(bootstrap_admin(settings, email=email, username=username, public_url=args.public_url))
    except BootstrapError as exc:
        print(f"bootstrap-admin: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"first admin invite for {result['username']} (expires {result['expires_at']})")
        print("open this link once, privately, and choose a password:")
        print(f"  {result['url']}")
    return 0


__all__ = ("BootstrapError", "bootstrap_admin", "main")
