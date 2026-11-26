# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The managed-artifact lifecycle, from a terminal.

Everything under here existed server-side and could be reached by nothing but
hand-written HTTP: the index, the quota accounting, the reverse-reference
index, legal holds, and the collectability analysis the collector consults.

That combination is the worst one for storage. An operator who cannot see what
is collectable, what references what, or what is held cannot answer the only
question that matters when a bucket is filling -- *what is safe to delete* --
and the safe answer in the absence of evidence is "nothing", which is how
managed storage grows without bound while the machinery to bound it sits
unused.

Holds are deliberately here rather than under a destructive-operations group.
Placing one is how a human stops the collector from touching evidence, and it
has to be the easiest thing in this file to find.
"""

from __future__ import annotations

import click

from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession


def _emit(payload: object) -> None:
    JsonRenderer().emit(payload)


@click.group("artifacts")
def artifacts() -> None:
    """Inspect and hold managed artifacts."""


@artifacts.command("list")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--kind", help="Restrict to one artifact kind (callables, workspaces, results, logs).")
@click.pass_obj
def list_artifacts(session: CliSession, org_id: str | None, project_id: str | None, kind: str | None) -> None:
    """List the managed artifacts this project owns."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    params: dict[str, str] = {"org_id": scope.org_id, "project_id": scope.project_id}
    if kind is not None:
        params["kind"] = kind
    _emit(session.client.request("GET", "/api/v1/artifacts", params=params))


@artifacts.command("usage")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def artifact_usage(session: CliSession, org_id: str | None, project_id: str | None) -> None:
    """Report stored bytes against this project's artifact quota."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _emit(
        session.client.request(
            "GET", "/api/v1/artifacts/usage", params={"org_id": scope.org_id, "project_id": scope.project_id}
        ),
    )


@artifacts.command("inventory")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--cursor", help="Resume after the canonical managed URI returned as next_cursor.")
@click.option("--limit", type=click.IntRange(1, 1000), default=100, show_default=True)
@click.option(
    "--reindex-missing",
    is_flag=True,
    help="Add metadata only for physical-only objects observed in this page; never delete or rewrite.",
)
@click.option(
    "--acknowledge-scope",
    help="Required with --reindex-missing; repeat the exact ORG/PROJECT being changed.",
)
@click.pass_obj
def inventory(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    cursor: str | None,
    limit: int,
    reindex_missing: bool,
    acknowledge_scope: str | None,
) -> None:
    """Compare physical objects with the scoped index, dry-run by default."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    if not reindex_missing:
        if acknowledge_scope is not None:
            raise click.UsageError("--acknowledge-scope is accepted only with --reindex-missing")
        params: dict[str, str | int] = {
            "org_id": scope.org_id,
            "project_id": scope.project_id,
            "limit": limit,
        }
        if cursor is not None:
            params["cursor"] = cursor
        _emit(session.client.request("GET", "/api/v1/artifacts/inventory", params=params))
        return

    expected = f"{scope.org_id}/{scope.project_id}"
    if acknowledge_scope != expected:
        raise click.UsageError(
            f"--reindex-missing requires --acknowledge-scope {expected!r}; run the dry-run inventory first"
        )
    payload: dict[str, str | int | None] = {
        "org_id": scope.org_id,
        "project_id": scope.project_id,
        "cursor": cursor,
        "limit": limit,
        "acknowledge_scope": acknowledge_scope,
    }
    _emit(session.client.request("POST", "/api/v1/artifacts/inventory/reindex", json=payload))


@artifacts.command("collectable")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def collectable(session: CliSession, org_id: str | None, project_id: str | None) -> None:
    """List artifacts the collector would remove, and why each qualifies.

    Read this before enabling collection, not after. It is the only view that
    distinguishes "unreferenced and past its window" from "still referenced by a
    queued workload", and those look identical in a bucket listing.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _emit(
        session.client.request(
            "GET", "/api/v1/artifacts/collectable", params={"org_id": scope.org_id, "project_id": scope.project_id}
        )
    )


@artifacts.command("references")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--uri", required=True, help="The managed artifact URI to trace.")
@click.pass_obj
def references(session: CliSession, org_id: str | None, project_id: str | None, uri: str) -> None:
    """List the workloads that reference one artifact.

    The question behind "is this safe to delete", answered from the reverse
    index rather than by scanning every spec.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _emit(
        session.client.request(
            "GET",
            "/api/v1/artifacts/references",
            params={"org_id": scope.org_id, "project_id": scope.project_id, "uri": uri},
        )
    )


@artifacts.command("collect")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--apply/--dry-run", "apply", default=False, show_default="dry-run")
@click.option("--grace-period-s", type=float, default=None)
@click.option("--limit", type=int, default=500)
@click.pass_obj
def collect(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    apply: bool,
    grace_period_s: float | None,
    limit: int,
) -> None:
    """Run one bounded collection pass over managed artifacts.

    Dry-run is the default because deletion here is real: claim,
    version-conditional delete, finalize. Read `collectable` first; collect
    when the list says what you expect.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    params: dict[str, str | bool | int | float] = {
        "org_id": scope.org_id,
        "project_id": scope.project_id,
        "dry_run": not apply,
        "limit": limit,
    }
    if grace_period_s is not None:
        params["grace_period_s"] = grace_period_s
    _emit(session.client.request("POST", "/api/v1/artifacts/collect", params=params))


@artifacts.command("hold")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--uri", required=True)
@click.option("--reason", required=True, help="Why this artifact must survive collection.")
@click.option("--idempotency-key")
@click.pass_obj
def place_hold(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    uri: str,
    reason: str,
    idempotency_key: str | None,
) -> None:
    """Stop the collector from removing one artifact.

    A reason is required because a hold outlives the incident that motivated it,
    and an unexplained hold is indistinguishable from a forgotten one -- which is
    how holds accumulate until nothing is collectable at all.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _emit(
        session.client.request(
            "POST",
            "/api/v1/artifacts/holds",
            json={"org_id": scope.org_id, "project_id": scope.project_id, "uri": uri, "reason": reason},
            headers=({} if idempotency_key is None else {"Idempotency-Key": idempotency_key}),
        ),
    )


@artifacts.command("release-hold")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--uri", required=True)
@click.option("--idempotency-key")
@click.pass_obj
def release_hold(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    uri: str,
    idempotency_key: str | None,
) -> None:
    """Release a legal hold, making the artifact collectable again."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    _emit(
        session.client.request(
            "DELETE",
            "/api/v1/artifacts/holds",
            params={"org_id": scope.org_id, "project_id": scope.project_id, "uri": uri},
            headers=({} if idempotency_key is None else {"Idempotency-Key": idempotency_key}),
        ),
    )


def register_artifact_commands(main: click.Group) -> None:
    """Attach the artifact command group to the CLI root."""

    main.add_command(artifacts)


__all__ = ("artifacts", "register_artifact_commands")
