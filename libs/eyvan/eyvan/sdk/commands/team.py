# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Team accounts from the CLI: ``eyvan login``/``logout``, ``eyvan users``, ``eyvan tokens``.

``eyvan login`` exchanges an email and password once for a named personal API
token, stores it owner-only (never in the context file), and every later
command and ``init()`` against that plane sends it. The password is never
stored and never accepted as a flag value, so it cannot land in shell history.
"""

from __future__ import annotations

import socket
import ssl
import sys
import typing as tp
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import click
import httpx

from eyvan._internal.json_types import is_json_object
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.connect import connect as connect_plane
from eyvan.sdk.commands.connect import resolve_url
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.team_credentials import (
    TRUST_CA,
    TRUST_PINNED,
    PlaneTrust,
    TeamCredential,
    TeamCredentialError,
    TeamCredentialStore,
    display_fingerprint,
    normalize_fingerprint,
    plane_trust_httpx_options,
)


def _object(payload: object) -> dict[str, tp.Any]:
    """Narrow one decoded API answer to a JSON object, or fail with a readable error."""

    if not isinstance(payload, dict):
        raise click.ClickException("the plane returned an unexpected answer")
    return payload


def _objects(value: object) -> list[dict[str, tp.Any]]:
    """Return the objects in a decoded JSON array."""

    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _http(url: str, trust: PlaneTrust | None = None) -> httpx.Client:
    """HTTP client for one plane: explicit trust, else trust stored by `eyvan login`, else the head CA."""

    from eyvan._internal.plane_tls import control_plane_httpx_options

    endpoint = ControlPlaneEndpoint.parse(url)
    verify: dict[str, tp.Any] = (
        {"verify": trust.ssl_context()}
        if trust is not None
        else (plane_trust_httpx_options(url) or control_plane_httpx_options())
    )
    return httpx.Client(base_url=url, timeout=20.0, trust_env=not endpoint.is_loopback, **verify)


def _host_port(url: str) -> tuple[str, int]:
    split = urlsplit(url)
    return split.hostname or "", split.port or (443 if split.scheme == "https" else 80)


def _verifies_with_default_trust(url: str) -> bool:
    """Return whether system trust (plus a configured head CA) accepts this plane's certificate."""

    from eyvan._internal.plane_tls import control_plane_ssl_context

    host, port = _host_port(url)
    context = control_plane_ssl_context() or ssl.create_default_context()
    try:
        with socket.create_connection((host, port), timeout=10) as raw, context.wrap_socket(raw, server_hostname=host):
            return True
    except ssl.SSLCertVerificationError:
        return False
    except OSError:
        # Unreachable, or another TLS failure: the request that follows reports
        # it with its real cause instead of this probe guessing.
        return True


def _resolve_trust(url: str, *, ca_file: str | None, fingerprint: str | None, can_prompt: bool) -> PlaneTrust | None:
    """Decide how to trust ``url`` before any password is sent to it.

    A private CA is taken as given. A certificate no trusted CA issued is
    pinned only after its SHA-256 fingerprint matched ``--fingerprint`` or a
    person confirmed it interactively; nothing is sent to an unconfirmed peer.
    """

    if ca_file is not None:
        try:
            return PlaneTrust.from_pem(TRUST_CA, Path(ca_file).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise click.ClickException(f"--ca-file {ca_file}: {exc}") from exc
    if not url.startswith("https://"):
        if fingerprint is not None:
            raise click.UsageError("--fingerprint only applies to an https:// plane")
        return None
    expected: str | None = None
    if fingerprint is not None:
        try:
            expected = normalize_fingerprint(fingerprint)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--fingerprint") from exc
    elif _verifies_with_default_trust(url):
        return None
    host, port = _host_port(url)
    try:
        trust = PlaneTrust.from_pem(TRUST_PINNED, ssl.get_server_certificate((host, port), timeout=10))
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"could not read the TLS certificate of {url}: {exc}") from exc
    shown = display_fingerprint(trust.fingerprint)
    if expected is not None:
        if trust.fingerprint != expected:
            raise click.ClickException(
                f"{url} presented certificate {shown}, which is not the --fingerprint you gave; "
                "refusing to send a password to it"
            )
        return trust
    message = (
        f"{url} uses a certificate that no trusted CA issued (self-signed or private).\n"
        f"  fingerprint {shown}\n"
        "  Compare it with the one your admin gave you (they can read it with "
        "`openssl x509 -in CERT -noout -fingerprint -sha256`)"
    )
    if not can_prompt:
        raise click.ClickException(f"{message};\n  then re-run with --fingerprint {shown} (or --ca-file FILE)")
    click.echo(message + ".", err=True)
    if not click.confirm("Trust this certificate for this plane?", default=False, err=True):
        raise click.ClickException("certificate not trusted; nothing was sent")
    return trust


def _granted_scope(me: tp.Any) -> tuple[str | None, str | None]:
    """Return the one org/project the account was granted, so later commands submit there."""

    grants = me.get("grants", []) if is_json_object(me) else []
    pairs = {
        (grant.get("org_id"), grant.get("project_id"))
        for grant in grants
        if is_json_object(grant) and grant.get("org_id") and grant.get("project_id")
    }
    if len(pairs) == 1:
        org, project = next(iter(pairs))
        return str(org), str(project)
    orgs = {grant.get("org_id") for grant in grants if is_json_object(grant) and grant.get("org_id")}
    if not pairs and len(orgs) == 1:
        return str(next(iter(orgs))), None
    return None, None


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300] or f"HTTP {response.status_code}"
    detail = payload.get("detail") if is_json_object(payload) else None
    return str(detail) if detail else f"HTTP {response.status_code}"


def _store() -> TeamCredentialStore:
    return TeamCredentialStore()


def _emit(as_json: bool, payload: tp.Any, text: str) -> None:
    if as_json:
        JsonRenderer().emit(payload)
    else:
        click.echo(text)


# ------------------------------------------------------------------- login


@click.command("login")
@click.argument("url")
@click.option(
    "--username",
    "--email",
    "email",
    help="Your user name or the email from your invite (prompted when omitted).",
)
@click.option("--password-stdin", is_flag=True, help="Read the password from stdin instead of prompting.")
@click.option("--fleet", help="Fleet to make active when the plane has more than one.")
@click.option(
    "--ca-file",
    type=click.Path(exists=True, dir_okay=False),
    help="PEM of the private CA that issued the plane's certificate.",
)
@click.option(
    "--fingerprint",
    help="SHA-256 fingerprint of a self-signed plane certificate to pin (ask your admin for it).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the result as JSON.")
@click.pass_obj
def login(
    session: CliSession,
    url: str,
    email: str | None,
    password_stdin: bool,
    fleet: str | None,
    ca_file: str | None,
    fingerprint: str | None,
    as_json: bool,
) -> None:
    """Sign in to a team Eyvan plane and remember a personal token for it.

    URL may be a host, host:port or full URL. Afterwards every command and
    init() on this machine targets that plane and is authenticated as you.
    """

    if ca_file is not None and fingerprint is not None:
        raise click.UsageError("pass --ca-file or --fingerprint, not both")
    try:
        base = resolve_url(url, None)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    endpoint = ControlPlaneEndpoint.parse(base)
    if not endpoint.is_loopback and endpoint.url.startswith("http://"):
        click.echo(
            "warning: this plane is plain HTTP; your password and token cross the network unencrypted. "
            "Ask the admin to enable TLS.",
            err=True,
        )
    can_prompt = not password_stdin and sys.stdin.isatty()
    trust = _resolve_trust(endpoint.url, ca_file=ca_file, fingerprint=fingerprint, can_prompt=can_prompt)
    me: tp.Any = None
    with _http(base, trust) as client:
        try:
            status_response = client.get("/api/v1/team/status")
        except httpx.HTTPError as exc:
            raise click.ClickException(f"could not reach {base}: {exc}") from exc
        if status_response.status_code != 200 or not status_response.json().get("team_mode"):
            raise click.ClickException(
                f"{base} does not run team accounts (start it with `eyvan-server --team` or set `team: true` "
                "in its platform config); use `eyvan connect` instead"
            )
        address = email or click.prompt("User name or email")
        if password_stdin:
            password = sys.stdin.readline().rstrip("\n")
        else:
            password = click.prompt("Password", hide_input=True)
        token_name = f"cli-{socket.gethostname()}"[:64] or "cli"
        response = client.post(
            "/api/v1/team/login", json={"email": address, "password": password, "token_name": token_name}
        )
        if response.status_code == 200:
            raw_token = response.json().get("token")
            mine = client.get("/api/v1/team/me", headers={"Authorization": f"Bearer {raw_token}"})
            me = mine.json() if mine.status_code == 200 else None
    if response.status_code != 200:
        raise click.ClickException(f"sign-in failed: {_detail(response)}")
    body = response.json()
    user = body.get("user") or {}
    credential = TeamCredential(
        url=endpoint.url,
        token=body["token"],
        token_id=body["record"]["id"],
        subject=user.get("subject", ""),
        email=user.get("email", address),
        trust=trust,
    )
    try:
        _store().put(credential)
    except (OSError, TeamCredentialError) as exc:
        raise click.ClickException(f"signed in, but could not save the token: {exc}") from exc
    org_id, project_id = _granted_scope(me)
    outcome = connect_plane(
        session,
        host=endpoint.url,
        port=None,
        fleet=fleet,
        token=credential.token,
        org_id=org_id,
        project_id=project_id,
    )
    notes = list(outcome.notes)
    saved_context = outcome.saved_context
    if saved_context is None:
        # No single fleet to pick: still make this plane the default, so later
        # commands need no --url; the fleet is inferred or named per command.
        saved_context, _ = session.save_plane_context(endpoint.url, org_id=org_id, project_id=project_id)
        notes = [note for note in notes if "eyvan connect" not in note and "re-run with --fleet" not in note]
        notes.append("later commands target this plane; pick a fleet with `eyvan use FLEET` when it has several")
    lines = [f"signed in to {endpoint.url} as {credential.subject} ({credential.email})"]
    lines.append(f"  token {credential.token_id} saved in {_store().path} (owner-only)")
    if trust is not None:
        lines.append(
            f"  trusting {'its CA' if trust.kind == TRUST_CA else 'pinned certificate'} "
            f"{display_fingerprint(trust.fingerprint)}"
        )
    lines.append(f"  active context: {saved_context} ({session.org_id}/{session.project_id})")
    lines.extend(f"  {note}" for note in notes)
    _emit(
        as_json,
        {
            "url": endpoint.url,
            "subject": credential.subject,
            "token_id": credential.token_id,
            "saved_context": saved_context,
            "scope": {"org_id": session.org_id, "project_id": session.project_id},
            "trust": None if trust is None else {"kind": trust.kind, "fingerprint": trust.fingerprint},
            "notes": notes,
        },
        "\n".join(lines),
    )


@click.command("logout")
@click.argument("url", required=False)
@click.pass_obj
def logout(session: CliSession, url: str | None) -> None:
    """Revoke this machine's token for a plane (default: the current one) and forget it."""

    base = ControlPlaneEndpoint.parse(resolve_url(url, None) if url else session.base_url).url
    try:
        credential = _store().get(base)
    except TeamCredentialError as exc:
        raise click.ClickException(str(exc)) from exc
    if credential is None:
        raise click.ClickException(f"not signed in to {base}")
    revoked = True
    try:
        with _http(base) as client:
            response = client.delete(
                f"/api/v1/team/tokens/{credential.token_id}",
                headers={"Authorization": f"Bearer {credential.token}"},
            )
        revoked = response.status_code in (200, 401, 404)
    except httpx.HTTPError:
        revoked = False
    _store().remove(base)
    if revoked:
        click.echo(f"signed out of {base}; token {credential.token_id} revoked")
    else:
        click.echo(
            f"forgot the local token for {base}, but could not reach the plane to revoke {credential.token_id}; "
            "revoke it with `eyvan tokens revoke` or from the dashboard",
            err=True,
        )


# ------------------------------------------------------------------- users


@click.group("users")
def users() -> None:
    """Team accounts on this plane (team mode)."""


@users.command("me")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def users_me(session: CliSession, as_json: bool) -> None:
    """Show who this CLI is signed in as."""

    payload = _object(session.client.request("GET", "/api/v1/team/me"))
    roles = ", ".join(str(grant.get("role")) for grant in _objects(payload.get("grants", [])))
    _emit(as_json, payload, f"{payload.get('subject')}  roles: {roles or 'none'}")


@users.command("list")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def users_list(session: CliSession, as_json: bool) -> None:
    """List every account (fleet admins)."""

    payload = session.client.request("GET", "/api/v1/team/users")
    rows = _objects(payload.get("users", [])) if isinstance(payload, dict) else []
    text = "\n".join(
        f"{row['subject']:<20} {row['email']:<32} {row.get('tier') or '-'!s:<8} "
        f"{'disabled' if row['disabled'] else 'active'}"
        for row in rows
    )
    _emit(as_json, payload, text or "no accounts yet; invite someone with `eyvan users invite`")


@users.command("invite")
@click.argument("email")
@click.option("--subject", help="User name (default: the part of the email before @).")
@click.option(
    "--role",
    type=click.Choice(
        ["fleet_admin", "org_admin", "project_admin", "project_operator", "project_submitter", "project_viewer"]
    ),
    help="Advanced: the exact role the account gets on acceptance (instead of --tier).",
)
@click.option(
    "--tier",
    type=click.Choice(["admin", "trusted", "user"]),
    help="admin (fleet admin), trusted (operator; may run @eyvan.execute) or user (submitter).",
)
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def users_invite(
    session: CliSession,
    email: str,
    subject: str | None,
    role: str | None,
    tier: str | None,
    org_id: str | None,
    project_id: str | None,
    as_json: bool,
) -> None:
    """Invite someone; prints a one-time link to send them privately."""

    if role is not None and tier is not None:
        raise click.UsageError("pass --tier or --role, not both")
    name = (subject or email.split("@", 1)[0]).strip().lower()
    body: dict[str, tp.Any] = {"email": email, "subject": name}
    if tier is not None:
        body["tier"] = tier
        if tier != "admin":
            scope = session.workload_scope(org_id=org_id, project_id=project_id)
            body["org_id"] = scope.org_id
            body["project_id"] = scope.project_id
    if role is not None:
        scope = session.workload_scope(org_id=org_id, project_id=project_id)
        body["role"] = role
        if role != "fleet_admin":
            body["org_id"] = scope.org_id
        if role.startswith("project_"):
            body["project_id"] = scope.project_id
    payload = _object(session.client.request("POST", "/api/v1/team/invites", json=body))
    _emit(
        as_json,
        payload,
        f"invited {email} as {name}\n  link (shown once, send it privately): {payload['url']}",
    )


@users.command("tier")
@click.argument("subject")
@click.argument("tier", type=click.Choice(["admin", "trusted", "user"]))
@click.option("--org", "org_id", help="Org of the project role (trusted/user; default: the active context).")
@click.option("--project", "project_id", help="Project of the project role (default: the active context).")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def users_tier(
    session: CliSession, subject: str, tier: str, org_id: str | None, project_id: str | None, as_json: bool
) -> None:
    """Make an account admin, trusted or user (fleet admins).

    admin: fleet admin everywhere. trusted: project operator, may run
    @eyvan.execute and is never held for approval. user: project submitter;
    runs @eyvan.execute only once a max TPU (`eyvan users rules --max-tpu`) is set.
    """

    body: dict[str, tp.Any] = {"tier": tier}
    if tier != "admin":
        scope = session.workload_scope(org_id=org_id, project_id=project_id)
        body["org_id"] = scope.org_id
        body["project_id"] = scope.project_id
    payload = _object(session.client.request("PUT", f"/api/v1/team/users/{subject}/tier", json=body))
    where = "" if tier == "admin" else f" in {body['org_id']}/{body['project_id']}"
    _emit(as_json, payload, f"{subject} is now {payload.get('tier', tier)}{where}")


@users.command("disable")
@click.argument("subject")
@click.pass_obj
def users_disable(session: CliSession, subject: str) -> None:
    """Disable an account and revoke all of its tokens."""

    session.client.request("POST", f"/api/v1/team/users/{subject}/disable")
    click.echo(f"disabled {subject}; all of its tokens are revoked")


@users.command("enable")
@click.argument("subject")
@click.pass_obj
def users_enable(session: CliSession, subject: str) -> None:
    """Re-enable an account (its old tokens stay revoked)."""

    session.client.request("POST", f"/api/v1/team/users/{subject}/enable")
    click.echo(f"enabled {subject}")


@users.command("invites")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def users_invites(session: CliSession, as_json: bool) -> None:
    """List invites (fleet admins)."""

    payload = session.client.request("GET", "/api/v1/team/invites")
    rows = _objects(payload.get("invites", [])) if isinstance(payload, dict) else []
    text = "\n".join(
        f"{row['id']}  {row['email']:<32} "
        f"{'accepted' if row['accepted_at'] else 'revoked' if row['revoked'] else 'open until ' + row['expires_at']}"
        for row in rows
    )
    _emit(as_json, payload, text or "no invites")


@users.command("revoke-invite")
@click.argument("invite_id")
@click.pass_obj
def users_revoke_invite(session: CliSession, invite_id: str) -> None:
    """Revoke an open invite."""

    session.client.request("DELETE", f"/api/v1/team/invites/{invite_id}")
    click.echo(f"revoked {invite_id}")


@users.command("rules")
@click.argument("subject")
@click.option("--fleet", "fleets", multiple=True, help="Allowed fleet; repeat for several. Omit for no fleet rule.")
@click.option("--approval-threshold", type=click.IntRange(min=0), help="Hold jobs above this many TPU units.")
@click.option("--guarantee", type=click.IntRange(min=0), help="Guaranteed TPU units; usage above it is borrowed.")
@click.option(
    "--max-tpu",
    "max_tpu",
    type=click.IntRange(min=0, max=1_000_000),
    help="Hard cap on TPU units in use or queued at once, across all projects. Omit for no cap.",
)
@click.option("--max-cpu", type=click.FloatRange(0, 1_000_000), help="Hard cap on CPU units (fractions allowed).")
@click.option("--max-gpu", type=click.IntRange(0, 1_000_000), help="Hard cap on GPU units in use or queued.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def users_rules(
    session: CliSession,
    subject: str,
    fleets: tuple[str, ...],
    approval_threshold: int | None,
    guarantee: int | None,
    max_tpu: int | None,
    max_cpu: float | None,
    max_gpu: int | None,
    as_json: bool,
) -> None:
    """Replace one account's governance rules (fleet admins); options left out are cleared."""

    payload = _object(
        session.client.request(
            "PUT",
            f"/api/v1/team/users/{subject}/rules",
            json={
                "allowed_fleets": list(fleets) if fleets else None,
                "approval_tpu_threshold": approval_threshold,
                "guaranteed_tpu_units": guarantee,
                "max_tpu_units": max_tpu,
                "max_cpu_units": max_cpu,
                "max_gpu_units": max_gpu,
            },
        )
    )
    rules = _object(payload.get("rules", {}))
    threshold = rules.get("approval_tpu_threshold")
    guaranteed = rules.get("guaranteed_tpu_units")
    caps = []
    for resource in ("tpu", "cpu", "gpu"):
        cap = rules.get(f"max_{resource}_units")
        caps.append(f"max {resource.upper()} {cap if cap is not None else 'no limit'}")
    allowed = rules.get("allowed_fleets")
    fleet_label = "any" if allowed is None else (", ".join(allowed) or "none")
    text = (
        f"{subject}: fleets {fleet_label}; "
        f"approval above {threshold if threshold is not None else 'never'}; "
        f"guarantee {guaranteed if guaranteed is not None else 'none'}; "
        f"{'; '.join(caps)}"
    )
    _emit(as_json, payload, text)


@users.command("reclaim-plan")
@click.argument("tpu_units", type=click.IntRange(min=1))
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def users_reclaim_plan(session: CliSession, tpu_units: int, as_json: bool) -> None:
    """Show what reclaiming TPU_UNITS of borrowed capacity for you would take (read-only)."""

    payload = _object(session.client.request("GET", "/api/v1/team/reclaim/plan", params={"tpu_units": tpu_units}))
    if payload.get("allowed"):
        victims = ", ".join(str(item) for item in payload.get("victims") or []) or "(hidden; ask a fleet admin)"
        text = f"allowed: frees {payload.get('freed_tpu_units')} TPU units from {victims}"
    else:
        text = f"not allowed: {payload.get('reason')}"
    _emit(as_json, payload, text)


@users.command("reclaim")
@click.argument("tpu_units", type=click.IntRange(min=1))
@click.option("--reason", default="", help="Why you need it; recorded in the audit trail.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def users_reclaim(session: CliSession, tpu_units: int, reason: str, as_json: bool) -> None:
    """Take back TPU_UNITS of borrowed capacity for you; each job is checkpointed, then requeued."""

    payload = _object(
        session.client.request(
            "POST",
            "/api/v1/team/reclaim",
            json={"tpu_units": tpu_units, "reason": reason},
            headers={"Idempotency-Key": f"reclaim-{uuid.uuid4().hex}"},
        )
    )
    _emit(
        as_json,
        payload,
        f"reclaiming {payload.get('freed_tpu_units')} TPU units from {payload.get('reclaimed')} job(s); "
        "each is checkpointed, stopped and requeued",
    )


@users.command("reclaim-workload")
@click.argument("workload_id")
@click.option("--reason", required=True, help="Why; recorded in the audit trail and the job's status.")
@click.pass_obj
def users_reclaim_workload(session: CliSession, workload_id: str, reason: str) -> None:
    """Checkpoint, stop and requeue one running workload (fleet admins)."""

    payload = _object(
        session.client.request(
            "POST",
            f"/api/v1/team/workloads/{workload_id}/reclaim",
            json={"reason": reason},
            headers={"Idempotency-Key": f"reclaim-{uuid.uuid4().hex}"},
        )
    )
    click.echo(f"{payload.get('id')} {payload.get('state')}; it requeues once its checkpoint and stop finish")


# ------------------------------------------------------------------ tokens


@click.group("tokens")
def tokens() -> None:
    """Your personal API tokens (team mode)."""


@tokens.command("list")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def tokens_list(session: CliSession, as_json: bool) -> None:
    payload = session.client.request("GET", "/api/v1/team/tokens")
    rows = _objects(payload.get("tokens", [])) if isinstance(payload, dict) else []
    text = "\n".join(
        f"{row['id']}  {row['name']:<24} {row['prefix']}…  "
        f"{'revoked' if row['revoked_at'] else 'last used ' + (row['last_used_at'] or 'never')}"
        for row in rows
    )
    _emit(as_json, payload, text or "no personal tokens")


@tokens.command("create")
@click.argument("name")
@click.option("--expires-in-days", type=click.IntRange(1, 3650))
@click.pass_obj
def tokens_create(session: CliSession, name: str, expires_in_days: int | None) -> None:
    """Mint a token for a script or CI; printed once."""

    body: dict[str, tp.Any] = {"name": name}
    if expires_in_days is not None:
        body["expires_in_days"] = expires_in_days
    payload = _object(session.client.request("POST", "/api/v1/team/tokens", json=body))
    click.echo(f"{payload['token']}\n(shown once; use it as EYVAN_TOKEN)", err=False)


@tokens.command("revoke")
@click.argument("token_id")
@click.pass_obj
def tokens_revoke(session: CliSession, token_id: str) -> None:
    session.client.request("DELETE", f"/api/v1/team/tokens/{token_id}")
    click.echo(f"revoked {token_id}")


def register_team_commands(group: click.Group) -> None:
    """Attach login/logout/users/tokens to the CLI root."""

    for command in (login, logout, users, tokens):
        group.add_command(command)


__all__ = ("register_team_commands",)
