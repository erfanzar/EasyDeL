# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`eyvan connect` -- point this machine at a control plane somebody else runs.

Eyvan's own setup path assumes the plane and the operator are the same person:
``runtime deploy`` supervises a local process, ``autoscale generate`` reads the
operator's gcloud, and the CLI falls back to loopback when nothing says
otherwise. That is right for one person on a laptop and wrong for the case this
command exists for -- a team where one CPU server runs Eyvan and everybody else
submits to it from their own checkout.

What a teammate had to type instead was ``eyvan --url http://host:8080 --token
... use SOME_FLEET``, which requires knowing a fleet name before ever having
seen the fleet, and repeats the URL on every later command until they do. So
this asks the plane what it has rather than requiring the answer up front.

**It writes one file and provisions nothing.** The saved context is the URL, the
scope, and the fleet chosen from what the plane reported; the credential is
never written here. A wrong address costs a re-run.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import click

from eyvan._internal.json_types import is_json_object
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.contexts import DEFAULT_ORG_ID, DEFAULT_PROJECT_ID
from eyvan.server.local.auth import LOCAL_DEVELOPMENT_CREDENTIAL_HEADER

#: What a control plane listens on when nobody says otherwise. Shared with the
#: server's own bind so `--port` is the exception rather than the ritual.
DEFAULT_CONTROL_PLANE_PORT = 8080


@dataclass(frozen=True, slots=True)
class ConnectTarget:
    """The plane this command resolved, before anything is saved."""

    url: str
    reachable: bool
    detail: str = ""
    fleets: tuple[str, ...] = ()
    pools: int = 0
    subject: str | None = None
    roles: tuple[str, ...] = ()


@dataclass
class ConnectOutcome:
    """What was found and what, if anything, was written."""

    target: ConnectTarget
    saved_context: str | None = None
    scope: tuple[str, str] = (DEFAULT_ORG_ID, DEFAULT_PROJECT_ID)
    notes: list[str] = field(default_factory=list)


def resolve_url(host: str, port: int | None, *, scheme: str | None = None) -> str:
    """Build the control-plane URL from what a person is likely to have.

    Accepts the three shapes people actually paste: a bare host, a host:port,
    and a full URL. Anything already carrying a scheme keeps it -- a team behind
    TLS types ``https://eyvan.internal`` and must not have it rewritten to http
    with a port bolted on.
    """

    value = host.strip()
    if not value:
        raise ValueError("a host is required")
    if "://" not in value:
        value = f"{scheme or 'http'}://{value}"
    split = urlsplit(value)
    if not split.hostname:
        raise ValueError(f"{host!r} does not name a host")
    # An explicit --port wins; otherwise keep whatever the address carried, and
    # only then fall back to the default. Defaulting first would silently
    # rewrite "host:9000" to port 8080.
    resolved_port = port or split.port or DEFAULT_CONTROL_PLANE_PORT
    hostname = f"[{split.hostname}]" if ":" in split.hostname else split.hostname
    netloc = f"{hostname}:{resolved_port}"
    return ControlPlaneEndpoint.parse(urlunsplit((split.scheme, netloc, split.path.rstrip("/"), "", ""))).url


def probe(
    url: str,
    *,
    token: str | None = None,
    headers: tp.Mapping[str, str] | None = None,
    org_id: str = DEFAULT_ORG_ID,
    project_id: str = DEFAULT_PROJECT_ID,
    timeout_s: float = 10.0,
) -> ConnectTarget:
    """Ask the plane what it is and what it holds, without changing anything.

    Authenticates the way every other command does. ``headers`` carries the
    identity the CLI already resolved -- a development plane is reached with
    identity headers and no bearer token at all, so probing with only a token
    reported "refused this identity" against a plane the very next command
    would have talked to fine.

    Every failure is reported as unreachable with its cause rather than raised:
    a wrong port, a plane that is down, and a credential the plane rejects are
    three different fixes, and a stack trace names none of them.
    """

    import httpx

    headers = {**(headers or {}), **({"Authorization": f"Bearer {token}"} if token else {})}
    base = url.rstrip("/")
    try:
        from eyvan._internal.plane_tls import control_plane_httpx_options
        from eyvan.sdk.team_credentials import plane_trust_httpx_options

        # Trust `eyvan login` stored for exactly this plane first (a private CA
        # or a pinned certificate), else a configured head CA, else defaults.
        verify = plane_trust_httpx_options(base) or control_plane_httpx_options()
        with httpx.Client(timeout=timeout_s, trust_env=False, **verify) as client:
            health = client.get(f"{base}/api/v1/health", headers=headers)
            if health.status_code != 200:
                return ConnectTarget(url, False, f"control plane answered {health.status_code} on /health")
            session = client.get(f"{base}/api/v1/session", headers=headers)
            if session.status_code in (401, 403):
                return ConnectTarget(
                    url,
                    False,
                    "the control plane is up but refused this identity; ask whoever runs it for a token "
                    "and pass --token",
                )
            # The route filters by ``fleet_name`` alone and ignores every
            # other query parameter, so the scope here is declarative -- the
            # probe names the org and project it is asking about -- rather
            # than something the route requires. An earlier comment claimed
            # the route mandated these and answered 422 without them; neither
            # was ever true of the server.
            pools = client.get(
                f"{base}/api/v1/capacity-pools",
                headers=headers,
                params={"org_id": org_id, "project_id": project_id},
            )
    except httpx.HTTPError as error:
        return ConnectTarget(url, False, f"could not reach {base}: {error}")

    # Fleets are read off the pools rather than from /fleets. A fleet with no
    # pool is not something anybody can submit to, and this command exists to
    # answer "what can I use here" -- so the list is the set of fleets that
    # actually own capacity.
    return ConnectTarget(
        url=base,
        reachable=True,
        fleets=_fleet_names(pools),
        pools=_count(pools),
        subject=_subject(session),
        roles=_roles(session),
    )


def _json(response: object) -> object:
    try:
        return response.json()  # type: ignore[attr-defined]
    except Exception:
        return None


def _fleet_names(response: object) -> tuple[str, ...]:
    """Return the distinct fleets owning the pools in this response."""

    payload = _json(response)
    if not isinstance(payload, list):
        return ()
    names = [item.get("fleet_name") for item in payload if is_json_object(item)]
    return tuple(sorted({name for name in names if isinstance(name, str) and name}))


def _count(response: object) -> int:
    payload = _json(response)
    return len(payload) if isinstance(payload, list) else 0


def _subject(response: object) -> str | None:
    payload = _json(response)
    subject = payload.get("subject") if is_json_object(payload) else None
    return subject if isinstance(subject, str) else None


def _roles(response: object) -> tuple[str, ...]:
    payload = _json(response)
    if not is_json_object(payload):
        return ()
    grants = payload.get("grants")
    if not isinstance(grants, list):
        return ()
    roles = [grant.get("role") for grant in grants if is_json_object(grant)]
    return tuple(sorted({role for role in roles if isinstance(role, str)}))


def choose_fleet(target: ConnectTarget, requested: str | None) -> tuple[str | None, str | None]:
    """Return the fleet to save, and a note when the choice was not obvious.

    A named fleet is honoured even when the plane does not report it: a
    teammate connecting before the fleet is applied is ordinary, and refusing
    would make the order of two people's work matter.
    """

    if requested:
        if target.fleets and requested not in target.fleets:
            return requested, f"{requested!r} is not applied on this plane yet; saved anyway"
        return requested, None
    if len(target.fleets) == 1:
        return target.fleets[0], None
    if not target.fleets:
        return None, "this plane has no fleet applied yet; run `eyvan connect --fleet NAME` once one exists"
    return None, "this plane has several fleets: " + ", ".join(target.fleets) + "; re-run with --fleet NAME"


def connect(
    session: CliSession,
    *,
    host: str,
    port: int | None,
    fleet: str | None,
    token: str | None,
    org_id: str | None,
    project_id: str | None,
    probe_plane: tp.Callable[..., ConnectTarget] = probe,
) -> ConnectOutcome:
    """Resolve a plane, and save it as the active context when one fleet is clear."""

    url = resolve_url(host, port)
    headers = session.identity.headers()
    # The root resolved its identity for the previous context, not this host.
    # Its owner credential is machine-local even when connect chooses a remote
    # plane; retain explicit bearer/development identity without that secret.
    if not ControlPlaneEndpoint.parse(url).is_loopback:
        headers.pop(LOCAL_DEVELOPMENT_CREDENTIAL_HEADER, None)
    target = probe_plane(
        url,
        token=token,
        headers=headers,
        org_id=org_id or session.org_id or DEFAULT_ORG_ID,
        project_id=project_id or session.project_id or DEFAULT_PROJECT_ID,
    )
    outcome = ConnectOutcome(target=target)
    if not target.reachable:
        return outcome

    chosen, note = choose_fleet(target, fleet)
    if note:
        outcome.notes.append(note)
    if chosen is None:
        return outcome

    # `ContextOverrides` is frozen, and rightly so -- it is the record of what
    # the invocation asked for. Replaced rather than mutated, so the saved
    # context takes the address this command resolved instead of whatever the
    # session defaulted to.
    from dataclasses import replace

    session.overrides = replace(session.overrides, base_url=url)
    context = session.select_context(chosen, org_id=org_id, project_id=project_id)
    outcome.saved_context = chosen
    outcome.scope = (context.org_id, context.project_id)
    return outcome


def render(outcome: ConnectOutcome) -> str:
    """Return what to tell somebody who just ran this."""

    target = outcome.target
    if not target.reachable:
        return f"could not connect to {target.url}: {target.detail}"

    lines = [f"connected to {target.url}"]
    if target.subject:
        roles = f" · {', '.join(target.roles)}" if target.roles else ""
        lines.append(f"  you are {target.subject}{roles}")
    lines.append(f"  {len(target.fleets)} fleet(s), {target.pools} capacity pool(s)")
    for note in outcome.notes:
        lines.append(f"  {note}")
    if outcome.saved_context:
        org, project = outcome.scope
        lines.append(f"  saved as the active context: {outcome.saved_context} ({org}/{project})")
        lines.append("")
        lines.append("  eyvan capacity list            what you can submit to")
        lines.append("  eyvan run --tpu SHAPE -- CMD   submit from this directory")
        lines.append(f"  open {target.url}/".ljust(33) + "the same dashboard everyone else sees")
    return "\n".join(lines)


@click.command("connect")
@click.option("--host", required=True, help="Control plane host, host:port, or full URL.")
@click.option(
    "--port",
    type=int,
    default=None,
    help=f"Port, when the host does not carry one  [default: {DEFAULT_CONTROL_PLANE_PORT}]",
)
@click.option("--fleet", help="Fleet to make active. Only needed when the plane has more than one.")
@click.option("--token", envvar="EYVAN_TOKEN", help="Bearer credential, when the plane requires one.")
@click.option("--org", "org_id", help="Organization to work in.")
@click.option("--project", "project_id", help="Project to work in.")
@click.option("--json", "as_json", is_flag=True, help="Emit the result as JSON.")
@click.pass_obj
def connect_command(
    session: CliSession,
    host: str,
    port: int | None,
    fleet: str | None,
    token: str | None,
    org_id: str | None,
    project_id: str | None,
    as_json: bool,
) -> None:
    """Point this machine at a control plane somebody else runs.

    For the team case Eyvan's own setup path does not cover: one CPU server runs
    the plane, everybody else submits to it from their own checkout. Saves the
    address so later commands need no flags, and reports what the plane holds so
    nobody has to know a fleet name before seeing the fleet.

    Reads only. The credential is never written to disk.
    """

    try:
        outcome = connect(
            session,
            host=host,
            port=port,
            fleet=fleet,
            token=token,
            org_id=org_id,
            project_id=project_id,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        JsonRenderer().emit(
            {
                "url": outcome.target.url,
                "reachable": outcome.target.reachable,
                "detail": outcome.target.detail or None,
                "fleets": list(outcome.target.fleets),
                "pools": outcome.target.pools,
                "subject": outcome.target.subject,
                "saved_context": outcome.saved_context,
                "notes": outcome.notes,
            }
        )
        if not outcome.target.reachable:
            raise SystemExit(1)
        return

    click.echo(render(outcome))
    if not outcome.target.reachable:
        raise SystemExit(1)


def register_connect_commands(group: click.Group) -> None:
    """Attach the connect command to the CLI root."""

    group.add_command(connect_command)


__all__ = (
    "DEFAULT_CONTROL_PLANE_PORT",
    "ConnectOutcome",
    "ConnectTarget",
    "choose_fleet",
    "connect",
    "connect_command",
    "probe",
    "register_connect_commands",
    "render",
    "resolve_url",
)
