# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Open the hosted dashboard, with a separate source-only Vite launcher.

The installed wheel cannot contain the repository's Node source tree. The
ordinary ``eyvan dashboard`` command therefore has a wheel-safe contract: open
the dashboard hosted by the configured control plane. Developers who are
changing the frontend use ``eyvan dashboard-dev`` explicitly; that command owns
the source lookup, npm requirement, loopback credential, and Vite environment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import click

import eyvan
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.commands.http import EyvanApiError
from eyvan.sdk.commands.state import CliSession

#: Written by ``eyvan runtime init`` for a loopback control plane in
#: LOOPBACK_DEFAULT auth mode, which 401s every request without it.
DEFAULT_LOCAL_CREDENTIAL_PATH = Path("~/.eyvan/local-development-credential")


def _dashboard_root() -> Path:
    """Return the checked-in dashboard directory for the development command."""

    root = Path(eyvan.__file__).resolve().parent.parent / "dashboard"
    if not (root / "package.json").is_file():
        raise click.ClickException(
            f"the Eyvan dashboard sources are not present at {root}; "
            "'eyvan dashboard-dev' requires a source checkout, while "
            "'eyvan dashboard' opens the hosted production UI"
        )
    return root


def _local_credential(path: Path) -> str | None:
    expanded = path.expanduser()
    try:
        value = expanded.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


@click.command("dashboard")
@click.option("--open/--no-open", "open_browser", default=True, help="Open the hosted dashboard in a browser.")
@click.pass_obj
def dashboard(session: CliSession, open_browser: bool) -> None:
    """Open the dashboard hosted by the configured control plane."""

    dashboard_url = f"{session.base_url.rstrip('/')}/"
    # A loopback plane demands its owner-only development credential from every
    # caller, the dashboard it serves included -- and a browser cannot read the
    # 0600 file the value lives in. This terminal can, so it bridges the value
    # in the URL *fragment*: fragments are never sent to a server, never appear
    # in a `Referer`, and the page strips this one from its address bar on
    # arrival. The credential never leaves this machine: a remote plane gets a
    # plain URL, and no local credential is ever printed for one.
    endpoint = ControlPlaneEndpoint.parse(session.base_url)
    local_credential = session.identity.local_development_credential if endpoint.is_loopback else None
    if local_credential:
        # The fragment parameter name is a wire contract with the page that
        # reads it: ``LOCAL_CREDENTIAL_FRAGMENT_PARAM`` in
        # ``dashboard/src/auth/oidc.ts``. It is stated once per side and each
        # side names the other, the same discipline as the refusal detail.
        dashboard_url = f"{dashboard_url}#eyvan-local-credential={quote(local_credential)}"
    # Asked before a browser is opened, because a control plane installed from a
    # wheel may serve no UI at all: the dashboard is a Node build tree that
    # cannot travel inside a Python distribution. This command used to open that
    # URL regardless and hand the operator a bare 404, with nothing to say which
    # half was missing -- the plane, the route, or their own credentials.
    #
    # Only a 404 is treated as "no dashboard here". Any other answer, including a
    # transport failure, leaves the old behaviour alone: guessing that a plane
    # hosts nothing because one probe was unhappy would be its own wrong answer.
    #
    # ``request`` decodes JSON and reports failures by raising, so a status is
    # only ever visible on the raised error -- the probe used to read
    # ``.status_code`` off a decoded document and could never see the 404.
    try:
        session.client.request("GET", "/")
    except EyvanApiError as error:
        if error.status_code == 404:
            raise click.ClickException(
                f"the control plane at {session.base_url} serves its API but hosts no dashboard. "
                "A head installed from a wheel carries no UI bundle: re-run `eyvan head up` from a checkout "
                "that has built one (npm --prefix dashboard run build), or use the container deployment, "
                "which builds it. `eyvan dashboard-dev` runs the UI locally against this plane."
            ) from error
    except Exception:
        pass
    click.echo(f"Eyvan dashboard: {dashboard_url}")
    if open_browser:
        click.launch(dashboard_url)


@click.command("dashboard-dev")
@click.option("--port", default=4173, show_default=True, help="Port for the dashboard Vite development server.")
@click.option("--role", "role_override", help="Dashboard role; defaults to this session's role.")
@click.option(
    "--credential-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_LOCAL_CREDENTIAL_PATH,
    show_default=True,
    help="Loopback development credential the browser cannot read itself.",
)
@click.option("--open/--no-open", "open_browser", default=True, help="Open the development dashboard in a browser.")
@click.option("--print-env", is_flag=True, help="Print the resolved Vite environment and exit without starting.")
@click.pass_obj
def dashboard_dev(
    session: CliSession,
    port: int,
    role_override: str | None,
    credential_path: Path,
    open_browser: bool,
    print_env: bool,
) -> None:
    """Run the source-checkout-only Vite dashboard against this session."""

    root = _dashboard_root()
    endpoint = ControlPlaneEndpoint.parse(session.base_url)
    roles = role_override or ",".join(session.identity.dev_roles) or "fleet_admin"
    environment = {
        "EYVAN_DASHBOARD_API": session.base_url,
        "VITE_EYVAN_DEV_AUTH": "true",
        "VITE_EYVAN_DEV_ROLE": roles,
    }
    if session.identity.dev_user:
        environment["VITE_EYVAN_DEV_USER"] = session.identity.dev_user
    if session.org_id:
        environment["VITE_EYVAN_ORG"] = session.org_id
    if session.project_id:
        environment["VITE_EYVAN_PROJECT"] = session.project_id
    # The browser sends this credential through Vite's API proxy, so only a
    # loopback API target may receive this workstation's owner credential.
    credential = _local_credential(credential_path) if endpoint.is_loopback else None
    if credential is not None:
        environment["VITE_EYVAN_LOCAL_CREDENTIAL"] = credential

    if print_env:
        for name, value in sorted(environment.items()):
            # The credential is the one value that must not be echoed: this
            # output is for pasting into a shell, and shells keep history.
            shown = "<redacted>" if name == "VITE_EYVAN_LOCAL_CREDENTIAL" else value
            click.echo(f"{name}={shown}")
        return

    if credential is None and endpoint.is_loopback:
        click.echo(
            f"No development credential at {credential_path}; a loopback control plane will reject "
            "every dashboard request. Run 'eyvan runtime init' first.",
            err=True,
        )

    npm = shutil.which("npm")
    if npm is None:
        raise click.ClickException("npm is required to run the dashboard development server")

    click.echo(f"Eyvan dashboard development server -> {session.base_url} as {roles}")
    click.echo(f"  http://localhost:{port}/")
    if open_browser:
        click.launch(f"http://localhost:{port}/")
    process_environment = {**os.environ, **environment}
    if not endpoint.is_loopback:
        # An inherited credential is just as sensitive as one read above.
        process_environment.pop("VITE_EYVAN_LOCAL_CREDENTIAL", None)
    try:
        subprocess.run(
            [npm, "run", "dev", "--", "--port", str(port)],
            cwd=root,
            env=process_environment,
            check=True,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive exit
        click.echo("dashboard stopped")
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"the dashboard development server exited with {exc.returncode}") from exc


def register_dashboard_commands(group: click.Group) -> None:
    """Attach hosted and source-development dashboard commands."""

    group.add_command(dashboard)
    group.add_command(dashboard_dev)


__all__ = ("dashboard", "dashboard_dev", "register_dashboard_commands")
