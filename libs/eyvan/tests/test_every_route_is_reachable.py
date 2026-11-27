# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every HTTP route is reachable from something a user has, or says why not.

The dominant defect shape in this package is code that is built, tested, and
never called. A route with no client is the purest form of it: the handler
works, the tests pass, and the feature does not exist for anyone who is not
willing to write HTTP by hand.

A one-time audit does not fix that, because the next route repeats it. So this
enumerates the application's routes, scans ``eyvan/sdk`` for the paths it
requests, and fails on any route neither reached nor *classified*. Adding a
route without a client becomes a failing test rather than a quiet gap.

Classification is the honest part, and there are exactly two legitimate reasons
a route has no SDK caller:

* it is redeemed by a **capability token** from inside a running job, so the
  worker runtime reaches it and the SDK deliberately does not; or
* it is reached through a **dynamically built path** the scanner cannot see.

Both are listed with the caller that does reach them. "Nobody has got to it
yet" is not on that list -- it is what this test exists to surface.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app

#: Routes a *job* redeems with a capability token. The worker runtime reaches
#: each one; the SDK must not, because holding the credential is the point.
CAPABILITY_ROUTES: dict[str, str] = {
    "GET /api/v1/artifacts/callables/{}/content": "minted as a transfer URL by artifacts/transfers.py",
    "GET /api/v1/artifacts/workspaces/{}/content": "minted as a transfer URL by artifacts/transfers.py",
    "PUT /api/v1/artifacts/results/{}/content": "minted as a transfer URL by artifacts/transfers.py",
    "GET /api/v1/secrets/content": "minted per mount by controller/workload_commands.py",
    "POST /api/v1/workloads/{}/progress-reports": "minted as EYVAN_PROGRESS_URL at dispatch",
    "POST /api/v1/workloads/{}/children": "minted as EYVAN_CHILDREN_URL at dispatch",
    "PUT /api/v1/workloads/{}/endpoint": "registered by the running service against its dispatch URL",
}

#: Routes the *browser* reaches. The dashboard is a first-class client and not
#: part of the Python SDK, so a Python caller would be the wrong thing to add.
BROWSER_ROUTES: dict[str, str] = {
    "GET /api/v1/dashboard/config": "dashboard/src/auth/oidc.ts, so one image can serve two identity providers",
    "GET /api/v1/dashboard/auth/login": "dashboard/src/auth/oidc.ts beginsLogin navigates to the server-side OIDC flow",
    "GET /api/v1/dashboard/auth/callback": "the provider redirects the browser here; the server completes the exchange",
    "POST /api/v1/dashboard/auth/logout": "dashboard/src/auth/oidc.ts logout drops the HttpOnly session cookie",
    "GET /api/v1/session": "dashboard/src/api/client.ts resolves effective durable authority and project scopes",
    "POST /api/v1/attachments/plan": "dashboard/src/components/AddTpuModal.tsx previews local CLI attachment steps",
    "POST /api/v1/attachments": "dashboard/src/components/AddTpuModal.tsx starts a durable local attachment",
    "GET /api/v1/attachments": "dashboard/src/components/AddTpuModal.tsx recovers recent local jobs",
    "GET /api/v1/attachments/{}": "dashboard/src/components/AddTpuModal.tsx polls durable attachment progress",
    "GET /api/v1/attachments/discovered": "dashboard/src/pages/FleetPage.tsx lists every TPU in the GCP account",
    "GET /api/v1/fleets/{}/init-script": "dashboard/src/components/FleetInitScript.tsx shows a fleet's bash init script",
    "PUT /api/v1/fleets/{}/init-script": "dashboard/src/components/FleetInitScript.tsx saves a fleet's bash init script",
    "POST /api/v1/team/invites/preview": "dashboard accept-invite screen shows who an invite is for",
    "POST /api/v1/team/invites/accept": "dashboard accept-invite screen sets the password and signs in",
    "POST /api/v1/team/logout": "dashboard sign-out drops the HttpOnly team session cookie",
    "POST /api/v1/assistant/oauth/openrouter/start": (
        "dashboard Settings 'Sign in with OpenRouter' sends the browser to OpenRouter's consent page"
    ),
}

#: Routes the SDK does reach, by a path this scanner cannot resolve statically.
DYNAMIC_ROUTES: dict[str, str] = {
    "POST /api/v1/workloads/{}/cancel": "built from _WORKLOAD_ACTIONS in sdk/commands/workloads.py",
    "POST /api/v1/workloads/{}/checkpoint": "built from _WORKLOAD_ACTIONS in sdk/commands/workloads.py",
    "POST /api/v1/workloads/{}/retry": "built from _WORKLOAD_ACTIONS in sdk/commands/workloads.py",
}

#: Routes with no client and a stated reason. Each entry is a decision, and the
#: bar for adding one is that a client would be *wrong*, not merely absent.
DEFERRED_ROUTES: dict[str, str] = {
    "POST /api/v1/assistant/chat": (
        "the one-shot JSON turn kept for compatibility; `eyvan ask` uses the streaming route so the "
        "user sees each step, and a second CLI path for the same turn would only drift"
    ),
    "GET /api/v1/assistant/oauth/openrouter/callback/{}": (
        "OpenRouter's OAuth redirect target; only OpenRouter sends the admin's browser here with the "
        "single-use state, so any first-party client calling it would be wrong"
    ),
    "GET /api/v1/workloads/{}/logs/stream": (
        "server-sent events; a CLI follower needs streaming request support the shared "
        "JSON client does not have, and half-implementing it would buffer the stream and "
        "defeat the point"
    ),
    "GET /api/v1/fleet/events": (
        "unscoped audit trail; the scoped `eyvan events` covers every tenant-facing use "
        "and this one exists for a fleet admin debugging cross-tenant ordering"
    ),
    "POST /api/v1/groups": (
        "heterogeneous group submission takes a member matrix no flag set expresses well; "
        "it belongs with the declarative submit-dag file format rather than as options"
    ),
    "POST /api/v1/secrets/rotate": (
        "break-glass rotation of the capability-signing credential; an operator drives it "
        "rarely, from the plane they are standing on, and the response is a ledger row whose "
        "fence must be kept by whoever may resolve a later inconclusive rotation"
    ),
    "POST /api/v1/capacity-pools": (
        "capacity pools are declared as part of a fleet document via PUT /api/v1/fleets/{fleet} "
        "(eyvan fleet apply / eyvan up); an imperative single-pool CLI upsert would bypass fleet ownership"
    ),
    "POST /api/v1/cells": (
        "cells are provisioned and registered by capacity drivers and cell agents during reconciliation; "
        "hand-creating cells from the CLI would bypass driver ownership, observation, and lease fencing"
    ),
    "GET /api/v1/mcp": (
        "MCP Streamable HTTP transport SSE stream spoken by external MCP clients configured via "
        "`eyvan mcp` (sdk/commands/mcp.py), not by the Eyvan Python HTTP client"
    ),
    "POST /api/v1/mcp": (
        "MCP Streamable HTTP JSON-RPC endpoint spoken by external MCP clients configured via "
        "`eyvan mcp` (sdk/commands/mcp.py), not by the Eyvan Python HTTP client"
    ),
    "DELETE /api/v1/mcp": (
        "MCP Streamable HTTP session termination endpoint spoken by external MCP clients configured via "
        "`eyvan mcp` (sdk/commands/mcp.py), not by the Eyvan Python HTTP client"
    ),
    "GET /api/v1/governance/summary": (
        "admin console governance overview route consumed by the Eyvan web dashboard UsersPage; "
        "CLI governance uses /api/v1/team/users and /api/v1/team/reclaim/plan"
    ),
}


def _routes() -> set[tuple[str, str]]:
    """Every API route the application serves, with parameters normalized."""

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        app = create_app(
            settings=EyvanSettings(dev_auth_enabled=True, artifact_token_key="k" * 32, artifact_store_uri=directory)
        )
    found: set[tuple[str, str]] = set()
    for route in app.routes:
        methods = getattr(route, "methods", None)
        if not methods or not route.path.startswith("/api/v1"):
            continue
        for method in sorted(methods - {"HEAD", "OPTIONS"}):
            found.add((method, re.sub(r"\{[^}]*\}", "{}", route.path)))
    return found


_HTTP_VERBS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})
_METHOD_FIRST_CALLS = frozenset(
    {"request", "request_checked", "mutate", "request_text", "request_bytes", "stream_lines", "_mutating_request"}
)


def _normalize_api_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        candidate = node.value
    elif isinstance(node, ast.JoinedStr):
        candidate = "".join(
            value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else "{}"
            for value in node.values
        )
    else:
        return None
    if candidate.startswith("/api/v1"):
        return re.sub(r"\{[^}]*\}", "{}", candidate)
    return None


def _resolve_api_path(node: ast.AST, bindings: dict[str, str]) -> str | None:
    direct = _normalize_api_string(node)
    if direct is not None:
        return direct
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    return None


def _resolve_http_verb(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        verb = node.value.upper()
        if verb in _HTTP_VERBS:
            return verb
    return None


def _collect_string_bindings(tree: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            normalized = _normalize_api_string(node.value)
            if normalized is not None:
                bindings[node.targets[0].id] = normalized
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            normalized = _normalize_api_string(node.value)
            if normalized is not None:
                bindings[node.target.id] = normalized
    return bindings


def _requested_routes(root: pathlib.Path) -> set[tuple[str, str]]:
    """Every ``(METHOD, /api/v1/...)`` request spelled under ``root``."""

    found: set[tuple[str, str]] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bindings = _collect_string_bindings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}

            if func_name.upper() in _HTTP_VERBS and node.args:
                resolved = _resolve_api_path(node.args[0], bindings)
                if resolved is not None:
                    found.add((func_name.upper(), resolved))
                continue

            if func_name == "read" and node.args:
                resolved = _resolve_api_path(node.args[0], bindings)
                if resolved is not None:
                    found.add(("GET", resolved))
                continue

            if func_name == "_put" and len(node.args) >= 2:
                resolved = _resolve_api_path(node.args[1], bindings)
                if resolved is not None:
                    found.add(("PUT", resolved))
                continue

            if func_name == "ResourceSpec":
                for key in ("list_path", "get_path"):
                    if key in keywords:
                        resolved = _resolve_api_path(keywords[key], bindings)
                        if resolved is not None:
                            found.add(("GET", resolved))
                continue

            if func_name in _METHOD_FIRST_CALLS or len(node.args) >= 2 or ("method" in keywords and "path" in keywords):
                method_node = node.args[0] if len(node.args) >= 1 else keywords.get("method")
                path_node = node.args[1] if len(node.args) >= 2 else keywords.get("path")
                verb = _resolve_http_verb(method_node) if method_node is not None else None
                resolved = _resolve_api_path(path_node, bindings) if path_node is not None else None
                if verb is not None and resolved is not None:
                    found.add((verb, resolved))
    return found


def _package_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "eyvan"


def test_every_route_is_reachable_or_classified() -> None:
    """The guard. A route with no client and no reason fails here.

    This is the whole value: the audit that produced the current classification
    was worth doing once, and worth nothing by next month. The test is what
    makes it hold.
    """

    classified = {*CAPABILITY_ROUTES, *BROWSER_ROUTES, *DYNAMIC_ROUTES, *DEFERRED_ROUTES}
    reachable = _requested_routes(_package_root() / "sdk")

    unreachable = sorted(
        f"{method} {path}"
        for method, path in _routes()
        if (method, path) not in reachable and f"{method} {path}" not in classified
    )

    assert not unreachable, (
        "these routes cannot be reached from the SDK or CLI and are not classified; add a client, "
        "or add an entry to CAPABILITY_ROUTES / DYNAMIC_ROUTES / DEFERRED_ROUTES with the reason: "
        + ", ".join(unreachable)
    )


def test_no_classification_outlives_the_route_it_describes() -> None:
    """A stale exemption silently re-permits a gap.

    An entry naming a route that no longer exists is one nobody will notice
    covering a *different* route that does, once paths shift.
    """

    live = {f"{method} {path}" for method, path in _routes()}
    stale = sorted({*CAPABILITY_ROUTES, *BROWSER_ROUTES, *DYNAMIC_ROUTES, *DEFERRED_ROUTES} - live)

    assert not stale, f"these classifications name routes that no longer exist: {', '.join(stale)}"


def test_every_capability_route_still_has_something_that_mints_its_address() -> None:
    """The exemption's premise, checked at the level the code is organized at.

    A job never spells these paths. It receives a *complete URL* built at
    dispatch -- deliberately, so a job holds an address rather than the knowledge
    of how to assemble one, which would put every route in two places. One
    builder serves all three artifact kinds by interpolating the kind, so there
    is no literal path fragment to search for and a path-level match would be
    pretending.

    So this checks the builders themselves. What it catches is the case that
    matters: a capability route whose address nothing constructs any more, which
    is unreachable by everything rather than merely by the SDK.
    """

    from eyvan.artifacts.transfers import ManagedArtifactAccess
    from eyvan.server.controller.workload_commands import _ControllerWorkloadCommands

    for builder in (
        "callable_download_transfer",
        "workspace_download_transfer",
        "result_upload_transfer",
        "progress_report_url",
    ):
        assert hasattr(ManagedArtifactAccess, builder), (
            f"a capability route is exempted because {builder} builds its address, and it is gone"
        )
    for builder in ("_children_url", "_secret_url"):
        assert hasattr(_ControllerWorkloadCommands, builder), (
            f"a capability route is exempted because {builder} builds its address, and it is gone"
        )


def test_the_deferred_list_states_a_reason_rather_than_a_placeholder() -> None:
    """ "TODO" as a justification is how a deferral becomes permanent.

    Each reason has to say why a client would be *wrong*, which is a much harder
    thing to write than "not yet" -- and that difficulty is the point.
    """

    for route, reason in DEFERRED_ROUTES.items():
        assert len(reason) > 40, f"{route} is deferred without a real reason"
        assert "todo" not in reason.lower()


def test_every_command_module_is_actually_attached_to_the_cli() -> None:
    """A command module that exists and is never registered passes every other check.

    The path scan reads files, so a group defined in ``sdk/commands`` makes its
    routes look reachable whether or not ``main`` ever attaches it -- the same
    built-and-never-called shape as the routes this file exists to catch, one
    level up. A mutation that dropped ``register_artifact_commands`` from the
    CLI root survived until this test existed.
    """

    import importlib
    import pkgutil

    import click

    from eyvan.sdk import cli, commands

    def reachable(command: click.Command) -> set[int]:
        found = {id(command)}
        if isinstance(command, click.Group):
            for child in command.commands.values():
                found |= reachable(child)
        return found

    attached = reachable(cli.main)
    orphans: list[str] = []
    for module_info in pkgutil.iter_modules(commands.__path__):
        module = importlib.import_module(f"{commands.__name__}.{module_info.name}")
        for name, value in vars(module).items():
            if isinstance(value, click.Group) and id(value) not in attached:
                orphans.append(f"{module_info.name}.{name}")

    assert not orphans, f"these command groups are defined and never attached to the CLI: {', '.join(orphans)}"


@pytest.mark.parametrize("route,caller", sorted(BROWSER_ROUTES.items()))
def test_a_browser_route_is_reached_by_the_dashboard(route: str, caller: str) -> None:
    """The exemption's premise: the dashboard actually requests it.

    "The browser reaches it" is the whole reason a Python caller would be wrong.
    A route exempted on that basis and requested by no TypeScript is unreachable
    by everything, which is the gap this file exists to prevent rather than an
    instance of it being handled.
    """

    source = (pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "src").rglob("*.ts*")
    path = route.split(" ", 1)[1]
    assert any(path in re.sub(r"\$\{[^}]*\}", "{}", file.read_text(encoding="utf-8")) for file in source), (
        f"{route} is exempted because {caller} reaches it, and no dashboard source requests it"
    )


def test_the_dashboard_calls_the_panels_it_gained() -> None:
    """A client method no page invokes is the same defect one level down.

    ``explainWorkload``, ``getWarmth`` and ``getBilling`` are testable in
    isolation and would pass their own tests forever while no page ever rendered
    them -- which is exactly the shape this file exists to catch at the route
    level, repeated inside the client.
    """

    pages = (pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "src" / "pages").rglob("*.tsx")
    source = "\n".join(page.read_text(encoding="utf-8") for page in pages)

    for method in ("explainWorkload", "getWarmth", "getBilling", "getTpuQuota"):
        assert f"apiClient.{method}" in source or f".{method}(" in source, (
            f"apiClient.{method} exists and no page calls it"
        )
