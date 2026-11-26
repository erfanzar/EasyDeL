# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generic MCP access to this plane's own HTTP API, dispatched in process.

An external agent (Xerxes, Claude Code, Codex, Cursor) holds its user's own
token. The curated MCP tools cover the common paths; everything else the user
may do through ``/api/v1`` is reachable through three generic tools backed by
this module:

* :func:`list_operations` describes the catalogued operations from the app's
  own OpenAPI document.
* :func:`dispatch` sends one request for a catalogued operation **through the
  same FastAPI app**, in process, with only the caller's authentication and
  scope headers. The route then runs its own authentication, tenancy,
  permission and audit code; nothing is re-implemented or bypassed here.

What is catalogued is decided structurally and then narrowed by a denylist:

* the route is in the OpenAPI schema, under ``/api/v1/``, and authenticates
  through ``authenticated_principal`` (so worker callbacks that authenticate
  with capability tokens, and unauthenticated routes, are never reachable);
* it declares no cookie parameter and no header parameter other than
  ``Idempotency-Key``;
* it is not in :data:`DENIED_PATHS` / :data:`DENIED_PREFIXES`.

Privilege boundary of the in-process request: it carries the ORIGINAL
``request.client`` (never the transport's loopback default), a deliberately
unresolvable ``Host`` (``eyvan-mcp-bridge.invalid``), no cookies, no
``Origin``/``Sec-Fetch-*`` headers and no caller-chosen header besides
``Idempotency-Key``. So the dashboard's same-origin defaults
(``is_own_dashboard_request``) can never apply, cookie sessions are never
used, and an identity that only the original browser request had is simply
missing (the forwarded call fails closed with 401).
"""

from __future__ import annotations

import asyncio
import json
import re
import typing as tp
import uuid
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from starlette.routing import Match

from eyvan.server.api.auth import authenticated_principal
from eyvan.server.local.auth import LOCAL_DEVELOPMENT_CREDENTIAL_HEADER

API_PREFIX: tp.Final = "/api/v1/"
#: ``.invalid`` never resolves (RFC 6761) and is not a loopback literal, so no
#: Host-based trust rule can match a bridged request.
BRIDGE_BASE_URL: tp.Final = "http://eyvan-mcp-bridge.invalid"
DISPATCH_TIMEOUT_S: tp.Final = 120.0
MAX_BODY_BYTES: tp.Final = 1_048_576
MAX_RESPONSE_BYTES: tp.Final = 8 * 1_048_576
MAX_PATH_CHARS: tp.Final = 1024
MAX_QUERY_ITEMS: tp.Final = 64
MAX_QUERY_VALUE_CHARS: tp.Final = 2000
#: Listings with more matches than this show paths only.
DETAILED_LISTING_LIMIT: tp.Final = 25
#: Character budget of one listing, below the MCP result bound so it is never cut.
LISTING_BUDGET_CHARS: tp.Final = 12_000
MAX_SCHEMA_CHARS: tp.Final = 2500
WRITE_METHODS: tp.Final = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_SIGN_IN = "a sign-in/session flow; agents act with the token they already hold"
_TOKENS = "personal token management stays with the user (`eyvan tokens` or the dashboard)"
_INTERACTIVE = "an interactive browser sign-in on the plane host"
_STREAM = "a streaming response; use the curated tools (get_workload_logs) or poll the JSON routes"
_BINARY = "a binary or plain-text transfer, not JSON"
_WORKER = "a worker/agent callback authenticated by a job capability token, not a user route"

#: Route templates never reachable through the bridge, whatever the method.
DENIED_PATHS: tp.Final[dict[str, str]] = {
    "/api/v1/mcp": "the MCP endpoint itself (no recursion)",
    "/api/v1/team/login": _SIGN_IN,
    "/api/v1/team/logout": _SIGN_IN,
    "/api/v1/team/invites/accept": _SIGN_IN,
    "/api/v1/team/invites/preview": _SIGN_IN,
    "/api/v1/team/status": _SIGN_IN,
    "/api/v1/team/tokens": _TOKENS,
    "/api/v1/team/tokens/{token_id}": _TOKENS,
    "/api/v1/assistant/oauth/openrouter/start": _INTERACTIVE,
    "/api/v1/assistant/oauth/openrouter/callback/{state}": _INTERACTIVE,
    "/api/v1/assistant/claude-code/login": _INTERACTIVE,
    "/api/v1/events/stream": _STREAM,
    "/api/v1/workloads/{workload_id}/logs/stream": _STREAM,
    "/api/v1/assistant/chat/stream": _STREAM,
    "/api/v1/workloads/{workload_id}/logs": _BINARY,
    "/api/v1/artifacts/results/{result_id}": _BINARY,
    "/api/v1/artifacts/callables/{digest}": _BINARY,
    "/api/v1/artifacts/workspaces/{digest}": _BINARY,
    "/api/v1/tracking/runs/{run_id}/blobs/{sha256}": _BINARY,
    "/api/v1/artifacts/results/{result_id}/content": _WORKER,
    "/api/v1/artifacts/callables/{digest}/content": _WORKER,
    "/api/v1/artifacts/workspaces/{digest}/content": _WORKER,
    "/api/v1/secrets/content": _WORKER,
    "/api/v1/workloads/{workload_id}/progress-reports": _WORKER,
    "/api/v1/workloads/{workload_id}/endpoint": _WORKER,
    "/metrics": "the Prometheus scrape endpoint",
    "/healthz": "an unauthenticated probe",
    "/readyz": "an unauthenticated probe",
}
#: Whole subtrees never reachable through the bridge: the prefix without its
#: trailing slash and everything below it.
DENIED_PREFIXES: tp.Final[tuple[tuple[str, str], ...]] = (
    ("/api/v1/dashboard/", "dashboard sign-in, session cookies and browser configuration"),
    (
        "/api/v1/attachments/",
        "local attachment operations (subprocesses and SSH on the plane host) that also need the plane's real "
        "URL, which an in-process call does not have",
    ),
)

_ACCOUNT = (
    "account creation: an invite link creates an account (possibly a fleet admin) that outlives the agent's token; "
    "invite from the dashboard or with `eyvan users invite`"
)
_AUTHORITY = (
    "a durable grant of authority to an account, which outlives the agent's token; do it from the dashboard or CLI"
)
_FEDERATION = (
    "durable authority for another control plane (peer registration, budget or quota grant) that outlives the "
    "agent's token"
)
#: Single (method, route template) operations never reachable through the bridge.
#: Other methods on the same template (for example ``GET /api/v1/team/invites``) stay available.
DENIED_OPERATIONS: tp.Final[dict[tuple[str, str], str]] = {
    ("POST", "/api/v1/team/invites"): _ACCOUNT,
    ("POST", "/api/v1/grants"): _AUTHORITY,
    ("PUT", "/api/v1/team/users/{subject}/tier"): _AUTHORITY,
    ("POST", "/api/v1/team/users/{subject}/enable"): (
        "re-activates an account whose password signs in again, which outlives the agent's token"
    ),
    ("PUT", "/api/v1/team/users/{subject}/rules"): (
        "per-account governance (TPU cap, approval threshold, allowed fleets) that outlives the agent's token"
    ),
    ("POST", "/api/v1/query"): (
        "raw SQL over the plane's database, which would read credential hashes and other tenants' rows into an "
        "agent's context"
    ),
    ("POST", "/api/v1/secrets/rotate"): "deployment-wide signing-credential rotation, an operator action",
    ("PUT", "/api/v1/federation/peers/{plane_id}"): _FEDERATION,
    ("POST", "/api/v1/federation/budget-grants"): _FEDERATION,
    ("POST", "/api/v1/federation/quota-grants"): _FEDERATION,
    ("POST", "/api/v1/workloads/{workload_id}/children"): _WORKER,
}

#: Last path segments of POST routes that stop, take away or discard something.
_DESTRUCTIVE_ACTIONS: tp.Final = frozenset(
    {"cancel", "drain", "force-finalize", "stop", "disable", "kick", "reclaim", "collect", "roll", "rotate"}
)
_SEGMENT: tp.Final = re.compile(r"^[A-Za-z0-9._~:@+=,-]+$")
_QUERY_KEY: tp.Final = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_IDEMPOTENCY_KEY: tp.Final = re.compile(r"^[\x21-\x7e]{1,200}$")
_DEV_IDENTITY_HEADERS: tp.Final = ("X-Eyvan-User", "X-Eyvan-Roles", LOCAL_DEVELOPMENT_CREDENTIAL_HEADER)
_SCOPE_HEADERS: tp.Final = ("X-Eyvan-Org", "X-Eyvan-Project")


#: Routes still running after their caller timed out; kept referenced so they finish.
_ORPHANED: set[asyncio.Future[tp.Any]] = set()


def _forget(task: asyncio.Future[tp.Any]) -> None:
    _ORPHANED.discard(task)
    if not task.cancelled():
        task.exception()  # retrieve it so an orphaned failure is not reported as never-retrieved


class ApiBridgeError(Exception):
    """A refusal or a failed route, with the HTTP status the agent should see."""

    def __init__(
        self, message: str, *, status: int, kind: str | None = None, idempotency_key: str | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.kind = kind
        #: The key a write was sent with, so the agent can retry it safely.
        self.idempotency_key = idempotency_key


def denial_reason(template: str, method: str | None = None) -> str | None:
    """Why ``template`` (or ``method template`` when a method is given) is never bridged, or ``None``."""

    if template in DENIED_PATHS:
        return DENIED_PATHS[template]
    for prefix, reason in DENIED_PREFIXES:
        if template == prefix.rstrip("/") or template.startswith(prefix):
            return reason
    if method is not None:
        return DENIED_OPERATIONS.get((method.upper(), template))
    return None


@dataclass(frozen=True, slots=True)
class Operation:
    """One catalogued (method, path template) pair, described for agents."""

    method: str
    path: str
    area: str
    summary: str
    description: str
    parameters: tuple[dict[str, tp.Any], ...]
    query_names: frozenset[str]
    body_schema: dict[str, tp.Any] | None
    body_required: bool
    idempotency: str | None
    read_only: bool
    destructive: bool

    def compact(self) -> str:
        return f"{self.method} {self.path}"

    def document(self, *, with_body: bool) -> dict[str, tp.Any]:
        document: dict[str, tp.Any] = {
            "method": self.method,
            "path": self.path,
            "area": self.area,
            "tool": "eyvan_api_get" if self.method == "GET" else "eyvan_api_call",
            "summary": self.summary,
            "read_only": self.read_only,
            "destructive": self.destructive,
        }
        if self.description:
            document["description"] = self.description
        if self.parameters:
            document["parameters"] = list(self.parameters)
        if self.idempotency is not None:
            document["idempotency_key"] = self.idempotency
        if self.body_schema is not None:
            document["body_required"] = self.body_required
            document["body_schema"] = (
                self.body_schema if with_body else "omitted to stay small; narrow with search to see it"
            )
        return document


@dataclass(frozen=True, slots=True)
class Catalogue:
    """The operations agents may call, and the routes they may not, for one app."""

    operations: tuple[Operation, ...]
    by_route: dict[tuple[int, str], Operation]


def _uses_user_authentication(dependant: Dependant) -> bool:
    return any(node.call is authenticated_principal for node in _walk(dependant))


def _walk(dependant: Dependant) -> tp.Iterator[Dependant]:
    yield dependant
    for child in dependant.dependencies:
        yield from _walk(child)


def _header_names(route: APIRoute) -> set[str]:
    return {str(field.alias).lower() for node in _walk(route.dependant) for field in node.header_params}


def _has_cookie_parameters(route: APIRoute) -> bool:
    return any(node.cookie_params for node in _walk(route.dependant))


def _inline(node: tp.Any, components: dict[str, tp.Any], depth: int, stack: frozenset[str]) -> tp.Any:
    """Resolve ``$ref``s up to ``depth``, dropping schema titles (never property names)."""

    if isinstance(node, list):
        return [_inline(item, components, depth, stack) for item in node]
    if not isinstance(node, dict):
        return node
    reference = node.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/components/schemas/"):
        name = reference.rsplit("/", 1)[1]
        target = components.get(name)
        if depth <= 0 or name in stack or not isinstance(target, dict):
            return {"$ref": name}
        return _inline(target, components, depth - 1, stack | {name})
    result: dict[str, tp.Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key == "properties" and isinstance(value, dict):
            result[key] = {name: _inline(item, components, depth, stack) for name, item in value.items()}
        else:
            result[key] = _inline(value, components, depth, stack)
    return result


def _bounded_schema(schema: tp.Any, components: dict[str, tp.Any], limit: int = MAX_SCHEMA_CHARS) -> tp.Any:
    """``schema`` with references inlined as deeply as fits in ``limit`` characters."""

    for depth in (6, 3, 1, 0):
        candidate = _inline(schema, components, depth, frozenset())
        if len(json.dumps(candidate, separators=(",", ":"), default=str)) <= limit:
            return candidate
    top = _inline(schema, components, 1, frozenset())
    properties = top.get("properties") if isinstance(top, dict) else None
    names = list(properties)[:40] if isinstance(properties, dict) else []
    return {"type": "object", "properties_too_large_to_list": names}


def _first_paragraph(text: tp.Any, limit: int = 240) -> str:
    if not isinstance(text, str):
        return ""
    paragraph = " ".join(text.strip().split("\n\n", 1)[0].split())
    return paragraph if len(paragraph) <= limit else paragraph[: limit - 3] + "..."


def _describe(route: APIRoute, method: str, operation: dict[str, tp.Any], components: dict[str, tp.Any]) -> Operation:
    parameters: list[dict[str, tp.Any]] = []
    query_names: set[str] = set()
    idempotency: str | None = None
    for parameter in operation.get("parameters") or ():
        if not isinstance(parameter, dict):
            continue
        location = parameter.get("in")
        name = str(parameter.get("name", ""))
        if location == "header":
            if name.lower() == "idempotency-key":
                idempotency = "required" if parameter.get("required") else "optional"
            continue
        if location not in {"path", "query"}:
            continue
        if location == "query":
            query_names.add(name)
        described: dict[str, tp.Any] = {
            "name": name,
            "in": location,
            "required": bool(parameter.get("required")),
            "schema": _bounded_schema(parameter.get("schema") or {}, components, 300),
        }
        if parameter.get("description"):
            described["description"] = _first_paragraph(parameter["description"], 160)
        parameters.append(described)
    body_schema: dict[str, tp.Any] | None = None
    body = operation.get("requestBody")
    if isinstance(body, dict):
        content = body.get("content") or {}
        json_body = content.get("application/json") if isinstance(content, dict) else None
        if isinstance(json_body, dict):
            body_schema = _bounded_schema(json_body.get("schema") or {}, components)
    path = route.path_format
    last = path.rstrip("/").rsplit("/", 1)[-1]
    tags = operation.get("tags") or route.tags or ["other"]
    return Operation(
        method=method,
        path=path,
        area=str(tags[0]),
        summary=str(operation.get("summary") or route.name),
        description=_first_paragraph(operation.get("description")),
        parameters=tuple(parameters),
        query_names=frozenset(query_names),
        body_schema=body_schema,
        body_required=bool(isinstance(body, dict) and body.get("required")),
        idempotency=idempotency,
        read_only=method == "GET",
        destructive=method == "DELETE" or (method in WRITE_METHODS and last in _DESTRUCTIVE_ACTIONS),
    )


def _build_catalogue(app: FastAPI) -> Catalogue:
    schema = app.openapi()
    paths = schema.get("paths") or {}
    components = (schema.get("components") or {}).get("schemas") or {}
    operations: list[Operation] = []
    by_route: dict[tuple[int, str], Operation] = {}
    for route in app.router.routes:
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue
        template = route.path_format
        if not template.startswith(API_PREFIX) or denial_reason(template) is not None:
            continue
        if not _uses_user_authentication(route.dependant) or _has_cookie_parameters(route):
            continue
        if _header_names(route) - {"idempotency-key"}:
            continue
        for method in sorted(route.methods or ()):
            if method == "HEAD" or denial_reason(template, method) is not None:
                continue
            described = (paths.get(template) or {}).get(method.lower())
            if not isinstance(described, dict):
                continue
            operation = _describe(route, method, described, components)
            operations.append(operation)
            by_route[(id(route), method)] = operation
    operations.sort(key=lambda item: (item.path, item.method))
    return Catalogue(operations=tuple(operations), by_route=by_route)


def catalogue(app: FastAPI) -> Catalogue:
    """The app's catalogue, built once from its routes and OpenAPI document."""

    cached = getattr(app.state, "mcp_api_catalogue", None)
    if isinstance(cached, Catalogue):
        return cached
    built = _build_catalogue(app)
    app.state.mcp_api_catalogue = built
    return built


def list_operations(app: FastAPI, *, area: str | None = None, search: str | None = None) -> dict[str, tp.Any]:
    """The catalogued operations matching ``area``/``search``, bounded in size.

    Many matches list ``METHOD path`` only; a narrow match (at most
    :data:`DETAILED_LISTING_LIMIT`) adds parameters and request-body schemas.
    """

    operations = catalogue(app).operations
    areas = sorted({operation.area for operation in operations})
    wanted_area = area.strip().lower() if area else None
    needle = search.strip().lower() if search else None
    matched = [
        operation
        for operation in operations
        if (wanted_area is None or operation.area.lower() == wanted_area)
        and (
            needle is None
            or needle in operation.path.lower()
            or needle in operation.summary.lower()
            or needle in operation.description.lower()
            or needle == operation.method.lower()
        )
    ]
    detailed = len(matched) <= DETAILED_LISTING_LIMIT
    shown: list[tp.Any] = []
    used = 0
    for operation in matched:
        entry: tp.Any = operation.document(with_body=True) if detailed else operation.compact()
        size = len(json.dumps(entry, separators=(",", ":"), default=str))
        if detailed and used + size > LISTING_BUDGET_CHARS:
            entry = operation.document(with_body=False)
            size = len(json.dumps(entry, separators=(",", ":"), default=str))
        if used + size > LISTING_BUDGET_CHARS:
            break
        shown.append(entry)
        used += size
    note = (
        "call GET operations with eyvan_api_get and the others with eyvan_api_call; every call runs with your own "
        "permissions. org_id/project_id query parameters default to your MCP scope when omitted."
    )
    if not detailed:
        note += f" {len(matched)} matches: pass area or search to see parameters and body schemas."
    if len(shown) < len(matched):
        note += f" Only {len(shown)} of {len(matched)} fit; narrow with area or search."
    return {"total": len(matched), "shown": len(shown), "areas": areas, "operations": shown, "note": note}


# ------------------------------------------------------------------ dispatch


def _validated_path(path: tp.Any) -> str:
    if not isinstance(path, str) or not path or len(path) > MAX_PATH_CHARS:
        raise ApiBridgeError(f"path must be a string of 1 to {MAX_PATH_CHARS} characters", status=400)
    if not path.startswith(API_PREFIX):
        raise ApiBridgeError("refused: only /api/v1/ operations are reachable through this tool", status=403)
    segments = path[1:].split("/")
    if path.endswith("/") or any(segment in {".", ".."} or not _SEGMENT.fullmatch(segment) for segment in segments):
        raise ApiBridgeError(
            "path must be a plain /api/v1/... path: no query string, fragment, percent-encoding, empty, '.' or '..' "
            "segments (pass query parameters in `query`)",
            status=400,
        )
    return path


def _query_pairs(query: tp.Any) -> list[tuple[str, str]]:
    if query is None:
        return []
    if not isinstance(query, dict):
        raise ApiBridgeError("query must be an object of parameter names to values", status=400)
    pairs: list[tuple[str, str]] = []

    def rendered(value: tp.Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, str) and len(value) <= MAX_QUERY_VALUE_CHARS:
            return value
        raise ApiBridgeError(
            f"query values must be strings (at most {MAX_QUERY_VALUE_CHARS} characters), numbers or booleans",
            status=400,
        )

    for key, value in query.items():
        if not isinstance(key, str) or not _QUERY_KEY.fullmatch(key):
            raise ApiBridgeError("query parameter names must match [A-Za-z0-9_.-]{1,64}", status=400)
        if value is None:
            continue
        for item in value if isinstance(value, list) else [value]:
            pairs.append((key, rendered(item)))
    if len(pairs) > MAX_QUERY_ITEMS:
        raise ApiBridgeError(f"at most {MAX_QUERY_ITEMS} query values", status=400)
    return pairs


def resolve(app: FastAPI, method: str, path: str) -> Operation:
    """The catalogued operation the app would route ``method path`` to, else a refusal.

    Uses the router's own matching (first full match wins, exactly as the
    request will be routed), so precedence between overlapping templates can
    never select a different route than the one checked here.
    """

    known = catalogue(app)
    probe: dict[str, tp.Any] = {"type": "http", "path": path, "root_path": "", "method": method, "headers": []}
    method_mismatch = False
    for route in app.router.routes:
        match, _ = route.matches(probe)
        if match is Match.PARTIAL:
            method_mismatch = True
            continue
        if match is not Match.FULL:
            continue
        template = getattr(route, "path_format", None)
        if not isinstance(route, APIRoute) or not isinstance(template, str):
            break  # the dashboard's static mount: no API operation here
        reason = denial_reason(template, method)
        if reason is not None:
            raise ApiBridgeError(f"refused: {method} {template} is {reason}", status=403, kind="Refused")
        operation = known.by_route.get((id(route), method))
        if operation is None:
            raise ApiBridgeError(
                f"refused: {method} {template} is not available to agents (not a user-authenticated JSON operation)",
                status=403,
                kind="Refused",
            )
        return operation
    if method_mismatch:
        raise ApiBridgeError(f"{method} is not allowed on {path}; see eyvan_api_operations", status=405)
    raise ApiBridgeError(f"no operation matches {path}; list them with eyvan_api_operations", status=404)


def forwarded_headers(request: Request, *, dev_auth: bool) -> dict[str, str]:
    """Only the caller's authentication and scope headers; never cookies, Origin or fetch metadata."""

    headers: dict[str, str] = {}
    names: list[str] = ["Authorization", *_SCOPE_HEADERS]
    if dev_auth:
        names.extend(_DEV_IDENTITY_HEADERS)
    for name in names:
        value = request.headers.get(name)
        if value is not None:
            headers[name] = value
    return headers


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, list):
        parts = []
        for item in detail[:10]:
            if isinstance(item, dict):
                location = ".".join(str(part) for part in item.get("loc", ()))
                parts.append(f"{location or 'value'}: {item.get('msg', 'invalid')}")
        detail = "; ".join(parts)
    if not detail:
        detail = f"HTTP {response.status_code}"
    return str(detail)[:1000]


async def dispatch(
    request: Request,
    *,
    method: str,
    path: tp.Any,
    query: tp.Any = None,
    body: tp.Any = None,
    idempotency_key: tp.Any = None,
    dev_auth: bool,
    default_scope: tuple[str | None, str | None] = (None, None),
) -> dict[str, tp.Any]:
    """Run one catalogued operation through ``request.app`` as the caller and return ``{status, body}``.

    Raises:
        ApiBridgeError: a refusal before dispatch, or the route's own error
            status and detail.
    """

    method = method.upper()
    if method != "GET" and method not in WRITE_METHODS:
        raise ApiBridgeError("method must be GET, POST, PUT, PATCH or DELETE", status=400)
    checked_path = _validated_path(path)
    pairs = _query_pairs(query)
    if method == "GET" and body is not None:
        raise ApiBridgeError("GET operations take no body", status=400)
    if body is not None and len(json.dumps(body, separators=(",", ":"), default=str)) > MAX_BODY_BYTES:
        raise ApiBridgeError(f"body is limited to {MAX_BODY_BYTES} bytes of JSON", status=413)
    if idempotency_key is not None and (
        not isinstance(idempotency_key, str) or not _IDEMPOTENCY_KEY.fullmatch(idempotency_key)
    ):
        raise ApiBridgeError("idempotency_key must be 1-200 printable ASCII characters", status=400)
    operation = resolve(request.app, method, checked_path)

    present = {key for key, _ in pairs}
    for name, default in zip(("org_id", "project_id"), default_scope, strict=True):
        if name in operation.query_names and name not in present and default:
            pairs.append((name, default))
    headers = forwarded_headers(request, dev_auth=dev_auth)
    key = idempotency_key
    if key is None and operation.idempotency is not None and method in WRITE_METHODS:
        key = f"mcp-{uuid.uuid4().hex}"
    if key is not None and operation.idempotency is not None:
        headers["Idempotency-Key"] = key
    sent_key = key if key is not None and operation.idempotency is not None else None
    retry = (
        f" Retrying with idempotency_key {sent_key!r} is safe: the plane replays the original outcome instead of "
        "acting twice."
        if sent_key is not None
        else ""
    )
    client = (request.client.host, request.client.port) if request.client is not None else ("unknown", 0)
    transport = httpx.ASGITransport(app=request.app, raise_app_exceptions=False, client=client)

    async def send() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=transport, base_url=BRIDGE_BASE_URL, follow_redirects=False, trust_env=False, timeout=None
        ) as bridge:
            return await bridge.request(method, checked_path, params=tuple(pairs), json=body, headers=headers)

    # The route runs as its own task and is never cancelled on timeout: cancelling
    # it mid-transaction could leave a write half-applied with no one to observe it.
    task = asyncio.ensure_future(send())
    done, _ = await asyncio.wait({task}, timeout=DISPATCH_TIMEOUT_S)
    if task not in done:
        _ORPHANED.add(task)
        task.add_done_callback(_forget)
        outcome = "the write may still complete; its result is unknown." if method in WRITE_METHODS else ""
        raise ApiBridgeError(
            f"{method} {operation.path} did not answer within {DISPATCH_TIMEOUT_S:.0f}s; {outcome}{retry}".rstrip(),
            status=504,
            idempotency_key=sent_key,
        )
    response = task.result()
    if response.status_code >= 300:
        raise ApiBridgeError(
            _error_detail(response) + retry,
            status=response.status_code,
            kind="HTTPError",
            idempotency_key=sent_key,
        )
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ApiBridgeError("the response is too large for an agent; narrow the request", status=413)
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    parsed: tp.Any = None
    if response.content:
        if content_type != "application/json" and not content_type.endswith("+json"):
            raise ApiBridgeError(
                f"{method} {operation.path} answered {content_type or 'an untyped body'}, not JSON; "
                "this tool returns JSON only",
                status=415,
            )
        try:
            parsed = response.json()
        except ValueError:
            raise ApiBridgeError(f"{method} {operation.path} returned invalid JSON", status=502) from None
    result: dict[str, tp.Any] = {"status": response.status_code, "body": parsed}
    if key is not None and operation.idempotency is not None:
        result["idempotency_key"] = key
    if response.headers.get("idempotent-replay") == "true":
        result["replayed"] = True
    return result


__all__ = (
    "DENIED_OPERATIONS",
    "DENIED_PATHS",
    "DENIED_PREFIXES",
    "WRITE_METHODS",
    "ApiBridgeError",
    "Catalogue",
    "Operation",
    "catalogue",
    "denial_reason",
    "dispatch",
    "forwarded_headers",
    "list_operations",
    "resolve",
)
