# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The plane as an MCP (streamable HTTP) tool server for agents such as Xerxes.

``POST /api/v1/mcp`` speaks JSON-RPC 2.0 the way the MCP Streamable HTTP
transport does, answering every request with a plain ``application/json``
body. The server is stateless: it issues no ``Mcp-Session-Id`` and opens no
server-initiated stream, so ``GET`` and ``DELETE`` answer 405.

Every call is authenticated by the ordinary ``authenticated_principal``
dependency and every tool runs as that principal, so an agent can never read
or do more than its user could with their own token. Unlike the dashboard
assistant, whose acting tools only draft, the MCP client *is* the user's own
agent acting on their instruction: ``submit_workload`` and ``cancel_workload``
act directly, through exactly the service calls the REST routes make.

Tools live in :data:`TOOLS`; adding one is one :class:`McpTool` entry with a
name, description, JSON schema and ``async handler(service, principal, scope,
args)``.
"""

from __future__ import annotations

import json
import logging
import re
import typing as tp
import uuid
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from eyvan._version import __version__
from eyvan.config import EyvanSettings
from eyvan.records import Permission, Principal, WorkloadRecord
from eyvan.server.api import mcp_api_bridge
from eyvan.server.api.assistant import TOOL_SPECS as ASSISTANT_TOOL_SPECS
from eyvan.server.api.assistant import AssistantTools
from eyvan.server.api.auth import DEFAULT_DEVELOPMENT_SCOPE, authenticated_principal
from eyvan.server.api.research_routes import plan_document
from eyvan.server.api.schemas import WorkloadSubmitRequest
from eyvan.server.local.auth import LOCAL_DEVELOPMENT_CREDENTIAL_HEADER
from eyvan.server.local.daemon import LocalDaemonError
from eyvan.server.local.tracker import LocalTrackerError, read_tracker_record, read_tracker_token
from eyvan.server.service.errors import (
    AuthorizationError,
    ConflictError,
    EyvanServiceError,
    NotFoundError,
    UnavailableError,
)
from eyvan.worker.agent.client import CellAgentUnavailableError

if tp.TYPE_CHECKING:
    from eyvan.server.service import ControlPlaneService

logger = logging.getLogger(__name__)

MCP_PATH = "/api/v1/mcp"
#: Revisions this server answers, newest first. ``initialize`` echoes the
#: client's revision when it is one of these, else offers the newest.
SUPPORTED_PROTOCOL_VERSIONS: tp.Final = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION: tp.Final = SUPPORTED_PROTOCOL_VERSIONS[0]
#: Largest JSON-RPC message accepted; a workload spec is far smaller.
MAX_REQUEST_BYTES: tp.Final = 1_048_576
#: Largest text a tool result carries; longer results are cut and say so.
MAX_RESULT_CHARS: tp.Final = 16_000
DEFAULT_LOG_LINES: tp.Final = 200
MAX_LOG_LINES: tp.Final = 2000
MAX_LOG_CHARS: tp.Final = 12_000

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

_LOOPBACK_HOSTS: tp.Final = frozenset({"127.0.0.1", "localhost", "::1"})
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_METRIC_KEY = re.compile(r"^[A-Za-z0-9_./-]{1,128}$")
_TRACKER_UNCONFIGURED = (
    "run tracking is not configured on this plane; run `eyvan track up --local` or set EYVAN_TRACKER_URL"
)
_INSTRUCTIONS = (
    "Eyvan control plane tools. Every tool runs with the signed-in user's own permissions in one org/project: "
    "pass org_id/project_id, or rely on the X-Eyvan-Org/X-Eyvan-Project headers or the user's single granted "
    "scope. submit_workload and cancel_workload act immediately; draft_workload and draft_easydel_launch only "
    "prepare a spec for submit_workload."
)


class McpToolError(Exception):
    """A tool failure reported to the agent as ``isError`` rather than a JSON-RPC error.

    ``status`` is the HTTP status the equivalent REST call answers, so an agent
    (and a test) can tell a refusal from a missing record.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        kind: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.idempotency_key = idempotency_key
        self.status = status
        self.kind = kind

    def document(self) -> dict[str, tp.Any]:
        payload: dict[str, tp.Any] = {"error": self.message}
        if self.kind is not None:
            payload["type"] = self.kind
        if self.status is not None:
            payload["status"] = self.status
        if self.idempotency_key is not None:
            payload["idempotency_key"] = self.idempotency_key
        return payload


@dataclass(frozen=True, slots=True)
class McpScope:
    """The org/project one tool call acts in, plus the plane it runs on.

    ``org_id``/``project_id`` are the call's arguments, else the request's
    ``X-Eyvan-Org``/``X-Eyvan-Project`` headers, else the principal's single
    granted scope; ``None`` when that is ambiguous. ``settings`` and ``app``
    give handlers the plane's configuration and live state (controller,
    tracker transport); they confer no authority.
    """

    org_id: str | None
    project_id: str | None
    team_mode: bool = False
    candidates: tuple[tuple[str, str], ...] = ()
    settings: EyvanSettings | None = None
    app: FastAPI | None = None
    #: Grants the credential itself carried, before durable grants were merged in.
    credential_grants: tuple[tp.Any, ...] | None = None
    #: The MCP HTTP request itself: the API bridge forwards only its authentication
    #: and scope headers and its original client address, never anything else.
    request: Request | None = None

    def require(self) -> tuple[str, str]:
        """Return the scope, or explain how to name one when it is ambiguous."""

        if self.org_id and self.project_id:
            return self.org_id, self.project_id
        known = ", ".join(f"{org}/{project}" for org, project in self.candidates[:10])
        hint = f"; you have access to {known}" if known else ""
        raise McpToolError(f"pass org_id and project_id: this tool needs one org/project scope{hint}", status=400)

    def or_default(self) -> tuple[str, str]:
        """Return the scope for tools that ignore it, without refusing."""

        return self.org_id or DEFAULT_DEVELOPMENT_SCOPE, self.project_id or DEFAULT_DEVELOPMENT_SCOPE


Handler = tp.Callable[["ControlPlaneService", Principal, McpScope, dict[str, tp.Any]], tp.Awaitable[tp.Any]]


@dataclass(frozen=True, slots=True)
class McpTool:
    """One published tool. ``scoped`` adds optional ``org_id``/``project_id`` arguments."""

    name: str
    description: str
    input_schema: dict[str, tp.Any]
    handler: Handler
    scoped: bool = False
    annotations: dict[str, tp.Any] = field(default_factory=dict)

    def listing(self) -> dict[str, tp.Any]:
        document: dict[str, tp.Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.annotations:
            document["annotations"] = self.annotations
        return document


#: The published tools, in ``tools/list`` order.
TOOLS: dict[str, McpTool] = {}


def register_tool(tool: McpTool) -> McpTool:
    """Add (or replace) one tool in :data:`TOOLS`; scoped tools gain org_id/project_id."""

    if tool.scoped:
        tool = replace(tool, input_schema=_with_scope(tool.input_schema))
    TOOLS[tool.name] = tool
    return tool


# ------------------------------------------------------------------ helpers


_STR: tp.Final = {"type": "string"}
_INT: tp.Final = {"type": "integer"}
_READ_ONLY: tp.Final = {"readOnlyHint": True, "openWorldHint": False}


def _schema(properties: dict[str, tp.Any], required: tp.Sequence[str] = ()) -> dict[str, tp.Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def _with_scope(schema: dict[str, tp.Any]) -> dict[str, tp.Any]:
    properties = dict(schema.get("properties") or {})
    properties.setdefault("org_id", {"type": "string", "description": "Org to act in (default: your scope)."})
    properties.setdefault("project_id", {"type": "string", "description": "Project to act in (default: your scope)."})
    return {**schema, "properties": properties}


#: Any team bearer token (``eyv_`` + base64url), wherever it appears in a result.
_TEAM_TOKEN_TEXT: tp.Final = re.compile(r"eyv_[A-Za-z0-9_-]{16,}")
REDACTED: tp.Final = "<redacted>"


def redact_secrets(text: str, secrets: tp.Iterable[str] = ()) -> str:
    """``text`` without the caller's credential values or any team token."""

    for secret in sorted(set(secrets), key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return _TEAM_TOKEN_TEXT.sub(f"eyv_{REDACTED}", text)


def _caller_secrets(request: Request | None) -> tuple[str, ...]:
    """The credential values the caller presented (Authorization and local development credential)."""

    if request is None:
        return ()
    values: list[str] = []
    authorization = request.headers.get("authorization", "").strip()
    if authorization:
        values.append(authorization)
        token = authorization.partition(" ")[2].strip()
        if token:
            values.append(token)
    credential = request.headers.get(LOCAL_DEVELOPMENT_CREDENTIAL_HEADER, "").strip()
    if credential:
        values.append(credential)
    # Very short values would redact ordinary words; real credentials are long.
    return tuple(value for value in values if len(value) >= 8)


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    note = f" ...[truncated: {len(text)} characters; narrow the request (filters, tail, keys, points)]"
    return text[: max(0, limit - len(note))] + note


def bounded_text(value: tp.Any, limit: int = MAX_RESULT_CHARS, *, secrets: tp.Iterable[str] = ()) -> str:
    """Serialize ``value`` as compact JSON with credentials redacted, cut to ``limit`` characters."""

    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)
    return _bounded(redact_secrets(text, secrets), limit)


def _redacted_result(value: tp.Any, *, error: bool = False, secrets: tp.Iterable[str] = ()) -> dict[str, tp.Any]:
    return {"content": [{"type": "text", "text": bounded_text(value, secrets=secrets)}], "isError": error}


def _service_status(exc: EyvanServiceError) -> int:
    """The status ``handle_service_error`` in app.py answers for this error."""

    if isinstance(exc, AuthorizationError):
        return status.HTTP_403_FORBIDDEN
    if isinstance(exc, NotFoundError):
        return status.HTTP_404_NOT_FOUND
    if isinstance(exc, ConflictError):
        return status.HTTP_409_CONFLICT
    if isinstance(exc, UnavailableError):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_400_BAD_REQUEST


#: The assistant reports service errors as ``"<ClassName>: message"``; map the
#: class name back to the REST status so both surfaces agree.
_ERROR_STATUS_BY_NAME: tp.Final = {
    "AuthorizationError": 403,
    "NotFoundError": 404,
    "NoMatchingCapacityPoolError": 404,
    "ConflictError": 409,
    "CapacityClassUnavailableError": 409,
    "UnavailableError": 503,
    "InputError": 400,
}


def _assistant_error(message: str) -> McpToolError:
    kind, separator, _ = message.partition(": ")
    if separator and kind in _ERROR_STATUS_BY_NAME:
        return McpToolError(message, status=_ERROR_STATUS_BY_NAME[kind], kind=kind)
    return McpToolError(message, status=400)


def _validation_text(exc: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(item) for item in error.get('loc', ())) or 'value'}: {error.get('msg', 'invalid')}"
        for error in exc.errors()[:10]
    ]
    return "; ".join(parts)[:1000]


def _string(args: dict[str, tp.Any], name: str, *, required: bool = False, limit: int = 200) -> str | None:
    value = args.get(name)
    if value is None:
        if required:
            raise McpToolError(f"{name} is required", status=400)
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise McpToolError(f"{name} must be a non-empty string of at most {limit} characters", status=400)
    return value.strip()


def _integer(args: dict[str, tp.Any], name: str, *, default: int, low: int, high: int) -> int:
    value = args.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise McpToolError(f"{name} must be an integer from {low} to {high}", status=400)
    return value


def _workload_summary(workload: WorkloadRecord) -> dict[str, tp.Any]:
    return {
        "id": workload.id,
        "name": workload.spec.name,
        "state": workload.state,
        "org_id": workload.org_id,
        "project_id": workload.project_id,
        "accelerator_type": workload.spec.accelerator_type,
        "created_by": workload.created_by,
        "created_at": workload.created_at.isoformat(),
    }


def _granted_scope(principal: Principal) -> tuple[str | None, str | None, tuple[tuple[str, str], ...]]:
    """The principal's single project scope, ``default`` for unscoped grants, else ``None``."""

    pairs = tuple(
        sorted({(grant.org_id, grant.project_id) for grant in principal.grants if grant.org_id and grant.project_id})
    )
    if len(pairs) == 1:
        return pairs[0][0], pairs[0][1], pairs
    if not pairs and not any(grant.org_id or grant.project_id for grant in principal.grants):
        # Only global grants (a fleet admin): the plane's conventional scope.
        return DEFAULT_DEVELOPMENT_SCOPE, DEFAULT_DEVELOPMENT_SCOPE, pairs
    return None, None, pairs


# ------------------------------------------------------------------ assistant tools

#: Assistant tools that read or write inside one org/project.
_SCOPED_ASSISTANT_TOOLS: tp.Final = frozenset(
    {
        "list_fleets",
        "list_capacity",
        "get_quota",
        "list_workloads",
        "draft_workload",
        "file_request",
        "propose_research_plan",
    }
)
_DRAFT_NOTE = "not submitted; call submit_workload with draft.spec (and draft.org_id/project_id) to submit it"


async def _run_assistant_tool(
    name: str, service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    org_id, project_id = scope.require() if name in _SCOPED_ASSISTANT_TOOLS else scope.or_default()
    tools = AssistantTools(service, principal, org_id=org_id, project_id=project_id)
    tools.team_mode = scope.team_mode
    outcome = await tools.call(name, args)
    result = outcome.result
    if not outcome.ok:
        message = result.get("error") if isinstance(result, dict) else None
        raise _assistant_error(str(message or result))
    if isinstance(result, dict) and set(result) == {"error"}:
        raise McpToolError(str(result["error"]), status=404)
    if name == "draft_workload" and isinstance(result, dict) and result.get("drafted") and tools.drafts:
        result = {**result, "note": _DRAFT_NOTE, "draft": tools.drafts[-1]}
    return result


def _assistant_handler(name: str) -> Handler:
    async def handle(
        service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
    ) -> tp.Any:
        return await _run_assistant_tool(name, service, principal, scope, args)

    return handle


_ASSISTANT_ANNOTATIONS: tp.Final[dict[str, dict[str, tp.Any]]] = {
    "file_request": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    "propose_research_plan": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
}

for _spec in ASSISTANT_TOOL_SPECS:
    _name = str(_spec["name"])
    _description = str(_spec["description"])
    if _name == "draft_workload":
        _description = _description.replace("Prepare (NOT submit)", "Prepare (not submit)") + (
            " Returns the draft; submit it with submit_workload."
        )
    register_tool(
        McpTool(
            name=_name,
            description=_description,
            input_schema=dict(_spec["parameters"]),
            handler=_assistant_handler(_name),
            scoped=_name in _SCOPED_ASSISTANT_TOOLS,
            annotations=dict(_ASSISTANT_ANNOTATIONS.get(_name, _READ_ONLY)),
        )
    )


def _draft_easydel_schema() -> dict[str, tp.Any]:
    draft = TOOLS["draft_workload"].input_schema
    properties = {
        key: value for key, value in (draft.get("properties") or {}).items() if key not in {"command", "org_id"}
    }
    properties.pop("project_id", None)
    return _schema(properties, ["name", "easydel_trainer", "easydel_launch"])


async def _draft_easydel_launch(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    _string(args, "easydel_trainer", required=True)
    if not isinstance(args.get("easydel_launch"), dict):
        raise McpToolError("easydel_launch must be an object with at least model and dataset", status=400)
    return await _run_assistant_tool("draft_workload", service, principal, scope, args)


register_tool(
    McpTool(
        name="draft_easydel_launch",
        description=(
            "Render a runnable EasyDeL eLarge launch for a trainer (validated like validate_easydel_arguments) and "
            "return a draft workload; nothing is submitted. Submit draft.spec with submit_workload."
        ),
        input_schema=_draft_easydel_schema(),
        handler=_draft_easydel_launch,
        scoped=True,
        annotations=dict(_READ_ONLY),
    )
)


# ------------------------------------------------------------------ acting workload tools


async def _submit_workload(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    org_id, project_id = scope.require()
    key = _string(args, "idempotency_key") or f"mcp-{uuid.uuid4().hex}"
    try:
        request = WorkloadSubmitRequest.model_validate(
            {
                "org_id": org_id,
                "project_id": project_id,
                "spec": args.get("spec"),
                "parent_workload_id": args.get("parent_workload_id"),
            }
        )
    except ValidationError as exc:
        raise McpToolError(f"invalid workload: {_validation_text(exc)}", status=422) from None
    workload, replayed = await service.submit_workload(
        principal,
        org_id=request.org_id,
        project_id=request.project_id,
        spec=request.spec,
        idempotency_key=key,
        parent_workload_id=request.parent_workload_id,
    )
    return {"submitted": True, "replayed": replayed, "idempotency_key": key, "workload": _workload_summary(workload)}


async def _get_workload(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    workload_id = _string(args, "workload_id", required=True)
    assert workload_id is not None
    workload = await service.get_workload(principal, workload_id)
    return workload.model_dump(mode="json")


async def _cancel_workload(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    workload_id = _string(args, "workload_id", required=True)
    assert workload_id is not None
    key = _string(args, "idempotency_key") or f"mcp-cancel-{uuid.uuid4().hex}"
    cascade = args.get("cascade", True)
    if not isinstance(cascade, bool):
        raise McpToolError("cascade must be a boolean", status=400)
    workload, replayed = await service.cancel_workload(principal, workload_id, idempotency_key=key, cascade=cascade)
    return {"cancel_requested": True, "replayed": replayed, "workload": _workload_summary(workload)}


async def _get_workload_logs(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    workload_id = _string(args, "workload_id", required=True)
    assert workload_id is not None
    tail = _integer(args, "tail", default=DEFAULT_LOG_LINES, low=1, high=MAX_LOG_LINES)
    workload = await service.get_workload(principal, workload_id)  # READ on the workload's scope
    controller = getattr(scope.app.state, "controller", None) if scope.app is not None else None
    if controller is None:
        raise McpToolError("this plane serves no workload logs", status=503)
    try:
        text = await controller.workload_logs(workload_id)
    except CellAgentUnavailableError:
        text = (
            "cell agent unreachable; live output is unavailable right now. "
            "Archived output remains available via 'eyvan workload logs' once the run settles."
        )
    lines = str(text).splitlines()
    shown = "\n".join(lines[-tail:])
    clipped = len(shown) > MAX_LOG_CHARS
    if clipped:
        shown = shown[-MAX_LOG_CHARS:]
    return {
        "workload_id": workload.id,
        "state": workload.state,
        "total_lines": len(lines),
        "lines_shown": min(tail, len(lines)),
        "truncated": clipped or len(lines) > tail,
        "logs": shown,
    }


register_tool(
    McpTool(
        name="submit_workload",
        description=(
            "Submit a workload NOW, as you, through the same path as POST /api/v1/workloads. spec is a workload "
            'definition, e.g. {"name": "hello", "execution": {"kind": "command", "command": "echo hi"}} or the '
            "spec of a draft_workload/draft_easydel_launch result. Reusing idempotency_key returns the original "
            "submission instead of a duplicate."
        ),
        input_schema=_schema(
            {
                "spec": {"type": "object", "description": "Workload definition (WorkloadDefinition)."},
                "idempotency_key": {"type": "string", "description": "Retry-safe key (default: a fresh one)."},
                "parent_workload_id": _STR,
            },
            ["spec"],
        ),
        handler=_submit_workload,
        scoped=True,
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
)
register_tool(
    McpTool(
        name="get_workload",
        description="One workload's full record: state, spec, attempts, scheduling reason.",
        input_schema=_schema({"workload_id": _STR}, ["workload_id"]),
        handler=_get_workload,
        annotations=dict(_READ_ONLY),
    )
)
register_tool(
    McpTool(
        name="cancel_workload",
        description=(
            "Cancel a workload NOW (and by default every child under it), with the same permission check as "
            "POST /api/v1/workloads/{id}/cancel."
        ),
        input_schema=_schema(
            {"workload_id": _STR, "cascade": {"type": "boolean"}, "idempotency_key": _STR}, ["workload_id"]
        ),
        handler=_cancel_workload,
        annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    )
)
register_tool(
    McpTool(
        name="get_workload_logs",
        description=f"The last `tail` lines (default {DEFAULT_LOG_LINES}, max {MAX_LOG_LINES}) of a workload's output.",
        input_schema=_schema({"workload_id": _STR, "tail": _INT}, ["workload_id"]),
        handler=_get_workload_logs,
        annotations=dict(_READ_ONLY),
    )
)


# ------------------------------------------------------------------ research plans


async def _approve_research_plan(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    plan_id = _string(args, "plan_id", required=True)
    assert plan_id is not None
    await service.approve_research_plan(principal, plan_id, credential_grants=scope.credential_grants)
    plan, workloads = await service.describe_research_plan(principal, plan_id)
    return plan_document(plan, workloads)


async def _stop_research_plan(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    plan_id = _string(args, "plan_id", required=True)
    assert plan_id is not None
    reason = _string(args, "reason", limit=2000) or ""
    await service.stop_research_plan(principal, plan_id, reason=reason)
    plan, workloads = await service.describe_research_plan(principal, plan_id)
    return plan_document(plan, workloads)


register_tool(
    McpTool(
        name="approve_research_plan",
        description=(
            "Approve one of your proposed research plans NOW (owner only): the plane then submits its jobs as you, "
            "monitors them and iterates inside the plan's budget. Only on the user's explicit instruction."
        ),
        input_schema=_schema({"plan_id": _STR}, ["plan_id"]),
        handler=_approve_research_plan,
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
)
register_tool(
    McpTool(
        name="stop_research_plan",
        description="Stop a research plan NOW (owner or fleet admin); its running workloads are cancelled.",
        input_schema=_schema({"plan_id": _STR, "reason": _STR}, ["plan_id"]),
        handler=_stop_research_plan,
        annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    )
)


# ------------------------------------------------------------------ run tracking


def _tracker_target(settings: EyvanSettings | None) -> tuple[str, str | None]:
    url = settings.tracker_url if settings is not None else None
    token = settings.tracker_token if settings is not None else None
    if url is None:
        try:
            record = read_tracker_record()
            if record is not None:
                url, token = record.url, read_tracker_token(record)
        except (OSError, ValueError, LocalDaemonError, LocalTrackerError):
            raise McpToolError(_TRACKER_UNCONFIGURED, status=503) from None
    if url is None:
        raise McpToolError(_TRACKER_UNCONFIGURED, status=503)
    return url, token


async def _tracker_get(scope: McpScope, path: str, params: list[tuple[str, str]]) -> tp.Any:
    url, token = _tracker_target(scope.settings)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    transport = getattr(scope.app.state, "tracking_transport", None) if scope.app is not None else None
    try:
        async with httpx.AsyncClient(
            transport=transport, timeout=30.0, trust_env=False, follow_redirects=False
        ) as client:
            # A tuple, not the list: tuples are covariant, so mypy accepts str-only pairs.
            response = await client.get(f"{url.rstrip('/')}{path}", params=tuple(params), headers=headers)
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        raise McpToolError("the run tracker is unreachable or returned invalid JSON", status=502) from None
    if response.status_code != 200:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        raise McpToolError(str(detail or f"tracker answered HTTP {response.status_code}"), status=response.status_code)
    return payload


async def _list_runs(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    org_id, project_id = scope.require()
    service.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
    limit = _integer(args, "limit", default=50, low=1, high=200)
    params = [("org_id", org_id), ("project_id", project_id), ("limit", str(limit))]
    state = _string(args, "status", limit=32)
    if state is not None:
        params.append(("status", state))
    payload = await _tracker_get(scope, "/api/v1/tracking/runs", params)
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    fields = ("run_id", "name", "group", "status", "workload_id", "attempts", "last_step", "updated_at", "keys_count")
    return {
        "org_id": org_id,
        "project_id": project_id,
        "runs": [{key: run.get(key) for key in fields} for run in runs if isinstance(run, dict)],
    }


async def _get_run_metrics(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    org_id, project_id = scope.require()
    service.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
    run_id = _string(args, "run_id", required=True, limit=128)
    if run_id is None or not _RUN_ID.fullmatch(run_id):
        raise McpToolError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}", status=400)
    points = _integer(args, "points", default=50, low=1, high=1000)
    x_axis = _string(args, "x", limit=128) or "step"
    keys = args.get("keys")
    if keys is not None and (
        not isinstance(keys, list)
        or not 1 <= len(keys) <= 16
        or not all(isinstance(key, str) and _METRIC_KEY.fullmatch(key) for key in keys)
    ):
        raise McpToolError("keys must be a list of 1 to 16 metric names", status=400)
    scope_params = [("org_id", org_id), ("project_id", project_id)]
    detail = await _tracker_get(scope, f"/api/v1/tracking/runs/{run_id}", scope_params)
    detail = detail if isinstance(detail, dict) else {}
    available = [str(key) for key in detail.get("keys", []) if isinstance(key, str)]
    selected = list(keys) if keys is not None else available[:8]
    series: list[tp.Any] = []
    if selected:
        payload = await _tracker_get(
            scope,
            f"/api/v1/tracking/runs/{run_id}/series",
            [*scope_params, ("x", x_axis), ("points", str(points)), *(("key", key) for key in selected)],
        )
        series = payload.get("series", []) if isinstance(payload, dict) else []
    run_fields = ("run_id", "name", "status", "workload_id", "attempts", "last_step", "updated_at", "summary")
    return {
        "run": {key: detail.get(key) for key in run_fields},
        "available_keys": available[:100],
        "x": x_axis,
        "series": [
            {
                key: item.get(key)
                for key in ("key", "rank", "points", "total", "downsampled", "nonfinite")
                if isinstance(item, dict)
            }
            for item in series
        ],
        "point_format": "[x, mean, min, max]",
    }


register_tool(
    McpTool(
        name="list_runs",
        description="Experiment-tracking runs (eyvan.track) in a project, newest first.",
        input_schema=_schema({"status": _STR, "limit": _INT}),
        handler=_list_runs,
        scoped=True,
        annotations=dict(_READ_ONLY),
    )
)
register_tool(
    McpTool(
        name="get_run_metrics",
        description=(
            "Metric series of one tracked run, downsampled to `points` (default 50). keys defaults to the run's "
            "first 8 metric keys; x is step (default), time or another logged axis."
        ),
        input_schema=_schema(
            {"run_id": _STR, "keys": {"type": "array", "items": _STR}, "points": _INT, "x": _STR}, ["run_id"]
        ),
        handler=_get_run_metrics,
        scoped=True,
        annotations=dict(_READ_ONLY),
    )
)


# ------------------------------------------------------------------ generic API access


async def _api_operations(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    if scope.app is None:
        raise McpToolError("this plane publishes no API catalogue", status=503)
    area = _string(args, "area", limit=64)
    search = _string(args, "search", limit=200)
    return mcp_api_bridge.list_operations(scope.app, area=area, search=search)


async def _api_dispatch(scope: McpScope, method: str, args: dict[str, tp.Any]) -> tp.Any:
    request = scope.request
    if request is None:
        raise McpToolError("the API tools are only available over the MCP HTTP endpoint", status=503)
    dev_auth = bool(scope.settings is not None and scope.settings.dev_auth_enabled)
    if not request.headers.get("authorization") and not dev_auth:
        # A cookie session is never forwarded, so the routed call would carry no identity.
        raise McpToolError(
            "the API tools act with your bearer token; connect with an `Authorization: Bearer <token>` header",
            status=401,
        )
    try:
        return await mcp_api_bridge.dispatch(
            request,
            method=method,
            path=args.get("path"),
            query=args.get("query"),
            body=args.get("body"),
            idempotency_key=args.get("idempotency_key"),
            dev_auth=dev_auth,
            default_scope=(scope.org_id, scope.project_id),
        )
    except mcp_api_bridge.ApiBridgeError as exc:
        raise McpToolError(exc.message, status=exc.status, kind=exc.kind, idempotency_key=exc.idempotency_key) from None


async def _api_get(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    return await _api_dispatch(scope, "GET", args)


async def _api_call(
    service: ControlPlaneService, principal: Principal, scope: McpScope, args: dict[str, tp.Any]
) -> tp.Any:
    method = args.get("method")
    if not isinstance(method, str) or method.upper() not in mcp_api_bridge.WRITE_METHODS:
        raise McpToolError("method must be POST, PUT, PATCH or DELETE (use eyvan_api_get to read)", status=400)
    return await _api_dispatch(scope, method.upper(), args)


_API_PATH: tp.Final = {
    "type": "string",
    "description": "A concrete /api/v1/... path matching a template from eyvan_api_operations (no query string).",
}
_API_QUERY: tp.Final = {
    "type": "object",
    "description": "Query parameters; org_id/project_id default to your MCP scope when the operation takes them.",
}
register_tool(
    McpTool(
        name="eyvan_api_operations",
        description=(
            "List the plane's HTTP API operations you can call through eyvan_api_get / eyvan_api_call: method, path "
            "template, summary, parameters and request-body schema (shown when the match is narrow). Filter by area "
            "(a tag such as workloads, fleets, governance) and/or a search string."
        ),
        input_schema=_schema(
            {
                "area": {"type": "string", "description": "Only this API area (tag)."},
                "search": {"type": "string", "description": "Case-insensitive text to find in path or summary."},
            }
        ),
        handler=_api_operations,
        annotations=dict(_READ_ONLY),
    )
)
register_tool(
    McpTool(
        name="eyvan_api_get",
        description=(
            "Call one GET operation of this plane's API as you (same permission checks as the REST API) and return "
            "{status, body}. Only operations listed by eyvan_api_operations are reachable."
        ),
        input_schema=_schema({"path": _API_PATH, "query": _API_QUERY}, ["path"]),
        handler=_api_get,
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    )
)
register_tool(
    McpTool(
        name="eyvan_api_call",
        description=(
            "Call one POST/PUT/PATCH/DELETE operation of this plane's API NOW, as you, with the REST API's own "
            "permission checks and audit, and return {status, body}. Only operations listed by eyvan_api_operations "
            "are reachable. Routes that take an Idempotency-Key get idempotency_key (default: a fresh one, returned)."
        ),
        input_schema=_schema(
            {
                "method": {"type": "string", "enum": ["POST", "PUT", "PATCH", "DELETE"]},
                "path": _API_PATH,
                "query": _API_QUERY,
                "body": {"description": "JSON request body, as the operation's body_schema describes."},
                "idempotency_key": {"type": "string", "description": "Retry-safe key; reuse it to retry safely."},
            },
            ["method", "path"],
        ),
        handler=_api_call,
        annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    )
)


# ------------------------------------------------------------------ JSON-RPC


def _rpc_result(message_id: tp.Any, result: dict[str, tp.Any]) -> dict[str, tp.Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _rpc_error(message_id: tp.Any, code: int, message: str) -> dict[str, tp.Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _http_error(status_code: int, code: int, message: str, message_id: tp.Any = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=_rpc_error(message_id, code, message))


def _valid_id(value: tp.Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, str)


async def call_tool(
    tool: McpTool,
    arguments: dict[str, tp.Any],
    *,
    service: ControlPlaneService,
    principal: Principal,
    scope: McpScope,
) -> dict[str, tp.Any]:
    """Run one tool as ``principal`` and shape its outcome as an MCP ``tools/call`` result.

    Every result and error text is redacted for the caller's own credential
    values and any team token, so nothing the agent sees echoes them.
    """

    secrets = _caller_secrets(scope.request)

    def _tool_result(value: tp.Any, *, error: bool = False) -> dict[str, tp.Any]:
        return _redacted_result(value, error=error, secrets=secrets)

    args = dict(arguments)
    properties = tool.input_schema.get("properties") or {}
    if tool.input_schema.get("additionalProperties") is False:
        unknown = sorted(set(args) - set(properties))
        if unknown:
            return _tool_result({"error": f"unknown argument(s): {', '.join(unknown)}", "status": 400}, error=True)
    missing = [name for name in tool.input_schema.get("required") or () if args.get(name) is None]
    if missing:
        return _tool_result({"error": f"missing argument(s): {', '.join(missing)}", "status": 400}, error=True)
    try:
        if tool.scoped:
            org_id = _string(args, "org_id", limit=128)
            project_id = _string(args, "project_id", limit=128)
            args.pop("org_id", None)
            args.pop("project_id", None)
            scope = replace(scope, org_id=org_id or scope.org_id, project_id=project_id or scope.project_id)
        value = await tool.handler(service, principal, scope, args)
    except McpToolError as exc:
        return _tool_result(exc.document(), error=True)
    except EyvanServiceError as exc:
        error = McpToolError(str(exc), status=_service_status(exc), kind=type(exc).__name__)
        return _tool_result(error.document(), error=True)
    except ValidationError as exc:
        return _tool_result({"error": f"invalid arguments: {_validation_text(exc)}", "status": 400}, error=True)
    except (TypeError, ValueError) as exc:
        return _tool_result({"error": f"invalid arguments: {str(exc)[:500]}", "status": 400}, error=True)
    except HTTPException as exc:
        return _tool_result({"error": str(exc.detail)[:500], "status": exc.status_code}, error=True)
    except Exception as exc:
        # Never echo an unexpected exception's text: it may carry internals.
        logger.error("mcp tool %s failed with %s", tool.name, type(exc).__name__, exc_info=True)
        return _tool_result({"error": f"internal error running {tool.name}", "status": 500}, error=True)
    return _tool_result(value)


def _normalized_origin(value: str) -> str | None:
    """``scheme://host[:port]`` in lower case with the default port dropped, or ``None``."""

    try:
        split = urlsplit(value.strip())
        port = split.port
    except ValueError:
        return None
    scheme = split.scheme.lower()
    host = (split.hostname or "").lower()
    if scheme not in {"http", "https"} or not host:
        return None
    if port in {None, 80 if scheme == "http" else 443}:
        port = None
    rendered = f"[{host}]" if ":" in host else host
    return f"{scheme}://{rendered}" + (f":{port}" if port is not None else "")


def origin_allowed(origin: str, *, public_url: str, host_header: str, scheme: str) -> bool:
    """Whether a browser ``Origin`` may call the MCP endpoint (DNS-rebinding guard).

    Allowed: the configured ``public_url`` origin, or this request's own origin
    when its ``Host`` is a loopback literal. A rebinding attacker's request
    carries the attacker's name in both ``Host`` and ``Origin``, so a loopback
    literal ``Host`` cannot be produced by one.
    """

    presented = _normalized_origin(origin)
    if presented is None:
        return False
    if presented == _normalized_origin(public_url):
        return True
    own = _normalized_origin(f"{scheme}://{host_header}") if host_header else None
    if own is None:
        return False
    hostname = urlsplit(own).hostname or ""
    return hostname in _LOOPBACK_HOSTS and presented == own


def require_safe_origin(request: Request) -> None:
    """Refuse cross-origin browser calls before authentication; non-browser callers send no Origin."""

    origin = request.headers.get("origin")
    if origin is None:
        return
    settings: EyvanSettings | None = getattr(request.app.state, "mcp_settings", None)
    public_url = settings.public_url if settings is not None else ""
    if not origin_allowed(
        origin, public_url=public_url, host_header=request.headers.get("host", ""), scheme=request.url.scheme
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="MCP requests from this origin are refused")


#: Module level: with postponed annotations FastAPI resolves these against module globals.
_Origin = tp.Annotated[None, Depends(require_safe_origin)]
_Principal = tp.Annotated[Principal, Depends(authenticated_principal)]


async def _read_body(request: Request) -> bytes | None:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_REQUEST_BYTES:
                return None
        except ValueError:
            return None
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            return None
        body.extend(chunk)
    return bytes(body)


def register_mcp_routes(app: FastAPI, *, settings: EyvanSettings, service: ControlPlaneService) -> None:
    """Mount ``POST /api/v1/mcp`` (JSON-RPC) and its stateless ``GET``/``DELETE`` 405 answers."""

    app.state.mcp_settings = settings

    @app.post(MCP_PATH, tags=["mcp"], response_model=None)
    async def mcp_endpoint(request: Request, _origin: _Origin, principal: _Principal) -> Response:
        """MCP Streamable HTTP endpoint: one JSON-RPC message per POST, tools run as the caller."""

        content_type = request.headers.get("content-type", "application/json").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            return _http_error(415, INVALID_REQUEST, "Content-Type must be application/json")
        version = request.headers.get("mcp-protocol-version")
        if version is not None and version not in SUPPORTED_PROTOCOL_VERSIONS:
            supported = ", ".join(SUPPORTED_PROTOCOL_VERSIONS)
            return _http_error(400, INVALID_REQUEST, f"unsupported MCP-Protocol-Version; supported: {supported}")
        body = await _read_body(request)
        if body is None:
            return _http_error(413, INVALID_REQUEST, f"MCP messages are limited to {MAX_REQUEST_BYTES} bytes")
        try:
            message = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return _http_error(400, PARSE_ERROR, "Parse error")
        if isinstance(message, list):
            return _http_error(400, INVALID_REQUEST, "JSON-RPC batches are not supported; send one message per POST")
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _http_error(400, INVALID_REQUEST, "Invalid Request: expected one JSON-RPC 2.0 object")
        method = message.get("method")
        if "id" not in message:
            if isinstance(method, str):
                return Response(status_code=status.HTTP_202_ACCEPTED)  # a notification
            return _http_error(400, INVALID_REQUEST, "Invalid Request: no method")
        message_id = message["id"]
        if not _valid_id(message_id):
            return _http_error(400, INVALID_REQUEST, "Invalid Request: id must be a string or integer")
        if not isinstance(method, str):
            if "result" in message or "error" in message:
                return Response(status_code=status.HTTP_202_ACCEPTED)  # a response; we never ask
            return _http_error(400, INVALID_REQUEST, "Invalid Request: no method", message_id)
        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return JSONResponse(_rpc_error(message_id, INVALID_PARAMS, "params must be an object"))

        if method == "initialize":
            requested = params.get("protocolVersion")
            negotiated = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
            return JSONResponse(
                _rpc_result(
                    message_id,
                    {
                        "protocolVersion": negotiated,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "eyvan", "version": __version__},
                        "instructions": _INSTRUCTIONS,
                    },
                )
            )
        if method == "ping":
            return JSONResponse(_rpc_result(message_id, {}))
        if method == "tools/list":
            return JSONResponse(_rpc_result(message_id, {"tools": [tool.listing() for tool in TOOLS.values()]}))
        if method != "tools/call":
            return JSONResponse(_rpc_error(message_id, METHOD_NOT_FOUND, f"Method not found: {method[:100]}"))

        name = params.get("name")
        tool = TOOLS.get(name) if isinstance(name, str) else None
        if tool is None:
            return JSONResponse(_rpc_error(message_id, INVALID_PARAMS, f"unknown tool {str(name)[:100]!r}"))
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return JSONResponse(_rpc_error(message_id, INVALID_PARAMS, "arguments must be an object"))
        granted_org, granted_project, candidates = _granted_scope(principal)
        header_org = request.headers.get("x-eyvan-org", "").strip()[:128] or None
        header_project = request.headers.get("x-eyvan-project", "").strip()[:128] or None
        scope = McpScope(
            org_id=header_org or granted_org,
            project_id=header_project or granted_project,
            team_mode=settings.team_mode,
            candidates=candidates,
            settings=settings,
            app=request.app,
            credential_grants=getattr(request.state, "credential_grants", None),
            request=request,
        )
        result = await call_tool(tool, arguments, service=service, principal=principal, scope=scope)
        return JSONResponse(_rpc_result(message_id, result))

    @app.api_route(MCP_PATH, methods=["GET", "DELETE"], tags=["mcp"], include_in_schema=False, response_model=None)
    async def mcp_no_stream() -> Response:
        """Stateless server: no server-initiated stream to open and no session to end."""

        return JSONResponse(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            content={"detail": "this MCP endpoint is stateless; POST JSON-RPC messages to it"},
            headers={"Allow": "POST"},
        )


__all__ = (
    "MAX_REQUEST_BYTES",
    "MAX_RESULT_CHARS",
    "MCP_PATH",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "TOOLS",
    "McpScope",
    "McpTool",
    "McpToolError",
    "bounded_text",
    "call_tool",
    "origin_allowed",
    "register_mcp_routes",
    "register_tool",
)
