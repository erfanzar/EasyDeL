# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FastAPI transport for the Eyvan cell agent."""

from __future__ import annotations

import hmac
import logging
import typing as tp
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from eyvan._internal.build_identity import build_identity
from eyvan._internal.json_types import JsonObject
from eyvan._internal.trace_context import remote_trace_parent
from eyvan.capacity.drivers.base import (
    CapacityProviderError,
    DriverBusyError,
    StaleMutationError,
    UnsupportedCapabilityError,
)
from eyvan.config import EyvanSettings
from eyvan.worker.agent.models import (
    AgentBuild,
    AgentCapabilities,
    AgentObservation,
    CapacityProviderFailure,
    CapacityRequest,
    CellStageActionRequest,
    CellStageActionResult,
    CheckpointRequest,
    ExecRequest,
    GangWorkloadMutationRequest,
    GangWorkloadRequest,
    MutationConflictCode,
    WorkloadMutationRequest,
    WorkloadObservation,
    WorkloadRequest,
)
from eyvan.worker.agent.service import CellAgentService

logger = logging.getLogger(__name__)


def create_agent_app(
    service: CellAgentService | None = None,
    settings: EyvanSettings | None = None,
    *,
    allow_unauthenticated: bool = False,
) -> FastAPI:
    """Create the authenticated cell-agent API.

    Args:
        service: Pre-built cell-agent service; one is built from ``settings`` when omitted.
        settings: Agent settings; loaded from the environment when omitted.
        allow_unauthenticated: Explicitly allow startup without ``agent_shared_secret`` for
            local development. Every endpoint is then reachable without credentials.

    Raises:
        ValueError: If no shared secret is configured and ``allow_unauthenticated`` is False.
    """

    # Pin this process's identity to the tree it started from; see build_identity.
    build_identity()
    settings = settings or EyvanSettings.from_env()
    if settings.agent_shared_secret is None:
        if not allow_unauthenticated:
            raise ValueError(
                "EYVAN_AGENT_SHARED_SECRET is required to start the cell agent; pass "
                "--allow-unauthenticated or set EYVAN_AGENT_ALLOW_UNAUTHENTICATED=1 to run "
                "without authentication for local development"
            )
        logger.warning(
            "EYVAN_AGENT_SHARED_SECRET is not set and unauthenticated access was explicitly allowed: "
            "every cell-agent endpoint (capacity apply, workload submit, release) is reachable "
            "WITHOUT credentials — do not use this outside local development"
        )
    service = service or CellAgentService(settings=settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> tp.AsyncIterator[None]:
        try:
            yield
        finally:
            service.close()

    app = FastAPI(
        title="Eyvan Cell Agent",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    def authenticate_credentials(authorization: str | None = Header(default=None)) -> None:
        secret = settings.agent_shared_secret
        if secret is None:
            return
        expected = f"Bearer {secret}"
        # compare_digest raises TypeError for non-ASCII str, and the header is
        # caller-controlled: unauthenticated garbage must answer 401, never a
        # 500.
        try:
            recognized = authorization is not None and hmac.compare_digest(authorization, expected)
        except TypeError:
            recognized = False
        if not recognized:
            raise HTTPException(status_code=401, detail="invalid cell-agent credential")

    async def authenticate(
        request: Request, _authenticated: None = Depends(authenticate_credentials)
    ) -> tp.AsyncIterator[None]:
        # Async dependency scope reaches sync handlers via their thread-pool
        # context copy, and unwinds after errors as well as successful responses.
        with remote_trace_parent(request.headers.get("traceparent"), enabled=settings.correlation_enabled):
            yield

    @app.exception_handler(StaleMutationError)
    async def stale_mutation_handler(_request: Request, exc: StaleMutationError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "code": MutationConflictCode.STALE_MUTATION.value},
        )

    @app.exception_handler(DriverBusyError)
    async def driver_busy_handler(_request: Request, exc: DriverBusyError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "code": MutationConflictCode.DRIVER_BUSY.value},
        )

    @app.exception_handler(UnsupportedCapabilityError)
    async def unsupported_capability_handler(_request: Request, exc: UnsupportedCapabilityError) -> JSONResponse:
        """Report a capability this backend does not have as 501, not 500."""

        return JSONResponse(status_code=501, content={"detail": str(exc), "code": "unsupported_capability"})

    @app.exception_handler(CapacityProviderError)
    async def capacity_provider_error_handler(_request: Request, exc: CapacityProviderError) -> JSONResponse:
        failure = CapacityProviderFailure(
            detail=exc.diagnostic,
            code=exc.code,
        )
        return JSONResponse(status_code=502, content=failure.model_dump(mode="json"))

    @app.exception_handler(ValueError)
    async def invalid_configuration_handler(_request: Request, exc: ValueError) -> JSONResponse:
        """Report a rejected configuration as the operator's to fix, not a crash.

        The capacity drivers validate provider options by raising ``ValueError``
        with a message that names the exact contradiction -- for example
        "internal-only eyvan-native Compute capacity cannot use
        bootstrap_egress=public-internet". Preserve that diagnostic rather than
        replacing it with FastAPI's generic 500 body.

        A 400 because the request is well-formed but the *configuration* it names
        is not, so retrying it unchanged can never succeed -- which also lets the
        outbox settle it terminally instead of burning its retry budget.
        """

        return JSONResponse(status_code=400, content={"detail": str(exc), "code": "invalid_configuration"})

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/capabilities", dependencies=[Depends(authenticate)])
    def capabilities() -> AgentCapabilities:
        return service.capabilities()

    @app.get("/api/v1/build", dependencies=[Depends(authenticate)])
    def build() -> AgentBuild:
        """Report the code this agent is running, not only its version string.

        Every component of this package ships one version, so an operator cannot
        use it to tell whether the agent holding their capacity is the build
        they deployed. A hardware proof has already been run against a stale
        agent and read as evidence.
        """

        identity = build_identity()
        return AgentBuild(
            version=identity.version,
            origin=identity.origin,
            origin_detail=identity.origin_detail,
            source_digest=identity.source_digest,
            digest_detail=identity.digest_detail,
            file_count=identity.file_count,
            python=identity.python,
            platform=identity.platform,
        )

    @app.get("/api/v1/cells/{cell_id}/worker-builds", dependencies=[Depends(authenticate)])
    def worker_builds(cell_id: str) -> list[dict[str, object]] | None:
        """Report the build of every native worker this cell holds.

        The processes that actually execute a workload, and the last components
        of this system that could not describe themselves. They install from the
        wheel their cell was provisioned with, which a re-staged head does not
        change -- so this is not a formality: it was observed drifting.
        """

        return service.worker_builds(cell_id)

    @app.get("/api/v1/cells", dependencies=[Depends(authenticate)])
    def inventory() -> list[dict[str, str | int | None]]:
        """List every cell this agent holds durable state for."""

        return service.inventory()

    @app.get(
        "/api/v1/cells/{cell_id}",
        dependencies=[Depends(authenticate)],
        response_model_exclude_none=True,
    )
    def observe(cell_id: str) -> AgentObservation:
        return service.observe(cell_id)

    @app.post(
        "/api/v1/cells/{cell_id}/reconcile",
        dependencies=[Depends(authenticate)],
        response_model_exclude_none=True,
    )
    def reconcile(cell_id: str, body: WorkloadMutationRequest) -> AgentObservation:
        return service.reconcile(cell_id, body)

    @app.put(
        "/api/v1/cells/{cell_id}/capacity",
        dependencies=[Depends(authenticate)],
        response_model_exclude_none=True,
    )
    def apply_capacity(cell_id: str, body: CapacityRequest) -> AgentObservation:
        return service.apply_capacity(cell_id, body)

    @app.post(
        "/api/v1/cells/{cell_id}/drain",
        dependencies=[Depends(authenticate)],
        response_model_exclude_none=True,
    )
    def drain(cell_id: str, body: WorkloadMutationRequest) -> AgentObservation:
        return service.drain(cell_id, body)

    @app.delete(
        "/api/v1/cells/{cell_id}",
        dependencies=[Depends(authenticate)],
        response_model_exclude_none=True,
    )
    def release(cell_id: str, body: WorkloadMutationRequest) -> AgentObservation:
        return service.release(cell_id, body)

    @app.post("/api/v1/cells/{cell_id}/stage-actions", dependencies=[Depends(authenticate)])
    def stage_asset(cell_id: str, body: CellStageActionRequest) -> CellStageActionResult:
        """Measure, materialize or remove one named asset on this cell's disk.

        The one wire method managed staging adds, and the reason it is one: an
        agent that predates it answers 404 for all three actions at once, so a
        control plane learns "this agent has no staging executor" from a single
        probe rather than inferring it three times.
        """

        return service.stage_asset(cell_id, body)

    @app.post("/api/v1/cells/{cell_id}/workloads", dependencies=[Depends(authenticate)])
    def submit(cell_id: str, body: WorkloadRequest) -> WorkloadObservation:
        return service.submit(cell_id, body)

    @app.post("/api/v1/cells/{cell_id}/workload-gangs", dependencies=[Depends(authenticate)])
    def submit_gang(cell_id: str, body: GangWorkloadRequest) -> WorkloadObservation:
        return service.submit_gang(cell_id, body)

    @app.get("/api/v1/cells/{cell_id}/workloads/{remote_id}", dependencies=[Depends(authenticate)])
    def workload(cell_id: str, remote_id: str) -> WorkloadObservation:
        return service.workload(cell_id, remote_id)

    @app.get(
        "/api/v1/cells/{cell_id}/workload-gangs/{workload_id}/{remote_id}",
        dependencies=[Depends(authenticate)],
    )
    def gang_workload(cell_id: str, workload_id: str, remote_id: str) -> WorkloadObservation:
        return service.gang_workload(cell_id, workload_id, remote_id)

    @app.post("/api/v1/cells/{cell_id}/workloads/{remote_id}/stop", dependencies=[Depends(authenticate)])
    def stop(cell_id: str, remote_id: str, body: WorkloadMutationRequest) -> WorkloadObservation:
        return service.stop(cell_id, remote_id, body)

    @app.post(
        "/api/v1/cells/{cell_id}/workload-gangs/{workload_id}/{remote_id}/stop",
        dependencies=[Depends(authenticate)],
    )
    def stop_gang(
        cell_id: str,
        workload_id: str,
        remote_id: str,
        body: GangWorkloadMutationRequest,
    ) -> WorkloadObservation:
        return service.stop_gang(cell_id, workload_id, remote_id, body)

    @app.post("/api/v1/cells/{cell_id}/workloads/{remote_id}/retire", dependencies=[Depends(authenticate)])
    def retire(cell_id: str, remote_id: str, body: WorkloadMutationRequest) -> WorkloadObservation:
        return service.retire(cell_id, remote_id, body)

    @app.post(
        "/api/v1/cells/{cell_id}/workload-gangs/{workload_id}/{remote_id}/retire",
        dependencies=[Depends(authenticate)],
    )
    def retire_gang(
        cell_id: str,
        workload_id: str,
        remote_id: str,
        body: GangWorkloadMutationRequest,
    ) -> WorkloadObservation:
        return service.retire_gang(cell_id, workload_id, remote_id, body)

    @app.post("/api/v1/cells/{cell_id}/workloads/{remote_id}/checkpoint", dependencies=[Depends(authenticate)])
    def checkpoint(cell_id: str, remote_id: str, body: CheckpointRequest) -> WorkloadObservation:
        return service.checkpoint(cell_id, remote_id, body)

    @app.get(
        "/api/v1/cells/{cell_id}/workloads/{remote_id}/logs",
        response_class=PlainTextResponse,
        dependencies=[Depends(authenticate)],
    )
    def logs(cell_id: str, remote_id: str) -> str:
        return service.logs(cell_id, remote_id)

    # response_model=None: the page is a plain JSON document, and building a
    # response model for it would restate LogRecord's own wire shape.
    # response_model=None: the result is backend-shaped JSON.
    @app.post(
        "/api/v1/cells/{cell_id}/workloads/{remote_id}/exec",
        dependencies=[Depends(authenticate)],
        response_model=None,
    )
    def exec_in_workload(cell_id: str, remote_id: str, body: ExecRequest) -> JsonObject:
        """Run one diagnostic command beside a live workload."""

        if body.runtime_rank is None:
            return service.exec_in_workload(cell_id, remote_id, body.command, timeout_s=body.timeout_s)
        return service.exec_in_workload(
            cell_id, remote_id, body.command, timeout_s=body.timeout_s, runtime_rank=body.runtime_rank
        )

    @app.get(
        "/api/v1/cells/{cell_id}/workloads/{remote_id}/log-page",
        dependencies=[Depends(authenticate)],
        response_model=None,
    )
    def log_page(cell_id: str, remote_id: str, after: str = "", limit: int = 200) -> JsonObject:
        """Return one resumable page of structured log records.

        Separate from the plain-text route rather than a format parameter on it:
        that route's response is ``PlainTextResponse`` and every existing
        consumer reads it as bytes, so its contract stays exactly as it was.
        """

        return service.log_page(cell_id, remote_id, after=after, limit=limit)

    @app.get(
        "/api/v1/cells/{cell_id}/workload-gangs/{workload_id}/{remote_id}/logs",
        response_class=PlainTextResponse,
        dependencies=[Depends(authenticate)],
    )
    def gang_logs(cell_id: str, workload_id: str, remote_id: str) -> str:
        return service.gang_logs(cell_id, workload_id, remote_id)

    @app.get(
        "/api/v1/cells/{cell_id}/workloads/{remote_id}/process-status",
        dependencies=[Depends(authenticate)],
        response_model=None,
    )
    def process_status(cell_id: str, remote_id: str) -> JsonObject:
        return service.process_status(cell_id, remote_id)

    return app
