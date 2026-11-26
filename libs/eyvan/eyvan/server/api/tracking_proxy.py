# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Authenticated, scope-authorized reads from the independent run tracker."""

from __future__ import annotations

import typing as tp

import httpx
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, Response

from eyvan.config import EyvanSettings
from eyvan.records import Permission, Principal
from eyvan.server.api.auth import authenticated_principal
from eyvan.server.local.daemon import LocalDaemonError
from eyvan.server.local.tracker import LocalTrackerError, read_tracker_record, read_tracker_token
from eyvan.server.service import ControlPlaneService
from eyvan.track._format import MEDIA_TYPES
from eyvan.track._rich import MAX_BLOB_BYTES

#: Headers on every proxied blob: inert, never sniffed, sandboxed if opened directly.
_BLOB_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
    "Cache-Control": "private, no-store",
    "Content-Disposition": "inline",
}
_Sha256 = tp.Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")]
_ItemId = tp.Annotated[str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")]

_UNCONFIGURED = "Run tracking is not configured; run `eyvan track up --local` or set EYVAN_TRACKER_URL."
_RunId = tp.Annotated[str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
_ScopeId = tp.Annotated[str, Query(min_length=1, max_length=128)]
_Principal = tp.Annotated[Principal, Depends(authenticated_principal)]


def register_tracking_routes(
    app: FastAPI,
    *,
    settings: EyvanSettings,
    service: ControlPlaneService,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Register the proxy, with an optional HTTP transport for isolated tests.

    ``app.state.tracking_transport`` can also supply a transport after the main
    application factory has run. Neither seam changes authentication or scope
    authorization. Upstream redirects are never followed with the bearer token.
    """

    async def forward(
        request: Request, *, principal: Principal, org_id: str | None = None, project_id: str | None = None
    ) -> JSONResponse:
        response = await upstream(request, principal=principal, org_id=org_id, project_id=project_id)
        try:
            payload = response.json()
        except ValueError:
            raise HTTPException(
                status_code=502, detail="Tracker is unreachable or returned invalid JSON; check its status."
            ) from None
        return JSONResponse(status_code=response.status_code, content=payload)

    async def forward_blob(request: Request, *, principal: Principal, org_id: str, project_id: str) -> Response:
        response = await upstream(request, principal=principal, org_id=org_id, project_id=project_id, blob=True)
        if response.status_code != 200:
            try:
                payload = response.json()
            except ValueError:
                payload = {"detail": "Tracker refused the blob request"}
            return JSONResponse(status_code=response.status_code, content=payload)
        return Response(
            content=response.content, media_type=response.headers["content-type"], headers=dict(_BLOB_HEADERS)
        )

    async def upstream(
        request: Request,
        *,
        principal: Principal,
        org_id: str | None = None,
        project_id: str | None = None,
        blob: bool = False,
    ) -> httpx.Response:
        if org_id is not None and project_id is not None:
            service.authorizer.require(principal, Permission.READ, org_id=org_id, project_id=project_id)
        url, token = settings.tracker_url, settings.tracker_token
        if url is None:
            try:
                record = read_tracker_record()
                if record is not None:
                    url, token = record.url, read_tracker_token(record)
            except (OSError, ValueError, LocalDaemonError, LocalTrackerError):
                raise HTTPException(status_code=503, detail=_UNCONFIGURED) from None
        if url is None:
            raise HTTPException(status_code=503, detail=_UNCONFIGURED)
        # Keep repeated metric keys, but bind the forwarded scope to exactly the
        # values we authorized, regardless of duplicate org/project parameters.
        params = [
            (key, value) for key, value in request.query_params.multi_items() if key not in {"org_id", "project_id"}
        ]
        if org_id is not None and project_id is not None:
            params.extend((("org_id", org_id), ("project_id", project_id)))
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        selected_transport = getattr(request.app.state, "tracking_transport", transport)
        try:
            async with httpx.AsyncClient(
                transport=selected_transport, timeout=30.0, trust_env=False, follow_redirects=False
            ) as client:
                target = f"{url.rstrip('/')}{request.url.path}"
                if not blob:
                    return await client.get(target, params=params, headers=headers)
                async with client.stream("GET", target, params=params, headers=headers) as response:
                    # Bound both successful blobs and error bodies before any
                    # eager read. Content-Length is an early refusal, not proof
                    # that the decoded body fits the limit.
                    try:
                        content_length = int(response.headers.get("content-length", ""))
                    except ValueError:
                        content_length = None
                    if content_length is not None and content_length > MAX_BLOB_BYTES:
                        raise HTTPException(status_code=502, detail="Tracker returned an oversized blob.")
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if response.status_code == 200 and content_type not in MEDIA_TYPES:
                        raise HTTPException(
                            status_code=502, detail="Tracker returned a media type the plane will not serve."
                        )
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > MAX_BLOB_BYTES:
                            raise HTTPException(status_code=502, detail="Tracker returned an oversized blob.")
                        content.extend(chunk)
                    # aiter_bytes has already decoded Content-Encoding; do not
                    # copy it into a new response and decode the bytes again.
                    return httpx.Response(
                        response.status_code, content=bytes(content), headers={"Content-Type": content_type}
                    )
        except httpx.HTTPError:
            raise HTTPException(
                status_code=502, detail="Tracker is unreachable or returned invalid JSON; check its status."
            ) from None

    @app.get("/api/v1/tracking/health", tags=["tracking"])
    async def tracking_health(request: Request, principal: _Principal) -> JSONResponse:
        """Read tracker readiness without granting access to any run's data."""

        return await forward(request, principal=principal)

    @app.get("/api/v1/tracking/runs", tags=["tracking"])
    async def tracking_runs(
        request: Request, principal: _Principal, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List runs visible to a project's readers."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}", tags=["tracking"])
    async def tracking_run(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read one run's metadata through its authorized scope."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/series", tags=["tracking"])
    async def tracking_series(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read run series, retaining repeated key query parameters."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/histograms", tags=["tracking"])
    async def tracking_histograms(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read one histogram key as columns over a common bin grid."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/episodes", tags=["tracking"])
    async def tracking_episodes(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List a run's RL episodes, newest first."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/episodes/curve", tags=["tracking"])
    async def tracking_episode_curve(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read bucketed episode return or length with percentile bands."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/episodes/breakdown", tags=["tracking"])
    async def tracking_episode_breakdown(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read per-actor, per-env, actor-lag and eval-by-checkpoint aggregates."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/rollouts", tags=["tracking"])
    async def tracking_rollouts(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List a run's sampled rollouts, newest first."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/rollouts/{rollout_id}", tags=["tracking"])
    async def tracking_rollout(
        request: Request,
        principal: _Principal,
        run_id: _RunId,
        rollout_id: _ItemId,
        org_id: _ScopeId,
        project_id: _ScopeId,
    ) -> JSONResponse:
        """Read one rollout's per-step tracks, actions and video reference."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/completions", tags=["tracking"])
    async def tracking_completions(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List a run's LLM-RL completion groups."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    # Registered before the {completion_id} route: "stats" also matches its pattern.
    @app.get("/api/v1/tracking/runs/{run_id}/completions/stats", tags=["tracking"])
    async def tracking_completion_stats(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read per-step completion aggregates (reward, zero-variance share, tokens)."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/completions/{completion_id}", tags=["tracking"])
    async def tracking_completion(
        request: Request,
        principal: _Principal,
        run_id: _RunId,
        completion_id: _ItemId,
        org_id: _ScopeId,
        project_id: _ScopeId,
    ) -> JSONResponse:
        """Read one completion group with every response."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/tables", tags=["tracking"])
    async def tracking_tables(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List table keys, or read the latest table for one key."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/media", tags=["tracking"])
    async def tracking_media(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List a run's logged media, newest first."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id)

    @app.get("/api/v1/tracking/runs/{run_id}/blobs/{sha256}", tags=["tracking"], response_model=None)
    async def tracking_blob(
        request: Request,
        principal: _Principal,
        run_id: _RunId,
        sha256: _Sha256,
        org_id: _ScopeId,
        project_id: _ScopeId,
    ) -> Response:
        """Stream one referenced media blob as inert, sandboxed bytes."""

        return await forward_blob(request, principal=principal, org_id=org_id, project_id=project_id)
