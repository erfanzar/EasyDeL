# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read-only tracking HTTP API and periodic object-store indexing."""

from __future__ import annotations

import asyncio
import hmac
import logging
import math
import typing as tp
from contextlib import asynccontextmanager, suppress

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from eyvan.track._format import validate_run_id
from eyvan.track._store import TrackingStore
from eyvan.track.tracker.index import TrackingIndex
from eyvan.track.tracker.phase3_api import add_routes

_LOG = logging.getLogger(__name__)
_BASE = "/api/v1/tracking"


def create_tracker_app(
    store: TrackingStore, index: TrackingIndex, *, token: str | None = None, scan_interval_s: float = 5.0
) -> FastAPI:
    """Create the JSON query and raw-media tracker with optional bearer auth.

    Scans run off the event loop. An initial store outage does not prevent the
    existing disposable index from being queried; later scans retry it. Error
    logs deliberately omit exception messages, which may contain store secrets.
    """

    if not math.isfinite(scan_interval_s) or scan_interval_s <= 0:
        raise ValueError("scan interval must be finite and positive")

    async def scan_once() -> None:
        try:
            await asyncio.to_thread(index.scan, store)
        except Exception:
            _LOG.warning("Tracking store scan failed; check store access and tracker configuration")

    async def scan_loop(stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=scan_interval_s)
            except TimeoutError:
                await scan_once()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> tp.AsyncIterator[None]:
        await scan_once()
        stop = asyncio.Event()
        task = asyncio.create_task(scan_loop(stop))
        try:
            yield
        finally:
            # Let any in-flight scan finish before shutting down; cancellation
            # cannot stop a thread that may still own the index lock.
            stop.set()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Eyvan tracker", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    async def invalid_query(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": "Invalid tracking request parameters"})

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        if not token:
            return
        scheme, _, supplied = (authorization or "").partition(" ")
        valid = hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8"))
        if scheme.lower() != "bearer" or not valid:
            raise HTTPException(status_code=401, detail="Bearer token required", headers={"WWW-Authenticate": "Bearer"})

    def scope(org_id: str, project_id: str, run_id: str | None = None) -> None:
        try:
            validate_run_id(org_id)
            validate_run_id(project_id)
            if run_id is not None:
                validate_run_id(run_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid organization, project or run id") from None

    @app.get(f"{_BASE}/health")
    def health() -> dict[str, tp.Any]:
        return {"status": "ok", "service": "eyvan-tracker", "store": store.root, **index.scan_state()}

    routes = APIRouter(prefix=_BASE, dependencies=[Depends(authenticate)])

    @routes.get("/runs")
    def runs(
        org_id: str, project_id: str, status: str | None = None, limit: int = Query(default=200, ge=1)
    ) -> dict[str, tp.Any]:
        scope(org_id, project_id)
        return {"runs": index.list_runs(org_id, project_id, status=status, limit=limit)}

    @routes.get("/runs/{run_id}")
    def run(run_id: str, org_id: str, project_id: str) -> dict[str, tp.Any]:
        scope(org_id, project_id, run_id)
        detail = index.run_detail(org_id, project_id, run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Tracking run not found")
        return detail

    @routes.get("/runs/{run_id}/series")
    def series(
        run_id: str,
        org_id: str,
        project_id: str,
        key: list[str] = Query(min_length=1, max_length=32),
        x: str = "step",
        points: int = Query(default=1000, ge=1, le=5000),
        rank: int | None = Query(default=None, ge=0, le=9999),
        include_replayed: bool = False,
    ) -> dict[str, tp.Any]:
        scope(org_id, project_id, run_id)
        try:
            result = index.series(
                org_id, project_id, run_id, key, x=x, points=points, rank=rank, include_replayed=include_replayed
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid metric keys, axis or series parameters") from None
        if result is None:
            raise HTTPException(status_code=404, detail="Tracking run not found")
        return result

    def phase_two_result(query: tp.Callable[..., dict[str, tp.Any] | None], **parameters: tp.Any) -> dict[str, tp.Any]:
        try:
            result = query(**parameters)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid tracking request parameters") from None
        if result is None:
            raise HTTPException(status_code=404, detail="Tracking run not found")
        return result

    @routes.get("/runs/{run_id}/histograms")
    def histograms(
        run_id: str,
        org_id: str,
        project_id: str,
        key: str,
        x: str = "step",
        points: int = Query(default=200, ge=1, le=1000),
        bins: int = Query(default=64, ge=1, le=256),
        rank: int | None = Query(default=None, ge=0, le=9999),
    ) -> dict[str, tp.Any]:
        scope(org_id, project_id, run_id)
        return phase_two_result(
            index.histograms,
            org=org_id,
            project=project_id,
            run_id=run_id,
            key=key,
            x=x,
            points=points,
            bins=bins,
            rank=rank,
        )

    @routes.get("/runs/{run_id}/episodes")
    def episodes(
        run_id: str,
        org_id: str,
        project_id: str,
        kind: str | None = None,
        actor: str | None = Query(default=None, max_length=64),
        env: str | None = Query(default=None, max_length=128),
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        include_replayed: bool = False,
    ) -> dict[str, tp.Any]:
        scope(org_id, project_id, run_id)
        return phase_two_result(
            index.episodes,
            org=org_id,
            project=project_id,
            run_id=run_id,
            kind=kind,
            actor=actor,
            env=env,
            limit=limit,
            offset=offset,
            include_replayed=include_replayed,
        )

    @routes.get("/runs/{run_id}/episodes/curve")
    def episode_curve(
        run_id: str,
        org_id: str,
        project_id: str,
        metric: str = "return",
        kind: str = "train",
        x: str = "step",
        points: int = Query(default=200, ge=1, le=2000),
        env: str | None = Query(default=None, max_length=128),
        actor: str | None = Query(default=None, max_length=64),
    ) -> dict[str, tp.Any]:
        scope(org_id, project_id, run_id)
        return phase_two_result(
            index.episode_curve,
            org=org_id,
            project=project_id,
            run_id=run_id,
            metric=metric,
            kind=kind,
            x=x,
            points=points,
            env=env,
            actor=actor,
        )

    @routes.get("/runs/{run_id}/episodes/breakdown")
    def episode_breakdown(
        run_id: str,
        org_id: str,
        project_id: str,
        kind: str = "train",
        last: int = Query(default=200, ge=1, le=5000),
    ) -> dict[str, tp.Any]:
        scope(org_id, project_id, run_id)
        return phase_two_result(
            index.episode_breakdown,
            org=org_id,
            project=project_id,
            run_id=run_id,
            kind=kind,
            last=last,
        )

    add_routes(routes, index, store, scope, phase_two_result)
    app.include_router(routes)
    return app


__all__ = ("create_tracker_app",)
