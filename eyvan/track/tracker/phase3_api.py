# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rich tracking query routes, including integrity-checked raw blob responses."""

from __future__ import annotations

import typing as tp

from fastapi import APIRouter, HTTPException, Query, Response

from eyvan.track._store import TrackingStore, TrackingStoreError
from eyvan.track.tracker.index import TrackingIndex
from eyvan.track.tracker.phase3 import BlobIntegrityError

Record = dict[str, tp.Any]


def add_routes(
    routes: APIRouter,
    index: TrackingIndex,
    store: TrackingStore,
    scope: tp.Callable[..., None],
    result: tp.Callable[..., Record],
) -> None:
    """Register phase-three routes under the existing authenticated router."""

    @routes.get("/runs/{run_id}/rollouts")
    def rollouts(
        run_id: str,
        org_id: str,
        project_id: str,
        kind: str | None = None,
        actor: str | None = Query(default=None, max_length=64),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        include_replayed: bool = False,
    ) -> Record:
        scope(org_id, project_id, run_id)
        return result(
            index.rollouts,
            org=org_id,
            project=project_id,
            run_id=run_id,
            kind=kind,
            actor=actor,
            limit=limit,
            offset=offset,
            include_replayed=include_replayed,
            store=store,
        )

    @routes.get("/runs/{run_id}/rollouts/{rollout_id}")
    def rollout_detail(run_id: str, rollout_id: str, org_id: str, project_id: str) -> Record:
        scope(org_id, project_id, run_id)
        return result(
            index.rollout_detail, org=org_id, project=project_id, run_id=run_id, rollout_id=rollout_id, store=store
        )

    @routes.get("/runs/{run_id}/completions")
    def completions(
        run_id: str,
        org_id: str,
        project_id: str,
        step: int | None = Query(default=None, ge=0),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        sort: str = "recent",
        include_replayed: bool = False,
    ) -> Record:
        scope(org_id, project_id, run_id)
        return result(
            index.completions,
            org=org_id,
            project=project_id,
            run_id=run_id,
            step=step,
            limit=limit,
            offset=offset,
            sort=sort,
            include_replayed=include_replayed,
        )

    # Register the literal route before the identifier route.
    @routes.get("/runs/{run_id}/completions/stats")
    def completion_stats(
        run_id: str,
        org_id: str,
        project_id: str,
        points: int = Query(default=200, ge=1, le=2000),
    ) -> Record:
        scope(org_id, project_id, run_id)
        return result(index.completion_stats, org=org_id, project=project_id, run_id=run_id, points=points)

    @routes.get("/runs/{run_id}/completions/{completion_id}")
    def completion_detail(run_id: str, completion_id: str, org_id: str, project_id: str) -> Record:
        scope(org_id, project_id, run_id)
        return result(
            index.completion_detail, org=org_id, project=project_id, run_id=run_id, completion_id=completion_id
        )

    @routes.get("/runs/{run_id}/tables")
    def tables(
        run_id: str,
        org_id: str,
        project_id: str,
        key: str | None = None,
        step: int | None = Query(default=None, ge=0),
    ) -> Record:
        scope(org_id, project_id, run_id)
        return result(index.tables, org=org_id, project=project_id, run_id=run_id, key=key, step=step, store=store)

    @routes.get("/runs/{run_id}/media")
    def media(
        run_id: str,
        org_id: str,
        project_id: str,
        key: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> Record:
        scope(org_id, project_id, run_id)
        return result(
            index.media, org=org_id, project=project_id, run_id=run_id, key=key, limit=limit, offset=offset, store=store
        )

    @routes.get("/runs/{run_id}/blobs/{sha256}")
    def blob(run_id: str, sha256: str, org_id: str, project_id: str) -> Response:
        scope(org_id, project_id, run_id)
        try:
            verified = index.blob(org_id, project_id, run_id, sha256, store=store)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid tracking request parameters") from None
        except BlobIntegrityError:
            raise HTTPException(status_code=500, detail="blob integrity check failed") from None
        except TrackingStoreError:
            raise HTTPException(status_code=502, detail="Tracking blob store unavailable") from None
        if verified is None:
            raise HTTPException(status_code=404, detail="blob not available")
        data, content_type = verified
        return Response(
            content=data,
            media_type=content_type,
            headers={
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'; sandbox",
                "Cache-Control": "private, max-age=31536000, immutable",
                "Content-Disposition": "inline",
            },
        )


__all__ = ("add_routes",)
