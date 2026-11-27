# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Authenticated, scope- and visibility-authorized reads and comparisons from the run tracker."""

from __future__ import annotations

import asyncio
import json
import math
import typing as tp

import httpx
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from eyvan.config import EyvanSettings
from eyvan.records import Permission, Principal
from eyvan.server.api.auth import authenticated_principal
from eyvan.server.local.daemon import LocalDaemonError
from eyvan.server.local.tracker import LocalTrackerError, read_tracker_record, read_tracker_token
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.collaboration import CollaborationStore, can_view
from eyvan.server.service.errors import AuthorizationError
from eyvan.track._format import MEDIA_TYPES, validate_run_id
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
_PRIVATE_RUN_DETAIL = "You do not have permission to view this run."
_RunId = tp.Annotated[str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
_ScopeId = tp.Annotated[str, Query(min_length=1, max_length=128)]
_Principal = tp.Annotated[Principal, Depends(authenticated_principal)]

_CONTROL_PLANE_RUN_FILTERS = frozenset({"org_id", "project_id", "owner", "visibility", "experiment_id", "group"})
_MAX_FLATTEN_DEPTH = 16


class RunCompareRequest(BaseModel):
    """Request payload for ``POST /api/v1/tracking/compare``."""

    model_config = ConfigDict(extra="forbid")

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    run_ids: list[str] = Field(min_length=1, max_length=12)
    keys: list[str] = Field(default_factory=list, max_length=32)
    x: str = Field(default="step", min_length=1, max_length=128)
    points: int = Field(default=300, ge=1, le=2000)


def require_project_member(
    service: ControlPlaneService,
    principal: Principal,
    *,
    org_id: str,
    project_id: str,
) -> None:
    """Require at least ``READ`` or ``SUBMIT`` permission on ``(org_id, project_id)``."""

    if service.authorizer.allows(principal, Permission.READ, org_id=org_id, project_id=project_id):
        return
    if service.authorizer.allows(principal, Permission.SUBMIT, org_id=org_id, project_id=project_id):
        return
    raise AuthorizationError(f"{principal.subject!r} lacks project access for {org_id}/{project_id}")


def is_project_admin(
    service: ControlPlaneService,
    principal: Principal,
    *,
    org_id: str,
    project_id: str,
) -> bool:
    """Return whether ``principal`` holds fleet or project management authority."""

    return bool(
        service.authorizer.allows(principal, Permission.MANAGE_FLEET)
        or service.authorizer.allows(principal, Permission.MANAGE_PROJECT, org_id=org_id, project_id=project_id)
    )


def resolve_tracker_credentials(settings: EyvanSettings) -> tuple[str, str | None]:
    """Resolve the upstream tracker URL and bearer token, or raise HTTP 503."""

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
    return url, token


async def resolve_workload_owners(
    service: ControlPlaneService,
    workload_ids: tp.Sequence[str],
    *,
    org_id: str,
    project_id: str,
) -> dict[str, str]:
    """Resolve ``{workload_id: created_by}`` for runs missing an explicit owner."""

    unique_ids = list(dict.fromkeys(wid for wid in workload_ids if isinstance(wid, str) and wid))
    if not unique_ids:
        return {}
    owners: dict[str, str] = {}
    try:
        async with service.repository.transaction() as tx:
            for workload_id in unique_ids:
                record = await tx.get_workload(workload_id)
                if (
                    record is not None
                    and record.org_id == org_id
                    and record.project_id == project_id
                    and record.created_by
                ):
                    owners[workload_id] = record.created_by
    except Exception:
        return owners
    return owners


def enrich_run_record(
    run: dict[str, tp.Any],
    overlay: dict[str, tp.Any] | None,
    *,
    workload_owner: str | None = None,
    notes_count: int = 0,
) -> dict[str, tp.Any]:
    """Merge a tracker run summary/detail dict with its durable collaboration overlay."""

    merged = dict(run)
    overlay_owner = overlay.get("owner") if overlay is not None else None
    merged_owner = overlay_owner or merged.get("owner") or workload_owner or None
    merged_visibility = overlay["visibility"] if overlay is not None else (merged.get("visibility") or "team")
    merged_shared = list(overlay["shared_with"]) if overlay is not None else list(merged.get("shared_with") or [])
    merged_experiment_id = overlay.get("experiment_id") if overlay is not None else merged.get("experiment_id")
    merged_pinned = bool(overlay["pinned"]) if overlay is not None else bool(merged.get("pinned", False))

    merged["owner"] = merged_owner
    merged["visibility"] = merged_visibility
    merged["shared_with"] = merged_shared
    merged["experiment_id"] = merged_experiment_id
    merged["pinned"] = merged_pinned
    merged["notes_count"] = notes_count
    return merged


async def enrich_and_filter_runs(
    runs: list[tp.Any],
    *,
    service: ControlPlaneService,
    store: CollaborationStore | None,
    principal: Principal,
    org_id: str,
    project_id: str,
    owner: str | None = None,
    visibility: str | None = None,
    experiment_id: str | None = None,
    group: str | None = None,
) -> list[dict[str, tp.Any]]:
    """Merge runs with overlays, workload owners, and note counts, then apply RBAC and filters."""

    overlays = store.list_run_overlays(org_id, project_id) if store is not None else {}
    note_counts = store.count_notes_by_target(org_id, project_id, "run") if store is not None else {}
    missing_owner_workloads: list[str] = []
    for item in runs:
        if not isinstance(item, dict):
            continue
        rid = item.get("run_id")
        overlay = overlays.get(rid) if isinstance(rid, str) else None
        has_owner = bool((overlay and overlay.get("owner")) or item.get("owner"))
        wid = item.get("workload_id")
        if not has_owner and isinstance(wid, str) and wid:
            missing_owner_workloads.append(wid)
    workload_owners = await resolve_workload_owners(
        service, missing_owner_workloads, org_id=org_id, project_id=project_id
    )
    admin = is_project_admin(service, principal, org_id=org_id, project_id=project_id)

    visible: list[dict[str, tp.Any]] = []
    for item in runs:
        if not isinstance(item, dict):
            continue
        rid = item.get("run_id")
        overlay = overlays.get(rid) if isinstance(rid, str) else None
        wid = item.get("workload_id")
        wl_owner = workload_owners.get(wid) if isinstance(wid, str) else None
        enriched = enrich_run_record(
            item,
            overlay,
            workload_owner=wl_owner,
            notes_count=note_counts.get(rid, 0) if isinstance(rid, str) else 0,
        )
        if not can_view(
            principal.subject,
            owner=enriched["owner"],
            visibility=enriched["visibility"],
            shared_with=enriched["shared_with"],
            is_admin=admin,
        ):
            continue
        if owner is not None and enriched.get("owner") != owner:
            continue
        if visibility is not None and enriched.get("visibility") != visibility:
            continue
        if experiment_id is not None and enriched.get("experiment_id") != experiment_id:
            continue
        if group is not None and enriched.get("group") != group:
            continue
        visible.append(enriched)
    return visible


async def tracker_get_json(
    app: FastAPI,
    settings: EyvanSettings,
    path: str,
    params: tp.Sequence[tuple[str, str | int]],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[int, tp.Any]:
    """Issue a GET request to the configured tracker and return ``(status_code, json_payload)``."""

    url, token = resolve_tracker_credentials(settings)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    target = f"{url.rstrip('/')}{path}"
    try:
        if client is not None:
            response = await client.get(target, params=list(params), headers=headers)
        else:
            selected_transport = getattr(app.state, "tracking_transport", transport)
            async with httpx.AsyncClient(
                transport=selected_transport, timeout=30.0, trust_env=False, follow_redirects=False
            ) as owned_client:
                response = await owned_client.get(target, params=list(params), headers=headers)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=502,
            detail="Run tracker timed out before responding; check `eyvan track status` or retry shortly.",
        ) from None
    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Run tracker is unreachable; check `eyvan track status` or restart with `eyvan track up --local`.",
        ) from None
    try:
        payload = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Run tracker returned an invalid JSON response; check `eyvan track status` or its logs.",
        ) from None
    return response.status_code, payload


def _flatten_config(config: tp.Any, prefix: str = "", depth: int = 0) -> dict[str, tp.Any]:
    """Flatten nested config mappings into dot-separated keys."""

    if not isinstance(config, dict) or depth >= _MAX_FLATTEN_DEPTH:
        return {}
    out: dict[str, tp.Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value and depth + 1 < _MAX_FLATTEN_DEPTH:
            out.update(_flatten_config(value, prefix=full_key, depth=depth + 1))
        else:
            out[full_key] = value
    return out


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def _run_keys_list(run: dict[str, tp.Any]) -> list[str]:
    raw_keys = run.get("keys")
    if isinstance(raw_keys, list) and all(isinstance(k, str) for k in raw_keys):
        return list(raw_keys)
    latest = run.get("latest_metrics")
    if isinstance(latest, dict):
        return sorted(str(k) for k in latest)
    return []


def register_tracking_routes(
    app: FastAPI,
    *,
    settings: EyvanSettings,
    service: ControlPlaneService,
    collaboration_store: CollaborationStore | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Register the tracker proxy and multi-run compare endpoint.

    ``app.state.tracking_transport`` can supply an HTTP transport for isolated
    tests after the main application factory has run. Neither seam changes
    authentication or scope/visibility authorization. Upstream redirects are
    never followed with the bearer token.
    """

    def _store(request: Request) -> CollaborationStore | None:
        return collaboration_store or getattr(request.app.state, "collaboration_store", None)

    def _enforce_overlay_access(
        request: Request,
        *,
        principal: Principal,
        org_id: str,
        project_id: str,
        run_id: str,
    ) -> dict[str, tp.Any] | None:
        store = _store(request)
        if store is None:
            return None
        overlay = store.get_run_overlay(org_id, project_id, run_id)
        if overlay is not None:
            admin = is_project_admin(service, principal, org_id=org_id, project_id=project_id)
            if not can_view(
                principal.subject,
                owner=overlay["owner"],
                visibility=overlay["visibility"],
                shared_with=overlay["shared_with"],
                is_admin=admin,
            ):
                raise HTTPException(status_code=403, detail=_PRIVATE_RUN_DETAIL)
        return overlay

    async def upstream(
        request: Request,
        *,
        principal: Principal,
        org_id: str | None = None,
        project_id: str | None = None,
        run_id: str | None = None,
        blob: bool = False,
        strip_params: frozenset[str] = frozenset({"org_id", "project_id"}),
    ) -> httpx.Response:
        if org_id is not None and project_id is not None:
            require_project_member(service, principal, org_id=org_id, project_id=project_id)
            if run_id is not None:
                _enforce_overlay_access(
                    request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id
                )
        url, token = resolve_tracker_credentials(settings)
        # Keep repeated metric keys, but bind the forwarded scope to exactly the
        # values we authorized, regardless of duplicate org/project parameters.
        params = [(key, value) for key, value in request.query_params.multi_items() if key not in strip_params]
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
                    return httpx.Response(
                        response.status_code, content=bytes(content), headers={"Content-Type": content_type}
                    )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=502,
                detail="Run tracker timed out before responding; check `eyvan track status` or retry shortly.",
            ) from None
        except httpx.HTTPError:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Run tracker is unreachable; check `eyvan track status` or restart with `eyvan track up --local`."
                ),
            ) from None

    async def forward(
        request: Request,
        *,
        principal: Principal,
        org_id: str | None = None,
        project_id: str | None = None,
        run_id: str | None = None,
    ) -> JSONResponse:
        response = await upstream(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)
        try:
            payload = response.json()
        except ValueError:
            raise HTTPException(
                status_code=502,
                detail="Run tracker returned an invalid JSON response; check `eyvan track status` or its logs.",
            ) from None
        return JSONResponse(status_code=response.status_code, content=payload)

    async def forward_blob(
        request: Request, *, principal: Principal, org_id: str, project_id: str, run_id: str
    ) -> Response:
        response = await upstream(
            request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id, blob=True
        )
        if response.status_code != 200:
            try:
                payload = response.json()
            except ValueError:
                payload = {"detail": "Tracker refused the blob request"}
            return JSONResponse(status_code=response.status_code, content=payload)
        return Response(
            content=response.content, media_type=response.headers["content-type"], headers=dict(_BLOB_HEADERS)
        )

    @app.get("/api/v1/tracking/health", tags=["tracking"])
    async def tracking_health(request: Request, principal: _Principal) -> JSONResponse:
        """Read tracker readiness without granting access to any run's data."""

        return await forward(request, principal=principal)

    @app.get("/api/v1/tracking/runs", tags=["tracking"])
    async def tracking_runs(
        request: Request,
        principal: _Principal,
        org_id: _ScopeId,
        project_id: _ScopeId,
        owner: str | None = Query(default=None, max_length=128),
        visibility: str | None = Query(default=None, max_length=32),
        experiment_id: str | None = Query(default=None, max_length=128),
        group: str | None = Query(default=None, max_length=256),
    ) -> JSONResponse:
        """List runs visible to a project member, enriched with collaboration metadata."""

        response = await upstream(
            request,
            principal=principal,
            org_id=org_id,
            project_id=project_id,
            strip_params=_CONTROL_PLANE_RUN_FILTERS,
        )
        try:
            payload = response.json()
        except ValueError:
            raise HTTPException(
                status_code=502,
                detail="Run tracker returned an invalid JSON response; check `eyvan track status` or its logs.",
            ) from None
        if response.status_code == 200 and isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            filtered = await enrich_and_filter_runs(
                payload["runs"],
                service=service,
                store=_store(request),
                principal=principal,
                org_id=org_id,
                project_id=project_id,
                owner=owner,
                visibility=visibility,
                experiment_id=experiment_id,
                group=group,
            )
            return JSONResponse(status_code=200, content={**payload, "runs": filtered})
        return JSONResponse(status_code=response.status_code, content=payload)

    @app.post("/api/v1/tracking/compare", tags=["tracking"])
    async def tracking_compare(
        request: Request,
        body: RunCompareRequest,
        principal: _Principal,
    ) -> JSONResponse:
        """Compare multiple visible runs: aligned metric series, summary stats, and hyperparameter diff."""

        require_project_member(service, principal, org_id=body.org_id, project_id=body.project_id)
        deduped_ids: list[str] = []
        seen_ids: set[str] = set()
        for raw_id in body.run_ids:
            try:
                valid = validate_run_id(raw_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            if valid not in seen_ids:
                seen_ids.add(valid)
                deduped_ids.append(valid)

        store = _store(request)
        admin = is_project_admin(service, principal, org_id=body.org_id, project_id=body.project_id)
        overlays: dict[str, dict[str, tp.Any] | None] = {}
        for rid in deduped_ids:
            overlay = store.get_run_overlay(body.org_id, body.project_id, rid) if store is not None else None
            overlays[rid] = overlay
            if overlay is not None and not can_view(
                principal.subject,
                owner=overlay["owner"],
                visibility=overlay["visibility"],
                shared_with=overlay["shared_with"],
                is_admin=admin,
            ):
                raise HTTPException(status_code=403, detail=_PRIVATE_RUN_DETAIL)

        resolve_tracker_credentials(settings)
        selected_transport = getattr(request.app.state, "tracking_transport", transport)
        scope_params: list[tuple[str, str | int]] = [("org_id", body.org_id), ("project_id", body.project_id)]

        async with httpx.AsyncClient(
            transport=selected_transport, timeout=30.0, trust_env=False, follow_redirects=False
        ) as client:
            detail_responses = await asyncio.gather(
                *(
                    tracker_get_json(
                        app,
                        settings,
                        f"/api/v1/tracking/runs/{rid}",
                        scope_params,
                        client=client,
                    )
                    for rid in deduped_ids
                )
            )

            raw_runs: list[dict[str, tp.Any]] = []
            missing_workloads: list[str] = []
            for rid, (status_code, payload) in zip(deduped_ids, detail_responses, strict=True):
                if status_code == 404:
                    raise HTTPException(status_code=404, detail=f"Tracking run {rid!r} not found")
                if status_code != 200 or not isinstance(payload, dict):
                    detail = payload.get("detail") if isinstance(payload, dict) else "Tracker returned an error"
                    raise HTTPException(status_code=status_code if 400 <= status_code < 600 else 502, detail=detail)
                raw_runs.append(payload)
                overlay = overlays.get(rid)
                has_owner = bool((overlay and overlay.get("owner")) or payload.get("owner"))
                wid = payload.get("workload_id")
                if not has_owner and isinstance(wid, str) and wid:
                    missing_workloads.append(wid)

            workload_owners = await resolve_workload_owners(
                service, missing_workloads, org_id=body.org_id, project_id=body.project_id
            )
            note_counts = store.count_notes_by_target(body.org_id, body.project_id, "run") if store is not None else {}

            enriched_runs: list[dict[str, tp.Any]] = []
            for rid, raw_run in zip(deduped_ids, raw_runs, strict=True):
                wid = raw_run.get("workload_id")
                enriched = enrich_run_record(
                    raw_run,
                    overlays.get(rid),
                    workload_owner=workload_owners.get(wid) if isinstance(wid, str) else None,
                    notes_count=note_counts.get(rid, 0),
                )
                if not can_view(
                    principal.subject,
                    owner=enriched["owner"],
                    visibility=enriched["visibility"],
                    shared_with=enriched["shared_with"],
                    is_admin=admin,
                ):
                    raise HTTPException(status_code=403, detail=_PRIVATE_RUN_DETAIL)
                enriched_runs.append(enriched)

            key_sets = [set(_run_keys_list(run)) for run in enriched_runs]
            all_keys = sorted(set().union(*key_sets)) if key_sets else []
            common_keys = sorted(set.intersection(*key_sets)) if key_sets else []
            if body.keys:
                selected_keys = list(dict.fromkeys(k for k in body.keys if isinstance(k, str) and k))[:32]
            else:
                selected_keys = common_keys[:12] if common_keys else all_keys[:12]

            series_by_key: dict[str, dict[str, dict[str, list[tp.Any]]]] = {k: {} for k in selected_keys}
            if selected_keys:

                async def _fetch_run_series(rid: str, run_keys: set[str]) -> tuple[str, list[tp.Any]]:
                    keys_for_run = [k for k in selected_keys if not run_keys or k in run_keys]
                    if not keys_for_run:
                        return rid, []
                    params: list[tuple[str, str | int]] = [
                        ("org_id", body.org_id),
                        ("project_id", body.project_id),
                        ("x", body.x),
                        ("points", body.points),
                        *(("key", k) for k in keys_for_run),
                    ]
                    code, data = await tracker_get_json(
                        app,
                        settings,
                        f"/api/v1/tracking/runs/{rid}/series",
                        params,
                        client=client,
                    )
                    if code == 200 and isinstance(data, dict) and isinstance(data.get("series"), list):
                        return rid, data["series"]
                    return rid, []

                series_results = await asyncio.gather(
                    *(_fetch_run_series(rid, run_keys) for rid, run_keys in zip(deduped_ids, key_sets, strict=True))
                )
                for rid, series_list in series_results:
                    for item in series_list:
                        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
                            continue
                        key = item["key"]
                        xs: list[tp.Any] = []
                        ys: list[tp.Any] = []
                        mins: list[tp.Any] = []
                        maxs: list[tp.Any] = []
                        for point in item.get("points") or []:
                            if not isinstance(point, list) or len(point) < 2:
                                continue
                            xs.append(point[0])
                            ys.append(point[1])
                            mins.append(point[2] if len(point) >= 3 else point[1])
                            maxs.append(point[3] if len(point) >= 4 else point[1])
                        series_by_key.setdefault(key, {})[rid] = {
                            "x": xs,
                            "y": ys,
                            "min": mins,
                            "max": maxs,
                        }

        # Config diff table (flattened dot-separated keys, differing keys first).
        flat_configs = {
            rid: _flatten_config(run.get("config")) for rid, run in zip(deduped_ids, enriched_runs, strict=True)
        }
        all_config_keys = sorted({key for flat in flat_configs.values() for key in flat})
        config_diff: list[dict[str, tp.Any]] = []
        for cfg_key in all_config_keys:
            values = {rid: flat_configs[rid].get(cfg_key) for rid in deduped_ids}
            signatures = {
                (cfg_key in flat_configs[rid], json.dumps(flat_configs[rid].get(cfg_key), sort_keys=True, default=str))
                for rid in deduped_ids
            }
            config_diff.append({"key": cfg_key, "values": values, "differs": len(signatures) > 1})
        config_diff.sort(key=lambda entry: (not entry["differs"], entry["key"]))

        # Metric summary table across selected keys.
        summary_keys = selected_keys if selected_keys else all_keys[:12]
        metric_summary: list[dict[str, tp.Any]] = []
        for key in summary_keys:
            per_run: dict[str, dict[str, tp.Any]] = {}
            for rid, run in zip(deduped_ids, enriched_runs, strict=True):
                series = series_by_key.get(key, {}).get(rid)
                finite_y: list[float] = []
                finite_min: list[float] = []
                finite_max: list[float] = []
                last_x: tp.Any = None
                if series:
                    xs = series["x"]
                    ys = series["y"]
                    mins = series["min"]
                    maxs = series["max"]
                    for idx, raw_y in enumerate(ys):
                        val_y = _finite_number(raw_y)
                        val_mn = _finite_number(mins[idx]) if idx < len(mins) else val_y
                        val_mx = _finite_number(maxs[idx]) if idx < len(maxs) else val_y
                        if val_y is not None:
                            finite_y.append(val_y)
                            last_x = xs[idx] if idx < len(xs) else last_x
                        if val_mn is not None:
                            finite_min.append(val_mn)
                        if val_mx is not None:
                            finite_max.append(val_mx)
                if finite_y:
                    per_run[rid] = {
                        "latest": finite_y[-1],
                        "min": min(finite_min) if finite_min else min(finite_y),
                        "max": max(finite_max) if finite_max else max(finite_y),
                        "last_step": last_x if body.x == "step" else run.get("last_step"),
                    }
                else:
                    fallback = None
                    latest_map = run.get("latest_metrics")
                    if isinstance(latest_map, dict):
                        fallback = _finite_number(latest_map.get(key))
                    if fallback is None and isinstance(run.get("summary"), dict):
                        fallback = _finite_number(run["summary"].get(key))
                    if fallback is not None:
                        per_run[rid] = {
                            "latest": fallback,
                            "min": fallback,
                            "max": fallback,
                            "last_step": run.get("last_step"),
                        }
                    else:
                        per_run[rid] = {"latest": None, "min": None, "max": None, "last_step": None}
            metric_summary.append({"key": key, "values": per_run})

        return JSONResponse(
            status_code=200,
            content={
                "org_id": body.org_id,
                "project_id": body.project_id,
                "x": body.x,
                "runs": enriched_runs,
                "common_keys": common_keys,
                "all_keys": all_keys,
                "selected_keys": selected_keys,
                "series_by_key": series_by_key,
                "config_diff": config_diff,
                "metric_summary": metric_summary,
            },
        )

    @app.get("/api/v1/tracking/runs/{run_id}", tags=["tracking"])
    async def tracking_run(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read one run's metadata through its authorized scope and visibility policy."""

        response = await upstream(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)
        try:
            payload = response.json()
        except ValueError:
            raise HTTPException(
                status_code=502,
                detail="Run tracker returned an invalid JSON response; check `eyvan track status` or its logs.",
            ) from None
        if response.status_code == 200 and isinstance(payload, dict) and "run_id" in payload:
            store = _store(request)
            overlay = store.get_run_overlay(org_id, project_id, run_id) if store is not None else None
            wl_owner: str | None = None
            has_owner = bool((overlay and overlay.get("owner")) or payload.get("owner"))
            wid = payload.get("workload_id")
            if not has_owner and isinstance(wid, str) and wid:
                resolved = await resolve_workload_owners(service, [wid], org_id=org_id, project_id=project_id)
                wl_owner = resolved.get(wid)
            note_counts = store.count_notes_by_target(org_id, project_id, "run") if store is not None else {}
            enriched = enrich_run_record(
                payload,
                overlay,
                workload_owner=wl_owner,
                notes_count=note_counts.get(run_id, 0),
            )
            admin = is_project_admin(service, principal, org_id=org_id, project_id=project_id)
            if not can_view(
                principal.subject,
                owner=enriched["owner"],
                visibility=enriched["visibility"],
                shared_with=enriched["shared_with"],
                is_admin=admin,
            ):
                raise HTTPException(status_code=403, detail=_PRIVATE_RUN_DETAIL)
            return JSONResponse(status_code=200, content=enriched)
        return JSONResponse(status_code=response.status_code, content=payload)

    @app.get("/api/v1/tracking/runs/{run_id}/series", tags=["tracking"])
    async def tracking_series(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read run series, retaining repeated key query parameters."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

    @app.get("/api/v1/tracking/runs/{run_id}/histograms", tags=["tracking"])
    async def tracking_histograms(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read one histogram key as columns over a common bin grid."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

    @app.get("/api/v1/tracking/runs/{run_id}/episodes", tags=["tracking"])
    async def tracking_episodes(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List a run's RL episodes, newest first."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

    @app.get("/api/v1/tracking/runs/{run_id}/episodes/curve", tags=["tracking"])
    async def tracking_episode_curve(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read bucketed episode return or length with percentile bands."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

    @app.get("/api/v1/tracking/runs/{run_id}/episodes/breakdown", tags=["tracking"])
    async def tracking_episode_breakdown(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read per-actor, per-env, actor-lag and eval-by-checkpoint aggregates."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

    @app.get("/api/v1/tracking/runs/{run_id}/rollouts", tags=["tracking"])
    async def tracking_rollouts(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List a run's sampled rollouts, newest first."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

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

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

    @app.get("/api/v1/tracking/runs/{run_id}/completions", tags=["tracking"])
    async def tracking_completions(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List a run's LLM-RL completion groups."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

    # Registered before the {completion_id} route: "stats" also matches its pattern.
    @app.get("/api/v1/tracking/runs/{run_id}/completions/stats", tags=["tracking"])
    async def tracking_completion_stats(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """Read per-step completion aggregates (reward, zero-variance share, tokens)."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

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

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

    @app.get("/api/v1/tracking/runs/{run_id}/tables", tags=["tracking"])
    async def tracking_tables(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List table keys, or read the latest table for one key."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

    @app.get("/api/v1/tracking/runs/{run_id}/media", tags=["tracking"])
    async def tracking_media(
        request: Request, principal: _Principal, run_id: _RunId, org_id: _ScopeId, project_id: _ScopeId
    ) -> JSONResponse:
        """List a run's logged media, newest first."""

        return await forward(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)

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

        return await forward_blob(request, principal=principal, org_id=org_id, project_id=project_id, run_id=run_id)
