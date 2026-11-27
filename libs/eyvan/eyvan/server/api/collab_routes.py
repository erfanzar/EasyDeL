# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Collaborative runs, experiments/speedruns, team notes, activity feed, and storage region check routes."""

from __future__ import annotations

import asyncio
import math
import typing as tp

import httpx
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from eyvan.config import EyvanSettings
from eyvan.records import Permission, Principal
from eyvan.server.api.auth import authenticated_principal
from eyvan.server.api.tracking_proxy import (
    authorize_run_access,
    enrich_runs_snapshot,
    fetch_tracker_run_metadata,
    is_project_admin,
    require_project_member,
    require_project_writer,
    resolve_run_workload_owner,
    resolve_tracker_credentials,
    sanitize_json_value,
    tracker_get_json,
)
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.collaboration import (
    _UNSET,
    VALID_NOTE_TARGETS,
    CollaborationStore,
    can_manage,
    can_view,
)
from eyvan.server.service.errors import AuthorizationError
from eyvan.storage import (
    check_gcs_uri_locality,
    default_gcp_token_provider,
    detect_gcp_context,
    extract_bucket_from_gcs_http_url,
    extract_bucket_from_gcs_uri,
    resolve_bucket_location,
    validate_bucket_name,
    validate_region,
    zone_to_region,
)

_Principal = tp.Annotated[Principal, Depends(authenticated_principal)]
_ScopeId = tp.Annotated[str, Query(min_length=1, max_length=128, pattern=r"\S")]
_RunId = tp.Annotated[str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
_ExpId = tp.Annotated[str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
_NoteId = tp.Annotated[str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]

_ACTIVE_PERMISSIONS: tuple[Permission, ...] = (
    Permission.READ,
    Permission.SUBMIT,
    Permission.OPERATE,
    Permission.MANAGE_PROJECT,
    Permission.MANAGE_FLEET,
)
_LEADERBOARD_CONCURRENCY = 16


def require_any_active_permission(service: ControlPlaneService, principal: Principal) -> None:
    """Require that ``principal`` holds at least one active Eyvan permission globally or in some scope."""

    if service.authorizer.allows(principal, Permission.MANAGE_FLEET):
        return
    for grant in principal.grants:
        for permission in _ACTIVE_PERMISSIONS:
            if service.authorizer.allows(principal, permission, org_id=grant.org_id, project_id=grant.project_id):
                return
    raise AuthorizationError(f"{principal.subject!r} holds no active Eyvan permissions")


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSharingUpdateRequest(_Body):
    """Partial update to one run's visibility, selective sharing, experiment link, or pin."""

    visibility: tp.Literal["private", "shared", "team"] | None = None
    shared_with: list[str] | None = Field(default=None, max_length=64)
    experiment_id: str | None = Field(default=None, max_length=128)
    pinned: bool | None = None
    owner: str | None = Field(default=None, max_length=128)


class ExperimentCreateRequest(_Body):
    """Create a collaborative experiment or speedrun leaderboard."""

    org_id: str = Field(min_length=1, max_length=128, pattern=r"\S")
    project_id: str = Field(min_length=1, max_length=128, pattern=r"\S")
    name: str = Field(min_length=1, max_length=160, pattern=r"\S")
    subject: str = Field(default="", max_length=120)
    kind: tp.Literal["experiment", "speedrun"] = "experiment"
    description: str = Field(default="", max_length=4000)
    metric_key: str | None = Field(default=None, max_length=128)
    goal: tp.Literal["min", "max"] = "min"
    target_value: float | None = None
    group_filter: str | None = Field(default=None, max_length=256)
    tag_filter: list[str] = Field(default_factory=list, max_length=32)
    visibility: tp.Literal["private", "shared", "team"] = "team"
    shared_with: list[str] = Field(default_factory=list, max_length=64)
    pinned_baseline_run_id: str | None = Field(default=None, max_length=128)


class ExperimentUpdateRequest(_Body):
    """Partial update to an experiment or speedrun."""

    name: str | None = Field(default=None, max_length=160)
    subject: str | None = Field(default=None, max_length=120)
    kind: tp.Literal["experiment", "speedrun"] | None = None
    description: str | None = Field(default=None, max_length=4000)
    metric_key: str | None = Field(default=None, max_length=128)
    goal: tp.Literal["min", "max"] | None = None
    target_value: float | None = None
    group_filter: str | None = Field(default=None, max_length=256)
    tag_filter: list[str] | None = Field(default=None, max_length=32)
    visibility: tp.Literal["private", "shared", "team"] | None = None
    shared_with: list[str] | None = Field(default=None, max_length=64)
    pinned_baseline_run_id: str | None = Field(default=None, max_length=128)


class NoteCreateRequest(_Body):
    """Create a team note attached to a run or experiment."""

    org_id: str = Field(min_length=1, max_length=128, pattern=r"\S")
    project_id: str = Field(min_length=1, max_length=128, pattern=r"\S")
    target_type: tp.Literal["run", "experiment"]
    target_id: str = Field(min_length=1, max_length=128, pattern=r"\S")
    body: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    metric_ref: str | None = Field(default=None, max_length=128)
    step_ref: int | None = Field(default=None, ge=0)


class StorageCheckRegionRequest(_Body):
    """Request body for ``POST /api/v1/storage/check-region``."""

    uri: str = Field(min_length=1, max_length=2048, pattern=r"\S")
    region: str | None = Field(default=None, max_length=128)
    strict: bool = False
    allowed_buckets: list[str] = Field(default_factory=list, max_length=64)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def _run_matches_experiment(run: dict[str, tp.Any], exp: dict[str, tp.Any]) -> bool:
    if run.get("experiment_id") == exp["id"]:
        return True
    if exp.get("group_filter") and run.get("group") == exp["group_filter"]:
        return True
    tag_filter = exp.get("tag_filter") or []
    if tag_filter:
        run_tags = set(run.get("tags") or [])
        if set(tag_filter) <= run_tags:
            return True
    if run.get("experiment") and run.get("experiment") == exp["name"]:
        return True
    return False


def _metric_from_run_snapshot(run: dict[str, tp.Any], metric_key: str) -> float | None:
    latest = run.get("latest_metrics")
    if isinstance(latest, dict):
        val = _finite_number(latest.get(metric_key))
        if val is not None:
            return val
    summary = run.get("summary")
    if isinstance(summary, dict):
        val = _finite_number(summary.get(metric_key))
        if val is not None:
            return val
    return None


def _target_reached(value: float, *, goal: str, target_value: float) -> bool:
    return value <= target_value if goal == "min" else value >= target_value


def _is_better(candidate: float, current: float, *, goal: str) -> bool:
    return candidate < current if goal == "min" else candidate > current


def _analyze_series_for_leaderboard(
    step_points: list[tp.Any],
    time_points: list[tp.Any],
    *,
    goal: str,
    target_value: float | None,
    started_at: float | None,
    duration_s: float | None,
) -> tuple[float | None, float | None, bool, int | None, float | None]:
    """Compute ``(best_value, latest_value, reached_target, steps_to_target, time_to_target_s)``."""

    best_value: float | None = None
    latest_value: float | None = None
    reached_target = False
    steps_to_target: int | None = None

    for point in step_points:
        if not isinstance(point, list) or len(point) < 2:
            continue
        x_val = _finite_number(point[0])
        y_mean = _finite_number(point[1])
        y_min = _finite_number(point[2]) if len(point) >= 3 else y_mean
        y_max = _finite_number(point[3]) if len(point) >= 4 else y_mean
        if y_mean is not None:
            latest_value = y_mean
        candidate = y_min if goal == "min" else y_max
        if candidate is None:
            candidate = y_mean
        if candidate is None:
            continue
        if best_value is None or _is_better(candidate, best_value, goal=goal):
            best_value = candidate
        if (
            target_value is not None
            and not reached_target
            and _target_reached(candidate, goal=goal, target_value=target_value)
        ):
            reached_target = True
            if x_val is not None:
                steps_to_target = int(x_val)

    time_to_target_s: float | None = None
    if reached_target and target_value is not None:
        first_t: float | None = None
        hit_t: float | None = None
        for point in time_points:
            if not isinstance(point, list) or len(point) < 2:
                continue
            t_val = _finite_number(point[0])
            if t_val is not None and first_t is None:
                first_t = t_val
            y_mean = _finite_number(point[1])
            y_min = _finite_number(point[2]) if len(point) >= 3 else y_mean
            y_max = _finite_number(point[3]) if len(point) >= 4 else y_mean
            candidate = y_min if goal == "min" else y_max
            if candidate is None:
                candidate = y_mean
            if candidate is not None and _target_reached(candidate, goal=goal, target_value=target_value):
                hit_t = t_val
                break
        if hit_t is not None:
            if started_at is not None and hit_t >= started_at:
                time_to_target_s = round(hit_t - started_at, 3)
            elif first_t is not None:
                time_to_target_s = round(max(0.0, hit_t - first_t), 3)
        if time_to_target_s is None and duration_s is not None:
            time_to_target_s = duration_s

    return best_value, latest_value, reached_target, steps_to_target, time_to_target_s


def _normalize_region_or_zone(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("region must be non-empty")
    try:
        return validate_region(cleaned)
    except ValueError:
        return zone_to_region(cleaned)


def register_collaboration_routes(
    app: FastAPI,
    *,
    settings: EyvanSettings,
    service: ControlPlaneService,
    store: CollaborationStore,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Register collaborative runs, experiments/speedruns, team notes, feed, and region-check routes."""

    async def _fetch_runs_snapshot(
        *,
        principal: Principal,
        org_id: str,
        project_id: str,
        limit: int = 500,
    ) -> tuple[bool, bool, dict[str, dict[str, tp.Any]], list[dict[str, tp.Any]]]:
        """Fetch and enrich runs from tracker.

        Returns ``(tracker_available, configured_unavailable, all_by_id, visible)``.
        """

        try:
            status_code, payload = await tracker_get_json(
                app,
                settings,
                "/api/v1/tracking/runs",
                [("org_id", org_id), ("project_id", project_id), ("limit", limit)],
                transport=transport,
            )
        except HTTPException as exc:
            if exc.status_code == 503:
                return False, False, {}, []
            if exc.status_code == 502:
                return False, True, {}, []
            raise
        if status_code != 200 or not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            return False, True, {}, []
        all_enriched, visible = await enrich_runs_snapshot(
            payload["runs"],
            service=service,
            store=store,
            principal=principal,
            org_id=org_id,
            project_id=project_id,
        )
        return True, False, all_enriched, visible

    async def _filter_visible_notes(
        notes: list[dict[str, tp.Any]],
        *,
        principal: Principal,
        admin: bool,
        org_id: str,
        project_id: str,
        overlays: dict[str, dict[str, tp.Any]],
        experiments_by_id: dict[str, dict[str, tp.Any]],
        tracker_runs_by_id: dict[str, dict[str, tp.Any]] | None = None,
        tracker_configured_unavailable: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, tp.Any]]:
        """Filter notes by effective target visibility across overlays, experiments, and tracker metadata."""

        if admin:
            return notes[:limit] if limit is not None else notes

        unverified_run_ids = list(
            dict.fromkeys(
                note["target_id"]
                for note in notes
                if note["target_type"] == "run"
                and note["target_id"] not in overlays
                and (tracker_runs_by_id is None or note["target_id"] not in tracker_runs_by_id)
            )
        )
        run_allowed: dict[str, bool] = {}
        if unverified_run_ids:
            if tracker_configured_unavailable:
                for rid in unverified_run_ids:
                    run_allowed[rid] = False
            else:
                selected_transport = getattr(app.state, "tracking_transport", transport)
                sem = asyncio.Semaphore(_LEADERBOARD_CONCURRENCY)
                async with httpx.AsyncClient(
                    transport=selected_transport, timeout=30.0, trust_env=False, follow_redirects=False
                ) as client:

                    async def _check_one(rid: str) -> tuple[str, bool]:
                        async with sem:
                            try:
                                await authorize_run_access(
                                    app=app,
                                    settings=settings,
                                    service=service,
                                    store=store,
                                    principal=principal,
                                    org_id=org_id,
                                    project_id=project_id,
                                    run_id=rid,
                                    overlay=None,
                                    overlay_loaded=True,
                                    transport=selected_transport,
                                    client=client,
                                    allow_unconfigured_tracker=True,
                                    allow_missing_tracker_run=True,
                                )
                                return rid, True
                            except HTTPException:
                                return rid, False

                    results = await asyncio.gather(*(_check_one(rid) for rid in unverified_run_ids))
                    for rid, ok in results:
                        run_allowed[rid] = ok

        visible_notes: list[dict[str, tp.Any]] = []
        for note in notes:
            if note["target_type"] == "experiment":
                exp = experiments_by_id.get(note["target_id"])
                if exp is not None and not can_view(
                    principal.subject,
                    owner=exp["owner"],
                    visibility=exp["visibility"],
                    shared_with=exp["shared_with"],
                    is_admin=False,
                ):
                    continue
            elif note["target_type"] == "run":
                rid = note["target_id"]
                ov = overlays.get(rid)
                if ov is not None:
                    if not can_view(
                        principal.subject,
                        owner=ov["owner"],
                        visibility=ov["visibility"],
                        shared_with=ov["shared_with"],
                        is_admin=False,
                    ):
                        continue
                elif tracker_runs_by_id is not None and rid in tracker_runs_by_id:
                    trun = tracker_runs_by_id[rid]
                    if not can_view(
                        principal.subject,
                        owner=trun.get("owner"),
                        visibility=trun.get("visibility") or "team",
                        shared_with=trun.get("shared_with") or (),
                        is_admin=False,
                    ):
                        continue
                elif not run_allowed.get(rid, False):
                    continue
            visible_notes.append(note)
            if limit is not None and len(visible_notes) >= limit:
                break
        return visible_notes

    # ------------------------------------------------------------------ run sharing / overlays

    @app.patch("/api/v1/collaboration/runs/{run_id}", tags=["collaboration"])
    async def update_run_sharing(
        run_id: _RunId,
        body: RunSharingUpdateRequest,
        principal: _Principal,
        org_id: _ScopeId,
        project_id: _ScopeId,
    ) -> JSONResponse:
        """Update sharing visibility, selective share list, experiment link, or baseline pin for one run."""

        norm_org = org_id.strip()
        norm_project = project_id.strip()
        require_project_writer(service, principal, org_id=norm_org, project_id=norm_project)
        admin = is_project_admin(service, principal, org_id=norm_org, project_id=norm_project)
        existing = await asyncio.to_thread(store.get_run_overlay, norm_org, norm_project, run_id)
        existing_owner = existing["owner"] if existing is not None else None
        if not existing_owner:
            tracker_wid: str | None = None
            status_code, meta = await fetch_tracker_run_metadata(
                app,
                settings,
                org_id=norm_org,
                project_id=norm_project,
                run_id=run_id,
                transport=transport,
                allow_unconfigured=True,
            )
            if status_code == 200 and isinstance(meta, dict) and "run_id" in meta:
                raw_owner = meta.get("owner")
                if isinstance(raw_owner, str) and raw_owner.strip():
                    existing_owner = raw_owner.strip()
                raw_wid = meta.get("workload_id")
                if isinstance(raw_wid, str) and raw_wid.strip():
                    tracker_wid = raw_wid.strip()
            if not existing_owner:
                existing_owner = await resolve_run_workload_owner(
                    service,
                    org_id=norm_org,
                    project_id=norm_project,
                    run_id=run_id,
                    workload_id=tracker_wid,
                )
        if not can_manage(principal.subject, owner=existing_owner, is_admin=admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the run owner or a project/fleet admin can update run sharing.",
            )

        fields = body.model_fields_set
        owner_arg: tp.Any = _UNSET
        if "owner" in fields:
            if existing_owner and body.owner != existing_owner and not admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only a project or fleet admin can reassign run ownership.",
                )
            if not admin and body.owner is not None and body.owner != principal.subject:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only a project or fleet admin can assign a run to another owner.",
                )
            owner_arg = body.owner
        elif existing is None:
            owner_arg = existing_owner

        experiment_arg: tp.Any = _UNSET
        if "experiment_id" in fields:
            if body.experiment_id is not None:
                exp = await asyncio.to_thread(store.get_experiment, norm_org, norm_project, body.experiment_id)
                if exp is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Experiment {body.experiment_id!r} not found",
                    )
                if not can_view(
                    principal.subject,
                    owner=exp["owner"],
                    visibility=exp["visibility"],
                    shared_with=exp["shared_with"],
                    is_admin=admin,
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You do not have permission to link runs to this experiment.",
                    )
            experiment_arg = body.experiment_id

        visibility_arg: tp.Any = body.visibility if "visibility" in fields and body.visibility is not None else _UNSET
        shared_arg: tp.Any = body.shared_with if "shared_with" in fields and body.shared_with is not None else _UNSET
        pinned_arg: tp.Any = body.pinned if "pinned" in fields and body.pinned is not None else _UNSET

        updated = await asyncio.to_thread(
            store.upsert_run_overlay,
            norm_org,
            norm_project,
            run_id,
            updated_by=principal.subject,
            owner=owner_arg,
            visibility=visibility_arg,
            shared_with=shared_arg,
            experiment_id=experiment_arg,
            pinned=pinned_arg,
        )
        return JSONResponse(status_code=200, content=updated)

    # ------------------------------------------------------------------ experiments & speedruns

    @app.get("/api/v1/collaboration/experiments", tags=["collaboration"])
    async def list_experiments(
        principal: _Principal,
        org_id: _ScopeId,
        project_id: _ScopeId,
        kind: str | None = Query(default=None, max_length=32),
        subject: str | None = Query(default=None, max_length=120),
        visibility: str | None = Query(default=None, max_length=32),
    ) -> JSONResponse:
        """List experiments and speedruns visible to the caller."""

        norm_org = org_id.strip()
        norm_project = project_id.strip()
        require_project_member(service, principal, org_id=norm_org, project_id=norm_project)
        admin = is_project_admin(service, principal, org_id=norm_org, project_id=norm_project)
        rows = await asyncio.to_thread(
            store.list_experiments,
            norm_org,
            norm_project,
            kind=kind,
            subject=subject,
            visibility=visibility,
        )
        note_counts = await asyncio.to_thread(store.count_notes_by_target, norm_org, norm_project, "experiment")
        visible = [
            {**exp, "notes_count": note_counts.get(exp["id"], 0)}
            for exp in rows
            if can_view(
                principal.subject,
                owner=exp["owner"],
                visibility=exp["visibility"],
                shared_with=exp["shared_with"],
                is_admin=admin,
            )
        ]
        return JSONResponse(status_code=200, content={"experiments": visible})

    @app.post("/api/v1/collaboration/experiments", status_code=status.HTTP_201_CREATED, tags=["collaboration"])
    async def create_experiment(
        body: ExperimentCreateRequest,
        principal: _Principal,
    ) -> JSONResponse:
        """Create a collaborative experiment or speedrun leaderboard."""

        norm_org = body.org_id.strip()
        norm_project = body.project_id.strip()
        require_project_writer(service, principal, org_id=norm_org, project_id=norm_project)
        created = await asyncio.to_thread(
            store.create_experiment,
            norm_org,
            norm_project,
            name=body.name,
            owner=principal.subject,
            subject=body.subject,
            kind=body.kind,
            description=body.description,
            metric_key=body.metric_key,
            goal=body.goal,
            target_value=body.target_value,
            group_filter=body.group_filter,
            tag_filter=body.tag_filter,
            visibility=body.visibility,
            shared_with=body.shared_with,
            pinned_baseline_run_id=body.pinned_baseline_run_id,
        )
        return JSONResponse(status_code=status.HTTP_201_CREATED, content={**created, "notes_count": 0})

    @app.get("/api/v1/collaboration/experiments/{exp_id}", tags=["collaboration"])
    async def get_experiment(
        exp_id: _ExpId,
        principal: _Principal,
        org_id: _ScopeId,
        project_id: _ScopeId,
    ) -> JSONResponse:
        """Return one experiment or speedrun record."""

        norm_org = org_id.strip()
        norm_project = project_id.strip()
        require_project_member(service, principal, org_id=norm_org, project_id=norm_project)
        exp = await asyncio.to_thread(store.get_experiment, norm_org, norm_project, exp_id)
        if exp is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {exp_id!r} not found")
        admin = is_project_admin(service, principal, org_id=norm_org, project_id=norm_project)
        if not can_view(
            principal.subject,
            owner=exp["owner"],
            visibility=exp["visibility"],
            shared_with=exp["shared_with"],
            is_admin=admin,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to view this experiment.",
            )
        note_counts = await asyncio.to_thread(store.count_notes_by_target, norm_org, norm_project, "experiment")
        return JSONResponse(status_code=200, content={**exp, "notes_count": note_counts.get(exp_id, 0)})

    @app.patch("/api/v1/collaboration/experiments/{exp_id}", tags=["collaboration"])
    async def update_experiment(
        exp_id: _ExpId,
        body: ExperimentUpdateRequest,
        principal: _Principal,
        org_id: _ScopeId,
        project_id: _ScopeId,
    ) -> JSONResponse:
        """Update mutable fields of an experiment or speedrun (owner or admin only)."""

        norm_org = org_id.strip()
        norm_project = project_id.strip()
        require_project_writer(service, principal, org_id=norm_org, project_id=norm_project)
        exp = await asyncio.to_thread(store.get_experiment, norm_org, norm_project, exp_id)
        if exp is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {exp_id!r} not found")
        admin = is_project_admin(service, principal, org_id=norm_org, project_id=norm_project)
        if not can_manage(principal.subject, owner=exp["owner"], is_admin=admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the experiment owner or a project/fleet admin can update it.",
            )
        fields = body.model_fields_set
        kwargs: dict[str, tp.Any] = {}
        for field_name in (
            "name",
            "subject",
            "kind",
            "description",
            "metric_key",
            "goal",
            "target_value",
            "group_filter",
            "tag_filter",
            "visibility",
            "shared_with",
            "pinned_baseline_run_id",
        ):
            if field_name in fields:
                val = getattr(body, field_name)
                if val is None and field_name in {"name", "kind", "goal", "visibility", "tag_filter", "shared_with"}:
                    continue
                kwargs[field_name] = val
        updated = await asyncio.to_thread(store.update_experiment, norm_org, norm_project, exp_id, **kwargs)
        if updated is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {exp_id!r} not found")
        note_counts = await asyncio.to_thread(store.count_notes_by_target, norm_org, norm_project, "experiment")
        return JSONResponse(status_code=200, content={**updated, "notes_count": note_counts.get(exp_id, 0)})

    @app.delete(
        "/api/v1/collaboration/experiments/{exp_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["collaboration"],
    )
    async def delete_experiment(
        exp_id: _ExpId,
        principal: _Principal,
        org_id: _ScopeId,
        project_id: _ScopeId,
    ) -> Response:
        """Delete an experiment or speedrun (owner or admin only)."""

        norm_org = org_id.strip()
        norm_project = project_id.strip()
        require_project_writer(service, principal, org_id=norm_org, project_id=norm_project)
        exp = await asyncio.to_thread(store.get_experiment, norm_org, norm_project, exp_id)
        if exp is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {exp_id!r} not found")
        admin = is_project_admin(service, principal, org_id=norm_org, project_id=norm_project)
        if not can_manage(principal.subject, owner=exp["owner"], is_admin=admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the experiment owner or a project/fleet admin can delete it.",
            )
        await asyncio.to_thread(store.delete_experiment, norm_org, norm_project, exp_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/v1/collaboration/experiments/{exp_id}/leaderboard", tags=["collaboration"])
    async def experiment_leaderboard(
        request: Request,
        exp_id: _ExpId,
        principal: _Principal,
        org_id: _ScopeId,
        project_id: _ScopeId,
    ) -> JSONResponse:
        """Rank visible runs belonging to an experiment or speedrun with baseline deltas and researcher stats."""

        norm_org = org_id.strip()
        norm_project = project_id.strip()
        require_project_member(service, principal, org_id=norm_org, project_id=norm_project)
        exp = await asyncio.to_thread(store.get_experiment, norm_org, norm_project, exp_id)
        if exp is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {exp_id!r} not found")
        admin = is_project_admin(service, principal, org_id=norm_org, project_id=norm_project)
        if not can_view(
            principal.subject,
            owner=exp["owner"],
            visibility=exp["visibility"],
            shared_with=exp["shared_with"],
            is_admin=admin,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to view this experiment.",
            )
        note_counts = await asyncio.to_thread(store.count_notes_by_target, norm_org, norm_project, "experiment")
        notes_count = note_counts.get(exp_id, 0)
        tracker_available, _, _, visible_runs = await _fetch_runs_snapshot(
            principal=principal, org_id=norm_org, project_id=norm_project
        )
        matched_runs = [run for run in visible_runs if _run_matches_experiment(run, exp)]
        metric_key: str | None = exp.get("metric_key")
        goal: str = exp["goal"]
        target_value: float | None = exp.get("target_value")

        raw_entries: list[dict[str, tp.Any]] = []
        if matched_runs and metric_key and tracker_available:
            selected_transport = getattr(request.app.state, "tracking_transport", transport)
            sem = asyncio.Semaphore(_LEADERBOARD_CONCURRENCY)
            try:
                resolve_tracker_credentials(settings)
                async with httpx.AsyncClient(
                    transport=selected_transport, timeout=30.0, trust_env=False, follow_redirects=False
                ) as client:

                    async def _fetch_metric_series(
                        run: dict[str, tp.Any],
                    ) -> tuple[list[tp.Any], list[tp.Any]]:
                        async with sem:
                            rid = run["run_id"]
                            step_pts: list[tp.Any] = []
                            time_pts: list[tp.Any] = []
                            try:
                                code, data = await tracker_get_json(
                                    app,
                                    settings,
                                    f"/api/v1/tracking/runs/{rid}/series",
                                    [
                                        ("org_id", norm_org),
                                        ("project_id", norm_project),
                                        ("key", metric_key),
                                        ("x", "step"),
                                        ("points", 1000),
                                    ],
                                    client=client,
                                )
                                if (
                                    code == 200
                                    and isinstance(data, dict)
                                    and isinstance(data.get("series"), list)
                                    and data["series"]
                                    and isinstance(data["series"][0], dict)
                                ):
                                    step_pts = list(data["series"][0].get("points") or [])
                            except HTTPException:
                                pass
                            if target_value is not None and step_pts:
                                try:
                                    tcode, tdata = await tracker_get_json(
                                        app,
                                        settings,
                                        f"/api/v1/tracking/runs/{rid}/series",
                                        [
                                            ("org_id", norm_org),
                                            ("project_id", norm_project),
                                            ("key", metric_key),
                                            ("x", "time"),
                                            ("points", 1000),
                                        ],
                                        client=client,
                                    )
                                    if (
                                        tcode == 200
                                        and isinstance(tdata, dict)
                                        and isinstance(tdata.get("series"), list)
                                        and tdata["series"]
                                        and isinstance(tdata["series"][0], dict)
                                    ):
                                        time_pts = list(tdata["series"][0].get("points") or [])
                                except HTTPException:
                                    pass
                            return step_pts, time_pts

                    series_pairs = await asyncio.gather(*(_fetch_metric_series(run) for run in matched_runs))
            except HTTPException:
                series_pairs = [([], []) for _ in matched_runs]
        else:
            series_pairs = [([], []) for _ in matched_runs]

        for run, (step_pts, time_pts) in zip(matched_runs, series_pairs, strict=True):
            best_value: float | None = None
            latest_value: float | None = None
            reached_target = False
            steps_to_target: int | None = None
            time_to_target_s: float | None = None

            if metric_key:
                started_at = _finite_number(run.get("started_at"))
                duration_s = _finite_number(run.get("duration_s"))
                if step_pts:
                    (
                        best_value,
                        latest_value,
                        reached_target,
                        steps_to_target,
                        time_to_target_s,
                    ) = _analyze_series_for_leaderboard(
                        step_pts,
                        time_pts,
                        goal=goal,
                        target_value=target_value,
                        started_at=started_at,
                        duration_s=duration_s,
                    )
                if best_value is None:
                    snap = _metric_from_run_snapshot(run, metric_key)
                    if snap is not None:
                        best_value = snap
                        latest_value = snap
                        if target_value is not None and _target_reached(snap, goal=goal, target_value=target_value):
                            reached_target = True
                            last_step = run.get("last_step")
                            if isinstance(last_step, int) and not isinstance(last_step, bool):
                                steps_to_target = last_step
                            time_to_target_s = duration_s

            raw_entries.append(
                {
                    "run_id": run["run_id"],
                    "name": run.get("name") or run["run_id"],
                    "owner": run.get("owner"),
                    "status": run.get("status", "unknown"),
                    "visibility": run.get("visibility", "team"),
                    "pinned": bool(run.get("pinned", False)),
                    "best_value": best_value,
                    "latest_value": latest_value,
                    "reached_target": reached_target,
                    "steps_to_target": steps_to_target,
                    "time_to_target_s": time_to_target_s,
                    "last_step": run.get("last_step"),
                    "duration_s": run.get("duration_s"),
                    "started_at": run.get("started_at"),
                    "updated_at": run.get("updated_at"),
                    "notes_count": run.get("notes_count", 0),
                    "latest_metrics": run.get("latest_metrics") or {},
                }
            )

        def _metric_sort_value(val: object) -> tuple[int, float]:
            finite = _finite_number(val)
            if finite is None:
                return (1, 0.0)
            return (0, finite if goal == "min" else -finite)

        def _sort_key(entry: dict[str, tp.Any]) -> tuple[tp.Any, ...]:
            updated_val = _finite_number(entry.get("updated_at"))
            updated_neg = -updated_val if updated_val is not None else math.inf
            if target_value is not None:
                reached_rank = 0 if entry["reached_target"] else 1
                time_finite = _finite_number(entry.get("time_to_target_s"))
                time_missing = 0 if time_finite is not None else 1
                time_val = time_finite if time_finite is not None else math.inf
                step_finite = _finite_number(entry.get("steps_to_target"))
                step_missing = 0 if step_finite is not None else 1
                step_val = step_finite if step_finite is not None else math.inf
                return (
                    reached_rank,
                    time_missing,
                    time_val,
                    step_missing,
                    step_val,
                    *_metric_sort_value(entry["best_value"]),
                    updated_neg,
                    entry["run_id"],
                )
            return (*_metric_sort_value(entry["best_value"]), updated_neg, entry["run_id"])

        raw_entries.sort(key=_sort_key)

        baseline_run_id: str | None = exp.get("pinned_baseline_run_id")
        if baseline_run_id is None:
            for entry in raw_entries:
                if entry["pinned"]:
                    baseline_run_id = entry["run_id"]
                    break
        baseline_value: float | None = None
        if baseline_run_id is not None:
            for entry in raw_entries:
                if entry["run_id"] == baseline_run_id:
                    baseline_value = _finite_number(entry["best_value"])
                    break

        entries: list[dict[str, tp.Any]] = []
        for rank, entry in enumerate(raw_entries, start=1):
            best_val = _finite_number(entry["best_value"])
            delta = round(best_val - baseline_value, 6) if best_val is not None and baseline_value is not None else None
            entries.append(
                {
                    **entry,
                    "rank": rank,
                    "is_record": rank == 1 and best_val is not None,
                    "is_baseline": entry["run_id"] == baseline_run_id,
                    "delta_vs_baseline": delta,
                }
            )

        # Researcher breakdown strip
        by_owner: dict[str, list[dict[str, tp.Any]]] = {}
        for entry in entries:
            owner_key = entry["owner"] or "unassigned"
            by_owner.setdefault(owner_key, []).append(entry)
        researcher_breakdown: list[dict[str, tp.Any]] = []
        for owner_key, group_entries in by_owner.items():
            best_vals = [finite for e in group_entries if (finite := _finite_number(e["best_value"])) is not None]
            best_for_owner = (min(best_vals) if goal == "min" else max(best_vals)) if best_vals else None
            best_rank = min(e["rank"] for e in group_entries)
            researcher_breakdown.append(
                {
                    "owner": owner_key,
                    "run_count": len(group_entries),
                    "best_value": best_for_owner,
                    "reached_target_count": sum(1 for e in group_entries if e["reached_target"]),
                    "running_count": sum(1 for e in group_entries if e["status"] == "running"),
                    "_best_rank": best_rank,
                }
            )
        researcher_breakdown.sort(key=lambda r: (r["_best_rank"], r["owner"]))
        for item in researcher_breakdown:
            item.pop("_best_rank", None)

        return JSONResponse(
            status_code=200,
            content=sanitize_json_value(
                {
                    "experiment": {**exp, "notes_count": notes_count},
                    "metric_key": metric_key,
                    "goal": goal,
                    "target_value": target_value,
                    "baseline_run_id": baseline_run_id,
                    "baseline_value": baseline_value,
                    "tracker_available": tracker_available,
                    "entries": entries,
                    "researcher_breakdown": researcher_breakdown,
                    "notes_count": notes_count,
                }
            ),
        )

    # ------------------------------------------------------------------ team notes

    @app.get("/api/v1/collaboration/notes", tags=["collaboration"])
    async def list_notes(
        principal: _Principal,
        org_id: _ScopeId,
        project_id: _ScopeId,
        target_type: str | None = Query(default=None, max_length=32),
        target_id: str | None = Query(default=None, max_length=128),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> JSONResponse:
        """List visible team notes for a run, experiment, or the whole project."""

        norm_org = org_id.strip()
        norm_project = project_id.strip()
        require_project_member(service, principal, org_id=norm_org, project_id=norm_project)
        admin = is_project_admin(service, principal, org_id=norm_org, project_id=norm_project)

        norm_target_type: str | None = None
        if target_type is not None:
            cleaned_type = target_type.strip().lower()
            if cleaned_type not in VALID_NOTE_TARGETS:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail='target_type must be "run" or "experiment"',
                )
            norm_target_type = cleaned_type

        norm_target_id: str | None = None
        if target_id is not None:
            cleaned_id = target_id.strip()
            if not cleaned_id:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="target_id must not be blank",
                )
            if norm_target_type is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="target_type is required when target_id is specified",
                )
            norm_target_id = cleaned_id

        if norm_target_type == "experiment" and norm_target_id is not None:
            exp = await asyncio.to_thread(store.get_experiment, norm_org, norm_project, norm_target_id)
            if exp is not None and not can_view(
                principal.subject,
                owner=exp["owner"],
                visibility=exp["visibility"],
                shared_with=exp["shared_with"],
                is_admin=admin,
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to view notes for this experiment.",
                )
        elif norm_target_type == "run" and norm_target_id is not None:
            await authorize_run_access(
                app=app,
                settings=settings,
                service=service,
                store=store,
                principal=principal,
                org_id=norm_org,
                project_id=norm_project,
                run_id=norm_target_id,
                transport=transport,
                allow_unconfigured_tracker=True,
                allow_missing_tracker_run=True,
                detail="You do not have permission to view notes for this run.",
            )

        fetch_limit = limit if admin else min(limit * 3, 1000)
        notes = await asyncio.to_thread(
            store.list_notes,
            norm_org,
            norm_project,
            target_type=norm_target_type,
            target_id=norm_target_id,
            limit=fetch_limit,
        )
        if not admin:
            overlays = await asyncio.to_thread(store.list_run_overlays, norm_org, norm_project)
            exp_rows = await asyncio.to_thread(store.list_experiments, norm_org, norm_project)
            exps = {e["id"]: e for e in exp_rows}
            notes = await _filter_visible_notes(
                notes,
                principal=principal,
                admin=False,
                org_id=norm_org,
                project_id=norm_project,
                overlays=overlays,
                experiments_by_id=exps,
                limit=limit,
            )
        return JSONResponse(status_code=200, content={"notes": notes})

    @app.post("/api/v1/collaboration/notes", status_code=status.HTTP_201_CREATED, tags=["collaboration"])
    async def create_note(
        body: NoteCreateRequest,
        principal: _Principal,
    ) -> JSONResponse:
        """Create a team note on a visible run or experiment."""

        norm_org = body.org_id.strip()
        norm_project = body.project_id.strip()
        norm_target_id = body.target_id.strip()
        require_project_writer(service, principal, org_id=norm_org, project_id=norm_project)
        admin = is_project_admin(service, principal, org_id=norm_org, project_id=norm_project)
        if body.target_type == "experiment":
            exp = await asyncio.to_thread(store.get_experiment, norm_org, norm_project, norm_target_id)
            if exp is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Experiment {norm_target_id!r} not found",
                )
            if not can_view(
                principal.subject,
                owner=exp["owner"],
                visibility=exp["visibility"],
                shared_with=exp["shared_with"],
                is_admin=admin,
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to comment on this experiment.",
                )
        else:
            await authorize_run_access(
                app=app,
                settings=settings,
                service=service,
                store=store,
                principal=principal,
                org_id=norm_org,
                project_id=norm_project,
                run_id=norm_target_id,
                transport=transport,
                allow_unconfigured_tracker=True,
                allow_missing_tracker_run=True,
                detail="You do not have permission to comment on this run.",
            )
        created = await asyncio.to_thread(
            store.create_note,
            norm_org,
            norm_project,
            target_type=body.target_type,
            target_id=norm_target_id,
            author=principal.subject,
            body=body.body,
            metric_ref=body.metric_ref,
            step_ref=body.step_ref,
        )
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=created)

    @app.delete(
        "/api/v1/collaboration/notes/{note_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["collaboration"],
    )
    async def delete_note(
        note_id: _NoteId,
        principal: _Principal,
        org_id: _ScopeId,
        project_id: _ScopeId,
    ) -> Response:
        """Delete a note (author or project/fleet admin only)."""

        norm_org = org_id.strip()
        norm_project = project_id.strip()
        require_project_writer(service, principal, org_id=norm_org, project_id=norm_project)
        note = await asyncio.to_thread(store.get_note, norm_org, norm_project, note_id)
        if note is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Note {note_id!r} not found")
        admin = is_project_admin(service, principal, org_id=norm_org, project_id=norm_project)
        if not admin and note["author"] != principal.subject:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the note author or a project/fleet admin can delete this note.",
            )
        await asyncio.to_thread(store.delete_note, norm_org, norm_project, note_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # ------------------------------------------------------------------ live team feed

    @app.get("/api/v1/collaboration/feed", tags=["collaboration"])
    async def collaboration_feed(
        request: Request,
        principal: _Principal,
        org_id: _ScopeId,
        project_id: _ScopeId,
        limit: int = Query(default=30, ge=1, le=200),
    ) -> JSONResponse:
        """Return combined live runs, enriched experiments/speedruns, recent notes, and researcher stats."""

        del request
        norm_org = org_id.strip()
        norm_project = project_id.strip()
        require_project_member(service, principal, org_id=norm_org, project_id=norm_project)
        admin = is_project_admin(service, principal, org_id=norm_org, project_id=norm_project)
        tracker_available, tracker_configured_unavailable, all_runs_by_id, visible_runs = await _fetch_runs_snapshot(
            principal=principal, org_id=norm_org, project_id=norm_project
        )

        active_runs = [run for run in visible_runs if run.get("status") == "running"][:limit]
        recent_runs = visible_runs[:limit]

        exp_rows = await asyncio.to_thread(store.list_experiments, norm_org, norm_project)
        exp_note_counts = await asyncio.to_thread(store.count_notes_by_target, norm_org, norm_project, "experiment")
        visible_exps = [
            exp
            for exp in exp_rows
            if can_view(
                principal.subject,
                owner=exp["owner"],
                visibility=exp["visibility"],
                shared_with=exp["shared_with"],
                is_admin=admin,
            )
        ]

        enriched_experiments: list[dict[str, tp.Any]] = []
        for exp in visible_exps:
            matched = [run for run in visible_runs if _run_matches_experiment(run, exp)]
            running_count = sum(1 for run in matched if run.get("status") == "running")
            leader_run_id: str | None = None
            leader_owner: str | None = None
            leader_value: float | None = None
            metric_key = exp.get("metric_key")
            goal = exp["goal"]
            if metric_key:
                for run in matched:
                    val = _metric_from_run_snapshot(run, metric_key)
                    if val is None:
                        continue
                    if leader_value is None or _is_better(val, leader_value, goal=goal):
                        leader_value = val
                        leader_run_id = run["run_id"]
                        leader_owner = run.get("owner")
            enriched_experiments.append(
                {
                    **exp,
                    "notes_count": exp_note_counts.get(exp["id"], 0),
                    "run_count": len(matched),
                    "running_count": running_count,
                    "leader_run_id": leader_run_id,
                    "leader_owner": leader_owner,
                    "leader_value": leader_value,
                }
            )

        # Recent visible notes (reusing the fetched tracker runs snapshot + overlays + experiments)
        overlays = await asyncio.to_thread(store.list_run_overlays, norm_org, norm_project)
        exps_by_id = {exp["id"]: exp for exp in exp_rows}
        raw_notes = await asyncio.to_thread(store.list_notes, norm_org, norm_project, limit=min(limit * 3, 500))
        recent_notes = await _filter_visible_notes(
            raw_notes,
            principal=principal,
            admin=admin,
            org_id=norm_org,
            project_id=norm_project,
            overlays=overlays,
            experiments_by_id=exps_by_id,
            tracker_runs_by_id=all_runs_by_id,
            tracker_configured_unavailable=tracker_configured_unavailable,
            limit=limit,
        )

        # Researcher activity breakdown
        researcher_map: dict[str, dict[str, tp.Any]] = {}
        for run in visible_runs:
            owner_key = run.get("owner") or "unassigned"
            entry = researcher_map.setdefault(
                owner_key,
                {
                    "owner": owner_key,
                    "total_runs": 0,
                    "running_runs": 0,
                    "shared_runs": 0,
                    "last_active_at": None,
                },
            )
            entry["total_runs"] += 1
            if run.get("status") == "running":
                entry["running_runs"] += 1
            if run.get("visibility") in {"shared", "team"}:
                entry["shared_runs"] += 1
            updated_at = _finite_number(run.get("updated_at"))
            if updated_at is not None and (entry["last_active_at"] is None or updated_at > entry["last_active_at"]):
                entry["last_active_at"] = updated_at
        researchers = sorted(
            researcher_map.values(),
            key=lambda r: (
                -r["running_runs"],
                -(r["last_active_at"] if r["last_active_at"] is not None else -math.inf),
                r["owner"],
            ),
        )

        return JSONResponse(
            status_code=200,
            content=sanitize_json_value(
                {
                    "org_id": norm_org,
                    "project_id": norm_project,
                    "tracker_available": tracker_available,
                    "active_runs": active_runs,
                    "recent_runs": recent_runs,
                    "experiments": enriched_experiments,
                    "recent_notes": recent_notes,
                    "researchers": researchers,
                }
            ),
        )

    # ------------------------------------------------------------------ storage region locality check

    @app.post("/api/v1/storage/check-region", tags=["storage"])
    async def check_storage_region(
        request: Request,
        body: StorageCheckRegionRequest,
        principal: _Principal,
    ) -> JSONResponse:
        """Evaluate whether accessing a ``gs://`` URI from a region is local or cross-region."""

        require_any_active_permission(service, principal)
        bucket = extract_bucket_from_gcs_uri(body.uri) or extract_bucket_from_gcs_http_url(body.uri)
        if bucket is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"expected a gs:// URI or GCS URL, got {body.uri!r}",
            )
        try:
            validated_allowlist = tuple(
                dict.fromkeys(validate_bucket_name(item) for item in body.allowed_buckets if item.strip())
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

        if body.region is not None and body.region.strip():
            try:
                host_region = _normalize_region_or_zone(body.region)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
        else:
            config_path = getattr(request.app.state, "storage_config_path", None)
            ctx = await asyncio.to_thread(detect_gcp_context, config_path=config_path)
            if not ctx.region:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No target region provided and no active GCP/TPU region detected; pass `region`.",
                )
            try:
                host_region = _normalize_region_or_zone(ctx.region)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

        custom_resolver = getattr(request.app.state, "storage_location_resolver", None)
        if custom_resolver is not None:
            resolver = custom_resolver
        else:

            def _default_resolver(target_bucket: str) -> tp.Any:
                return resolve_bucket_location(target_bucket, token_provider=default_gcp_token_provider())

            resolver = _default_resolver

        result = await asyncio.to_thread(
            check_gcs_uri_locality,
            body.uri,
            host_region=host_region,
            operation="check",
            allowed_buckets=validated_allowlist,
            allow_multiregion=not body.strict,
            location_resolver=resolver,
            on_unknown="allow",
        )

        allowlisted = bucket in set(validated_allowlist)
        return JSONResponse(
            status_code=200,
            content={
                "uri": result.uri,
                "bucket": result.bucket,
                "host_region": result.host_region,
                "bucket_location": result.bucket_location,
                "location_type": result.location_type,
                "allowed": result.allowed,
                "allowlisted": allowlisted,
                "reason": result.reason,
            },
        )


__all__ = (
    "ExperimentCreateRequest",
    "ExperimentUpdateRequest",
    "NoteCreateRequest",
    "RunSharingUpdateRequest",
    "StorageCheckRegionRequest",
    "register_collaboration_routes",
)
