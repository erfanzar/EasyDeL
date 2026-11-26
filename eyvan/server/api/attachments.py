# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fleet-admin-only attachment bridge for a local loopback development plane."""

from __future__ import annotations

import asyncio
import subprocess
import typing as tp
from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi import Path as ApiPath

from eyvan.config import EyvanSettings
from eyvan.records import CapacityPoolRecord, CellRecord, CellState, Permission, Principal, utcnow
from eyvan.server.api.auth import authenticated_principal
from eyvan.server.api.server_bind import ServerBindPolicy
from eyvan.server.local.attach_job import JOB_ID_PATTERN, AttachmentJobs, AttachmentRequest, run_plan
from eyvan.server.service import ControlPlaneService

_Principal = tp.Annotated[Principal, Depends(authenticated_principal)]
_JobId = tp.Annotated[str, ApiPath(pattern=JOB_ID_PATTERN)]


def _fleet_cell_states(
    pools: tp.Sequence[CapacityPoolRecord], cells: tp.Sequence[CellRecord]
) -> dict[str, dict[str, tp.Any]]:
    """Return what each connected fleet's cells are actually doing, keyed by fleet name.

    ``ready`` means a cell can take work now; ``connecting`` means a cell exists
    but is not ready yet; ``failing`` means the newest attempts failed and the
    pool is retrying (with the last error and when it retries); ``waiting``
    means the pool exists but has no live cell and no recent failure.
    """

    from eyvan.capacity.acquisition_backoff import pool_acquisition_backoffs

    now = utcnow()
    pool_fleet = {pool.id: pool.fleet_name for pool in pools}
    backoffs = pool_acquisition_backoffs(pools, (cell for cell in cells if cell.pool_id in pool_fleet), now)
    states: dict[str, dict[str, tp.Any]] = {}
    for fleet in set(pool_fleet.values()):
        fleet_cells = [cell for cell in cells if pool_fleet.get(cell.pool_id or "") == fleet]
        live = [cell for cell in fleet_cells if cell.state is not CellState.DELETED]
        streaks = [
            streak for pool_id, streak in backoffs.items() if pool_fleet.get(pool_id) == fleet and streak.recent(now)
        ]
        streak = min(streaks, key=lambda s: s.next_attempt_at) if streaks else None
        if any(cell.state is CellState.READY for cell in live):
            state = "ready"
        elif streak is not None and not any(
            cell.state is CellState.REGISTERING and not cell.availability.consecutive_failures for cell in live
        ):
            state = "failing"
        elif live:
            state = "connecting"
        else:
            state = "waiting"
        failing = state == "failing" and streak is not None
        states[fleet] = {
            "cell_state": state,
            "last_error": streak.last_error if failing and streak is not None else None,
            "next_attempt_at": (streak.next_attempt_at.isoformat() if failing and streak is not None else None),
        }
    return states


def _summary(plan: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Extract display facts from the CLI's fleet YAML, without inventing defaults."""

    try:
        fleet = yaml.safe_load(plan.get("fleet_yaml", ""))
        pools = fleet.get("pools", {})
        pool = next(iter(pools.values()), {})
        return {
            "shape": pool.get("accelerator_type"),
            "hosts": len(plan.get("endpoints", [])),
            "capacity_class": fleet.get("capacity_class"),
            "endpoint_mode": pool.get("provider_options", {}).get("endpoint_mode"),
        }
    except (yaml.YAMLError, AttributeError, TypeError):
        return {"shape": None, "hosts": len(plan.get("endpoints", [])), "capacity_class": None, "endpoint_mode": None}


def is_loopback_development(server_bind: ServerBindPolicy, settings: EyvanSettings) -> bool:
    """Whether this plane supports local attachments, independent of caller permissions."""

    return server_bind.is_loopback and settings.dev_auth_enabled


def register_attachment_routes(
    app: FastAPI, *, settings: EyvanSettings, service: ControlPlaneService, server_bind: ServerBindPolicy
) -> None:
    """Register local CLI operations, always guarded before disk or subprocess I/O."""

    def authorize(principal: Principal) -> None:
        service.authorizer.require(principal, Permission.MANAGE_FLEET)
        if not is_loopback_development(server_bind, settings):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Attachments require a local loopback development plane. Run `eyvan up` on this machine, "
                    "or use `eyvan gcp connect` on the plane's host."
                ),
            )

    def plane_url(request: Request) -> str:
        # The ASGI listening socket is authoritative, never an untrusted Host or
        # forwarded-host header that could redirect a privileged CLI operation.
        server = request.scope.get("server")
        if server is None:
            raise HTTPException(status_code=503, detail="Cannot resolve the local plane's listening port.")
        host = f"[{server_bind.host}]" if ":" in server_bind.host else server_bind.host
        return f"{'https' if settings.tls_enabled else 'http'}://{host}:{server[1]}"

    def jobs(request: Request) -> AttachmentJobs:
        # A test/embedding seam, not a new environment configuration surface.
        directory: Path | None = getattr(request.app.state, "attachments_directory", None)
        return AttachmentJobs(directory)

    def payload(request: Request, body: AttachmentRequest, principal: Principal) -> dict[str, tp.Any]:
        return {"request": body.model_dump(), "url": plane_url(request), "user": principal.subject}

    @app.post("/api/v1/attachments/plan", tags=["attachments"])
    async def attachment_plan(request: Request, body: AttachmentRequest, principal: _Principal) -> dict[str, tp.Any]:
        """Preview an existing TPU/SSH attachment without mutating the host or fleet."""

        authorize(principal)
        try:
            plan = await asyncio.to_thread(run_plan, payload(request, body, principal))
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504, detail="Attachment preview timed out after 120 seconds; check SSH access."
            ) from None
        except (OSError, ValueError) as error:
            detail = str(error) if isinstance(error, ValueError) else "Could not start the local attachment CLI."
            raise HTTPException(status_code=502, detail=detail) from None
        return {**plan, "summary": _summary(plan)}

    @app.post("/api/v1/attachments", status_code=202, tags=["attachments"])
    async def start_attachment(request: Request, body: AttachmentRequest, principal: _Principal) -> dict[str, tp.Any]:
        """Start a durable detached CLI job that survives local plane redeployment."""

        authorize(principal)
        return await asyncio.to_thread(jobs(request).start, payload(request, body, principal))

    @app.get("/api/v1/attachments", tags=["attachments"])
    async def recent_attachments(request: Request, principal: _Principal) -> dict[str, tp.Any]:
        """List recent local attachment jobs with their progress and terminal result."""

        authorize(principal)
        return {"jobs": await asyncio.to_thread(jobs(request).recent, url=plane_url(request))}

    @app.get("/api/v1/attachments/discovered", tags=["attachments"])
    async def discovered_tpus(request: Request, principal: _Principal) -> dict[str, tp.Any]:
        """List every TPU VM the machine owner's gcloud account can see, with its fleet name.

        Served from the read-only discovery index the local plane refreshes every
        few minutes; a stale index is refreshed in the background, never inline.
        """

        authorize(principal)
        from eyvan.capacity.providers.gcp.tpu_discovery import DiscoveryIndex, refresh_in_background

        index: DiscoveryIndex = getattr(request.app.state, "discovery_index", None) or DiscoveryIndex()
        snapshot = await asyncio.to_thread(index.load)
        if getattr(request.app.state, "discovery_refresh", True):
            await asyncio.to_thread(refresh_in_background, index)
        pools = [pool for pool in await service.list_capacity_pools(principal) if pool.enabled]
        if snapshot is None:
            return {"discovered_at": None, "tpus": [], "skipped": {}, "projects": []}
        states = _fleet_cell_states(pools, await service.list_cells(principal))
        document = snapshot.as_document()
        for tpu in document["tpus"]:
            state = states.get(tpu["fleet"])
            tpu["connected"] = state is not None
            tpu["cell_state"] = None if state is None else state["cell_state"]
            tpu["last_error"] = None if state is None else state["last_error"]
            tpu["next_attempt_at"] = None if state is None else state["next_attempt_at"]
        document.pop("version", None)
        return document

    @app.get("/api/v1/attachments/{attachment_id}", tags=["attachments"])
    async def get_attachment(request: Request, attachment_id: _JobId, principal: _Principal) -> dict[str, tp.Any]:
        """Read durable progress even after the application process has restarted."""

        authorize(principal)
        try:
            return await asyncio.to_thread(jobs(request).get, attachment_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Attachment job not found on this machine.") from None
