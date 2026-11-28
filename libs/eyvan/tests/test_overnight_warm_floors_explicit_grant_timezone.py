# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regressions for overnight warm floors and explicit grant timezone contracts."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from eyvan._internal.platform import EyvanPlatformConfig
from eyvan.capacity.planning import CapacityPlanner, PoolDemand
from eyvan.config import EyvanSettings
from eyvan.records import CapacityPoolRecord, Principal, ResourceLimits, Role, RoleGrant, RoleGrantRecord, WarmWindow
from eyvan.server.api.app import create_app
from eyvan.server.api.schemas import RoleGrantIssueRequest
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService


@pytest.mark.parametrize(
    ("days", "offset", "instant", "active"),
    [
        ((1,), 0, "2026-01-05T02:00:00Z", False),
        ((1,), 0, "2026-01-05T21:59:59Z", False),
        ((1,), 0, "2026-01-05T22:00:00Z", True),
        ((1,), 0, "2026-01-05T23:00:00Z", True),
        ((1,), 0, "2026-01-06T00:00:00Z", True),
        ((1,), 0, "2026-01-06T02:00:00Z", True),
        ((1,), 0, "2026-01-06T03:59:59Z", True),
        ((1,), 0, "2026-01-06T04:00:00Z", False),
        ((1,), 0, "2026-01-06T23:00:00Z", False),
        ((1, 2, 3, 4, 5), 0, "2026-01-05T02:00:00Z", False),
        ((1, 2, 3, 4, 5), 0, "2026-01-10T02:00:00Z", True),
        ((7,), 0, "2026-01-05T02:00:00Z", True),
        ((7,), 0, "2026-01-04T02:00:00Z", False),
        ((1,), 180, "2026-01-05T23:00:00Z", True),
        ((1,), 180, "2026-01-04T23:00:00Z", False),
        ((1,), -300, "2026-01-06T07:00:00Z", True),
        ((1,), -300, "2026-01-05T07:00:00Z", False),
        # A moment's own offset must not alter the configured fixed-offset clock.
        ((1,), 0, "2026-01-06T05:00:00+03:00", True),
        ((1,), 0, "2026-01-05T19:00:00-05:00", True),
    ],
)
async def test_overnight_floor_and_reports_follow_the_window_start_day(
    days: tuple[int, ...], offset: int, instant: str, active: bool
) -> None:
    moment = datetime.fromisoformat(instant)
    window = WarmWindow(min_cells=3, start_hour=22, end_hour=4, days=days, utc_offset_minutes=offset)
    pool = CapacityPoolRecord(
        id="research/cpu",
        fleet_name="research",
        node_type="cpu",
        provider="gcp",
        provider_project="project-a",
        region="us-east5",
        zone="us-east5-a",
        capacity=ResourceLimits(cpu=8),
        allocatable=ResourceLimits(cpu=8),
        agent_endpoint="http://agent:8090",
        min_cells=0,
        max_cells=10,
        initialized_at=datetime(2026, 1, 1, tzinfo=UTC),
        warm_windows=(window,),
    )
    demand = PoolDemand(real_cell_ids=frozenset(), virtual_worker_counts=(), virtual_workload_ids=())
    plan = CapacityPlanner(pools=(pool,), cells_by_pool={pool.id: []}, now=moment).plan({pool.id: demand})
    repository = InMemoryRepository()
    async with repository.transaction() as tx:
        await tx.put_capacity_pool(pool)
    report = await ControlPlaneService(repository).warmth_report(
        Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),)), now=moment
    )

    # Inspect observable consumers as well as the predicate; no provider I/O occurs.
    expected_floor = 3 if active else 0
    assert (window.covers(moment), pool.warm_floor(moment), plan[pool.id].target_worker_units) == (
        active,
        expected_floor,
        expected_floor,
    )
    assert (report.pools[0].window_active, report.pools[0].floor) == (active, expected_floor)


@pytest.mark.parametrize("field", ["expires_at", "granted_at"])
def test_grant_record_rejects_naive_timestamps_as_validation_errors(field: str) -> None:
    timestamps = {"granted_at": datetime(2098, 1, 1, tzinfo=UTC), "expires_at": datetime(2099, 1, 1, tzinfo=UTC)}
    timestamps[field] = timestamps[field].replace(tzinfo=None)
    with pytest.raises(ValidationError, match="timezone"):
        RoleGrantRecord(
            subject="alice",
            grant=RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),
            granted_by="root",
            **timestamps,
        )


@pytest.mark.parametrize("expires", [None, datetime(2099, 1, 1, tzinfo=UTC)])
def test_grant_active_at_requires_an_aware_instant(expires: datetime | None) -> None:
    record = RoleGrantRecord(
        subject="alice",
        grant=RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),
        granted_by="root",
        expires_at=expires,
    )
    with pytest.raises(ValueError, match="timezone"):
        record.active_at(datetime(2098, 1, 1))


def test_naive_grant_expiry_returns_422_without_storing_authority(tmp_path: Path) -> None:
    repository = InMemoryRepository()
    app = create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            controller_enabled=False,
            artifact_token_key="k" * 32,
            artifact_store_uri=str(tmp_path / "artifacts"),
        ),
        repository=repository,
        platform_config=EyvanPlatformConfig(),
    )
    # The controller is disabled; requests use real in-process ASGI routing, not sockets.
    with TestClient(app, raise_server_exceptions=False) as client:
        headers = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin", "Idempotency-Key": "invalid"}
        request = {
            "subject": "alice",
            "grant": {"role": "project_viewer", "org_id": "acme", "project_id": "research"},
            "expires_at": "2099-01-01T00:00:00",
        }
        rejected = client.post("/api/v1/grants", json=request, headers=headers)
        assert rejected.status_code == 422
        assert client.get("/api/v1/grants", headers=headers).json() == []

        for index, expiry in enumerate(("2099-01-01T00:00:00Z", "2099-01-01T03:00:00+03:00")):
            response = client.post(
                "/api/v1/grants",
                json={**request, "subject": f"aware-{index}", "expires_at": expiry},
                headers={**headers, "Idempotency-Key": f"aware-{index}"},
            )
            assert response.status_code == 201
            stored = RoleGrantRecord.model_validate(response.json())
            assert stored.expires_at == datetime(2099, 1, 1, tzinfo=UTC)
            assert stored.active_at(stored.expires_at - timedelta(microseconds=1))
            assert not stored.active_at(stored.expires_at)
            assert not stored.active_at(datetime.fromisoformat("2099-01-01T03:00:00+03:00"))
        grants = client.get("/api/v1/grants", headers=headers).json()
        assert {item["subject"] for item in grants} == {"aware-0", "aware-1"}


def test_request_conversion_rejects_naive_expiry_with_validation_error() -> None:
    request = RoleGrantIssueRequest.model_validate(
        {
            "subject": "alice",
            "grant": {"role": "project_viewer", "org_id": "acme", "project_id": "research"},
            "expires_at": "2099-01-01T00:00:00",
        }
    )
    with pytest.raises(ValidationError, match="timezone"):
        request.to_record()
