# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""An attached TPU recovers from a credential problem without anyone restarting or re-running anything.

Three parts: the TPU API client re-mints its gcloud token on a 403 (an account
fixed on the host is used on the next request, not after an agent restart); a
pool of existing hardware never waits an hour between re-reads of a TPU that
already exists; and the discovered-TPU listing reports the fleet's real cell
state instead of "connected" whenever a pool exists.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from eyvan.capacity.drivers.base import CapacityProviderFailureCode, CapacityStatus
from eyvan.capacity.providers.gcp.tpu_nodes import TpuNodeApi
from eyvan.records import AvailabilityPolicy, CapacityPoolRecord, CellState

_T0 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
_NODE = {
    "name": "projects/project-a/locations/us-central1-a/nodes/train-a",
    "id": "1",
    "acceleratorType": "v5p-8",
    "state": "READY",
    "health": "HEALTHY",
    "networkEndpoints": [{"ipAddress": "10.0.0.2", "accessConfig": {"externalIp": "192.0.2.2"}}],
}


def test_a_403_is_retried_once_with_a_freshly_minted_token() -> None:
    minted: list[str] = []

    def token_source() -> str:
        # The host's gcloud account changes between the first and second mint.
        minted.append("wrong-account" if not minted else "right-account")
        return minted[-1]

    def handle(request: httpx.Request) -> httpx.Response:
        if request.headers["Authorization"] == "Bearer wrong-account":
            return httpx.Response(403, json={"error": {"message": "Permission 'tpu.nodes.get' denied"}})
        return httpx.Response(200, json=_NODE)

    api = TpuNodeApi(token_source=token_source, transport=httpx.MockTransport(handle))
    node = api.get_node("project-a", "us-central1-a", "train-a")

    assert node is not None and node.name == "train-a"
    assert minted == ["wrong-account", "right-account"]


def test_a_persistent_403_still_fails_after_one_refresh() -> None:
    minted: list[int] = []

    def token_source() -> str:
        minted.append(1)
        return f"tok-{len(minted)}"

    api = TpuNodeApi(
        token_source=token_source,
        transport=httpx.MockTransport(lambda request: httpx.Response(403, json={"error": {"message": "denied"}})),
    )
    from eyvan.capacity.providers.gcp.tpu_nodes import TpuApiError

    with pytest.raises(TpuApiError) as caught:
        api.get_node("project-a", "us-central1-a", "train-a")
    assert caught.value.status == 403
    assert len(minted) == 2, "exactly one refresh, not a loop"


def test_many_forbidden_projects_do_not_each_mint_a_token() -> None:
    minted: list[int] = []

    def token_source() -> str:
        minted.append(1)
        return f"tok-{len(minted)}"

    def handle(request: httpx.Request) -> httpx.Response:
        if "/projects/locked-" in request.url.path:
            return httpx.Response(403, json={"error": {"message": "denied"}})
        return httpx.Response(200, json={"nodes": []})

    from eyvan.capacity.providers.gcp.tpu_discovery import discover

    api = TpuNodeApi(token_source=token_source, transport=httpx.MockTransport(handle))
    snapshot = discover(api, projects=[f"locked-{index}" for index in range(10)] + ["open"], concurrency=1)

    assert set(snapshot.skipped) == {f"locked-{index}" for index in range(10)}
    assert len(minted) <= 2, f"{len(minted)} gcloud token mints for one listing"


def _attach_pool() -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id="tpu-train-a-1a2b3c4d-v5p-8",
        fleet_name="tpu-train-a-1a2b3c4d",
        node_type="v5p-8",
        provider="gcp",
        capacity_backend="gcp-attach",
        provider_project="project-a",
        region="us-central1",
        zone="us-central1-a",
        capacity={"cpu": 208, "tpu": 4},
        allocatable={"cpu": 208, "tpu": 4},
        accelerator_type="v5p-8",
        capacity_class="on_demand",
        executor="eyvan-native",
        agent_endpoint="http://agent:8090",
        labels={"eyvan_resource_kind": "tpu"},
        min_cells=1,
        max_cells=1,
        availability_policy=AvailabilityPolicy(max_backoff_s=3600),
    )


def _failed(pool: CapacityPoolRecord, index: int, observed_at: datetime, *, detail: str | None = "denied (HTTP 403)"):
    cell = pool.make_cell(f"{pool.id}-{index:04d}")
    belief = cell.availability.model_copy(update={"consecutive_failures": 1, "last_observed_at": observed_at})
    return cell.model_copy(
        update={
            "state": CellState.DELETED,
            "provider_phase": CapacityStatus.PROVIDER_ERROR,
            "provider_failure_code": CapacityProviderFailureCode.CONFIGURATION,
            "unavailable_detail": detail,
            "availability": belief,
        }
    )


def test_existing_hardware_never_waits_an_hour_between_re_reads() -> None:
    from eyvan.capacity.acquisition_backoff import EXISTING_HARDWARE_MAX_BACKOFF_S, pool_acquisition_backoffs

    pool = _attach_pool()
    cells = [_failed(pool, index, _T0 + timedelta(seconds=index)) for index in range(20)]
    streak = pool_acquisition_backoffs((pool,), cells, _T0 + timedelta(seconds=21))[pool.id]

    assert streak.failed_attempts == 20
    assert streak.backoff_s == EXISTING_HARDWARE_MAX_BACKOFF_S
    assert EXISTING_HARDWARE_MAX_BACKOFF_S <= 300


def test_created_capacity_keeps_its_configured_cap() -> None:
    from eyvan.capacity.acquisition_backoff import pool_acquisition_backoffs

    pool = _attach_pool().model_copy(update={"capacity_backend": "gcp-tpu"})
    cells = [_failed(pool, index, _T0 + timedelta(seconds=index)) for index in range(20)]
    assert pool_acquisition_backoffs((pool,), cells, _T0 + timedelta(seconds=21))[pool.id].backoff_s == 3600


@pytest.mark.parametrize(
    "case,expected",
    [("ready", "ready"), ("failing", "failing"), ("connecting", "connecting"), ("waiting", "waiting")],
)
def test_discovered_tpus_report_the_fleets_real_cell_state(tmp_path: Path, case: str, expected: str) -> None:
    import asyncio

    from eyvan.capacity.providers.gcp.tpu_discovery import DiscoveredTpu, DiscoverySnapshot
    from eyvan.records import utcnow
    from tests.test_attachments_api import ADMIN, _discovery_client

    pool = _attach_pool()
    now = utcnow()
    cells = []
    if case in {"failing", "connecting"} or case == "ready":
        if case == "failing":
            cells = [_failed(pool, index, now - timedelta(seconds=30 - index)) for index in range(3)]
        live = pool.make_cell(f"{pool.id}-0099")
        if case == "ready":
            live = live.model_copy(update={"state": CellState.READY})
        if case != "failing":
            cells.append(live)

    tpu = DiscoveredTpu(
        fleet=pool.fleet_name,
        project="project-a",
        zone="us-central1-a",
        name="train-a",
        node_id="1",
        accelerator_type="v5p-8",
        capacity_class="on_demand",
        state="READY",
        hosts=1,
    )
    http = _discovery_client(tmp_path, DiscoverySnapshot(discovered_at=1000.0, tpus=(tpu,), projects=("project-a",)))
    repository = http.app.state.service.repository

    async def seed() -> None:
        async with repository.transaction() as tx:
            await tx.put_capacity_pool(pool)
            for cell in cells:
                await tx.put_cell(cell)

    asyncio.run(seed())
    body = http.get("/api/v1/attachments/discovered", headers=ADMIN).json()
    row = body["tpus"][0]

    assert row["connected"] is True
    assert row["cell_state"] == expected
    if expected == "failing":
        assert row["last_error"] == "denied (HTTP 403)"
        assert row["next_attempt_at"] is not None
    else:
        assert row["last_error"] is None


def test_discovery_records_the_account_it_listed_as_and_old_indexes_still_load(tmp_path: Path) -> None:
    from eyvan.capacity.providers.gcp.tpu_discovery import DiscoveryIndex, DiscoverySnapshot, discover

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nodes": [_NODE]})

    api = TpuNodeApi(
        token_source=lambda: "tok",
        transport=httpx.MockTransport(handle),
        account_source=lambda: "someone@example.com",
    )
    snapshot = discover(api, projects=["project-a"])
    assert snapshot.account == "someone@example.com"

    index = DiscoveryIndex(tmp_path / "discovered.json")
    index.save(snapshot)
    loaded = index.load()
    assert loaded is not None and loaded.account == "someone@example.com"

    # An index written before the account was recorded still loads, account unknown.
    document = snapshot.as_document()
    document.pop("account")
    old = DiscoverySnapshot.from_document(document)
    assert old.account is None and len(old.tpus) == 1


def test_a_custom_token_source_never_shells_out_to_name_the_account() -> None:
    api = TpuNodeApi(token_source=lambda: "tok", transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    assert api.account() is None


def test_the_discovered_route_names_the_listing_account(tmp_path: Path) -> None:
    from eyvan.capacity.providers.gcp.tpu_discovery import DiscoverySnapshot
    from tests.test_attachments_api import ADMIN, _discovery_client

    snapshot = DiscoverySnapshot(discovered_at=1000.0, projects=("project-a",), account="someone@example.com")
    body = _discovery_client(tmp_path, snapshot).get("/api/v1/attachments/discovered", headers=ADMIN).json()
    assert body["account"] == "someone@example.com"
