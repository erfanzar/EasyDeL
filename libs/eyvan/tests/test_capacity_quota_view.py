# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provider TPU quota, answered for the operator choosing where to run.

``quota_discovery.py`` has produced per-zone limits and live usage since
``eyvan autoscale generate`` was written, and nothing an operator looks at ever
showed it. Picking a zone meant running ``gcloud`` by hand.

The property every test here defends is the one that decides whether the answer
is worth having: a quota limit is not free capacity. Cloud Quotas reports a
ceiling for a zone whether or not Cloud Monitoring has a usage series for it,
so a row that renders a missing usage as zero claims the whole quota is
available -- and unlike an empty page, that mistake is confidently wrong and
costs money to act on. Unknown stays unknown here, on the wire and in the
terminal.
"""

from __future__ import annotations

import asyncio
import time
import typing as tp
from datetime import datetime, timedelta

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from pydantic import ValidationError

import eyvan.sdk.cli as cli
import eyvan.server.api.app as app_module
from eyvan._internal.compat import UTC
from eyvan.capacity.providers.gcp.quota_discovery import GcloudQuotaError, TpuQuotaAvailability
from eyvan.config import EyvanSettings
from eyvan.records import CapacityClass
from eyvan.server.api.app import create_app
from eyvan.server.api.schemas import TpuQuotaEntry, TpuQuotaState

OBSERVED_AT = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _headers(role: str = "fleet_admin") -> dict[str, str]:
    return {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
        "X-Eyvan-Roles": role,
    }


def _availability(
    *,
    zone: str = "us-central2-b",
    accelerator_type: str = "v6e-8",
    quota_tpus: int | None = 32,
    available_tpus: int | None = 16,
    preemptible: bool = True,
    usage_observed_at: datetime | None = OBSERVED_AT,
) -> TpuQuotaAvailability:
    """Build one provider observation with the discovery's own invariants."""

    generation = accelerator_type.partition("-")[0]
    suffix = "-preemptible" if preemptible else ""
    return TpuQuotaAvailability(
        zone=zone,
        accelerator_type=accelerator_type,
        generation=generation,
        quota_tpus=quota_tpus,
        preemptible=preemptible,
        metric=f"tpu.googleapis.com/tpu-{generation}{suffix}",
        available_tpus=available_tpus,
        usage_observed_at=usage_observed_at if available_tpus is not None and quota_tpus is not None else None,
    )


class FakeDiscovery:
    """Answer one purchase class at a time, exactly as the provider reader does."""

    def __init__(
        self,
        *,
        spot: tuple[TpuQuotaAvailability, ...] = (),
        on_demand: tuple[TpuQuotaAvailability, ...] = (),
        spot_error: str | None = None,
        on_demand_error: str | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.spot = spot
        self.on_demand = on_demand
        self.spot_error = spot_error
        self.on_demand_error = on_demand_error
        self.delay_s = delay_s
        self.projects: list[str] = []
        self.zones: list[tuple[str, ...]] = []

    def discover(
        self,
        *,
        project: str,
        zones: tp.Iterable[str] = (),
        generations: tp.Iterable[str] = (),
        accelerator_types: tp.Iterable[str] = (),
        preemptible: bool = True,
    ) -> tuple[TpuQuotaAvailability, ...]:
        del generations, accelerator_types
        self.projects.append(project)
        self.zones.append(tuple(zones))
        if self.delay_s:
            time.sleep(self.delay_s)
        error = self.spot_error if preemptible else self.on_demand_error
        if error is not None:
            raise GcloudQuotaError(error)
        return self.spot if preemptible else self.on_demand


def _client(discovery: object) -> TestClient:
    return TestClient(create_app(settings=EyvanSettings(dev_auth_enabled=True), quota_discovery=discovery))


def _quota(client: TestClient, **params: str) -> dict[str, tp.Any]:
    response = client.get("/api/v1/capacity/quota", headers=_headers(), params={"project": "gcp-project", **params})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def _rows(payload: dict[str, tp.Any], zone: str) -> list[dict[str, tp.Any]]:
    return [row for row in payload["entries"] if row["zone"] == zone]


def _apply_gcp_fleet(client: TestClient, *, project: str = "declared-project") -> None:
    """Apply one real GCP pool so the route can resolve a provider project."""

    applied = client.put(
        "/api/v1/fleets/research-fleet",
        headers={**_headers(), "Idempotency-Key": f"fleet-{project}"},
        json={
            "pools": [
                {
                    "id": "v6e-pool",
                    "fleet_name": "research-fleet",
                    "node_type": "v6e-workers",
                    "provider": "gcp",
                    "capacity_backend": "gcp-tpu",
                    "provider_project": project,
                    "region": "us-central2",
                    "zone": "us-central2-b",
                    "capacity": {"cpu": 180, "memory_gib": 720, "tpu": 8},
                    "allocatable": {"cpu": 180, "memory_gib": 720, "tpu": 8},
                    "accelerator_type": "v6e-8",
                    "capacity_class": "spot",
                    "executor": "native-local",
                    "agent_endpoint": "http://agent:8090",
                    "min_cells": 0,
                    "max_cells": 4,
                }
            ]
        },
    )
    assert applied.status_code == 200, applied.text


def test_quota_reports_limit_usage_and_free_per_zone_type_and_purchase_class() -> None:
    """The three numbers the panel exists for, with Spot kept separate.

    Spot and on-demand are distinct provider quota metrics -- the ``-preemptible``
    metric suffix -- so merging them would report capacity that cannot be bought
    the way the reader intends to buy it.
    """

    client = _client(
        FakeDiscovery(
            spot=(_availability(zone="us-central2-b", accelerator_type="v6e-8", quota_tpus=32, available_tpus=16),),
            on_demand=(
                _availability(
                    zone="us-central2-b",
                    accelerator_type="v6e-8",
                    quota_tpus=8,
                    available_tpus=8,
                    preemptible=False,
                ),
            ),
        )
    )

    payload = _quota(client)

    assert payload["status"] == "ok"
    assert payload["project"] == "gcp-project"
    by_class = {row["capacity_class"]: row for row in _rows(payload, "us-central2-b")}
    assert by_class["spot"]["limit_chips"] == 32
    assert by_class["spot"]["in_use_chips"] == 16
    assert by_class["spot"]["free_chips"] == 16
    # The question is "how many of these fit", so the division is already done.
    assert by_class["spot"]["free_slices"] == 2
    assert by_class["spot"]["region"] == "us-central2"
    assert by_class["on_demand"]["limit_chips"] == 8
    assert by_class["on_demand"]["in_use_chips"] == 0
    assert by_class["on_demand"]["free_slices"] == 1


def test_a_zone_with_no_usage_series_is_unknown_and_never_free() -> None:
    """The one behaviour that makes this data safe to act on.

    Cloud Quotas gives a limit for the zone; Cloud Monitoring has no matching
    usage series. Reporting the limit as the free balance would advertise a full
    zone as entirely available.
    """

    client = _client(
        FakeDiscovery(spot=(_availability(zone="europe-west4-a", quota_tpus=64, available_tpus=None),)),
    )

    row = _rows(_quota(client), "europe-west4-a")[0]

    assert row["state"] == "usage_unknown"
    assert row["limit_chips"] == 64
    assert row["free_chips"] is None
    assert row["free_slices"] is None
    assert row["in_use_chips"] is None
    assert row["usage_observed_at"] is None


def test_an_unlimited_quota_is_reported_as_unlimited_rather_than_a_number() -> None:
    """An unbounded ceiling has no balance to report and no usage to age."""

    client = _client(FakeDiscovery(spot=(_availability(zone="us-east5-a", quota_tpus=None, available_tpus=None),)))

    row = _rows(_quota(client), "us-east5-a")[0]

    assert row["state"] == "unlimited"
    assert row["limit_chips"] is None
    assert row["free_chips"] is None


def test_a_usage_observation_carries_its_own_age() -> None:
    """A free balance measured hours ago is not a live one.

    The allocation-usage gauge lags reality by minutes at best, so the row
    states when it was measured rather than letting the reader assume "now".
    """

    client = _client(FakeDiscovery(spot=(_availability(usage_observed_at=OBSERVED_AT),)))

    row = _rows(_quota(client), "us-central2-b")[0]

    assert row["usage_observed_at"].startswith("2026-07-29T12:00:00")
    expected = (datetime.now(UTC) - OBSERVED_AT).total_seconds()
    assert row["usage_age_s"] == pytest.approx(expected, abs=60)


def test_one_unreadable_purchase_class_does_not_hide_the_other() -> None:
    """A fleet whose Spot quota is unreadable still has readable on-demand quota."""

    client = _client(
        FakeDiscovery(
            on_demand=(_availability(preemptible=False, quota_tpus=8, available_tpus=8),),
            spot_error="gcloud beta quotas info list failed: PERMISSION_DENIED",
        )
    )

    payload = _quota(client)

    assert payload["status"] == "ok"
    assert [row["capacity_class"] for row in payload["entries"]] == ["on_demand"]
    assert len(payload["failures"]) == 1
    failure = payload["failures"][0]
    assert failure["capacity_class"] == "spot"
    assert failure["reason"] == "gcloud beta quotas info list failed: PERMISSION_DENIED"
    assert failure["code"] == "permission_denied"
    assert failure["summary"] is not None
    assert "gcp-project" in failure["summary"]
    assert failure["remediation"] is not None
    assert "roles/cloudquotas.viewer" in failure["remediation"]


def test_a_snapshot_with_nothing_readable_says_so_instead_of_looking_empty() -> None:
    """An empty table reads as "no capacity anywhere", which is a different fact."""

    client = _client(
        FakeDiscovery(
            spot_error="gcloud auth token lookup failed: no credentials",
            on_demand_error="gcloud auth token lookup failed: no credentials",
        )
    )

    payload = _quota(client)

    assert payload["status"] == "unavailable"
    assert payload["entries"] == []
    assert {failure["capacity_class"] for failure in payload["failures"]} == {"spot", "on_demand"}
    assert all("no credentials" in failure["reason"] for failure in payload["failures"])


def test_a_missing_gcloud_binary_is_reported_rather_than_crashing_the_panel() -> None:
    """The realistic control-plane host: no provider CLI installed at all."""

    class MissingGcloud:
        def discover(self, **_: object) -> tuple[TpuQuotaAvailability, ...]:
            raise GcloudQuotaError("gcloud beta quotas info list failed: [Errno 2] No such file or directory")

    payload = _quota(_client(MissingGcloud()))

    assert payload["status"] == "unavailable"
    assert all("No such file or directory" in failure["reason"] for failure in payload["failures"])


def test_a_reused_reading_states_its_age_instead_of_being_presented_as_live() -> None:
    """Serving a cached number as a live one is the failure this field prevents."""

    discovery = FakeDiscovery(spot=(_availability(),))
    client = _client(discovery)

    first = _quota(client)
    second = _quota(client)

    assert first["cached"] is False
    assert second["cached"] is True
    # Same reading, and it says so: the second response ages the first read
    # rather than restating it as a fresh observation.
    assert second["refreshed_at"] == first["refreshed_at"]
    assert second["age_s"] >= first["age_s"]
    assert discovery.projects == ["gcp-project", "gcp-project"]


def test_refresh_reads_the_provider_again() -> None:
    """The escape hatch for an operator who just raised a quota."""

    discovery = FakeDiscovery(spot=(_availability(),))
    client = _client(discovery)

    _quota(client)
    refreshed = _quota(client, refresh="true")

    assert refreshed["cached"] is False
    assert len(discovery.projects) == 4


def test_zone_selection_is_passed_to_the_provider_and_keyed_separately() -> None:
    """A zone-scoped read is a different question with a different answer."""

    discovery = FakeDiscovery(spot=(_availability(zone="us-central2-b"),))
    client = _client(discovery)

    _quota(client, zone="us-central2-b")

    assert discovery.zones[0] == ("us-central2-b",)
    assert all(zones == ("us-central2-b",) for zones in discovery.zones)


def test_the_project_defaults_to_the_one_this_fleet_declares() -> None:
    """An operator should not have to retype what the pools already state."""

    discovery = FakeDiscovery(spot=(_availability(),))
    client = _client(discovery)
    _apply_gcp_fleet(client, project="declared-project")

    response = client.get("/api/v1/capacity/quota", headers=_headers())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["project"] == "declared-project"
    assert payload["available_projects"] == ["declared-project"]
    assert discovery.projects == ["declared-project", "declared-project"]


def test_no_declared_project_is_unconfigured_rather_than_an_empty_fleet() -> None:
    """ "Nothing is configured" and "nothing is available" need different fixes."""

    discovery = FakeDiscovery(spot=(_availability(),))
    client = _client(discovery)

    response = client.get("/api/v1/capacity/quota", headers=_headers())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "unconfigured"
    assert payload["project"] is None
    assert payload["entries"] == []
    # No provider read is attempted when there is nothing to ask about.
    assert discovery.projects == []


def test_reading_quota_requires_the_same_authority_as_reading_the_pools() -> None:
    """The route resolves its project from pool records, so it is a fleet read."""

    client = _client(FakeDiscovery(spot=(_availability(),)))

    response = client.get(
        "/api/v1/capacity/quota",
        headers=_headers("project_viewer"),
        params={"project": "gcp-project"},
    )

    assert response.status_code == 403
    assert "manage_fleet" in response.json()["detail"]


async def test_concurrent_readers_share_one_provider_reading() -> None:
    """One page open in three tabs must not become three bursts of gcloud calls."""

    class CountingDiscovery:
        def __init__(self) -> None:
            self.calls = {True: 0, False: 0}

        def discover(self, *, project: str, preemptible: bool = True, **_: object):
            del project
            self.calls[preemptible] += 1
            time.sleep(0.2)
            return (_availability(quota_tpus=8 * self.calls[preemptible], available_tpus=0, preemptible=preemptible),)

    discovery = CountingDiscovery()
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True), quota_discovery=discovery)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first, second = await asyncio.gather(
            client.get("/api/v1/capacity/quota", headers=_headers(), params={"project": "p"}),
            client.get("/api/v1/capacity/quota", headers=_headers(), params={"project": "p"}),
        )

    assert first.status_code == 200 and second.status_code == 200
    # Both responses describe the *first* read of each class. A second concurrent
    # provider read would have reported a limit of 16 to one of them.
    assert {row["limit_chips"] for row in first.json()["entries"]} == {8}
    assert {row["limit_chips"] for row in second.json()["entries"]} == {8}
    assert first.json()["refreshed_at"] == second.json()["refreshed_at"]


async def test_a_slow_provider_answers_with_a_stated_failure_rather_than_hanging(monkeypatch) -> None:
    """A browser panel must not hold a request open for minutes.

    Past the deadline the request answers with the reason, and the read it
    started keeps running -- so the next caller gets the rows rather than
    starting the same slow read again.
    """

    monkeypatch.setattr(app_module, "TPU_QUOTA_READ_DEADLINE_S", 0.05)
    discovery = FakeDiscovery(spot=(_availability(),), on_demand=(), delay_s=0.6)
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True), quota_discovery=discovery)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:

        async def quota() -> dict[str, tp.Any]:
            response = await client.get("/api/v1/capacity/quota", headers=_headers(), params={"project": "gcp-project"})
            assert response.status_code == 200, response.text
            return response.json()

        payload = await quota()

        assert payload["status"] == "unavailable"
        assert any("did not answer within" in failure["reason"] for failure in payload["failures"])

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            later = await quota()
            if later["entries"]:
                assert later["entries"][0]["free_chips"] == 16
                return
            await asyncio.sleep(0.1)
    raise AssertionError("the background provider read never landed in the cache")


def test_an_unknown_usage_row_cannot_carry_an_invented_balance() -> None:
    """The wire model refuses the shape this whole feature exists to prevent."""

    with pytest.raises(ValidationError, match="must not carry a usage number"):
        TpuQuotaEntry(
            zone="us-central2-b",
            region="us-central2",
            generation="v6e",
            accelerator_type="v6e-8",
            capacity_class=CapacityClass.SPOT,
            metric="tpu.googleapis.com/tpu-v6e-preemptible",
            chips_per_slice=8,
            limit_chips=64,
            free_chips=64,
            state=TpuQuotaState.USAGE_UNKNOWN,
        )


def test_a_known_row_cannot_report_more_free_capacity_than_its_own_limit() -> None:
    with pytest.raises(ValidationError, match="must not exceed its own limit"):
        TpuQuotaEntry(
            zone="us-central2-b",
            region="us-central2",
            generation="v6e",
            accelerator_type="v6e-8",
            capacity_class=CapacityClass.SPOT,
            metric="tpu.googleapis.com/tpu-v6e-preemptible",
            chips_per_slice=8,
            limit_chips=8,
            in_use_chips=0,
            free_chips=16,
            free_slices=2,
            state=TpuQuotaState.KNOWN,
        )


def test_an_entry_derived_from_a_slice_type_divides_by_that_slice() -> None:
    """v4 and v5p bill two quota units per schedulable resource; the suffix is the unit."""

    entry = TpuQuotaEntry.from_availability(
        _availability(accelerator_type="v4-32", quota_tpus=128, available_tpus=96),
        capacity_class=CapacityClass.SPOT,
        observed_at=OBSERVED_AT + timedelta(seconds=300),
    )

    assert entry.chips_per_slice == 32
    assert entry.free_slices == 3
    assert entry.usage_age_s == 300


class FakeCliClient:
    """Answer the quota route with one readable and one unreadable zone."""

    def __init__(self, payload: dict[str, tp.Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, dict[str, tp.Any]]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, tp.Any] | None = None,
        json: dict[str, tp.Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tp.Any:
        del json, headers
        self.calls.append((method, path, params or {}))
        return self.payload

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        del headers


def _cli_payload() -> dict[str, tp.Any]:
    return {
        "observed_at": "2026-07-29T12:05:00Z",
        "provider": "gcp",
        "project": "gcp-project",
        "available_projects": ["gcp-project"],
        "status": "ok",
        "refreshed_at": "2026-07-29T12:04:00Z",
        "age_s": 60.0,
        "cached": True,
        "ttl_s": 120.0,
        "entries": [
            {
                "zone": "us-central2-b",
                "region": "us-central2",
                "generation": "v6e",
                "accelerator_type": "v6e-8",
                "capacity_class": "spot",
                "metric": "tpu.googleapis.com/tpu-v6e-preemptible",
                "chips_per_slice": 8,
                "limit_chips": 32,
                "in_use_chips": 16,
                "free_chips": 16,
                "free_slices": 2,
                "state": "known",
                "usage_observed_at": "2026-07-29T12:00:00Z",
                "usage_age_s": 300.0,
            },
            {
                "zone": "europe-west4-a",
                "region": "europe-west4",
                "generation": "v6e",
                "accelerator_type": "v6e-8",
                "capacity_class": "spot",
                "metric": "tpu.googleapis.com/tpu-v6e-preemptible",
                "chips_per_slice": 8,
                "limit_chips": 64,
                "in_use_chips": None,
                "free_chips": None,
                "free_slices": None,
                "state": "usage_unknown",
                "usage_observed_at": None,
                "usage_age_s": None,
            },
        ],
        "failures": [],
    }


def _invoke(fake: FakeCliClient, monkeypatch, argv: list[str]):
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    return CliRunner().invoke(cli.main, ["--user", "alice", "--org", "acme", "--project", "research", *argv])


def test_the_cli_reaches_the_route_and_never_prints_an_unknown_balance_as_zero(monkeypatch) -> None:
    """``test_every_route_is_reachable_or_classified`` demands a caller; this is it."""

    fake = FakeCliClient(_cli_payload())

    result = _invoke(fake, monkeypatch, ["capacity", "quota"])

    assert result.exit_code == 0, result.output
    assert fake.calls[0][:2] == ("GET", "/api/v1/capacity/quota")
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.strip().startswith(("us-", "europe-"))}
    assert "16" in lines["us-central2-b"]
    unknown_row = lines["europe-west4-a"]
    # In use, free, slices, and the age of an observation that does not exist.
    assert unknown_row.count("unknown") == 4
    assert "0" not in unknown_row.replace("europe-west4-a", "")
    # The reading's real age, so a cached number is not read as a live one.
    assert "provider read 1m ago (cached)" in result.output


def test_the_cli_keeps_unknown_zones_out_of_a_fit_answer_without_dropping_them(monkeypatch) -> None:
    """ "We could not tell" is a different answer from "it does not fit"."""

    fake = FakeCliClient(_cli_payload())

    result = _invoke(fake, monkeypatch, ["capacity", "quota", "--fits", "v6e-8:2"])

    assert result.exit_code == 0, result.output
    assert "Fits 2 x v6e-8:" in result.output
    assert "Usage unknown -- not proven empty and not proven full:" in result.output
    fits_section, _, unknown_section = result.output.partition("Usage unknown")
    assert "us-central2-b" in fits_section
    assert "europe-west4-a" not in fits_section
    assert "europe-west4-a" in unknown_section


def test_the_cli_refuses_to_claim_a_fit_when_nothing_was_readable(monkeypatch) -> None:
    payload = {
        **_cli_payload(),
        "status": "unavailable",
        "entries": [],
        "failures": [
            {"capacity_class": "spot", "reason": "gcloud auth token lookup failed"},
            {"capacity_class": "on_demand", "reason": "gcloud auth token lookup failed"},
        ],
    }
    fake = FakeCliClient(payload)

    result = _invoke(fake, monkeypatch, ["capacity", "quota"])

    assert result.exit_code != 0
    assert "gcloud auth token lookup failed" in result.output
    assert "no TPU quota could be read" in result.output


def test_the_cli_names_an_unconfigured_fleet_rather_than_printing_an_empty_table(monkeypatch) -> None:
    payload = {
        "observed_at": "2026-07-29T12:05:00Z",
        "provider": "gcp",
        "project": None,
        "available_projects": [],
        "status": "unconfigured",
        "refreshed_at": None,
        "age_s": None,
        "cached": False,
        "ttl_s": 120.0,
        "entries": [],
        "failures": [],
    }

    result = _invoke(FakeCliClient(payload), monkeypatch, ["capacity", "quota"])

    assert result.exit_code == 0, result.output
    assert "No GCP capacity pool declares a provider project" in result.output


def test_the_cli_lists_a_measured_zero_and_never_lists_an_unknown_zone_twice(monkeypatch) -> None:
    """A measured zero is a real answer. An unmeasured zone belongs in one place.

    Both halves matter: dropping a full zone hides the fact that it is full, and
    printing an unmeasured zone in the free-capacity table would put a row with
    no balance next to rows that have one.
    """

    payload = _cli_payload()
    payload["entries"][0] = {
        **payload["entries"][0],
        "zone": "us-central2-b",
        "in_use_chips": 32,
        "free_chips": 0,
        "free_slices": 0,
    }
    fake = FakeCliClient(payload)

    result = _invoke(fake, monkeypatch, ["capacity", "quota"])

    assert result.exit_code == 0, result.output
    free_section, _, unknown_section = result.output.partition("Usage unknown")
    assert "us-central2-b" in free_section
    assert "europe-west4-a" not in free_section
    assert unknown_section.count("europe-west4-a") == 1


def test_the_cli_rejects_a_fit_request_it_cannot_parse(monkeypatch) -> None:
    result = _invoke(FakeCliClient(_cli_payload()), monkeypatch, ["capacity", "quota", "--fits", "v6e-8:0"])

    assert result.exit_code != 0
    assert "--fits count must be a positive integer" in result.output


def _record_refusal(cache_dir, *, zone: str = "us-central2-b", accelerator_type: str = "v6e-8") -> None:
    from eyvan.capacity.autoscale.cache import AutoscaleConfigCache

    AutoscaleConfigCache(cache_dir).record_queue_admission_denial(
        project="gcp-project",
        zone=zone,
        accelerator_type=accelerator_type,
        capacity_class=CapacityClass.SPOT,
    )


def test_the_cli_never_lists_a_queue_that_refused_us_as_free_capacity(monkeypatch, tmp_path) -> None:
    """Quota is not access.

    The provisioning path records queue-admission refusals locally. The quota
    view must use those refusals rather than present an inaccessible queue's
    limit as capacity somebody can take.
    """

    _record_refusal(tmp_path)
    fake = FakeCliClient(_cli_payload())

    result = _invoke(fake, monkeypatch, ["capacity", "quota", "--cache-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Spot queue refused this project" in result.output
    free_section, _, refused_section = result.output.partition("Spot queue refused this project")
    assert "us-central2-b" not in free_section
    assert "us-central2-b" in refused_section
    # The refusal is exact: an unrelated zone keeps its own answer.
    assert "europe-west4-a" in free_section


def test_a_refused_queue_never_answers_a_fit_question(monkeypatch, tmp_path) -> None:
    _record_refusal(tmp_path)

    result = _invoke(
        FakeCliClient(_cli_payload()),
        monkeypatch,
        ["capacity", "quota", "--fits", "v6e-8:2", "--cache-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    fits_section, _, _ = result.output.partition("Spot queue refused this project")
    assert "us-central2-b" not in fits_section.partition("Fits 2 x v6e-8:")[2]


def test_a_refusal_for_another_type_or_project_changes_nothing(monkeypatch, tmp_path) -> None:
    _record_refusal(tmp_path, accelerator_type="v6e-16")
    from eyvan.capacity.autoscale.cache import AutoscaleConfigCache

    AutoscaleConfigCache(tmp_path).record_queue_admission_denial(
        project="other-project",
        zone="us-central2-b",
        accelerator_type="v6e-8",
        capacity_class=CapacityClass.SPOT,
    )

    result = _invoke(FakeCliClient(_cli_payload()), monkeypatch, ["capacity", "quota", "--cache-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Spot queue refused this project" not in result.output
    assert "us-central2-b" in result.output


def test_an_unreadable_refusal_cache_does_not_break_the_quota_view(monkeypatch, tmp_path) -> None:
    (tmp_path / "queue-admission-denials.json").write_text("{not json", encoding="utf-8")

    result = _invoke(FakeCliClient(_cli_payload()), monkeypatch, ["capacity", "quota", "--cache-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "us-central2-b" in result.output


def _apply_two_gcp_projects(client: TestClient, *, first: str, second: str) -> None:
    for idx, project in enumerate((first, second), start=1):
        applied = client.put(
            f"/api/v1/fleets/fleet-{idx}",
            headers={**_headers(), "Idempotency-Key": f"fleet-{project}-{idx}"},
            json={
                "pools": [
                    {
                        "id": f"v6e-pool-{idx}",
                        "fleet_name": f"fleet-{idx}",
                        "node_type": "v6e-workers",
                        "provider": "gcp",
                        "capacity_backend": "gcp-tpu",
                        "provider_project": project,
                        "region": "us-central2",
                        "zone": "us-central2-b",
                        "capacity": {"cpu": 180, "memory_gib": 720, "tpu": 8},
                        "allocatable": {"cpu": 180, "memory_gib": 720, "tpu": 8},
                        "accelerator_type": "v6e-8",
                        "capacity_class": "spot",
                        "executor": "native-local",
                        "agent_endpoint": "http://agent:8090",
                        "min_cells": 0,
                        "max_cells": 4,
                    }
                ]
            },
        )
        assert applied.status_code == 200, applied.text


def test_quota_defaults_to_active_operator_project_instead_of_alphabetical_first(monkeypatch, tmp_path) -> None:
    """Prefer the operator's active project from ~/.eyvan config when multiple fleet projects exist."""

    import json

    from eyvan.storage import StorageConfig, StorageConfigStore, clear_gcp_context_cache

    monkeypatch.setenv("HOME", str(tmp_path))
    clear_gcp_context_cache()

    discovery = FakeDiscovery(spot=(_availability(),))
    client = _client(discovery)
    _apply_two_gcp_projects(client, first="example-project-a", second="example-project-b")

    # 1. Native runtime profile selects example-project-b over alphabetical example-project-a.
    runtime_file = tmp_path / ".eyvan" / "gcp-native-runtime.json"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text(json.dumps({"project_id": "example-project-b"}), encoding="utf-8")

    resp = client.get("/api/v1/capacity/quota", headers=_headers())
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["project"] == "example-project-b"
    assert payload["available_projects"] == ["example-project-a", "example-project-b"]

    # 2. Removing native runtime falls back to storage.json project when present in available_projects.
    runtime_file.unlink()
    StorageConfigStore(tmp_path / ".eyvan" / "storage.json").save(StorageConfig(project="example-project-b"))
    clear_gcp_context_cache()

    resp2 = client.get("/api/v1/capacity/quota", headers=_headers())
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["project"] == "example-project-b"

    # 3. Explicit ?project= always wins over the smart default.
    resp3 = client.get("/api/v1/capacity/quota", headers=_headers(), params={"project": "example-project-a"})
    assert resp3.status_code == 200, resp3.text
    assert resp3.json()["project"] == "example-project-a"


def test_tpu_quota_failure_parses_protobuf_permission_denied_wall_and_all_codes() -> None:
    """Raw gcloud ErrorInfo walls become clean 1-sentence diagnostics with actionable remediation."""

    from eyvan.server.api.schemas import TpuQuotaFailure

    raw_gcloud_wall = (
        "gcloud beta quotas info list failed: ERROR: (gcloud.beta.quotas.info.list) PERMISSION_DENIED: "
        "Permission 'cloudquotas.quotas.get' denied on resource "
        "'//cloudquotas.googleapis.com/projects/example-project-a/locations/global/services/tpu.googleapis.com'. "
        "This command is authenticated as alice@example.com which is the active account specified by the "
        "[core/account] property.\n"
        "- '@type': type.googleapis.com/google.rpc.ErrorInfo\n"
        "  domain: iam.googleapis.com\n"
        "  metadata:\n"
        "    permission: cloudquotas.quotas.get\n"
        "    resource: projects/example-project-a\n"
        "  reason: IAM_PERMISSION_DENIED\n"
        "- '@type': type.googleapis.com/google.iam.v1.logging.AuditData\n"
        "  error_info_id: Cg5pYW0uZ29vZ2xlYXBpcy5jb21BQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWjAxMjM0NTY3ODk=\n"
        "  troubleshooter_url: https://console.cloud.google.com/iam-admin/troubleshooter;permissions=cloudquotas.quotas.get\n"
    )
    failure = TpuQuotaFailure.from_reason(capacity_class=CapacityClass.SPOT, reason=raw_gcloud_wall)
    assert failure.code == "permission_denied"
    assert failure.permission == "cloudquotas.quotas.get"
    assert failure.account == "alice@example.com"
    assert failure.summary is not None
    assert "alice@example.com" in failure.summary
    assert "cloudquotas.quotas.get" in failure.summary
    assert "example-project-a" in failure.summary
    for noise in ("@type", "type.googleapis.com", "error_info_id", "troubleshooter_url", "Cg5pYW0"):
        assert noise not in failure.summary
    assert failure.remediation is not None
    assert "roles/cloudquotas.viewer" in failure.remediation
    assert "--member=user:alice@example.com" in failure.remediation
    assert failure.reason.startswith("gcloud beta quotas info list failed:")

    auth_failure = TpuQuotaFailure.from_reason(
        capacity_class=CapacityClass.SPOT,
        reason="gcloud auth token lookup failed: UNAUTHENTICATED: no credentials found",
        project="my-proj",
    )
    assert auth_failure.code == "auth_required"
    assert auth_failure.remediation is not None and "gcloud auth login" in auth_failure.remediation

    disabled_failure = TpuQuotaFailure.from_reason(
        capacity_class=CapacityClass.ON_DEMAND,
        reason=(
            "PERMISSION_DENIED: Cloud Quotas API has not been used in project my-proj before or it is disabled. "
            "Enable it by visiting https://console.developers.google.com/apis/api/cloudquotas.googleapis.com"
        ),
        project="my-proj",
    )
    assert disabled_failure.code == "api_disabled"
    assert disabled_failure.remediation is not None
    assert "gcloud services enable cloudquotas.googleapis.com --project=my-proj" in disabled_failure.remediation

    not_found = TpuQuotaFailure.from_reason(
        capacity_class=CapacityClass.SPOT,
        reason="ERROR: NOT_FOUND: Requested project 'ghost-proj' does not exist.",
    )
    assert not_found.code == "project_not_found"
    assert not_found.summary is not None and "ghost-proj" in not_found.summary

    timed_out = TpuQuotaFailure.from_reason(
        capacity_class=CapacityClass.SPOT,
        reason="provider quota discovery for 'my-proj' did not answer within 25s",
    )
    assert timed_out.code == "timeout"

    missing_cli = TpuQuotaFailure.from_reason(
        capacity_class=CapacityClass.SPOT,
        reason="gcloud beta quotas info list failed: [Errno 2] No such file or directory: 'gcloud'",
    )
    assert missing_cli.code == "gcloud_missing"

    generic = TpuQuotaFailure.from_reason(
        capacity_class=CapacityClass.SPOT,
        reason=(
            "gcloud beta quotas info list failed: unexpected upstream gateway error\n"
            "- '@type': type.googleapis.com/google.rpc.DebugInfo\n"
            "  error_info_id: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        ),
        project="my-proj",
    )
    assert generic.code == "quota_read_failed"
    assert generic.summary is not None
    assert "@type" not in generic.summary
    assert "error_info_id" not in generic.summary


def test_storage_status_endpoint_and_backend_error_handlers(monkeypatch, tmp_path) -> None:
    """Verify GET /api/v1/storage/status auth/payload and clean 503 handlers for storage/cell errors."""

    from eyvan.storage import (
        MonitoringBucketConfig,
        RegionalBucketConfig,
        StorageConfig,
        StorageConfigStore,
        clear_gcp_context_cache,
    )
    from eyvan.worker.agent.client import CellAgentUnavailableError

    monkeypatch.setenv("HOME", str(tmp_path))
    clear_gcp_context_cache()

    cfg_path = tmp_path / ".eyvan" / "storage.json"
    StorageConfigStore(cfg_path).save(
        StorageConfig(
            project="example-project-a",
            auto_create=True,
            monitoring=MonitoringBucketConfig(
                bucket="eyvan_monitoring_example-project-a",
                location="us-central1",
                prefix="tracking",
            ),
            regional={
                "us-central2": RegionalBucketConfig(
                    bucket="eyvan_us-central2",
                    location="us-central2",
                )
            },
        )
    )

    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, controller_enabled=False))
    app.state.storage_config_path = cfg_path
    with TestClient(app) as client:
        unauth = client.get("/api/v1/storage/status")
        assert unauth.status_code == 401

        resp = client.get("/api/v1/storage/status", headers=_headers("project_viewer"))
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["configured"] is True
        assert data["auto_create"] is True
        assert data["config_path"] == str(cfg_path)
        assert data["project"] == "example-project-a"
        assert data["monitoring"]["bucket"] == "eyvan_monitoring_example-project-a"
        assert data["monitoring"]["uri"] == "gs://eyvan_monitoring_example-project-a/tracking"
        assert data["regional"]["us-central2"]["bucket"] == "eyvan_us-central2"
        assert set(data["context"]) >= {"on_gcp", "project", "zone", "region", "accelerator_type", "source"}

        # Corrupt storage config yields clean 503 JSON instead of a 500 traceback.
        cfg_path.write_text("{corrupt", encoding="utf-8")
        corrupt = client.get("/api/v1/storage/status", headers=_headers())
        assert corrupt.status_code == 503
        assert corrupt.json()["code"] == "storage_unavailable"

        # Unhandled CellAgentUnavailableError returns clean 503 JSON without leaking secrets.
        async def unreachable_orphans():
            raise CellAgentUnavailableError("connect refused with bearer super-secret-token")

        monkeypatch.setattr(app.state.controller, "capacity_orphans", unreachable_orphans)
        cell_err = client.get("/api/v1/capacity/orphans", headers=_headers())
        assert cell_err.status_code == 503
        assert cell_err.headers.get("retry-after") == "5"
        assert cell_err.json()["code"] == "cell_agent_unavailable"
        assert "super-secret-token" not in cell_err.text
