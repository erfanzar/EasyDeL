# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every TPU in the account is a usable fleet name before anyone connects it.

Discovery is read-only (projects and node listings only), keeps an unreadable
project *unknown* rather than empty, and ``init(fleet=...)`` connects a
discovered TPU the first time a job names it -- by fleet name or TPU name --
and never re-examines a fleet the plane already serves.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from eyvan.capacity.providers.gcp.tpu_discovery import (
    DiscoveryIndex,
    DiscoverySnapshot,
    discover,
    refresh_in_background,
)
from eyvan.capacity.providers.gcp.tpu_nodes import (
    GcpProject,
    TpuApiError,
    attach_fleet_name,
    parse_tpu_node,
)
from eyvan.sdk.client.transport import EyvanHTTPError


def _node(project: str, zone: str, name: str, *, spot: bool = False, kind: str = "v5p-8"):
    return parse_tpu_node(
        {
            "name": f"projects/{project}/locations/{zone}/nodes/{name}",
            "id": f"{abs(hash((project, zone, name))) % 10**9}",
            "acceleratorType": kind,
            "state": "READY",
            "schedulingConfig": {"spot": spot},
            "networkEndpoints": [{"ipAddress": "10.0.0.2", "accessConfig": {"externalIp": "34.0.0.2"}}],
        },
        project=project,
    )


class FakeApi:
    """Records every call; any method beyond the two listings would fail the test."""

    def __init__(self, nodes: dict[str, list], failures: dict[str, Exception] | None = None) -> None:
        self.nodes = nodes
        self.failures = failures or {}
        self.calls: list[tuple[str, str | None]] = []

    def list_projects(self):
        self.calls.append(("list_projects", None))
        return tuple(GcpProject(project, None, "ACTIVE") for project in sorted({*self.nodes, *self.failures}))

    def list_nodes(self, project, *, zone="-"):
        self.calls.append(("list_nodes", project))
        if project in self.failures:
            raise self.failures[project]
        return tuple(self.nodes[project]), ()

    def __getattr__(self, name):  # get_node, create, delete... are never expected
        raise AssertionError(f"discovery must only list; it called {name}")


def _account() -> FakeApi:
    return FakeApi(
        {
            "example-spot-project": [
                _node("example-spot-project", "us-east5-a", "spot-train", spot=True, kind="v6e-16")
            ],
            "example-research-project": [
                _node("example-research-project", "us-central1-a", "lab_spot_m"),
                _node("example-research-project", "us-central1-a", "v5p-trainer-1", kind="v5p-32"),
            ],
        },
        failures={"no-tpu-api": TpuApiError("Cloud TPU API has not been used in project", retryable=False, status=403)},
    )


def test_discovery_lists_every_projects_tpus_with_the_connect_fleet_name() -> None:
    api = _account()
    snapshot = discover(api, clock=lambda: 1000.0)

    names = {(tpu.project, tpu.name): tpu for tpu in snapshot.tpus}
    assert set(names) == {
        ("example-spot-project", "spot-train"),
        ("example-research-project", "lab_spot_m"),
        ("example-research-project", "v5p-trainer-1"),
    }
    # The very name `eyvan gcp connect` gives the fleet.
    spot = names[("example-spot-project", "spot-train")]
    assert spot.fleet == attach_fleet_name(project="example-spot-project", zone="us-east5-a", name="spot-train")
    assert spot.capacity_class == "spot" and spot.accelerator_type == "v6e-16"
    # Unreadable is unknown, not "no TPUs", and does not stop the rest.
    assert snapshot.skipped == {"no-tpu-api": "Cloud TPU API is not enabled"}
    assert {call for call, _ in api.calls} == {"list_projects", "list_nodes"}


def test_resolve_accepts_a_fleet_name_or_an_unambiguous_tpu_name() -> None:
    snapshot = discover(_account())
    tpu = snapshot.resolve("lab_spot_m")
    assert tpu is not None and tpu.project == "example-research-project"
    assert snapshot.resolve(tpu.fleet) == tpu
    assert snapshot.resolve("not-a-tpu") is None

    twins = FakeApi(
        {
            "a-project-1": [_node("a-project-1", "us-east5-a", "dev")],
            "b-project-1": [_node("b-project-1", "europe-west4-b", "dev")],
        }
    )
    doubled = discover(twins)
    assert doubled.resolve("dev") is None
    assert len(doubled.ambiguous("dev")) == 2


def test_the_index_is_owner_only_and_round_trips(tmp_path) -> None:
    index = DiscoveryIndex(tmp_path / "attached" / "discovered.json")
    assert index.load() is None
    saved = index.refresh(_account())

    mode = stat.S_IMODE(os.stat(index.path).st_mode)
    assert mode == 0o600
    assert index.load() == saved
    assert json.loads(index.path.read_text())["version"] == 1


def test_a_failed_refresh_keeps_the_previous_index(tmp_path) -> None:
    index = DiscoveryIndex(tmp_path / "discovered.json")
    saved = index.refresh(_account())

    class Down(FakeApi):
        def list_projects(self):
            raise TpuApiError("could not list GCP projects: HTTP 503", retryable=True, status=503)

    with pytest.raises(TpuApiError):
        index.refresh(Down({}))
    assert index.load() == saved


def test_fresh_reuses_a_young_index_and_background_refresh_skips_it(tmp_path) -> None:
    index = DiscoveryIndex(tmp_path / "discovered.json")
    index.save(DiscoverySnapshot(discovered_at=__import__("time").time()))
    untouched = FakeApi({})
    assert index.fresh(api=untouched).tpus == ()
    assert untouched.calls == []
    assert refresh_in_background(index) is None


def test_a_stale_index_is_refreshed_in_the_background_and_sync_is_idempotent(tmp_path) -> None:
    api = _account()
    index = DiscoveryIndex(tmp_path / "discovered.json", api=api)
    index.save(DiscoverySnapshot(discovered_at=0.0))  # decades old

    thread = refresh_in_background(index)
    assert thread is not None
    thread.join(timeout=10)
    first = index.load()
    assert first is not None and len(first.tpus) == 3

    # A second pass over the same account yields the same inventory.
    second = index.refresh()
    assert second.tpus == first.tpus and second.skipped == first.skipped
    assert {call for call, _ in api.calls} == {"list_projects", "list_nodes"}


def test_periodic_discovery_refreshes_on_every_interval_and_survives_failures(tmp_path) -> None:
    import threading

    from eyvan.capacity.providers.gcp.tpu_discovery import PeriodicDiscovery

    calls = {"n": 0}
    third = threading.Event()

    class Flaky(FakeApi):
        def list_projects(self):
            calls["n"] += 1
            if calls["n"] >= 3:
                third.set()
            if calls["n"] == 2:
                raise TpuApiError("HTTP 503", retryable=True, status=503)
            return super().list_projects()

    errors: list[Exception] = []
    index = DiscoveryIndex(tmp_path / "discovered.json", api=Flaky(_account().nodes))
    periodic = PeriodicDiscovery(index, interval_s=0.05, on_error=errors.append)
    periodic.start()
    try:
        assert third.wait(10), "the refresher stopped after a failed pass"
    finally:
        periodic.stop(timeout=5)
    # First pass ran at once, the failure was reported and did not end the loop,
    # and the index still holds the last good listing.
    assert len(errors) >= 1
    snapshot = index.load()
    assert snapshot is not None and len(snapshot.tpus) == 3


def test_periodic_discovery_refuses_a_non_positive_interval() -> None:
    from eyvan.capacity.providers.gcp.tpu_discovery import PeriodicDiscovery

    with pytest.raises(ValueError):
        PeriodicDiscovery(interval_s=0)


def test_a_failed_background_refresh_never_raises_and_keeps_the_index(tmp_path) -> None:
    class Down(FakeApi):
        def list_projects(self):
            raise TpuApiError("HTTP 503", retryable=True, status=503)

    index = DiscoveryIndex(tmp_path / "discovered.json", api=Down({}))
    stale = DiscoverySnapshot(discovered_at=0.0)
    index.save(stale)
    thread = refresh_in_background(index)
    assert thread is not None
    thread.join(timeout=10)
    assert index.load() == stale


# ------------------------------------------------------------ lazy connect


class _Transport:
    def __init__(self, fleets: set[str]) -> None:
        self.fleets = fleets
        self.reads: list[str] = []

    def request_checked(self, method, path, **kwargs):
        import httpx

        assert method == "GET"
        name = path.removeprefix("/api/v1/fleets/")
        self.reads.append(name)
        if name not in self.fleets:
            raise EyvanHTTPError(httpx.Response(404, request=httpx.Request("GET", "http://plane" + path)), "not found")
        return httpx.Response(200, json={"pools": []})


class _Client:
    def __init__(self, fleets: set[str]) -> None:
        self._transport = _Transport(fleets)


def _lazy(tmp_path, fleet, *, connected=(), exit_code=0, api=None):
    from eyvan.sdk.execution import _connect_discovered_tpu

    # The same fake answers any refresh the lookup makes: no test reaches gcloud.
    index = DiscoveryIndex(tmp_path / "discovered.json", api=api or _account())
    index.refresh()
    connects: list[tuple[str, str, str, str]] = []
    notes: list[str] = []
    client = _Client(set(connected))

    def connect(url, project, zone, name):
        connects.append((url, project, zone, name))
        client._transport.fleets.add(attach_fleet_name(project=project, zone=zone, name=name))
        return exit_code

    result = _connect_discovered_tpu(
        client, fleet, "http://127.0.0.1:8080", index=index, connect=connect, note=notes.append
    )
    return result, connects, notes, client


def test_a_discovered_tpu_is_connected_on_first_use_by_tpu_name(tmp_path) -> None:
    fleet, connects, notes, _ = _lazy(tmp_path, "v5p-trainer-1")

    assert fleet == attach_fleet_name(project="example-research-project", zone="us-central1-a", name="v5p-trainer-1")
    assert connects == [("http://127.0.0.1:8080", "example-research-project", "us-central1-a", "v5p-trainer-1")]
    assert any("connecting it now" in note for note in notes)


def test_a_discovered_tpu_is_connected_on_first_use_by_fleet_name(tmp_path) -> None:
    name = attach_fleet_name(project="example-spot-project", zone="us-east5-a", name="spot-train")
    fleet, connects, _, _ = _lazy(tmp_path, name)
    assert fleet == name
    assert connects == [("http://127.0.0.1:8080", "example-spot-project", "us-east5-a", "spot-train")]


def test_a_connected_fleet_is_never_looked_up_or_reconnected(tmp_path) -> None:
    name = attach_fleet_name(project="example-research-project", zone="us-central1-a", name="lab_spot_m")
    fleet, connects, notes, client = _lazy(tmp_path, name, connected={name})
    assert fleet == name and connects == [] and notes == []
    assert client._transport.reads == [name]

    # Named by its TPU name: mapped to the existing fleet, still not reconnected.
    fleet, connects, notes, _ = _lazy(tmp_path, "lab_spot_m", connected={name})
    assert fleet == name and connects == []
    assert any("using its fleet" in note for note in notes)


def test_an_unknown_name_is_left_to_the_placement_error(tmp_path) -> None:
    api = _account()
    fleet, connects, _, _ = _lazy(tmp_path, "research-us-east5-a", api=api)
    assert fleet == "research-us-east5-a" and connects == []
    # One re-listing in case the TPU is new, and only listings.
    assert [call for call, _ in api.calls].count("list_projects") == 2


def test_an_ambiguous_tpu_name_is_refused_with_both_fleet_names(tmp_path) -> None:
    twins = FakeApi(
        {
            "a-project-1": [_node("a-project-1", "us-east5-a", "dev")],
            "b-project-1": [_node("b-project-1", "europe-west4-b", "dev")],
        }
    )
    with pytest.raises(ValueError, match="more than one place") as raised:
        _lazy(tmp_path, "dev", api=twins)
    assert attach_fleet_name(project="a-project-1", zone="us-east5-a", name="dev") in str(raised.value)


def test_a_failed_connect_names_the_retry_command(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="eyvan gcp connect --tpu v5p-trainer-1 --project example-research-project"):
        _lazy(tmp_path, "v5p-trainer-1", exit_code=1)
