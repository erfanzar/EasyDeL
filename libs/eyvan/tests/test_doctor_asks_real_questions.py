# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Two ways ``eyvan doctor`` was not answering the question it exists for.

The fleet check called ``GET /api/v1/fleets`` with no query parameters, but the
route requires ``org_id`` and ``project_id``. It therefore returned 422 on every
invocation in every scope, so the check always failed, ``doctor`` always exited
non-zero, and whether a fleet was actually submittable was never established.

The existing doctor fake accepts ``**_kwargs`` and drops them, which is why no
test caught it -- the request looked identical with or without the parameters.
These tests record what was sent.

The second gap is drift rather than a defect: in a source checkout the runtime
profile pins one content-addressed wheel and nothing re-stages it when the
source changes, so a run silently deploys whatever was staged last. That bit
this session three separate times, so ``doctor`` now says it out loud.
"""

from __future__ import annotations

import os
import time
import typing as tp
from pathlib import Path

import pytest
from test_doctor import _healthy, _Store

from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.doctor import CheckStatus, DoctorCommandService
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContextConfig


class _RecordingClient:
    """A client that keeps the query parameters each call actually sent."""

    def __init__(self, handler: tp.Callable[[str], tp.Any]) -> None:
        self._handler = handler
        self.calls: list[tuple[str, dict[str, tp.Any]]] = []

    def request(self, method: str, path: str, **kwargs: tp.Any) -> tp.Any:
        del method
        self.calls.append((path, dict(kwargs.get("params") or {})))
        return self._handler(path)


def _session(client: _RecordingClient, *, org: str = "acme", project: str = "research") -> CliSession:
    return CliSession(
        client=client,
        identity=ClientIdentity(),
        context_store=_Store(),
        context_config=EyvanContextConfig(),
        base_url="http://test",
        org_id=org,
        project_id=project,
        overrides=ContextOverrides(),
    )


def test_the_fleet_check_sends_the_scope_the_route_requires() -> None:
    """Without both parameters the route answers 422, never a fleet list."""

    client = _RecordingClient(_healthy())

    report = DoctorCommandService(_session(client)).report()

    fleet_params = next(params for path, params in client.calls if path.endswith("/fleets"))
    assert fleet_params == {"org_id": "acme", "project_id": "research"}
    assert {check.name: check.status for check in report.checks}["submittable-fleet"] is CheckStatus.PASS


def test_the_fleet_check_uses_the_session_scope_not_a_constant() -> None:
    """A different context must ask about that context's fleets."""

    client = _RecordingClient(_healthy())

    DoctorCommandService(_session(client, org="other-org", project="other-project")).report()

    fleet_params = next(params for path, params in client.calls if path.endswith("/fleets"))
    assert fleet_params == {"org_id": "other-org", "project_id": "other-project"}


def test_a_wheel_staged_after_the_source_is_reported_as_current(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No warning when nothing has changed, or the warning becomes noise."""

    profile = tmp_path / "runtime.json"
    profile.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("eyvan.sdk.commands.doctor_preflights.DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH", str(profile))

    monkeypatch.setattr("eyvan.sdk.commands.doctor.stale_components", lambda **_kwargs: ())

    check = DoctorCommandService(_session(_RecordingClient(_healthy())))._source_drift_check()

    assert check.status is CheckStatus.PASS


def test_a_wheel_older_than_the_source_is_detected(tmp_path: Path) -> None:
    """Exercise the real drift computation, not a stub standing in for it.

    ``stale_components`` takes its inputs explicitly precisely so this can be
    asserted without depending on what the machine running it has deployed.
    """

    from eyvan.sdk.commands.deployment import WORKER_WHEEL_COMPONENT, stale_components

    profile = tmp_path / "runtime.json"
    profile.write_text("{}", encoding="utf-8")
    old = time.time() - 86_400
    os.utime(profile, (old, old))

    stale = stale_components(runtime_profile_path=profile, newest_source=time.time(), component_pids={})

    assert stale == (WORKER_WHEEL_COMPONENT,)


def test_a_wheel_newer_than_the_source_is_not_reported(tmp_path: Path) -> None:
    """A current deployment must stay silent or the warning becomes noise."""

    from eyvan.sdk.commands.deployment import stale_components

    profile = tmp_path / "runtime.json"
    profile.write_text("{}", encoding="utf-8")

    stale = stale_components(
        runtime_profile_path=profile,
        newest_source=time.time() - 86_400,
        component_pids={},
    )

    assert stale == ()


def test_a_running_component_older_than_the_source_is_detected(tmp_path: Path, monkeypatch) -> None:
    """The surface that actually bit: a daemon left behind by a partial deploy."""

    from eyvan.sdk.commands import deployment

    profile = tmp_path / "runtime.json"
    profile.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(deployment, "process_started_at", lambda pid: 0.0)

    stale = deployment.stale_components(
        runtime_profile_path=profile,
        newest_source=time.time(),
        component_pids={"cell agent (http://127.0.0.1:8090)": 4242},
    )

    assert "cell agent (http://127.0.0.1:8090)" in stale


def test_no_runtime_profile_is_not_a_finding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Most deployments never stage a native GCP worker at all."""

    monkeypatch.setattr(
        "eyvan.sdk.commands.doctor_preflights.DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH",
        str(tmp_path / "absent.json"),
    )
    monkeypatch.setattr("eyvan.sdk.commands.doctor.stale_components", lambda **_kwargs: ())

    check = DoctorCommandService(_session(_RecordingClient(_healthy())))._source_drift_check()

    assert check.status is CheckStatus.PASS


def test_component_discovery_finds_the_real_pid_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise discovery itself, not a stub standing in for it.

    The other drift tests substitute ``_running_component_pids`` so their
    outcome does not depend on what this machine is running -- which means none
    of them would notice if discovery stopped finding anything. It already did
    once: the first version derived each record path from the control plane's
    endpoint, but the cell agent is recorded under *its own* port, so it found
    neither and a broad ``except`` turned that into a green check.
    """

    import json as _json

    from eyvan.server.local import daemon as daemon_module

    for name, port in (("control-plane-8081-abc", 8081), ("cell-agent-8090-def", 8090)):
        (tmp_path / f"{name}.pid").write_text(
            _json.dumps({"endpoint": f"http://127.0.0.1:{port}", "pid": 1000 + port}),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        daemon_module.LocalDaemonPaths,
        "resolved_directory",
        property(lambda self: tmp_path),
    )

    from eyvan.sdk.commands.deployment import running_component_pids

    discovered = running_component_pids()

    assert set(discovered.values()) == {9081, 9090}
    assert any("control plane" in label for label in discovered)
    assert any("cell agent" in label for label in discovered), "the agent is recorded under its own port"


def _fleet_with(cells: list[dict]) -> tp.Callable[[str], tp.Any]:
    """A control plane holding one pool and whatever cells a test needs."""

    pool = {
        "id": "research/cpu",
        "fleet_name": "research",
        "node_type": "cpu",
        "provider": "local",
        "region": "local",
        "capacity": {"cpu": 4.0, "memory_gib": 16.0},
        "allocatable": {"cpu": 4.0, "memory_gib": 16.0},
        "agent_endpoint": "http://agent:8090",
    }

    def handler(path: str) -> tp.Any:
        if path.endswith("/health"):
            return {"status": "ok", "controller": "active"}
        if path.endswith("/capacity-pools"):
            return [pool]
        if path.endswith("/cells"):
            return cells
        return [{"name": "research"}]

    return handler


def _cell(cell_id: str, *, prices: dict | None = None, state: str = "ready") -> dict:
    return {
        "id": cell_id,
        "provider": "local",
        "region": "local",
        "state": state,
        "pool_id": "research/cpu",
        "capacity": {"cpu": 4.0, "memory_gib": 16.0},
        "allocatable": {"cpu": 4.0, "memory_gib": 16.0},
        "endpoint": "http://agent:8090",
        "unit_prices": prices or {},
    }


def test_a_fleet_where_nothing_is_priced_is_not_reported_as_healthy_pricing() -> None:
    """Warn when no fleet cells are priced rather than treating a vacuous price-drift check as healthy."""

    client = _RecordingClient(_fleet_with([_cell("cell-a"), _cell("cell-b")]))

    report = DoctorCommandService(_session(client)).report()
    drift = next(check for check in report.checks if check.name == "price-drift")

    assert drift.status is CheckStatus.WARN
    assert "structurally zero" in drift.detail
    assert "2 cell(s)" in drift.detail
    assert drift.remedy and "unit_prices" in drift.remedy


def test_a_priced_fleet_still_passes() -> None:
    """The warning must not fire on the configuration it exists to encourage."""

    client = _RecordingClient(_fleet_with([_cell("cell-a", prices={"cpu": 0.05})]))

    report = DoctorCommandService(_session(client)).report()
    drift = next(check for check in report.checks if check.name == "price-drift")

    assert drift.status is CheckStatus.PASS


def test_deleted_cells_do_not_trigger_the_warning() -> None:
    """A fleet that has scaled to zero is not an unpriced fleet.

    Counting released cells would make the warning fire on every idle fleet,
    which is how a warning becomes noise and then becomes muted.
    """

    client = _RecordingClient(_fleet_with([_cell("cell-a", state="deleted")]))

    report = DoctorCommandService(_session(client)).report()
    drift = next(check for check in report.checks if check.name == "price-drift")

    assert drift.status is CheckStatus.PASS


def test_a_partly_priced_fleet_is_not_this_warning_s_business() -> None:
    """One unpriced cell among priced ones is drift, not a structural zero.

    The warning exists for the fleet that cannot compute cost *at all*. Firing
    it whenever any single cell lacks a price would duplicate the drift check
    below it and make both harder to act on -- and a one-cell fixture cannot
    tell the two rules apart, which is how the looser one survived a mutation.
    """

    client = _RecordingClient(_fleet_with([_cell("cell-a", prices={"cpu": 0.05}), _cell("cell-b")]))

    report = DoctorCommandService(_session(client)).report()
    drift = next(check for check in report.checks if check.name == "price-drift")

    assert "structurally zero" not in drift.detail


def _fleet_with_pools(pools: list[dict]) -> tp.Callable[[str], tp.Any]:
    """A control plane holding the given pools and no cells."""

    def handler(path: str) -> tp.Any:
        if path.endswith("/health"):
            return {"status": "ok", "controller": "active"}
        if path.endswith("/capacity-pools"):
            return pools
        if path.endswith("/cells"):
            return []
        return [{"name": "research"}]

    return handler


def _pool(pool_id: str, *, cpu: float = 32.0, memory_gib: float, enabled: bool = True) -> dict:
    return {
        "id": pool_id,
        "fleet_name": "research",
        "node_type": "tpu",
        "provider": "local",
        "region": "local",
        "enabled": enabled,
        "capacity": {"cpu": cpu, "memory_gib": memory_gib, "tpu": 8},
        "allocatable": {"cpu": cpu, "memory_gib": memory_gib, "tpu": 8},
        "agent_endpoint": "http://agent:8090",
    }


def test_a_pool_that_advertises_no_memory_is_reported() -> None:
    """Legacy rows stay visible and direct operators to the actual repair.

    New catalog and API paths reject zero memory, but hiding an old durable row
    would make it impossible for doctor to explain why the default 1 GiB
    workload still cannot place after a binary upgrade.
    """

    client = _RecordingClient(_fleet_with_pools([_pool("research/v5p", memory_gib=0.0)]))

    report = DoctorCommandService(_session(client)).report()
    check = next(c for c in report.checks if c.name == "pool-schedulability")

    assert check.status is CheckStatus.WARN
    assert "research/v5p" in check.detail
    assert "capacity.memory_gib" in check.detail
    assert check.remedy and "remove explicit zero" in check.remedy
    assert "eyvan up <fleet.yaml>" in check.remedy
    assert "eyvan autoscale generate" in check.remedy
    assert "quota/catalog cache is regenerated" in check.remedy
    assert "--memory-gib 0" not in check.remedy


def test_an_accelerator_pool_with_zero_cpu_and_positive_memory_is_reported() -> None:
    """Accelerator count and RAM do not make a zero-CPU host schedulable."""

    client = _RecordingClient(_fleet_with_pools([_pool("research/v5p", cpu=0.0, memory_gib=448.0)]))

    report = DoctorCommandService(_session(client)).report()
    check = next(c for c in report.checks if c.name == "pool-schedulability")

    assert check.status is CheckStatus.WARN
    assert "research/v5p (capacity.cpu, allocatable.cpu)" in check.detail
    assert check.remedy and "remove explicit zero" in check.remedy


def test_a_pool_with_memory_passes() -> None:
    """The gate must not fire on the configuration it exists to encourage."""

    client = _RecordingClient(_fleet_with_pools([_pool("research/cpu", memory_gib=16.0)]))

    report = DoctorCommandService(_session(client)).report()

    assert next(c for c in report.checks if c.name == "pool-schedulability").status is CheckStatus.PASS


def test_a_disabled_pool_is_not_reported() -> None:
    """A pool nobody can submit to cannot be the reason a submission fails."""

    client = _RecordingClient(_fleet_with_pools([_pool("research/old", memory_gib=0.0, enabled=False)]))

    report = DoctorCommandService(_session(client)).report()

    assert next(c for c in report.checks if c.name == "pool-schedulability").status is CheckStatus.PASS


def test_a_cpu_only_pool_is_outside_the_accelerator_repair_check() -> None:
    """This migration warning must not claim every legacy CPU pool is catalog-backed."""

    pool = _pool("research/cpu", cpu=0.0, memory_gib=0.0)
    pool["node_type"] = "cpu"
    pool["capacity"] = {"cpu": 0.0, "memory_gib": 0.0}
    pool["allocatable"] = {"cpu": 0.0, "memory_gib": 0.0}
    client = _RecordingClient(_fleet_with_pools([pool]))

    report = DoctorCommandService(_session(client)).report()

    assert next(c for c in report.checks if c.name == "pool-schedulability").status is CheckStatus.PASS
