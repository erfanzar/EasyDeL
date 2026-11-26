# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import shlex
import sys
import time

import pytest

from eyvan._internal.json_types import JsonValue
from eyvan.capacity.drivers.base import CapacityStatus, ExecutionWorkloadStatus, MutationContext
from eyvan.capacity.drivers.local import LocalCapacityDriver
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.config import EyvanSettings
from eyvan.worker.agent.models import CapacityRequest, MutationModel, WorkloadRequest
from eyvan.worker.agent.service import CellAgentService


def _context(key: str, generation: int = 1) -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=f"cell:local:{generation}")


def _desired() -> dict[str, JsonValue]:
    return {
        "provider": "local",
        "capacity_backend": "local",
        "region": "local",
        "accelerator_type": "local-tpu",
        "capacity_class": "on_demand",
        "executor": "native-local",
        "agent_endpoint": "http://127.0.0.1:18081",
    }


def test_local_capacity_driver_materializes_and_releases_a_development_cell(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    driver = LocalCapacityDriver(state)

    applied = driver.apply_capacity("local-cell", _desired(), _context("apply"))

    assert applied.state is CapacityStatus.HEALTHY
    assert applied.execution_endpoint == "local"
    assert applied.runtime_process_count == 1
    assert applied.metadata["capacity_backend"] == "local"
    assert applied.metadata["region"] == "local"
    assert driver.reconcile("local-cell", _context("reconcile")).state is CapacityStatus.HEALTHY
    assert driver.drain("local-cell", _context("drain")).state is CapacityStatus.DRAINING
    assert driver.release("local-cell", _context("release", generation=2)).state is CapacityStatus.RELEASED
    assert driver.observe("local-cell").state is CapacityStatus.ABSENT


def test_local_capacity_driver_rejects_a_non_local_executor(tmp_path) -> None:
    driver = LocalCapacityDriver(FleetStateStore(LocalBackend(tmp_path / "fleet.json")))
    desired = _desired()
    desired["executor"] = "eyvan-native"

    with pytest.raises(ValueError, match="requires executor 'native-local'"):
        driver.apply_capacity("local-cell", desired, _context("apply"))


def test_local_agent_runs_a_native_local_workload_without_gcp(tmp_path) -> None:
    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    service = CellAgentService(
        state=state,
        settings=EyvanSettings(capacity_backend="local", agent_state_dir=str(tmp_path / "runtime")),
    )
    try:
        applied = service.apply_capacity(
            "local-cell",
            CapacityRequest(
                mutation=MutationModel(idempotency_key="apply", generation=1, fencing_token="cell:local:1"),
                desired=_desired(),
            ),
        )
        assert applied.state is CapacityStatus.HEALTHY
        assert service.capabilities().executor == "native-local"

        command = shlex.join((sys.executable, "-c", "print('local-eyvan-workload')"))
        submitted = service.submit(
            "local-cell",
            WorkloadRequest(
                mutation=MutationModel(idempotency_key="submit", generation=1, fencing_token="lease:local:1"),
                spec={
                    "entrypoint": command,
                    "submission_id": "local-workload",
                    "environment": {},
                    "gang": {"replicas": 1, "per_replica": {"tpu": 1}, "required_topology": {}},
                    "metadata": {"eyvan_workload_id": "local-workload"},
                },
            ),
        )
        assert submitted.status is ExecutionWorkloadStatus.SUBMITTED

        deadline = time.monotonic() + 10
        while True:
            observation = service.workload("local-cell", "local-workload")
            if observation.status in {ExecutionWorkloadStatus.SUCCEEDED, ExecutionWorkloadStatus.FAILED}:
                break
            assert time.monotonic() < deadline
            time.sleep(0.05)

        assert observation.status is ExecutionWorkloadStatus.SUCCEEDED
        assert service.logs("local-cell", "local-workload") == "local-eyvan-workload\n"
    finally:
        service.close()
