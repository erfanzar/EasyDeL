# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolver-to-wire-to-JAX contracts for independent elastic TPU allocations.

HTTP and JAX are test doubles; these tests qualify emitted initialization
arguments, not metadata availability, network rendezvous, or TPU execution.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import httpx
import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CellObservation, MutationContext
from eyvan.records import WorkloadDefinition
from eyvan.worker.runtime import _multihost_site, native_remote
from eyvan.worker.runtime.registry import ExecutionDriverResolver, NativeGangCell


def _cells(*, gce_tpu: bool) -> tuple[NativeGangCell, ...]:
    cells = []
    for index, cell_id in enumerate(("cell-a", "cell-b")):
        endpoints = [f"http://203.0.113.{10 + index * 10 + host}:8100" for host in range(2)]
        metadata = {"management_uid": f"owned-{cell_id}", "native_worker_endpoints": endpoints}
        if gce_tpu:
            metadata["tpu_config"] = {"node_count": 1}
        cells.append(
            NativeGangCell(
                cell_id=cell_id,
                observation=CellObservation(
                    cell_id=cell_id,
                    state=CapacityStatus.HEALTHY,
                    generation=7,
                    execution_endpoint=endpoints[0],
                    metadata=metadata,
                ),
                context=MutationContext(idempotency_key=f"submit-{cell_id}", generation=7, fencing_token=cell_id),
            )
        )
    return tuple(cells)


def _capture_submission(monkeypatch, *, gce_tpu: bool, placement: str, single_cell: bool = False):
    definition = WorkloadDefinition.model_validate(
        {
            "name": "elastic-tpu-worlds",
            "execution": {"kind": "command", "command": "python train.py"},
            "provider": "gcp",
            "required_executor": "eyvan-native",
            "capacity_class": "spot",
            "gang": {"per_replica": {"tpu": 4}},
            "worker_count": 4,
            "worker_resource_scope": "per_worker",
            "placement": placement,
        }
    )
    submitted = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        payload = json.loads(request.content)
        submitted.append(payload["spec"]["environment"])
        return httpx.Response(200, json={"remote_id": "gang-job"})

    real_client = httpx.Client

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(native_remote.httpx, "Client", client_factory)
    resolver = ExecutionDriverResolver(native_worker_token="test-root-secret")
    cells = _cells(gce_tpu=gce_tpu)
    spec = {
        "submission_id": "gang-job",
        "entrypoint": definition.execution.command,
        "placement": definition.placement.value,
        # Even internal dispatch callers must not override the resolver's mode.
        "environment": {"EYVAN_JAX_CLUSTER_DETECTION": "gcetpu", "JAX_PROCESS_COUNT": "999"},
    }
    if single_cell:
        executor = resolver.resolve(cells[0].cell_id, cells[0].observation)
        try:
            assert executor.submit(spec, cells[0].context) == "gang-job"
        finally:
            resolver.close()
    else:
        executor = resolver.resolve_gang(cells)
        try:
            assert executor.submit(spec) == "gang-job"
        finally:
            executor.close()
            resolver.close()
    return submitted


def _jax_arguments(monkeypatch, environments):
    arguments = []

    def initialize(**kwargs):
        arguments.append(kwargs)

    distributed = SimpleNamespace(initialize=initialize, is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))
    for environment in environments:
        with monkeypatch.context() as process:
            for key in tuple(os.environ):
                if key.startswith(("JAX_", "EYVAN_JAX_")):
                    process.delenv(key)
            for key, value in environment.items():
                process.setenv(key, value)
            _multihost_site.initialize_worker_runtime()
    return arguments


def test_single_cell_gce_tpu_control_uses_metadata(monkeypatch) -> None:
    environments = _capture_submission(monkeypatch, gce_tpu=True, placement="elastic", single_cell=True)
    assert len(environments) == 2
    arguments = _jax_arguments(monkeypatch, environments)
    assert len(arguments) == 2
    assert all(argument["cluster_detection_method"] == "gcetpu" for argument in arguments)
    assert all("coordinator_address" not in argument and "num_processes" not in argument for argument in arguments)


def test_elastic_gce_tpu_gang_preserves_each_cells_metadata_discovery(monkeypatch) -> None:
    environments = _capture_submission(monkeypatch, gce_tpu=True, placement="elastic")
    assert len(environments) == 4
    for rank, environment in enumerate(environments):
        assert environment["EYVAN_DISTRIBUTED_PROCESS_COUNT"] == "4"
        assert environment["EYVAN_DISTRIBUTED_PROCESS_ID"] == str(rank)
        assert environment["EYVAN_GANG_CELL_INDEX"] == str(rank // 2)
    arguments = _jax_arguments(monkeypatch, environments)
    assert len(arguments) == 4
    assert [argument["cluster_detection_method"] for argument in arguments] == ["gcetpu"] * 4
    assert all("coordinator_address" not in argument and "num_processes" not in argument for argument in arguments)
    assert all("process_id" not in argument for argument in arguments)


@pytest.mark.parametrize("gce_tpu,placement,count", [(False, "elastic", 2), (True, "atomic", 4)])
def test_explicit_and_atomic_controls_ignore_submitted_discovery_override(
    monkeypatch, gce_tpu, placement, count
) -> None:
    environments = _capture_submission(monkeypatch, gce_tpu=gce_tpu, placement=placement)
    arguments = _jax_arguments(monkeypatch, environments)
    assert len(arguments) == 4
    assert [argument["cluster_detection_method"] for argument in arguments] == ["deactivate"] * 4
    assert [argument["num_processes"] for argument in arguments] == [count] * 4
    assert [argument["process_id"] for argument in arguments] == ([0, 1, 0, 1] if count == 2 else [0, 1, 2, 3])
    expected_coordinators = (
        ["203.0.113.10:12345"] * 2 + ["203.0.113.20:12345"] * 2 if count == 2 else ["203.0.113.10:12345"] * 4
    )
    assert [argument["coordinator_address"] for argument in arguments] == expected_coordinators
