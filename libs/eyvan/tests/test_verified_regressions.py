# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression cover for defects an adversarial verification pass found.

Every test here exercises a path the original tests for these features missed.
They are grouped by what the original test got wrong, because that is the
reusable lesson: each feature was covered by a test that would still have passed
with the feature broken.
"""

from __future__ import annotations

import pytest

from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.batch import BatchError, BatchRequest, expand
from eyvan.server.service.control_plane import _capacity_apply_payload
from eyvan.server.service.errors import ConflictError


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


def _spec(name: str = "job") -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
    )


async def _running_tree(repository: InMemoryRepository, service: ControlPlaneService):
    """Build a parent/child pair that are both RUNNING.

    The original cascade tests used only PENDING children, which cancel
    synchronously and therefore never needed a stop command at all -- so they
    would have passed with the asynchronous path entirely absent.
    """

    root, _ = await service.submit_workload(
        _principal(), org_id="acme", project_id="research", spec=_spec(), idempotency_key="root"
    )
    child, _ = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec(),
        idempotency_key="child",
        parent_workload_id=root.id,
    )
    async with repository.transaction() as transaction:
        for workload in (root, child):
            stored = await transaction.get_workload(workload.id)
            await transaction.put_workload(stored.model_copy(update={"state": WorkloadState.RUNNING}))
    return root, child


# ------------------------------------------------- cascade actually stops work


async def test_a_running_child_receives_a_stop_command_not_just_a_state_change() -> None:
    """A state change alone leaves the child holding its accelerator.

    The `workload.cancel` outbox record is the only thing the controller's
    dispatcher acts on, so without one the remote process runs to completion and
    settles SUCCEEDED -- the exact budget leak the cascade exists to prevent.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    root, child = await _running_tree(repository, service)
    async with repository.transaction() as transaction:
        before = {command.id for command in await transaction.list_outbox(limit=100)}

    await service.cancel_workload(_principal(), root.id, idempotency_key="cancel")

    async with repository.transaction() as transaction:
        issued = {
            command.aggregate_id
            for command in await transaction.list_outbox(limit=100)
            if command.id not in before and command.topic == "workload.cancel"
        }
    assert issued == {root.id, child.id}


async def test_a_cancelling_parent_cannot_gain_new_children() -> None:
    """The in-flight cascade has already committed; a new child is a permanent orphan."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    root, _child = await _running_tree(repository, service)
    await service.cancel_workload(_principal(), root.id, idempotency_key="cancel")

    async with repository.transaction() as transaction:
        assert (await transaction.get_workload(root.id)).state is WorkloadState.CANCELLING

    with pytest.raises(ConflictError, match="cannot gain new children"):
        await service.submit_workload(
            _principal(),
            org_id="acme",
            project_id="research",
            spec=_spec(),
            idempotency_key="late",
            parent_workload_id=root.id,
        )


# ------------------------------------------------- batch children stay valid


def test_a_derived_child_name_cannot_exceed_the_schema_limit() -> None:
    """model_copy(update=...) skips validation, so a legal template made an illegal child.

    Such a record persists happily and then fails to re-validate on read -- and
    the Postgres backend re-validates every row in list_workloads, which the
    controller calls unfiltered. One bad batch stops placement control-plane
    wide, not just for its tenant.
    """

    template = _spec("x" * 200)

    with pytest.raises(BatchError, match="not a valid workload"):
        expand(BatchRequest(template=template, count=3))


def test_every_expanded_task_survives_a_round_trip_through_its_own_schema() -> None:
    tasks = expand(BatchRequest(template=_spec(), count=4))

    for task in tasks:
        WorkloadDefinition.model_validate(task.model_dump())


# ------------------------------------------------- cgroup ambiguity


def test_an_unreadable_cgroup_is_ambiguous_not_empty() -> None:
    """Returning () makes a hung workload look SUCCEEDED.

    _cgroup_has_processes(...) is False for an empty tuple, and
    _status_from_properties maps False to SUCCEEDED. A permission-denied or
    namespace-hidden cgroup on a stalled multi-host collective would therefore
    be reported as a completed run.
    """

    from eyvan.worker.runtime.systemd_transient import _cgroup_has_processes, _cgroup_pids

    unreadable = "/system.slice/eyvan-does-not-exist.service"

    assert _cgroup_pids(unreadable) is None
    assert _cgroup_has_processes(unreadable) is None


# ------------------------------------------------- manual capacity is reachable


def test_the_manual_driver_reads_hosts_from_the_payload_the_controller_sends() -> None:
    """Every original test hand-built a `desired` dict no production caller produces.

    The control plane nests provider-specific fields under `provider_options`,
    so reading `hosts` from the top level made the whole driver unreachable
    through the controller regardless of how well it worked in isolation.
    """

    from eyvan.capacity.drivers.manual import ManualCapacityDriver

    cell = CellRecord(
        id="cell-a",
        provider="manual",
        region="on-prem",
        state=CellState.REGISTERING,
        capacity=ResourceVector(cpu=8),
        allocatable=ResourceVector(cpu=8),
        capacity_backend="manual",
        executor="eyvan-native",
        accelerator_type="cpu",
        capacity_class="on-demand",
        provider_options={"hosts": ["http://host-a:8100"]},
    )

    desired = _capacity_apply_payload(cell)

    assert "hosts" not in desired, "the controller nests provider fields; this test guards that assumption"
    assert ManualCapacityDriver._nested_options(desired).get("hosts") == ["http://host-a:8100"]
